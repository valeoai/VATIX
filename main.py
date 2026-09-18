"""Entry point for training, evaluation, and evaluation-video export."""

import os
import random
import subprocess
from argparse import Namespace
from datetime import timedelta

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    """Initialize distributed process group from torchrun or SLURM env vars.

    Returns:
        tuple[int, int, int]: (local_rank, global_rank, world_size).

    Raises:
        RuntimeError: If required distributed environment variables are missing.
    """
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
        os.environ["LOCAL_RANK"] = str(local_rank)
    elif "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ and "SLURM_NODELIST" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = max(torch.cuda.device_count(), 1)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1").strip()
        local_rank = rank % num_gpus

        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        raise RuntimeError("Distributed setup requested but missing torchrun/SLURM environment variables")

    timeout_min = int(os.environ.get("DIST_TIMEOUT_MIN", "60"))
    timeout = timedelta(minutes=max(1, timeout_min))

    torch.cuda.set_device(local_rank)
    init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    print(f"Distributed timeout set to {timeout_min} minute(s)")
    return local_rank, rank, world_size


def launch_multi_main(args):
    """Launch distributed training/evaluation on the current process group.

    Args:
        args (Namespace): Runtime configuration.
    """

    local_rank, rank, world_size = ddp_setup()

    args.device = local_rank
    args.global_rank = rank
    args.is_master = args.global_rank == 0
    args.nb_gpus = world_size
    args.bsize = args.global_bsize // args.nb_gpus
    local_gpus = max(torch.cuda.device_count(), 1)
    args.num_nodes = int(os.environ.get("SLURM_NNODES", max(1, world_size // local_gpus)))
    if args.is_master:
        dist_backend = str(getattr(args, "dist_backend", "ddp")).upper()
        print(f"{args.nb_gpus} GPU(s) found, launch {dist_backend}")
        print(f"Detected {args.num_nodes} node(s)")

    try:
        main(args)
        if args.is_master:
            print("################ END OF JOBS ########################")
    finally:
        destroy_process_group()


def main(args):
    """Run training/evaluation logic for the selected mode.

    Args:
        args (Namespace): Runtime configuration.

    Returns:
        int: Process return code.
    """
    if args.mode == "img-to-vid":
        from vatix.trainer.img2vid_trainer import FM
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    fm = FM(args)
    if args.debug:
        return 0

    if args.gen_eval_video_only:
        if args.eval_folder == "":
            raise ValueError("eval_folder must be set when gen_eval_video_only=true")
        fm.evaluation_helper.generate_eval_videos(
            num_video=args.gen_eval_num_video,
            tot_frames=args.gen_eval_tot_frames,
            num_step=args.gen_eval_num_step,
            output_dir=args.gen_eval_out_dir,
        )
        return 0

    if not args.test_only:
        fm.fit(metrics_eval=args.metrics_eval)

    # Standard evaluation preset for released benchmarks.
    if args.eval_folder != "":
        m = fm.evaluation_helper.compute_score(num_video=500, tot_frames=25, num_step=fm.args.step, compute_pr=False, use_precomputed_feat="", save_sample=True)
        if fm.args.is_master:
            fm.log_add_scalar('Metrics/FID', m["FID"], fm.args.iter)
            fm.log_add_scalar('Metrics/FVD', m["FVD"], fm.args.iter)
    
    return 0


def run(args):
    """Prepare runtime state and dispatch single-GPU or distributed execution.

    Args:
        args (Namespace): Runtime configuration.
    """
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0

    # Enable deterministic behavior when a positive seed is provided.
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    # Fail fast on every rank, before any collective, on settings that would otherwise be silently ignored.
    if (args.trajectory_fuse_mode != "chunk_sum" or args.traj_aux_head_norm) and not args.use_trajectory_cond:
        raise ValueError("trajectory_fuse_mode/traj_aux_head_norm require use_trajectory_cond=true")
    if args.pretrained_ckpt and not os.path.isfile(args.pretrained_ckpt):
        raise FileNotFoundError(f"pretrained_ckpt not found: {args.pretrained_ckpt}")

    env_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local_world_size = torch.cuda.device_count()

    # Only enter DDP/FSDP launch path when a distributed launcher is active.
    if env_world_size > 1:
        args.is_multi_gpus = True
        print(f"Distributed launch detected (WORLD_SIZE={env_world_size})")
        launch_multi_main(args)
    else:
        print(f"Single-process launch, detected {local_world_size} visible GPU(s)")
        args.global_rank = 0
        args.num_nodes = 1
        args.is_master = True
        args.is_multi_gpus = False
        args.nb_gpus = 1
        args.bsize = args.global_bsize // args.nb_gpus
        main(args)


def cfg_to_args(cfg: DictConfig) -> Namespace:
    """Convert Hydra config to a mutable argparse-like namespace.

    Args:
        cfg (DictConfig): Hydra config object.

    Returns:
        Namespace: Mutable runtime namespace.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return Namespace(**cfg_dict)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig):
    """Hydra entrypoint that builds args and starts execution."""
    args = cfg_to_args(cfg)
    run(args)

if __name__ == "__main__":
    hydra_main()

