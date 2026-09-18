"""9B command panels (left/right/straight/static) for every image in real_videos/context_frames/.

Usage, from the repository root: python launch/lumi/generate_9b.py <ema_state_dict.pt>
The EMA weights are loaded as the model (use_ema=false) so one 64 GB GPU holds a single copy.
"""
import sys
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
import torch
from hydra import compose, initialize_config_dir

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

from main import cfg_to_args  # noqa: E402
from vatix.trainer import FM  # noqa: E402
from vatix.trajectory_utils import command_trajectories  # noqa: E402

with initialize_config_dir(config_dir=str(repo / "conf"), version_base=None):
    cfg = compose(config_name="config", overrides=[
        "experiment=base", "exp_name=inference_9b", "test_only=true", "writer_log=",
        "use_trajectory_cond=true", "vit_size=flagship9b",
        "trajectory_fuse_mode=chunk_sum_rms", "traj_aux_head_norm=true",
        "resume=false", "use_ema=false", f"pretrained_ckpt={sys.argv[1]}",
        "vit_folder=./ckpt/inference_9b/", "vqgan_folder=./ckpt/wan21/", "global_bsize=1",
    ])
args = cfg_to_args(cfg)
args.device = torch.device("cuda")
args.iter = args.global_epoch = args.global_rank = 0
args.is_master, args.is_multi_gpus = True, False
args.nb_gpus = args.num_nodes = args.bsize = 1

fm = FM(args)
names = ["left", "right", "straight", "static"]
commands = command_trajectories(args.trajectory_length)
trajectory = torch.from_numpy(np.stack([commands[n] for n in names])).to(args.device)

for ctx_path in sorted((repo / "real_videos" / "context_frames").glob("*.png")):
    frame = iio.imread(ctx_path)[..., :3]
    ctx = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None, :, None].float()
    x_ctx = ((ctx / 127.5) - 1.0).repeat(len(names), 1, args.n_frames, 1, 1).to(args.device)
    torch.manual_seed(0)
    with torch.no_grad():
        sample = fm.generation_helper.generate_samples(
            x_ctx=x_ctx, latent_context=1, num_steps=50, alpha=0.0,
            trajectory=trajectory, cfg_w=3.0, shared_noise=True,
        )
    video = ((sample.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy()
    panel = np.concatenate(list(video), axis=2)  # (T, H, 4W, 3): one column per command
    imageio.mimsave(f"command_panel_9B_{ctx_path.stem}.mp4", list(panel), fps=4)
