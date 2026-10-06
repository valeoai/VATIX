import numpy as np
from tqdm import tqdm
from pathlib import Path
import imageio.v2 as imageio
from einops import rearrange

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from vatix.dataset.dataloader import get_data
from vatix.metrics.inception_metrics import MultiInceptionMetrics
from vatix.metrics.pixel_metrics import PixelMetrics


class EvaluationHelper:
    """Utility class for evaluation-time metric computation and sample export."""

    def __init__(self, trainer, generation_helper):
        """Bind trainer state and generation helper used during evaluation.

        Args:
            trainer: Trainer instance exposing args, ema_scope and model state.
            generation_helper: Helper exposing ``generate_samples``.
        """
        self.fm = trainer
        self.generation_helper = generation_helper

    @torch.no_grad()
    def compute_score(self, num_video=1_000, num_step=50, tot_frames=16, save_exemple=False, compute_pr=False, use_precomputed_feat="", save_sample=False, cfg_w=1.0):
        """Compute video/image/pixel metrics on generated validation rollouts.

        Args:
            num_video (int): Target number of generated videos used for metrics.
            num_step (int): Number of denoising/sampling steps per rollout.
            tot_frames (int): Number of frames evaluated per sample.
            save_exemple (bool): If True, save computed metric dict to ``./results``.
            compute_pr (bool): If True, enable manifold-based precision/recall metrics.
            use_precomputed_feat (str): Optional path/key to reuse precomputed features.
            save_sample (bool): If True, save one side-by-side generated/real sample video.
            cfg_w (float): Classifier-free guidance weight; the unconditional pass drops the clean
                context frame (and the trajectory, given with ``data=mp4_traj`` on a
                trajectory-conditioned model). <= 1 disables guidance.

        Returns:
            dict[str, float]: Rounded metric values aggregated from all metric modules.

        Notes:
            - Runs under ``ema_scope`` and ``torch.inference_mode``.
            - Real samples are remapped from [0, 255] to [-1, 1] before metric updates.
            - In multi-camera mode, tensors are reshaped to evaluate each camera stream.
        """
        video_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="i3d",
            use_precomputed_feat=use_precomputed_feat,
        )

        images_metrics = MultiInceptionMetrics(
            device=self.fm.args.device,
            compute_manifold=compute_pr,
            num_inception_chunks=10,
            manifold_k=3,
            model="inception",
            use_precomputed_feat=use_precomputed_feat,
        )

        pixel_metrics = PixelMetrics(device=self.fm.args.device)

        # Pixels are needed: latent datasets (e.g. natix_feat) fall back to their raw-video variant,
        # while mp4_traj is kept so that the clips come with their trajectories.
        data = self.fm.args.data if self.fm.args.data == "mp4_traj" else self.fm.args.data.split("_")[0]
        use_traj = bool(getattr(self.fm.args, "use_trajectory_cond", False))
        split = getattr(self.fm.args, "eval_split", "val")
        loaders = get_data(
            data=data,
            img_size=self.fm.args.img_size,
            n_frames=tot_frames,
            seed=self.fm.args.seed,
            data_folder=self.fm.args.eval_folder,
            cameras=self.fm.args.cameras,
            bsize=self.fm.args.bsize,
            num_workers=max(2, self.fm.args.bsize),
            is_multi_gpus=self.fm.args.is_multi_gpus,
            trajectory_length=getattr(self.fm.args, "trajectory_length", 25),
            smooth_trajectory=use_traj,  # same SavGol smoothing as trajectory-conditioned training
        )
        if split == "all":
            # Every clip of eval_folder (both splits), in a seeded random order.
            dataset = ConcatDataset([loaders[0].dataset, loaders[1].dataset])
            seed = max(int(self.fm.args.seed), 0)
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if self.fm.args.is_multi_gpus else None
            eval_data = DataLoader(dataset, batch_size=self.fm.args.bsize, shuffle=sampler is None, sampler=sampler,
                                   generator=torch.Generator().manual_seed(seed),
                                   num_workers=max(2, self.fm.args.bsize), drop_last=True)
        elif split in ("train", "val"):
            eval_data = loaders[0] if split == "train" else loaders[1]
        else:
            raise ValueError(f"eval_split must be 'val', 'train' or 'all', got {split!r}")
        if self.fm.args.is_master:
            traj = "GT trajectory" if use_traj and data == "mp4_traj" else "no trajectory"
            guidance = (f"cfg {cfg_w:g} (unconditional pass drops the context frame and the trajectory)"
                        if cfg_w > 1 else "no guidance")
            mode = f"{traj}, {guidance}"
            print(f"Metrics evaluation on {self.fm.args.eval_folder} ({data}, split {split}, "
                  f"{len(eval_data.dataset)} clips): {mode}")

        # Per-rank target after accounting for distributed setup and camera packing.
        total_videos = num_video // (self.fm.args.nb_gpus * self.fm.args.nb_cam)
        bar = tqdm(total=total_videos, leave=False, desc="Metrics Evaluation") if self.fm.args.is_master else None
        rollout_steps = (tot_frames - 1) // self.fm.args.n_frames + 1
        with self.fm.ema_scope(), torch.inference_mode():
            for i, batch in enumerate(eval_data):
                real_sample = batch["images"].to(self.fm.args.device)
                b, c, t, h, w_total = real_sample.size()
                w = w_total // self.fm.args.nb_cam
                trajectory = batch["trajectory"].to(self.fm.args.device) if use_traj and "trajectory" in batch else None

                gen_sample = self.generation_helper.generate_samples(
                    x_ctx=real_sample,
                    nb_video=b,
                    latent_context=1,
                    num_steps=num_step,
                    alpha=0.0,
                    scheduler_mode=self.fm.args.scheduler_mode,
                    rollout_steps=rollout_steps,
                    trajectory=trajectory,
                    cfg_w=cfg_w,
                )

                # Keep only the requested horizon and flatten camera dimension for metrics.
                gen_sample = gen_sample[:, :, :tot_frames, :, :]
                gen_sample = rearrange(gen_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                gen_sample = torch.clamp(gen_sample, -1, 1).contiguous().float()

                real_sample = rearrange(real_sample, "b c t h (w cam) -> (b cam) t c h w", b=b, c=c, t=tot_frames, h=h, w=w, cam=self.fm.args.nb_cam)
                real_sample = torch.clamp(((real_sample.float() / 255.0) * 2) - 1, -1, 1).contiguous().float()

                # Drop the context frame: it is the VAE reconstruction of the real frame, not generated.
                gen_eval, real_eval = gen_sample[:, 1:].contiguous(), real_sample[:, 1:].contiguous()

                video_metrics.update(gen_eval, image_type="fake")
                video_metrics.update(real_eval, image_type="real")

                images_metrics.update(gen_eval, image_type="fake")
                images_metrics.update(real_eval, image_type="real")

                pixel_metrics.update(gen_eval, real_eval)

                if save_sample and i == 0 and self.fm.args.is_master:
                    video = torch.cat([gen_sample[0], real_sample[0]], dim=-1)
                    video = video.permute(0, 2, 3, 1).cpu().numpy()
                    video = ((video + 1) * 127.5).astype(np.uint8)
                    Path("./saved_video").mkdir(parents=True, exist_ok=True)
                    imageio.mimsave("./saved_video/generated_video.mp4", video, fps=10)

                if self.fm.args.is_master:
                    bar.update(b)

                if video_metrics.count >= total_videos:
                    break

        video_m = video_metrics.compute()
        images_m = images_metrics.compute()
        pixel_m = pixel_metrics.compute()

        metrics = {**video_m, **images_m, **pixel_m}
        metrics = {f"{k}": round(v, 4) for k, v in metrics.items()}

        if self.fm.args.is_master:
            bar.close()
            print(metrics)
            if save_exemple:
                name = str(self.fm.sampler).replace(" ", "_").replace(",", "").replace(":", "")
                with open(f"./results/" + name, "w") as file:
                    file.write(str(metrics))

        # Drop references before empty_cache to release GPU tensors promptly.
        batch = real_sample = gen_sample = gen_eval = real_eval = video = None
        video_m = images_m = pixel_m = None
        eval_data = video_metrics = images_metrics = pixel_metrics = None

        if self.fm.args.device != "cpu":
            torch.cuda.empty_cache()

        return metrics

    @torch.no_grad()
    def generate_eval_videos(self, num_video=500, num_step=50, tot_frames=25, output_dir="./saved_video/eval_videos", cfg_w=1.0):
        """Generate validation videos and save them to disk without metrics.

        Args:
            num_video (int): Total number of videos to generate.
            num_step (int): Number of denoising/sampling steps per rollout.
            tot_frames (int): Number of frames produced per video.
            output_dir (str): Destination directory for generated ``.mp4`` files.
            cfg_w (float): Classifier-free guidance weight; the unconditional pass drops the clean
                context frame. <= 1 disables guidance.

        Returns:
            dict[str, int | str]: Number of saved videos and output directory path.

        Notes:
            - In multi-GPU runs, each rank writes unique filenames with a rank prefix.
            - ``num_video`` is split approximately across ranks using ceil division.
        """
        eval_data = get_data(
            data=self.fm.args.data.split("_")[0],
            img_size=self.fm.args.img_size,
            n_frames=tot_frames,
            seed=self.fm.args.seed,
            data_folder=self.fm.args.eval_folder,
            cameras=self.fm.args.cameras,
            bsize=self.fm.args.bsize,
            num_workers=max(2, self.fm.args.bsize),
            is_multi_gpus=self.fm.args.is_multi_gpus,
        )[1]

        # Keep each rank writing unique files; aggregate target is approximate in multi-GPU.
        rank = int(getattr(self.fm.args, "global_rank", 0))
        world_size = max(1, int(getattr(self.fm.args, "nb_gpus", 1)))
        per_rank_target = max(1, int(np.ceil(num_video / world_size)))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        bar = (
            tqdm(total=per_rank_target, leave=False, desc="Generating eval videos")
            if self.fm.args.is_master
            else None
        )

        rollout_steps = (tot_frames - 1) // self.fm.args.n_frames + 1
        with self.fm.ema_scope(), torch.inference_mode():
            for batch in eval_data:
                real_sample = batch["images"].to(self.fm.args.device)
                b, c, t, h, w_total = real_sample.size()
                w = w_total // self.fm.args.nb_cam

                gen_sample = self.generation_helper.generate_samples(
                    x_ctx=real_sample,
                    nb_video=b,
                    latent_context=1,
                    num_steps=num_step,
                    alpha=0.0,
                    scheduler_mode=self.fm.args.scheduler_mode,
                    rollout_steps=rollout_steps,
                    cfg_w=cfg_w,
                )

                gen_sample = gen_sample[:, :, :tot_frames, :, :]
                gen_sample = rearrange(
                    gen_sample,
                    "b c t h (w cam) -> (b cam) t c h w",
                    b=b,
                    c=c,
                    t=tot_frames,
                    h=h,
                    w=w,
                    cam=self.fm.args.nb_cam,
                )
                gen_sample = torch.clamp(gen_sample, -1, 1).contiguous().float()

                for k in range(gen_sample.shape[0]):
                    if saved >= per_rank_target:
                        break
                    video = gen_sample[k].permute(0, 2, 3, 1).cpu().numpy()
                    video = ((video + 1.0) * 127.5).astype(np.uint8)
                    out_path = out_dir / f"rank{rank:02d}_video_{saved:05d}.mp4"
                    imageio.mimsave(str(out_path), video, fps=9)
                    saved += 1
                    if bar is not None:
                        bar.update(1)

                if saved >= per_rank_target:
                    break

        if bar is not None:
            bar.close()
            print(f"Saved {saved} generated validation videos to {out_dir}")

        if self.fm.args.device != "cpu":
            torch.cuda.empty_cache()

        return {"saved_videos": int(saved), "output_dir": str(out_dir)}

