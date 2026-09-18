# Trainer for Cls-to-Img MaskGIT
import os
import time
import random
from tqdm import tqdm
from datetime import datetime
from functools import partial
from collections import deque
from contextlib import nullcontext, contextmanager

import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from vatix.trainer.abstract_trainer import Trainer
from vatix.trainer.generation_helper import GenerationHelper
from vatix.trainer.evaluation_helper import EvaluationHelper
from vatix.dataset.dataloader import get_data
from vatix.utils import Checkpointer
from vatix.network.ema import EMA
from vatix.network.video_transformer import Block, Transformer


# Aux waypoint head: loss weight, and fraction of steps with one shared noise level per sample
# (the only steps where the head gets gradient).
TRAJ_AUX_WEIGHT = 0.1


class FM(Trainer):

    def __init__(self, args):
        """Initialize training state for flow-matching image-to-video.

        Args:
            args: Namespace-like runtime config produced from Hydra settings.

        Notes:
            - Builds the transformer, optimizer, EMA state, and VAE.
            - Handles checkpoint resume for both DDP and FSDP paths.
            - Prepares helper objects for generation and evaluation.
        """
        super().__init__(args)
        print(f"Init Img-to-Vid Flow Matching on [GPU{args.global_rank}]")

        self.args = args  # Main argument see main.py
        if self.args.cameras:
            self.args.cameras = self.args.cameras.split(" ")
            self.args.nb_cam = len(self.args.cameras)
            self.args.input_size[-1] *= self.args.nb_cam

        # Load transformer (Bidirectional Transformer) and VQGAN models
        self.vit = self.get_network("vit")  # Load Masked Bidirectional Transformer
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay, mode=self.args.optimizer) 
        self.ema = EMA(self._ema_model(), decay=self.args.ema_decay) if self.args.use_ema else None
        self.lr_warmup_start_iter = 0
        
        # Set up checkpointer for saving/loading model and optimizer states
        self.checkpointer = Checkpointer(self.args.vit_folder)

        if self.args.resume and not self.checkpointer.is_empty():
            if self.args.is_master:
                print(f"Resuming from checkpoint in {self.args.vit_folder}")
            start_time = time.time()
            self.args.iter, self.args.global_epoch = self.checkpointer.load_model(self.vit)
            if self.args.use_ema:
                ema_loaded = self.checkpointer.load_ema(self.ema, self.vit)
                if self.args.is_master and not ema_loaded:
                    print("No EMA checkpoint found, using freshly initialized EMA state.")
            if not self.args.test_only and self.args.resume_load_optimizer:
                try:
                    self.checkpointer.load_optim(self.vit, self.optim)
                except Exception as e:
                    if self.args.is_master:
                        print("Cannot load optimizer state; continuing with a fresh optimizer.", e)
            elif not self.args.test_only and self.args.is_master:
                print("Skipping optimizer-state resume because resume_load_optimizer=false")

            if self.args.is_master:
                print(f"Resumed from checkpoint in {time.time() - start_time:.2f} seconds")
        elif getattr(self.args, "pretrained_ckpt", ""):
            # Weights only: iter stays 0, so warmup and schedule restart.
            self.checkpointer.load_pretrained(self.vit, self.args.pretrained_ckpt)
            if self.args.use_ema:
                self.ema = None  # free the random-init shadow, then re-seed it from the loaded weights
                self.ema = EMA(self._ema_model(), decay=self.args.ema_decay)
        
        self.ae = self.get_network(self.args.vq_type)  # Load VQGAN 

        # Set up automatic mixed precision for training efficiency
        if self.args.device != 'cpu' and self.args.dtype == "bfloat16":
            self.autocast = autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        # print logs
        if self.args.is_master:
            self.full_training_bar = None
            args_items = sorted(vars(args).items(), key=lambda x: x[0])
            key_width = max((len(k) for k, _ in args_items), default=10)
            args_str = "\n".join([f"{k:<{key_width}} : {v}" for k, v in args_items])
            print("\n=== Run Parameters ===")
            print(args_str)
            print("======================\n")
            self.log_add_txt("Parameters", args_str, self.args.iter)

        # Keep a persistent train iterator so virtual epochs do not restart mid-pass.
        self._train_iter = None
        self._train_sampler_epoch = int(self.args.global_epoch)
        self.generation_helper = GenerationHelper(self)
        self.evaluation_helper = EvaluationHelper(self, self.generation_helper)

    def _ema_model(self):
        """Return the model object that EMA should track.

        Returns:
            nn.Module: Wrapped model in single-GPU mode, or the underlying
            `.module` when distributed wrappers are active.
        """
        return self.vit.module if self.args.is_multi_gpus else self.vit

    @contextmanager
    def ema_scope(self):
        """Temporarily swap model weights with EMA weights.

        Yields:
            Context manager scope where inference uses EMA parameters.

        Notes:
            If EMA is disabled, this acts as a no-op context.
        """
        if not self.args.use_ema:
            yield
            return

        model = self._ema_model()
        self.ema.store(model)
        self.ema.copy_to(model)
        try:
            yield
        finally:
            self.ema.restore(model)

    def _reset_train_iterator(self):
        """Reset the persistent training iterator.

        Notes:
            In distributed mode, this also advances the sampler epoch to keep
            shuffling consistent across workers.
        """
        if self.args.is_multi_gpus and hasattr(self.train_data, "sampler") and self.train_data.sampler is not None:
            self.train_data.sampler.set_epoch(self._train_sampler_epoch)
        self._train_iter = iter(self.train_data)
        self._train_sampler_epoch += 1

    def _next_train_batch(self):
        """Fetch the next training batch with automatic iterator rollover.

        Returns:
            dict: Next batch yielded by the dataloader.
        """
        if self._train_iter is None:
            self._reset_train_iterator()

        while True:
            try:
                return next(self._train_iter)
            except StopIteration:
                self._reset_train_iterator()

    def _get_trajectory_condition(self, batch):
        """Extract the ego-trajectory for a batch (its shape is validated by the embedder).

        Args:
            batch (dict): Batch yielded by the dataloader.

        Returns:
            torch.Tensor | None: Waypoints `(B, trajectory_length, 2)`, or None when disabled.

        Raises:
            ValueError: If conditioning is enabled but the batch carries no trajectory.
        """
        if not getattr(self.args, "use_trajectory_cond", False):
            return None

        trajectory = batch.get("trajectory")
        if trajectory is None:
            raise ValueError(
                "Trajectory conditioning is enabled, but the current batch does not provide trajectory data. "
                "Use a trajectory-aware dataset such as data=mp4_traj."
            )
        return trajectory.to(self.args.device)

    def _sample_trajectory_keep_mask(self, batch_size):
        """Draw the per-sample trajectory dropout mask used for classifier-free guidance.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            torch.Tensor | None: Bool mask `(B,)`, or None when every sample keeps its trajectory.
        """
        keep_prob = float(getattr(self.args, "trajectory_cond_prob", 1.0))
        if not getattr(self.args, "use_trajectory_cond", False) or keep_prob >= 1.0:
            return None
        return torch.rand(batch_size, device=self.args.device) < keep_prob

    def get_network(self, archi):
        """Build and wrap a network component for training/inference.

        Args:
            archi (str): Component key, currently `vit` or `wan21`.

        Returns:
            nn.Module: Constructed module, optionally wrapped in DDP/FSDP.

        Notes:
            - Applies model compilation when enabled.
            - Applies distributed wrapping according to `dist_backend`.
        """
        if archi == "vit":
            
            if self.args.nb_cam > 1:
                raise NotImplementedError("Multi-camera support is not implemented for the transformer model.")
            
            if self.args.is_master:
                if not os.path.exists(self.args.vit_folder):
                    os.makedirs(self.args.vit_folder)
                    print(f"Folder created: {self.args.vit_folder}")

            # Define transformer architecture parameters
            hidden_dim, depth, heads = self.transformer_size(self.args.vit_size)
            model_kwargs = dict(
                input_size=self.args.input_size,
                hidden_dim=hidden_dim,
                proj=self.args.proj,
                depth=depth,
                heads=heads,
                mlp_dim=hidden_dim * 4,
                dropout=self.args.dropout,
                is_causal=self.args.is_causal,
                use_trajectory_cond=getattr(self.args, "use_trajectory_cond", False),
                trajectory_length=getattr(self.args, "trajectory_length", 25),
                use_trajectory_aux_head=getattr(self.args, "use_trajectory_cond", False),
                trajectory_fuse_mode=getattr(self.args, "trajectory_fuse_mode", "chunk_sum"),
                traj_aux_head_norm=getattr(self.args, "traj_aux_head_norm", False),
            )

            # If vit_size starts with 'flagship', instantiate directly on device to avoid CPU RAM spike
            if str(self.args.vit_size).lower().startswith("flagship"):
                if hasattr(torch, "set_default_device"):
                    previous_device = "cpu"
                    if hasattr(torch, "get_default_device"):
                        previous_device = torch.get_default_device()
                    try:
                        torch.set_default_device(self.args.device)
                        model = Transformer(**model_kwargs).to(self.args.device)
                    finally:
                        torch.set_default_device(previous_device)
                else:
                    model = Transformer(**model_kwargs)
            else:
                model = Transformer(**model_kwargs)
        
        elif archi == "wan21":
            from vatix.network.wan21.wan.modules.vae import WanVAE
           
            class WANWrapper(nn.Module):
                def __init__(self, vae_pth):
                    """Create a thin adapter around WanVAE.

                    Args:
                        vae_pth (str): Path to `wan_2.1_vae.pth`.
                    """
                    super().__init__()
                    self.model = WanVAE(vae_pth=vae_pth, dtype=torch.bfloat16)

                @torch.no_grad()
                def encode(self, x):
                    # heuristic based on value range
                    xmin = x.min()
                    xmax = x.max()

                    if xmax > 1.5:  
                        # assume [0,255]
                        x = x / 255.0
                    elif xmin < 0:
                        # assume [-1,1]
                        x = (x + 1.0) / 2.0
                    else:
                        # assume already [0,1]
                        pass
                    
                    x = torch.clamp(x, 0.0, 1.0)
                    return torch.stack(self.model.encode(x))

                @torch.no_grad()
                def decode(self, z):
                    """Decode WAN latents back to normalized video frames.

                    Args:
                        z (Tensor): Latent tensor with shape `(B, C, T, H, W)`.

                    Returns:
                        Tensor: Reconstructed video in `[-1, 1]`.
                    """
                    x = (torch.stack(self.model.decode(z)).clamp(0, 1)*2) - 1  # Ensure output is in valid range [-1,1]
                    return x
                
            return WANWrapper(self.args.vqgan_folder + "wan_2.1_vae.pth")
           
        else:
            model = None

        model = model.to(self.args.device)
        trainable_params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6
        if self.args.compile:  # Enable model compilation if using PyTorch 2.0+
            model = torch.compile(model)

        if self.args.is_multi_gpus:  # Enable multi-GPU training if available
            if self.args.dist_backend == "fsdp":
                if self.args.fsdp_activation_checkpointing:
                    non_reentrant_wrapper = partial(
                        checkpoint_wrapper,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    )
                    apply_activation_checkpointing(
                        model,
                        checkpoint_wrapper_fn=non_reentrant_wrapper,
                        check_fn=lambda submodule: isinstance(submodule, Block),
                    )

                sharding_map = {
                    "full_shard": ShardingStrategy.FULL_SHARD,
                    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                    "no_shard": ShardingStrategy.NO_SHARD,
                }
                sharding_strategy = sharding_map.get(
                    str(self.args.fsdp_sharding).lower(),
                    ShardingStrategy.FULL_SHARD,
                )

                mixed_precision = None
                if self.args.dtype == "bfloat16":
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32,
                    )

                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={Block},
                )

                model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision,
                    sharding_strategy=sharding_strategy,
                    use_orig_params=self.args.fsdp_use_orig_params,
                    limit_all_gathers=self.args.fsdp_limit_all_gathers,
                    cpu_offload=CPUOffload(offload_params=self.args.fsdp_cpu_offload),
                    device_id=self.args.device,
                    sync_module_states=True,
                )
            else:
                model = DDP(model, device_ids=[self.args.device])

        if self.args.is_master:
            print(f"Size of model {archi}: {trainable_params_m:.3f}M")

        return model

    def train_one_epoch(self, virtual_epoch=5_000):
        """Run one virtual epoch of flow-matching optimization.

        Args:
            virtual_epoch (int): Maximum number of optimization iterations to
                execute in this call (unless `max_iter` is reached first).

        Returns:
            float: Mean training loss over processed mini-batches.

        Notes:
            This method handles gradient accumulation, EMA updates, periodic
            visualization, and checkpoint save triggers.
        """
        # Flow goes from t=0 (noise) to t=1 (data)
        self.vit.train()
        cum_loss = 0.
        num_batches = 0
        traj_aux_sum = 0.
        traj_aux_count = 0
        use_traj_aux_head = bool(getattr(self.args, "use_trajectory_cond", False))
        iter_start = int(self.args.iter)
        last_update_time = time.time()
        # Deque to store loss and accuracy over a moving window
        window_loss = deque(maxlen=self.args.grad_cum)
        self.optim.zero_grad(set_to_none=True)

        while True:
            if self.args.iter >= self.args.max_iter:
                break
            if (self.args.iter - iter_start) >= virtual_epoch:
                break

            batch = self._next_train_batch()
            num_batches += 1

            # Determine whether to update gradients based on gradient accumulation steps
            update_grad = (num_batches % self.args.grad_cum) == 0
            # Adjust the learning rate with warmup and cosine decay
            self.adapt_learning_rate(self.args.lr_scheduler_mode)

            # Encode videos
            if "images" in batch: # If the batch contains raw images, encode them using the autoencoder
                video_tensor = batch["images"].to(self.args.device)
                B, C, T, H, W_total = video_tensor.shape     #  (B, C, T, H, W*nb_cam), [0, 1]
                with torch.no_grad(), self.autocast:
                    x = self.ae.encode(video_tensor.clone())

            else: # the batch containt the latents
                x = batch["latents"][:, :, :self.args.input_size[1]].to(self.args.device)
                B = x.shape[0]

            trajectory_cond = self._get_trajectory_condition(batch)
            trajectory_keep_mask = self._sample_trajectory_keep_mask(B)

            context = [random.randint(0, 1) for _ in range(B)] # randomly drop the context (first frame)
            # Shared noise level on a shared_timestep_prob fraction of steps; the aux head is supervised on those.
            used_shared_timestep = use_traj_aux_head and torch.rand(1).item() < getattr(self.args, "shared_timestep_prob", 0.1)
            z_t, e, timestep = self.flow_noising(x, context=context, mu=self.args.mu, sigma=self.args.sigma,
                                                 shared_timestep=used_shared_timestep)

            # Predicted flow
            with self.autocast:
                out = self.vit(x=z_t, ada_cond=timestep,
                               return_feat=use_traj_aux_head,
                               trajectory_cond=trajectory_cond,
                               trajectory_keep_mask=trajectory_keep_mask)
                pred, traj_aux_pred = out if use_traj_aux_head else (out, None)
                loss_flow_total = self.flow_loss(pred=pred, x=x, z_t=z_t, e=e, t=timestep, context=context)

            # Aux head loss: computed every step (DDP), scaled to zero except on shared-timestep
            # steps, where the head cannot read direction off cleaner neighbouring frames.
            loss_traj_aux = 0
            if traj_aux_pred is not None and trajectory_cond is not None:
                # Waypoints 1..T-1 reshape onto latent frames 1..; normalised by the x/y scale.
                T_lat = timestep.shape[1]
                tgt = (trajectory_cond[:, 1:].reshape(B, T_lat - 1, -1, 2)
                       / trajectory_cond.new_tensor([2.0, 10.0])).flatten(2)

                # Noise-weighted: high at t~0 (pure noise), zero at t~1 (clean).
                w = (1.0 - timestep).clamp(0, 1).pow(2)                    # (B, T_lat)
                if trajectory_keep_mask is not None:
                    w = w * trajectory_keep_mask.float().view(-1, 1)

                aux_scale = TRAJ_AUX_WEIGHT if used_shared_timestep else 0.0
                loss_traj_aux = aux_scale * ((traj_aux_pred - tgt).pow(2).mean(-1) * w[:, 1:]).mean()
                if used_shared_timestep:
                    traj_aux_sum += float(loss_traj_aux.detach())
                    traj_aux_count += 1

            loss = (loss_flow_total + loss_traj_aux) / self.args.grad_cum
            loss.backward()

            # Perform gradient update if gradient accumulation step is reached
            if update_grad:
                nn.utils.clip_grad_norm_(self.vit.parameters(), self.args.grad_clip)  # Clip gradient
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
                if self.args.use_ema:
                    self.ema.update(self._ema_model())

            cum_loss += loss.cpu().item() * self.args.grad_cum
            window_loss.append(loss.cpu().item() * self.args.grad_cum)
            
            # Logging and visualization
            if update_grad:
                is_fsdp = getattr(self.args, "dist_backend", "ddp").lower() == "fsdp"
                if self.args.is_multi_gpus:  # Synchronize logs across multiple GPUs if applicable
                    mini_batch_loss = self.all_gather(torch.tensor(window_loss).mean())
                else:
                    mini_batch_loss = torch.tensor(window_loss).mean()

                if self.args.is_master:  # Master process logs metrics
                    now = time.time()
                    elapsed = max(now - last_update_time, 1e-6)
                    speed_samples_per_sec = self.args.global_bsize / elapsed
                    last_update_time = now

                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)
                    self.log_add_scalar('Train/LossTot', mini_batch_loss, self.args.iter)
                    self.log_add_scalar('Train/SpeedSamplesPerSec', speed_samples_per_sec, self.args.iter)
                
                # Save model and visualize samples periodically
                if self.args.iter > 0 and self.args.iter % self.args.log_iter == 0 and (self.args.is_master or is_fsdp):
                    self.vit.eval()
                    with torch.no_grad():
                        with self.autocast:
                            vid_x = self.ae.decode(x[:1])
                            vid_z_t = self.ae.decode(z_t[:1])

                        if self.args.is_master:
                            self.log_add_img("Images/Reconstruction", vid_x.cpu().float(), self.args.iter, nrow=self.args.n_frames)
                            self.log_add_img("Images/Noise", vid_z_t.cpu().float(), self.args.iter, nrow=self.args.n_frames)


                        with self.ema_scope():
                            samples = self.generation_helper.generate_samples(num_steps=self.args.step, x_ctx=vid_x[:1], latent_context=1, rollout_steps=1, alpha=0.)
                            if self.args.is_master:
                                self.log_add_vid("Video/Short", samples.cpu(), self.args.iter)
                            
                            samples = self.generation_helper.generate_samples(num_steps=self.args.step, x_ctx=vid_x[:1], latent_context=2, rollout_steps=3, alpha=0.)
                            if self.args.is_master:
                                self.log_add_vid("Video/Long", samples.cpu(), self.args.iter)

                    del vid_x, vid_z_t, samples
                    torch.cuda.empty_cache()
                        
                    self.vit.train()

                # Increment global iteration counter
                self.args.iter += 1
                # Save the current model state.
                # For FSDP this must be called on all ranks because state_dict gathering is collective.
                if self.args.save_model and self.args.iter > 0 and self.args.iter % self.args.save_iter == 0:
                    if is_fsdp or self.args.is_master:
                        # start_time = time.time()
                        self.checkpointer.save(
                            self.vit,
                            self.optim,
                            current_epoch=self.args.global_epoch,
                            current_iter=self.args.iter,
                            ema=self.ema,
                        )
                        # if self.args.is_master:
                        #     print(f"Checkpoint saved at iteration {self.args.iter} in {self.args.vit_folder} in {time.time() - start_time:.2f} seconds")


                if self.args.is_master:
                   self.full_training_bar.update()

        # Averaged over aux-head steps only; the others are scaled to zero.
        if traj_aux_count > 0 and self.args.is_master:
            self.log_add_scalar('Train/LossTrajAux', traj_aux_sum / traj_aux_count, self.args.iter)

        # Return average loss for the epoch
        return cum_loss / max(1, num_batches)

    def flow_noising(self, x, context=None, mu=-0.6, sigma=1, shared_timestep=False):
        """Sample noisy interpolation points for flow matching.

        Args:
            x (Tensor): Clean latent tensor `(B, C, T, H, W)`.
            context (int | list[int] | None): Number of context frames to keep unnoised.
            mu (float): Mean of logistic-normal timestep sampling.
            sigma (float): Std of logistic-normal timestep sampling.
            shared_timestep (bool): Draw one timestep per sample and share it across frames, so
                the trajectory aux head cannot read direction off cleaner neighbouring frames.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - `z_t`: noised latent,
                - `e`: sampled Gaussian noise,
                - `t`: timesteps per frame `(B, T)`.
        """
        device = x.device
        b, c, t_dim, h, w = x.shape

        # Sample timestep from shifted distribution
        if shared_timestep:
            s = (sigma * torch.randn(b, 1, device=device) + mu).expand(b, t_dim)
        else:
            s = sigma * torch.randn(b, t_dim, device=device) + mu
        t = torch.sigmoid(s)                     # (b, t)
        t_view = t.view(b, 1, t_dim, 1, 1)       # broadcast over C,H,W

        # Sample noise
        e = torch.randn_like(x)

        # Compute noised latent
        z_t = t_view * x + (1.0 - t_view) * e

        # Apply context masking if specified
        if context is not None:
            if isinstance(context, int):
                if context > 0:
                    z_t[:, :, :context] = x[:, :, :context].clone()
                    t[:, :context] = 1  # Set timesteps of context frames to 1 (no noise)
            elif isinstance(context, (list, tuple)):
                for idx, ctx in enumerate(context):
                    z_t[idx, :, :ctx] = x[idx, :, :ctx].clone()
                    t[idx, :ctx] = 1  # Set timesteps of context frames to 1 (no noise)
            
        return z_t, e, t 

    def flow_loss(self, pred, x, z_t, e, t, context=None):
        """Compute training loss for the selected prediction and target modes.

        Args:
            pred (Tensor): Model output.
            x (Tensor): Ground-truth clean latent.
            z_t (Tensor): Noised latent at timestep `t`.
            e (Tensor): Ground-truth Gaussian noise used to create `z_t`.
            t (Tensor): Timestep tensor with shape `(B, T)`.
            context (int | list[int] | None): Frames excluded from loss.

        Returns:
            Tensor: Scalar loss tensor.
        """
        mask = torch.ones_like(x)        # 0→don't compute the loss, 1→compute the loss
        if isinstance(context, int):
            if context > 0:
                mask[:, :, :context] = 0
        elif isinstance(context, (list, tuple)):
            for idx, ctx in enumerate(context):
                mask[idx, :, :ctx] = 0
        else:
            pass  # No context masking applied
        
        mask = mask.bool() 
        # Expand t for spatial broadcasting: (B, 1, 1, 1, 1)
        t_v = t.view(t.shape[0], 1, t.shape[1], 1, 1)
        
        # Derive x_pred, v_pred, e_pred based on what the model actually outputted
        if self.args.pred_mode == "x":
            x_pred = pred
            # Formula: v = (x - z_t) / (1 - t)
            v_pred = (x_pred - z_t) / torch.clamp(1 - t_v, min=0.05)
            e_pred = (z_t - t_v * x_pred) / torch.clamp(1 - t_v, min=0.05)

        elif self.args.pred_mode == "v":
            v_pred = pred
            # Formula: x = z_t + (1 - t)v
            x_pred = z_t + (1 - t_v) * v_pred
            e_pred = (z_t - t_v * x_pred) / torch.clamp(1 - t_v, min=0.05)

        elif self.args.pred_mode == "e":
            e_pred = pred
            # Formula: v = (z_t - e) / t
            x_pred = (z_t - (1 - t_v) * e_pred) / torch.clamp(t_v, min=1e-5)
            v_pred = (x_pred - z_t) / torch.clamp(1 - t_v, min=0.05)

        # Calculate Loss based on the desired loss_mode
        if self.args.loss_mode == "x":
            loss = ((x_pred - x) ** 2)[mask].mean()

        elif self.args.loss_mode == "v":
            v = (x - z_t) / torch.clamp(1 - t_v, min=0.05)
            loss = ((v_pred - v) ** 2)[mask].mean()

        elif self.args.loss_mode == "e":
            loss = ((e_pred - e) ** 2)[mask].mean()
            
        return loss

    @torch.no_grad()
    def eval_one_epoch(self, max_iter=-1):
        """Run validation over the evaluation dataloader.

        Args:
            max_iter (int): Optional cap on validation iterations. Use `-1`
                for full pass.

        Returns:
            Tensor: Mean validation loss on current device.
        """
        
        cum_loss = 0.0
        i_bar = -1  # stays -1 when the val loader yields no batch (drop_last with a small val split)
        self.vit.eval()
        bar = tqdm(self.test_data, leave=False, dynamic_ncols=True, desc="Eval eta") if self.args.is_master else self.test_data
        with self.ema_scope():
            for i_bar, batch in enumerate(bar):
                if "images" in batch:
                    video_tensor = batch["images"].to(self.args.device)
                    B, _, _, _, _ = video_tensor.shape
                    with self.autocast:
                        x = self.ae.encode(video_tensor.clone())
                else:
                    x = batch["latents"][:, :, :self.args.input_size[1]].to(self.args.device)
                    B = x.shape[0]

                # Validation is always conditioned: no keep mask, so the trajectory is never dropped.
                trajectory_cond = self._get_trajectory_condition(batch)

                # Keep validation stochasticity aligned with train-time context masking.
                context = [random.randint(0, 1) for _ in range(B)]
                z_t, e, timestep = self.flow_noising(x, context=context, mu=self.args.mu, sigma=self.args.sigma)

                with self.autocast:
                    pred = self.vit(x=z_t, ada_cond=timestep, trajectory_cond=trajectory_cond)
                    loss_flow = self.flow_loss(pred=pred, x=x, z_t=z_t, e=e, t=timestep, context=context)

                cum_loss += loss_flow.detach().float().item()
                if max_iter > 0 and (i_bar + 1) >= max_iter:
                    break
        # Ensure train mode is restored for next optimization steps.
        self.vit.train()
        return torch.tensor(cum_loss / max(1, i_bar + 1), device=self.args.device)

    def fit(self, metrics_eval=10):
        """Execute the full training loop.

        Args:
            metrics_eval (int): Evaluation frequency in epochs for heavy metrics
                (FVD/FID path via `EvaluationHelper`).

        Notes:
            - Builds train/validation dataloaders from config.
            - Alternates train and validation phases.
            - Synchronizes losses in distributed mode.
            - Logs epoch metrics and optionally computes generation metrics.
        """

        use_trajectory_cond = getattr(self.args, "use_trajectory_cond", False)
        self.train_data, self.test_data = get_data(
                data=self.args.data, img_size=self.args.img_size, data_folder=self.args.data_folder, bsize=self.args.bsize, 
                num_workers=self.args.num_workers, is_multi_gpus=self.args.is_multi_gpus, seed=self.args.seed, n_frames=self.args.n_frames,
                cameras=self.args.cameras, data_fraction=self.args.data_fraction,
                train_list=self.args.train_list,
                val_list=self.args.val_list,
                trajectory_length=getattr(self.args, "trajectory_length", 25),
                smooth_trajectory=use_trajectory_cond,
                horizontal_flip_aug=use_trajectory_cond,
            )
        
        if self.args.is_master:
            print(f"Dataset loaded... Number of train videos: {len(self.train_data.dataset)}, Number of test videos: {len(self.test_data.dataset)}")
            print("Start training:")
            self.full_training_bar = tqdm(
                initial=self.args.iter, total=self.args.max_iter, desc="Training eta", leave=False, dynamic_ncols=True
            )

        start = time.time()

        # Main training loop across epochs
        for e in range(self.args.global_epoch, self.args.epoch + 1):
            # Stop training if the maximum number of iterations is reached
            if self.args.iter >= self.args.max_iter:
                print("End of training: reached max iterations")
                break
   
            # Synchronize dataset shuffling across multiple GPUs if applicable
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)
                self.test_data.sampler.set_epoch(e)

            # Train/Eval the model and get test loss and accuracy
            train_loss = self.train_one_epoch(virtual_epoch=self.args.virtual_epoch)
            test_loss = self.eval_one_epoch(max_iter=self.args.eval_max_iter)

            # Synchronize loss and accuracy across GPUs if applicable
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss)
                test_loss = self.all_gather(test_loss)

            # Logging and printing progress
            if self.args.is_master:
                clock_time = (time.time() - start)
                self.log_add_scalar('Epoch/Loss', {"Train": train_loss, "Eval": test_loss}, self.args.global_epoch)
                now = datetime.now()
                print(f"\r\033[KEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter},"
                      f" Train: {train_loss:.4f}, Eval: {test_loss:.4f},"
                      f" Time: {int(clock_time // 3600):.0f}:{int((clock_time % 3600) // 60):02d}:{int(clock_time % 60):02d},"
                      f" Date: {now.date()} {now.hour:02}:{now.minute:02}")


            # Compute FID/FVD/SSIM/PSNR
            if metrics_eval > 0 and e % metrics_eval == metrics_eval - 1 and self.args.eval_folder!= "":
                m = self.evaluation_helper.compute_score(num_video=self.args.gen_eval_num_video, tot_frames=self.args.gen_eval_tot_frames, num_step=self.args.gen_eval_num_step)
                if self.args.is_master:
                    self.log_add_scalar('Metrics/FID', m["FID"], self.args.iter)
                    self.log_add_scalar('Metrics/FVD', m["FVD"], self.args.iter)

            self.args.global_epoch += 1

        return
