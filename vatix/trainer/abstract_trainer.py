import numpy as np
import torch

import torch.optim as optim
from einops import rearrange
import math
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*FSDP.set_state_dict_type()*"
)

warnings.filterwarnings(
    "ignore",
    message=".*DTensor*"
)

warnings.filterwarnings(
    "ignore",
    message=".*FullyShardedDataParallel.full_optim_state_dict*"
)


class Trainer(object):
    """Base trainer utilities shared across training implementations."""

    vit = None
    ae = None
    optim = None
    sampler = None
    writer = None
    ema = None

    def __init__(self, args):
        """Initialize trainer state and optional TensorBoard logging.

        Args:
            args: Runtime configuration object (Hydra/OmegaConf-like namespace).
        """
        self.args = args

        # Initialize logging writer (TensorBoard)
        if self.args.is_master and not args.debug and self.args.writer_log != "":
            self.writer = SummaryWriter(log_dir=args.writer_log)

    def get_network(self, archi):
        """Build and return the network for a given architecture key.

        Args:
            archi: Architecture name or identifier.

        Returns:
            Network module selected by the subclass.
        """
        pass

    def transformer_size(self, size):
        """Map a size preset to transformer width/depth/heads.

        Args:
            size (str): Preset name such as "tiny", "base", "flagship2b", "flagship9b".

        Returns:
            tuple[int, int, int]: (hidden_dim, depth, heads).

        Notes:
            Unknown presets fall back to the "base" configuration.
        """

        if size == "pico":
            hidden_dim, depth, heads = 128, 3, 2
        elif size == "nano":
            hidden_dim, depth, heads = 192, 4, 3
        elif size == "micro":
            hidden_dim, depth, heads = 256, 6, 4
        elif size == "tiny":
            hidden_dim, depth, heads = 384, 6, 6
        elif size == "small":
            hidden_dim, depth, heads = 384, 12, 6
        elif size == "medium":
            hidden_dim, depth, heads = 512, 12, 8
        elif size == "base":
            hidden_dim, depth, heads = 768, 12, 12
        elif size == "big":
            hidden_dim, depth, heads = 896, 20, 14
        elif size == "large":
            hidden_dim, depth, heads = 1024, 24, 16
        elif size == "xlarge":
            hidden_dim, depth, heads = 1152, 28, 16
        elif size == "giant":
            hidden_dim, depth, heads = 1536, 28, 16
        elif size == "flagship2b":
            hidden_dim, depth, heads = 2048, 32, 16
        elif size == "flagship9b":
            hidden_dim, depth, heads = 4096, 32, 32
        else:
            hidden_dim, depth, heads = 768, 12, 12
            if self.args.is_master:
                print("Size of the transformer not understood, initialize a Base VIT")

        return hidden_dim, depth, heads

    def log_add_img(self, names, img, iteration, nrow=None):
        """Log image batches to TensorBoard as an 8-bit grid.

        Args:
            names (str): TensorBoard tag.
            img (torch.Tensor): Image/video tensor in [-1, 1]. Supports BCHW or BCTHW.
            iteration (int): Global step.
            nrow (int | None): Number of images per grid row.
        """
        if self.writer is None:
            return
        
        if img.dim() == 5:
            b, c, t, h, w = img.size()
            img = rearrange(img, 'b c t h w -> (b t) c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        if img.size(0) > 8:
            idx = torch.linspace(0, img.size(0) - 1, steps=8).long()
            img = img[idx]
        b, c, h, w = img.size()

        img = vutils.make_grid(img, nrow=nrow if nrow is not None else min(10, len(img)), padding=2, normalize=False)
        img = (img + 1) / 2 # Scale image from [-1,1] to [0,1]
        img = torch.clip(img * 255, 0, 255).to(torch.uint8) # Convert to 8-bit format

        self.writer.add_image(tag=names, img_tensor=img, global_step=iteration)

    def log_add_txt(self, names, txt, iteration):
        """Log text content to TensorBoard.

        Args:
            names (str): TensorBoard tag.
            txt (str): Text payload.
            iteration (int): Global step.
        """
        if self.writer is None:
            return
        self.writer.add_text(tag=names, text_string=txt, global_step=iteration)

    def log_add_scalar(self, names, scalar, iteration):
        """Log one scalar or a scalar dictionary to TensorBoard.

        Args:
            names (str): TensorBoard tag.
            scalar (float | dict): Scalar value or mapping of scalar values.
            iteration (int): Global step.
        """
        if self.writer is None:
            return
        if isinstance(scalar, dict):
            self.writer.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iteration)
        else:
            self.writer.add_scalar(tag=names, scalar_value=scalar, global_step=iteration)

    def log_add_vid(self, names, vid, iteration):
        """Log a video tensor to TensorBoard.

        Args:
            names (str): TensorBoard tag.
            vid (torch.Tensor): Video tensor in BCTHW format and [-1, 1] range.
            iteration (int): Global step.
        """
        if self.writer is None:
            return
        b, c, t, h, w = vid.size()

        vid = rearrange(vid, 'b c t h w -> b t c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        vid = (vid + 1) / 2 # Scale image from [-1,1] to [0,1]
        vid = torch.clip(vid * 255, 0, 255).to(torch.uint8)  # Convert to 8-bit format
        
        self.writer.add_video(names,
                              vid,  # Shape: (1, T, C, H, W)
                              global_step=iteration,
                              fps=1)

    def get_optim(self, net, lr, mode="AdamW", **kwargs):
        """Create an optimizer for one module or a list of modules.

        Args:
            net: Module or list of modules to optimize.
            lr (float): Base learning rate.
            mode (str): One of "adamw", "adam", "sgd", "muon".
            **kwargs: Extra optimizer keyword arguments.

        Returns:
            torch.optim.Optimizer | DualOptimizer | None: Configured optimizer.

        Notes:
            In "muon" mode, matrix-like parameters (ndim == 2) are optimized with
            Muon, and remaining trainable parameters are optimized with AdamW.
        """

        # Extract parameters from network(s)
        if isinstance(net, list):
            params = []
            for n in net:
                params += list(n.parameters())
        else:
            params = list(net.parameters())

        # Trajectory params (trajectory_embed / trajectory_aux_head / trajectory_aux_norm) are
        # zero-initialised and get weak gradients, so they get their own lr multiplier and weight decay.
        if bool(getattr(self.args, "use_trajectory_cond", False)) and mode in ("adamw", "adam", "sgd"):
            nets = net if isinstance(net, list) else [net]
            traj_params, base_params = [], []
            for n in nets:
                for name, p in n.named_parameters():
                    (traj_params if "trajectory_" in name else base_params).append(p)
            if traj_params:
                traj_lr_mult = float(getattr(self.args, "trajectory_embed_lr_mult", 1.0))
                traj_wd = float(getattr(self.args, "trajectory_weight_decay", 0.0))
                traj_group = {"params": traj_params, "weight_decay": traj_wd}
                if traj_lr_mult != 1.0:
                    traj_group["lr"] = lr * traj_lr_mult
                params = [traj_group]
                if base_params:
                    params.insert(0, {"params": base_params})
                if getattr(self.args, "is_master", False):
                    print(f"[optim] trajectory param group: {len(traj_params)} tensors at "
                          f"lr x{traj_lr_mult:g}, weight_decay={traj_wd:g}; base group: {len(base_params)} tensors")

        # Choose an optimizer type
        if mode == "adamw":
            optimizer = optim.AdamW(params, lr=lr, **kwargs)
        elif mode == "adam":
            optimizer = optim.Adam(params, lr=lr, **kwargs)
        elif mode == "sgd":
            optimizer = optim.SGD(params, lr=lr, **kwargs)
        elif mode == "muon":
            from torch.optim import Muon
            muon_lr = kwargs.pop("muon_lr", 1e-4)
            muon_momentum = kwargs.pop("muon_momentum", 0.95)

            muon_params = [p for p in params if p.ndim == 2 and p.requires_grad]
            adam_params = [p for p in params if p.ndim != 2 and p.requires_grad]

            muon_optim = Muon(muon_params, lr=muon_lr, momentum=muon_momentum) if len(muon_params) > 0 else None
            adamw_optim = optim.AdamW(adam_params, lr=lr, **kwargs) if len(adam_params) > 0 else None

            optimizer = DualOptimizer(muon_optim, adamw_optim)
        else:
            optimizer = None

        if optimizer is not None:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])       

        return optimizer


    def train_one_epoch(self):
        """Run one training epoch."""
        return

    def fit(self):
        """Run the full training loop."""
        pass

    @staticmethod
    def all_gather(obj, reduce="mean"):
        """Gather a scalar-like object from all ranks and reduce it.

        Args:
            obj: Object to gather from each process.
            reduce (str): Reduction type in {"mean", "sum", "none"}.

        Returns:
            torch.Tensor: Reduced tensor containing gathered values.
        """
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather_object(tensor_list, obj)
        obj = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            obj = obj.mean()
        elif reduce == "sum":
            obj = obj.sum()
        elif reduce == "none":
            pass
        else:
            raise NameError("reduction not known")

        return obj

    def adapt_learning_rate(self, mode=None):
        """Update optimizer learning rates according to warmup and optional decay.

        Args:
            mode (str | None):
                - None: linear warmup then constant LR.
                - "constant_then_cosine": linear warmup, constant phase, then cosine
                  decay during the last 10% of training iterations.
                - "cosine": linear warmup then cosine decay until max_iter.

        Notes:
            - Each optimizer param group is expected to store its base LR in
              ``initial_lr``. If missing, the current ``lr`` is treated as base LR.
            - A minimum multiplicative floor of 0.01 is applied for cosine decay
              branches to avoid collapsing to zero.
        """
        warmup_start_iter = int(getattr(self, "lr_warmup_start_iter", 0))
        warmup_progress_iter = max(0, int(self.args.iter) - warmup_start_iter)

        # 1) Linear warmup phase.
        if warmup_progress_iter < self.args.warm_up:
            if self.args.warm_up > 0:
                scale = warmup_progress_iter / self.args.warm_up
            else:
                scale = 1.0
        # 2) Keep LR constant, then apply cosine decay on the final 10%.
        elif (self.args.iter > (self.args.max_iter - (.1 * self.args.max_iter))) and mode=="constant_then_cosine":
            decay_step = .1 * self.args.max_iter
            decay = (self.args.iter - (self.args.max_iter - decay_step)) / decay_step
            cosine_decay = (1 + np.cos(np.pi * decay))
            scale = max(0.5 * cosine_decay, 0.01)
        # 3) Cosine decay immediately after warmup.
        elif mode=="cosine":
            progress = (self.args.iter - self.args.warm_up) / (self.args.max_iter - self.args.warm_up)
            progress = min(max(progress, 0.0), 1.0)
            scale = max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)
        # 4) Constant LR after warmup.
        else:
            scale = 1.0

        for group in self.optim.param_groups:
            base_lr = group.get('initial_lr', group['lr'])
            group['lr'] = base_lr * scale

class DualOptimizer:
    """Wrap Muon and AdamW optimizers behind a single optimizer-like interface."""

    def __init__(self, muon_optimizer=None, adamw_optimizer=None):
        """Initialize optional inner optimizers.

        Args:
            muon_optimizer: Optimizer handling matrix-like parameters.
            adamw_optimizer: Optimizer handling all remaining parameters.
        """
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer

    @property
    def param_groups(self):
        groups = []
        if self.muon_optimizer is not None:
            groups.extend(self.muon_optimizer.param_groups)
        if self.adamw_optimizer is not None:
            groups.extend(self.adamw_optimizer.param_groups)
        return groups

    def step(self, closure=None):
        loss = None
        if self.muon_optimizer is not None:
            loss = self.muon_optimizer.step(closure=closure)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step(closure=closure)
        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": None if self.muon_optimizer is None else self.muon_optimizer.state_dict(),
            "adamw": None if self.adamw_optimizer is None else self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict or "adamw" in state_dict:
            if self.muon_optimizer is not None and state_dict.get("muon") is not None:
                self.muon_optimizer.load_state_dict(state_dict["muon"])
            if self.adamw_optimizer is not None and state_dict.get("adamw") is not None:
                self.adamw_optimizer.load_state_dict(state_dict["adamw"])
            return

        if self.muon_optimizer is not None:
            self.muon_optimizer.load_state_dict(state_dict)
