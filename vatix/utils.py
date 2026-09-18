import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
EMA_CHECKPOINT = "ema_state_dict.pt"

def resize_positional_embeddings(model, checkpoint_state):
    """
    Resize temporal_pos and spatial_pos embeddings if shape mismatch.
    """

    new_state = checkpoint_state.copy()

    # ---- Temporal embedding ----
    if "temporal_pos.weight" in checkpoint_state:
        old_temporal = checkpoint_state["temporal_pos.weight"]   # [T_old, D]
        new_temporal = model.temporal_pos.weight                  # [T_new, D]

        if old_temporal.shape[0] != new_temporal.shape[0]:
            print(f"Resizing temporal_pos: {old_temporal.shape[0]} -> {new_temporal.shape[0]}")

            old = old_temporal.unsqueeze(0).permute(0, 2, 1)  # [1, D, T_old]
            new = F.interpolate(old, size=new_temporal.shape[0], mode="linear", align_corners=False)
            new = new.permute(0, 2, 1).squeeze(0)             # [T_new, D]

            new_state["temporal_pos.weight"] = new


    # ---- Spatial embedding ----
    if "spatial_pos.weight" in checkpoint_state:
        old_spatial = checkpoint_state["spatial_pos.weight"]    # [N_old, D]
        new_spatial = model.spatial_pos.weight                  # [N_new, D]

        if old_spatial.shape[0] != new_spatial.shape[0]:
            print(f"Resizing spatial_pos: {old_spatial.shape[0]} -> {new_spatial.shape[0]}")
            print(old_spatial.shape, new_spatial.shape)
            D = old_spatial.shape[1]

            old_h, old_w = 20, 26
            new_h, new_w = 20, 26*4

            old = old_spatial.reshape(old_h, old_w, D)
            new = old.repeat(1, 4, 1).reshape(new_h * new_w, D)
            
            # The interpolation-based resizing is commented out as it may cause performance drop. The tile-based resizing is a simple alternative that preserves the original values without introducing interpolation artifacts.
            # old = old_spatial.reshape(1, old_h, old_w, D).permute(0, 3, 1, 2)  # [1,D,H,W]
            # new = F.interpolate(old, size=(new_h, new_w), mode="bicubic", align_corners=False)
            # new = new.permute(0, 2, 3, 1).reshape(new_h * new_w, D)


            new_state["spatial_pos.weight"] = new

    return new_state


def build_transformer_on_device(model, device, **kwargs):
    """Build transformer directly on target device when supported.

    This avoids a temporary CPU allocation spike from CPU->GPU transfer.
    """

    if hasattr(torch, "set_default_device"):
        previous_device = "cpu"
        if hasattr(torch, "get_default_device"):
            previous_device = torch.get_default_device()
        try:
            torch.set_default_device(device)
            model = model(**kwargs)
        finally:
            torch.set_default_device(previous_device)
        return model

    return model(**kwargs).to(device)


def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


class Checkpointer:
    def __init__(self, folder: str):
        self.folder = folder
        self.last_training_time = get_latest_checkpoint_folder(f"{folder}/ckpt")

    def is_empty(self):
        return self.last_training_time is None

    @staticmethod
    def _is_distributed_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _is_rank_zero(self) -> bool:
        if not self._is_distributed_initialized():
            return True
        return dist.get_rank() == 0

    def _should_broadcast_from_rank0(self) -> bool:
        if not self._is_distributed_initialized():
            return False
        return dist.get_world_size() > 1

    @staticmethod
    def _state_dict_model(model: nn.Module) -> nn.Module:
        """Use the wrapped module for DDP checkpoints to avoid key-prefix mismatch."""
        if isinstance(model, DDP):
            return model.module
        return model

    @staticmethod
    def _looks_like_plain_optimizer_state_dict(state_dict) -> bool:
        """Return True for plain ``optimizer.state_dict()`` payloads.

        Legacy DDP checkpoints may store int-indexed optimizer state that FSDP
        remapping cannot consume directly.
        """
        if not isinstance(state_dict, dict):
            return False
        if "state" not in state_dict or "param_groups" not in state_dict:
            return False
        if not isinstance(state_dict["state"], dict) or not isinstance(state_dict["param_groups"], list):
            return False
        if len(state_dict["state"]) == 0:
            return True
        return all(isinstance(k, int) for k in state_dict["state"].keys())

    @staticmethod
    def _clean_wrapper_fqn(key: str) -> str:
        """Strip compile/FSDP/activation-checkpoint wrapper segments from a state-dict key."""
        for marker in ("_orig_mod.", "_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
            key = key.replace(marker, "")
        return key

    def load_model(self, model: nn.Module):
        # Always load from the latest checkpoint (highest iteration number)
        base_ckpt_dir = f"{self.folder}/ckpt"
        latest_iter = get_latest_checkpoint_folder(base_ckpt_dir)
        if latest_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {base_ckpt_dir}")

        new_checkpoint_folder = os.path.join(base_ckpt_dir, str(latest_iter))
        last_model_checkpoint = os.path.join(new_checkpoint_folder, MODEL_CHECKPOINT)
        full_sd = torch.load(
            last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )

        iter = 0
        global_epoch = 0
        meta_path = os.path.join(new_checkpoint_folder, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            iter = meta.get("iter", 0)
            global_epoch = meta.get("global_epoch", 0)
            
        target_model = self._state_dict_model(model)

        # Warm start: an unconditional checkpoint may lack only the (zero-initialised) trajectory
        # embedder; that case loads non-strictly, any other mismatch raises. Keys are compared
        # explicitly because set_model_state_dict does not always report missing ones.
        model_keys = {self._clean_wrapper_fqn(k) for k in target_model.state_dict().keys()}
        ckpt_keys = set(full_sd.keys())

        # The training-only trajectory aux head (and its norm) is dropped when the model was built without it.
        aux_head_keys = {
            k for k in ckpt_keys - model_keys if k.startswith(("trajectory_aux_head.", "trajectory_aux_norm."))
        }
        if aux_head_keys:
            for k in aux_head_keys:
                full_sd.pop(k)
            ckpt_keys -= aux_head_keys
            print(
                f"Dropped {len(aux_head_keys)} training-only trajectory_aux_head weight(s) from "
                "the checkpoint; this model was built without the aux head."
            )

        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        warm_start = bool(missing) and not unexpected and all(
            k.startswith("trajectory_embed.") for k in missing
        )
        if (missing or unexpected) and not warm_start:
            raise RuntimeError(
                f"Checkpoint keys do not match the model: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        set_model_state_dict(
            model=target_model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=self._should_broadcast_from_rank0(),
                strict=not warm_start,
            ),
        )
        if warm_start:
            print(
                "Warm start: checkpoint has no trajectory embedder weights; left "
                f"{sorted(missing)} at their zero-init values."
            )
        return iter, global_epoch

    def load_pretrained(self, model: nn.Module, path: str):
        """Initialize model weights from a checkpoint file (raw, {"model_state_dict"} or EMA {"shadow"})."""
        target_model = self._state_dict_model(model)
        broadcast = self._should_broadcast_from_rank0()
        full_sd = {}
        if self._is_rank_zero() or not broadcast:  # other ranks receive the weights by broadcast
            full_sd = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
            full_sd = full_sd.get("model_state_dict", full_sd.get("shadow", full_sd))
            full_sd = {self._clean_wrapper_fqn(k).removeprefix("module."): v for k, v in full_sd.items()}

            model_keys = {self._clean_wrapper_fqn(k) for k in target_model.state_dict().keys()}
            unexpected = set(full_sd) - model_keys
            missing = model_keys - set(full_sd)
            # Only trajectory weights may differ: warm-starting a trajectory model from an unconditional one.
            if not all(k.startswith("trajectory_") for k in missing | unexpected):
                raise RuntimeError(
                    f"pretrained_ckpt keys do not match the model: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            for k in unexpected:
                full_sd.pop(k)
            print(f"Loaded pretrained weights from {path}; left at init: {sorted(missing)}, dropped: {sorted(unexpected)}")

        if not broadcast:
            # In-place copy; set_model_state_dict would first stage a full copy on the GPU (OOM for the 9B).
            getattr(target_model, "_orig_mod", target_model).load_state_dict(full_sd, strict=False)
            return
        set_model_state_dict(
            model=target_model,
            model_state_dict=full_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=broadcast, strict=False),
        )

    def load_optim(self, model: nn.Module, opt: torch.optim.Optimizer):
        # Always load from the latest checkpoint (highest iteration number)
        base_ckpt_dir = f"{self.folder}/ckpt"
        latest_iter = get_latest_checkpoint_folder(base_ckpt_dir)
        if latest_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {base_ckpt_dir}")
        last_optim_checkpoint = os.path.join(base_ckpt_dir, str(latest_iter), OPTIM_CHECKPOINT)
        full_sd = torch.load(
            last_optim_checkpoint, mmap=True, weights_only=False, map_location="cpu"
        )

        target_model = self._state_dict_model(model)
        try:
            set_optimizer_state_dict(
                model=target_model,
                optimizers=opt,
                optim_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=self._should_broadcast_from_rank0(),
                    strict=False,  # allow missing keys since optimizer state may have changed
                ),
            )
            return
        except Exception as e:
            if not self._looks_like_plain_optimizer_state_dict(full_sd):
                raise

            # Fallback for legacy plain optimizer checkpoints (e.g., DDP save -> FSDP load).
            opt.load_state_dict(full_sd)
            if self._is_rank_zero():
                print(
                    "Loaded optimizer from legacy plain state_dict fallback; "
                    f"distributed remapping was skipped ({type(e).__name__}: {e})."
                )

    def load_ema(self, ema, model: nn.Module):
        if ema is None:
            return False

        base_ckpt_dir = f"{self.folder}/ckpt"
        latest_iter = get_latest_checkpoint_folder(base_ckpt_dir)
        if latest_iter is None:
            return False

        ckpt_dir = os.path.join(base_ckpt_dir, str(latest_iter))
        ema_checkpoint = os.path.join(ckpt_dir, EMA_CHECKPOINT)
        if not os.path.isfile(ema_checkpoint):
            return False

        ema_state = torch.load(ema_checkpoint, map_location="cpu", weights_only=False)

        # Packed format for distributed EMA: one file with one EMA state per rank.
        if (
            isinstance(ema_state, dict)
            and ema_state.get("format") == "packed_per_rank_ema"
            and isinstance(ema_state.get("states"), list)
        ):
            states = ema_state["states"]
            if len(states) == 0:
                return False
            rank = dist.get_rank() if self._is_distributed_initialized() else 0
            rank = min(rank, len(states) - 1)
            ema_state = states[rank]

        # FSDP stores EMA as full tensors; rebuild rank-local EMA shards safely.
        if isinstance(model, FSDP) and isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
            target_model = self._state_dict_model(model)
            base_model = target_model.module if hasattr(target_model, "module") else target_model

            ema.store(base_model)
            try:
                set_model_state_dict(
                    model=target_model,
                    model_state_dict=ema_state["shadow"],
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=self._should_broadcast_from_rank0(),
                    ),
                )

                for name, p in base_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name in ema.shadow:
                        ema.shadow[name].copy_(p.detach())
            finally:
                ema.restore(base_model)

            ema.decay = ema_state.get("decay", ema.decay)
            return True

        ema.load_state_dict(ema_state, model)
        return True

    def _get_full_model_state_dict(self, model: nn.Module):
        target_model = self._state_dict_model(model)
        return get_model_state_dict(
            model=target_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    def _get_full_optimizer_state_dict(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
    ):
        target_model = self._state_dict_model(model)
        return get_optimizer_state_dict(
            model=target_model,
            optimizers=opt,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    def _get_full_ema_state_dict(self, model: nn.Module, ema):
        """Build a full, unsharded EMA state dict matching model parameter shapes.

        For FSDP/DDP, this temporarily copies EMA weights onto the live model,
        gathers a full model state dict, then restores original weights.
        """
        if ema is None:
            return None

        # Match EMA key naming: Trainer builds EMA on underlying module.
        base_model = model.module if hasattr(model, "module") else model

        ema.store(base_model)
        try:
            ema.copy_to(base_model)
            full_shadow = self._get_full_model_state_dict(model)
        finally:
            ema.restore(base_model)

        if not self._is_rank_zero():
            return None

        return {
            "decay": ema.decay,
            "shadow": full_shadow,
        }

    def save(self, model: nn.Module, optim: torch.optim.Optimizer, current_epoch: int, current_iter: int, ema=None):
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        ema_state_dict = self._get_full_ema_state_dict(model, ema) if ema is not None else None
        if self._is_rank_zero():
            # Use iteration as the checkpoint folder name
            new_checkpoint_folder = f"{self.folder}/ckpt/{current_iter}"
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)

            meta = {
                "iter": current_iter,
                "global_epoch": current_epoch
            }
            with open(os.path.join(new_checkpoint_folder, "meta.json"), "w") as f:
                json.dump(meta, f)

        # Save EMA as a full, unsharded state dict on rank0.
        if ema_state_dict is not None:
            new_checkpoint_folder = f"{self.folder}/ckpt/{current_iter}"
            new_ema_checkpoint = os.path.join(new_checkpoint_folder, EMA_CHECKPOINT)

            if self._is_rank_zero():
                torch.save(ema_state_dict, new_ema_checkpoint)
