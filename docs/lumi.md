# Running VATIX on LUMI

[LUMI](https://www.lumi-supercomputer.eu/) GPU nodes (`LUMI-G`) carry 4 AMD MI250X cards, exposed
as 8 GPUs of 64 GB each, with ROCm instead of CUDA. The 9B trajectory-conditioned model
(`9B_traj`) was fine-tuned on LUMI: 16 nodes × 8 GPUs (128 GPUs), FSDP, 100k iterations from the
unconditional 9B model.

Everything lives in `launch/lumi/`:

| File | Purpose |
| --- | --- |
| `lumi_install.sh` | one-time native install (login node) |
| `env_lumi.sh` | environment to `source` in every job |
| `inference_9b.slurm` + `generate_9b.py` | 9B command panels on 1 GPU (`dev-g`) |
| `train_9b_traj.slurm` | 9B trajectory fine-tune on 16 nodes (`standard-g`) |

Set `LUMI_PROJECT` (or edit the `project_XXXXXXXXX` placeholders) and run everything from the
repository root.

## 1. Install (once)

```bash
LUMI_PROJECT=project_XXXXXXXXX bash launch/lumi/lumi_install.sh
```

The install is native (no container): inside the LUMI PyTorch container the Slingshot RCCL plugin
cannot load, so multi-node collectives silently fall back to TCP. The script creates a Python 3.12
env under `/scratch/$LUMI_PROJECT/$USER/native`, installs `torch 2.5.1+rocm6.2` and the project
dependencies, copies MIOpen's precompiled kernel database and the RCCL plugin out of the container,
and downloads the WAN VAE to `./ckpt/wan21/`.

## 2. Environment (every job)

`source launch/lumi/env_lumi.sh` activates the env and sets the Slingshot/RCCL, FFmpeg, and MIOpen
variables. MIOpen's on-disk cache is disabled (it fails on LUMI); each `srun` task still creates its
own directory on node-local `/tmp` via `$MIOPEN_TASK_SETUP`, as the Slurm scripts do.

Interactive session:

```bash
srun --account=$LUMI_PROJECT --partition=dev-g --nodes=1 --gpus-per-node=1 --cpus-per-task=7 \
  --mem=120G --time=01:00:00 --pty bash
source launch/lumi/env_lumi.sh
```

## 3. Inference (1 GPU)

The 9B model fits on one 64 GB GPU when the EMA weights are loaded directly as the model
(`use_ema=false pretrained_ckpt=.../ema_state_dict.pt`).

```bash
huggingface-cli download llvictorll/Vatix 9B_traj/ckpt/100000/ema_state_dict.pt --repo-type model --local-dir ./ckpt
mkdir -p logs
sbatch launch/lumi/inference_9b.slurm
```

`generate_9b.py` writes `command_panel_9B_<name>.mp4` (`left`, `right`, `straight`, `static`,
`cfg_w=3`) for every image in `real_videos/context_frames/`. Loading takes about 5 minutes, then
about 5 minutes per context frame.

## 4. Training (FSDP, multi-node)

`experiment=flagship9b_traj` holds the 9B fine-tuning recipe. The script runs one task per GPU;
`main.py` builds the process group from the Slurm variables, so no `torchrun` is needed.

```bash
mkdir -p logs
PRETRAINED=/path/to/unconditional_9B/model_state_dict.pt \
DATA_FOLDER=/scratch/$LUMI_PROJECT/data/my_trajectory_clips \
sbatch launch/lumi/train_9b_traj.slurm
```

Notes:

- `global_bsize` is set to the number of GPUs (one clip per GPU); dataloader workers are 4 and
  `OMP_NUM_THREADS=1`, since each rank has 7 cores.
- Ranks are bound to the cores nearest their GPU (`--cpu-bind` mask); do not add `--gpu-bind`.
- To continue after the time limit, resubmit the same script: `resume=true` picks up the latest
  checkpoint and `pretrained_ckpt` is ignored. A 9B checkpoint (model, EMA, optimizer) is about
  152 GB, so keep `EXP_ROOT` on `/scratch`.
- Process-group initialization on 128 GPUs can take 15-20 minutes with no output; the default
  `DIST_TIMEOUT_MIN=60` covers it.
- Shake down on `dev-g` first (`--nodes=2 --time=3:00:00`, small `max_iter`).
