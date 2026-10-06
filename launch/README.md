# Launch scripts

Minimal launch entry points:

- `launch/base.sh`: local launch (single GPU by default, multi-GPU with `NPROC_PER_NODE>1`)
- `launch/slurm.sh`: Slurm launch (single or multi-node, multi-GPU)
- `launch/lumi/`: install, environment, and Slurm scripts for LUMI (AMD GPUs), see [docs/lumi.md](../docs/lumi.md)

Supported experiments:

- `base`
- `multi_gpu_ddp`
- `multi_gpu_fsdp`
- `flagship9b_traj`

## Local

```bash
bash launch/base.sh
bash launch/base.sh base
NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_ddp
NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_fsdp
```

## Slurm

```bash
sbatch --export=EXPERIMENT=multi_gpu_ddp launch/slurm.sh
sbatch --nodes=2 --gres=gpu:4 --export=EXPERIMENT=multi_gpu_fsdp launch/slurm.sh
sbatch --export=EXPERIMENT=multi_gpu_fsdp,HYDRA_OVERRIDES='max_iter=10000 lr=5e-5' launch/slurm.sh
```
