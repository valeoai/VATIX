#!/usr/bin/env bash
#SBATCH --job-name=flowmatching
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# Slurm launcher.
# Choose experiment with:
#   sbatch --export=EXPERIMENT=multi_gpu_ddp launch/slurm.sh
#   sbatch --export=EXPERIMENT=multi_gpu_fsdp launch/slurm.sh
# Extra Hydra overrides:
#   sbatch --export=EXPERIMENT=multi_gpu_fsdp,HYDRA_OVERRIDES='max_iter=10000' launch/slurm.sh

EXPERIMENT=${EXPERIMENT:-multi_gpu_ddp}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}

case "${EXPERIMENT}" in
  base|multi_gpu_ddp|multi_gpu_fsdp|flagship9b_traj)
    ;;
  *)
    echo "Unsupported experiment: ${EXPERIMENT}" >&2
    echo "Allowed values: base, multi_gpu_ddp, multi_gpu_fsdp, flagship9b_traj" >&2
    exit 1
    ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
mkdir -p logs

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-120}

NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
MASTER_PORT=${MASTER_PORT:-29500}

if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^gpu:([0-9]+)$ ]]; then
  NPROC_PER_NODE=${BASH_REMATCH[1]}
elif [[ "${SLURM_GPUS_PER_NODE:-}" =~ ^([0-9]+) ]]; then
  NPROC_PER_NODE=${BASH_REMATCH[1]}
else
  NPROC_PER_NODE=1
fi

echo "[launch] experiment=${EXPERIMENT}"
echo "[launch] nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc "NODE_RANK=\${SLURM_PROCID}; \
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=${NPROC_PER_NODE} \
      --node_rank=\${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      main.py \
      experiment=${EXPERIMENT} \
      ${HYDRA_OVERRIDES}"
