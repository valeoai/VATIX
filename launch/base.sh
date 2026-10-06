#!/usr/bin/env bash
set -euo pipefail

# Local launcher.
# Usage examples:
#   bash launch/base.sh
#   bash launch/base.sh base
#   NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_ddp
#   NPROC_PER_NODE=4 bash launch/base.sh multi_gpu_fsdp

EXPERIMENT=${1:-base}
shift || true

case "${EXPERIMENT}" in
  base|multi_gpu_ddp|multi_gpu_fsdp|flagship9b_traj)
    ;;
  *)
    echo "Unsupported experiment: ${EXPERIMENT}" >&2
    echo "Allowed values: base, multi_gpu_ddp, multi_gpu_fsdp, flagship9b_traj" >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=1

if [ "${NPROC_PER_NODE}" -eq 1 ]; then
  python main.py experiment="${EXPERIMENT}" "$@"
else
  torchrun \
    --standalone \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    main.py \
    experiment="${EXPERIMENT}" \
    "$@"
fi
