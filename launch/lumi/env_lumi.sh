#!/bin/bash
# LUMI job environment; `source launch/lumi/env_lumi.sh` in every job. See docs/lumi.md.
export LUMI_PROJECT="${LUMI_PROJECT:-project_XXXXXXXXX}"
export VATIX_NATIVE="${VATIX_NATIVE:-/scratch/$LUMI_PROJECT/$USER/native}"

module purge
source "$VATIX_NATIVE/miniforge/bin/activate" "$VATIX_NATIVE/vatix"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=7                       # 56 allocatable cores / 8 GPUs
export PYTORCH_ROCM_ARCH=gfx90a                # MI250X
export HF_HOME=/scratch/$LUMI_PROJECT/$USER/hf_cache

# Slingshot / RCCL
export LD_LIBRARY_PATH="$VATIX_NATIVE/ofi-rccl:/opt/cray/libfabric/1.22.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_CQ_HUGETLB=1
# Do NOT set CXI_FORK_SAFE / CXI_FORK_SAFE_HP: they hang small-message collectives.

# FFmpeg libraries for torchcodec
EB=/appl/lumi/SW/LUMI-25.03/G/EB
for lib in FFmpeg/7.1.1 LAME/3.100 X11/25.03 XZ/5.6.3 bzip2/1.0.8 x264/20250619 x265/4.1 zlib/1.3.1; do
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$EB/$lib-cpeGNU-25.03/lib"
done

# MIOpen's on-disk kernel cache fails on LUMI (miopenStatusInternalError); keep it off.
# Each srun task must still create its own writable dir on node-local /tmp; the Slurm scripts do it
# inline (MIOPEN_TASK_SETUP) because /tmp of the batch node is not the task's.
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_TASK_SETUP='R=/tmp/miopen-$USER-$SLURM_JOB_ID-$SLURM_PROCID; rm -rf $R; export MIOPEN_USER_DB_PATH=$R/db MIOPEN_CUSTOM_CACHE_DIR=$R/cache; mkdir -p $R/db $R/cache'
