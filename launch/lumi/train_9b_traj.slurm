#!/bin/bash
#SBATCH --job-name=vatix_9b_traj
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

# 9B trajectory fine-tune, FSDP, one task per GPU. From the repository root:
#   mkdir -p logs; PRETRAINED=/path/to/unconditional_9B/model_state_dict.pt sbatch launch/lumi/train_9b_traj.slurm
# Resubmit the same script to resume: resume=true picks up the latest checkpoint and pretrained_ckpt is ignored.
source launch/lumi/env_lumi.sh
export PRETRAINED="${PRETRAINED:?set PRETRAINED=/path/to/unconditional_9B/model_state_dict.pt}"
export DATA_FOLDER="${DATA_FOLDER:-/scratch/$LUMI_PROJECT/data/my_trajectory_clips}"
export EXP_ROOT="${EXP_ROOT:-/scratch/$LUMI_PROJECT/$USER}"
export OMP_NUM_THREADS=1                                   # 4 dataloader workers already share the 7 cores
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH    # shows the transport and ring setup if init hangs
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300                # long eval steps at save_iter must not trip the watchdog

# Bind each rank to the CPU cores closest to its GPU (LUMI-G numbering is not linear). No --gpu-bind:
# each rank selects its GPU from its local rank.
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

srun --cpu-bind="$CPU_BIND" bash -c "$MIOPEN_TASK_SETUP"'
  python main.py experiment=flagship9b_traj \
    global_bsize=$SLURM_NTASKS \
    num_workers=4 \
    data_folder=$DATA_FOLDER \
    vit_folder=$EXP_ROOT/ckpt/flagship9b_traj/ \
    writer_log=$EXP_ROOT/tensorboard/flagship9b_traj/ \
    vqgan_folder=./ckpt/wan21/ \
    pretrained_ckpt=$PRETRAINED'
