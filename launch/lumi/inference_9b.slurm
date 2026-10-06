#!/bin/bash
#SBATCH --job-name=vatix_9b_inference
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# 9B command panels on one GPU. From the repository root: mkdir -p logs; sbatch launch/lumi/inference_9b.slurm
source launch/lumi/env_lumi.sh
CKPT=${CKPT:-./ckpt/9B_traj/ckpt/100000/ema_state_dict.pt}

srun bash -c "$MIOPEN_TASK_SETUP; python launch/lumi/generate_9b.py $CKPT"
