#!/bin/bash -l
# SECONDARY inference-only: 5 realizations of test training-distribution corruption. sbatch --array=0-32 wahab_extra.sh
#SBATCH --job-name=ecg_multireal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
PTBXL=${PTBXL:-"$HOME/cardio ML/ptbxl"}
RUNS=${RUNS:-/scratch/spath004/ecg_lock_runs}
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
CONFIGS=(cnn_full cnn_diagonly_aug cnn_diagonly_clean cnn_diag_lead cnn_diag_rec cnn_full_leadwise
         cnn_full_cleanfrac20 cnn_full_leadwise_cleanfrac20 tx_full resnet_aug inception_aug)
i=${SLURM_ARRAY_TASK_ID:-0}; c=$(( i / 3 )); s=$(( i % 3 ))
cd "$SLURM_SUBMIT_DIR"
$E python export_multireal.py --run "$RUNS/${CONFIGS[$c]}_s$s" --ptbxl "$PTBXL" || exit 1
