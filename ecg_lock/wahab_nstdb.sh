#!/bin/bash -l
# SECONDARY recorded-artifact evaluation. Submit:  sbatch --array=0-38 wahab_nstdb.sh
# Tasks 0-38 = the 13 strat-split configurations x 3 seeds (same order as wahab_jobs.sh).
#SBATCH --job-name=ecg_nstdb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
PTBXL=${PTBXL:-"$HOME/cardio ML/ptbxl"}
RUNS=${RUNS:-/scratch/spath004/ecg_lock_runs}
BANK=${BANK:-$HOME/ecg_lock/nstdb_100hz.npy}
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
CONFIGS=(cnn_full cnn_diagonly_aug cnn_diagonly_clean cnn_diag_lead cnn_diag_rec cnn_full_leadwise
         cnn_full_cleanfrac20 cnn_full_leadwise_cleanfrac20 tx_full resnet_aug resnet_clean inception_aug inception_clean)
i=${SLURM_ARRAY_TASK_ID:-0}; c=$(( i / 3 )); s=$(( i % 3 ))
cd "$SLURM_SUBMIT_DIR"
$E python export_nstdb.py --run "$RUNS/${CONFIGS[$c]}_s$s" --ptbxl "$PTBXL" --bank "$BANK" || exit 1
