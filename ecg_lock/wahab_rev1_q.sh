#!/bin/bash -l
#SBATCH --job-name=ecg_rev1_q
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
module load container_env pytorch-gpu/2.1.0
cd "$SLURM_SUBMIT_DIR"
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python rev1_quality_metrics.py && echo Q_DONE
