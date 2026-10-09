#!/bin/bash -l
#SBATCH --job-name=ecg_rev1_b1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
module load container_env pytorch-gpu/2.1.0
cd "$SLURM_SUBMIT_DIR"
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python rev1_b1_fast.py
