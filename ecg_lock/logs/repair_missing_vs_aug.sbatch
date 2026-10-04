#!/bin/bash
#SBATCH --job-name=repair_va
#SBATCH --partition=timed-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/spath004/xfer/repair_missing_vs_aug_%j.out
module load container_env pytorch-gpu/2.1.0
cd /home/spath004/xfer
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python -u repair_missing_vs_aug.py
