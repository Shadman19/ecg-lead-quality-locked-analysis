#!/bin/bash
#SBATCH --job-name=scale_ev
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/spath004/xfer/scale_eval_%j.out
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
module load container_env pytorch-gpu/2.1.0
cd /home/spath004/xfer
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python -u scale_eval.py
