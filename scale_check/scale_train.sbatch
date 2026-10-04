#!/bin/bash
#SBATCH --job-name=scale_tr
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/spath004/xfer/scale_train_%A_%a.out
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
module load container_env pytorch-gpu/2.1.0
cd /home/spath004/xfer
S=$((SLURM_ARRAY_TASK_ID % 3))
if [ $SLURM_ARRAY_TASK_ID -lt 3 ]; then
  NAME=scale_w4_diagonly_aug; ARGS="--tasks diag --augment 1"
else
  NAME=scale_w4_proposed; ARGS="--tasks diag,lead,rec --lead-head leadwise --augment 1 --sigma-min 0.02 --clean-frac 0.2"
fi
export ECG_WIDTH=4
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python -u scale_train.py --ptbxl "/home/spath004/cardio ML/ptbxl" --out /scratch/spath004/ecg_scale_runs/${NAME}_s${S} --arch cnn --seed $S $ARGS
