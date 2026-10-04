#!/bin/bash -l
module load container_env pytorch-gpu/2.1.0
cd ~/ecg_lock
E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then $E python export_gated.py --runs /scratch/spath004/ecg_lock_runs --seed $SLURM_ARRAY_TASK_ID --ptbxl "$HOME/cardio ML/ptbxl" --nstdb-bank $HOME/ecg_lock/nstdb_100hz.npy --out $HOME/ecg_lock_results/gated; else $E python analyze_gated.py --dir $HOME/ecg_lock_results/gated; fi
