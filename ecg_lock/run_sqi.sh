#!/bin/bash -l
module load container_env pytorch-gpu/2.1.0
cd ~/ecg_lock
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python sqi_baselines.py --which $1 --out $HOME/ecg_lock_results/quality --procs 32
