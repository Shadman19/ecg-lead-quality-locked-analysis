#!/bin/bash -l
module load container_env pytorch-gpu/2.1.0
cd ~/ecg_lock
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python compare_quality.py
