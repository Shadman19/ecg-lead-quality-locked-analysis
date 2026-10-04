#!/bin/bash -l
module load container_env pytorch-gpu/2.1.0
cd ~/ecg_lock
i=$SLURM_ARRAY_TASK_ID; K=(lp lpx ae sup); F=(gauss nstdb flat clip hf spikes burst)
if [ $i -lt 12 ]; then A="--kind ${K[$((i/3))]} --seed $((i%3))"; else A="--kind lpx --seed 0 --drop ${F[$((i-12))]}"; fi
crun -p $HOME/envs/default-pytorch-gpu-2.1.0 python sqa.py $A
