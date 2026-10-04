#!/bin/bash -l
set -e
cd ~/ecg_lock
module load container_env pytorch-gpu/2.1.0
E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
$E pip install -q wfdb scipy matplotlib
$E python -c "import torch,wfdb,sklearn,pandas;print('OK', torch.__version__)"
$E python -c "from data import *; from pathlib import Path; r=Path.home()/'cardio ML'/'ptbxl'; x=load_signals(r, load_labels(r), r/'cache_x100.npy'); print('DATA', x.shape)"
sbatch --array=0-47 wahab_jobs.sh
squeue -u spath004 | head
