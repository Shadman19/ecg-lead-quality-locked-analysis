#!/bin/bash -l
#SBATCH --job-name=ecg_rev1_an
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
cd "$SLURM_SUBMIT_DIR"
$E python analyze_rev1.py --runs /scratch/spath004/ecg_lock_runs --out $HOME/ecg_lock_results_rev1 --bootstrap 1000 || exit 1
echo ANALYSIS_DONE
