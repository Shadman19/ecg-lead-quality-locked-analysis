#!/bin/bash -l
# Round-5 SECONDARY: external replication on CPSC 2018 (see PREREG_ANALYSIS_PLAN.md, Round-5 entry).
# 4 configurations x 3 seeds = 12 tasks:  sbatch --array=0-11 wahab_cpsc.sh
# Identical recipe to wahab_jobs.sh (train.py, export.py, export_multireal.py unchanged); only the data root,
# the class list (ECG_CLASSES) and the output root differ.
#SBATCH --job-name=ecg_cpsc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
export ECG_CLASSES=NORM,AF,AVB,BBB,STC
DATA=${DATA:-$HOME/data/cpsc2018_shaped}
RUNS=${RUNS:-/scratch/spath004/ecg_cpsc_runs}
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
CONFIGS=(
  "cnn_full_leadwise_cleanfrac20 cnn diag,lead,rec leadwise 1 --clean-frac 0.2 --sigma-min 0.02"
  "cnn_diagonly_aug              cnn diag          pooled   1 "
  "cnn_diagonly_clean            cnn diag          pooled   0 "
  "cnn_full                      cnn diag,lead,rec pooled   1 "
)
SEEDS=(0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
c=$(( i / 3 )); s=${SEEDS[$(( i % 3 ))]}
read -r NAME ARCH TASKS LH AUG EXTRA <<< "${CONFIGS[$c]}"
OUT=$RUNS/${NAME}_s${s}
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$RUNS"
if [ ! -f "$OUT/train_log.json" ]; then
  $E python train.py --ptbxl "$DATA" --out "$OUT" --arch $ARCH --tasks $TASKS --lead-head $LH --augment $AUG --seed $s $EXTRA || exit 1
fi
if [ ! -f "$OUT/lock.json" ] || ! grep -q test_stress "$OUT/lock.json"; then
  $E python export.py --run "$OUT" --ptbxl "$DATA" --mc 20 --realizations 5 || exit 1
fi
$E python export_multireal.py --run "$OUT" --ptbxl "$DATA" || exit 1
echo DONE $NAME $s
