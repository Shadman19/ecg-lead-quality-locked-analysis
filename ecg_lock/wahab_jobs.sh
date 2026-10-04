#!/bin/bash -l
# Locked experiment matrix for the JBHI revision. Submit on Wahab with:
#   sbatch --array=0-47 wahab_jobs.sh
# Edit the three paths below once. Each array task trains one (config, seed)
# and immediately exports its bundles; nothing reads the test set during training.
#SBATCH --job-name=ecg_lock
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

PTBXL=${PTBXL:-"$HOME/cardio ML/ptbxl"}      # confirmed location on Wahab (note the space)
RUNS=${RUNS:-/scratch/spath004/ecg_lock_runs}
NSTDB=${NSTDB:-$HOME/data/nstdb_100hz.npy}         # built once with nstdb_bank.py
# Environment: adapt to Wahab (e.g. `module load container_env pytorch-gpu` or a conda env with torch, wfdb, sklearn, pandas)
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"

CONFIGS=(
  # name                         arch         tasks          lead_head augment extra
  "cnn_full                      cnn          diag,lead,rec  pooled    1       "
  "cnn_diagonly_aug              cnn          diag           pooled    1       "
  "cnn_diagonly_clean            cnn          diag           pooled    0       "
  "cnn_diag_lead                 cnn          diag,lead      pooled    1       "
  "cnn_diag_rec                  cnn          diag,rec       pooled    1       "
  "cnn_full_leadwise             cnn          diag,lead,rec  leadwise  1       "
  "cnn_full_cleanfrac20          cnn          diag,lead,rec  pooled    1       --clean-frac 0.2"
  "cnn_full_leadwise_cleanfrac20 cnn          diag,lead,rec  leadwise  1       --clean-frac 0.2 --sigma-min 0.02"
  "tx_full                       tx           diag,lead,rec  pooled    1       "
  "resnet_aug                    resnet_wang  diag           pooled    1       "
  "resnet_clean                  resnet_wang  diag           pooled    0       "
  "inception_aug                 inception    diag           pooled    1       "
  "inception_clean               inception    diag           pooled    0       "
  "cnn_full_gss                  cnn          diag,lead,rec  pooled    1       --split gss"
  "cnn_diagonly_aug_gss          cnn          diag           pooled    1       --split gss"
  "resnet_aug_gss                resnet_wang  diag           pooled    1       --split gss"
)
SEEDS=(0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
c=$(( i / ${#SEEDS[@]} )); s=${SEEDS[$(( i % ${#SEEDS[@]} ))]}
[ $c -ge ${#CONFIGS[@]} ] && exit 0
read -r NAME ARCH TASKS LH AUG EXTRA <<< "${CONFIGS[$c]}"
OUT=$RUNS/${NAME}_s${s}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
if [ ! -f "$OUT/train_log.json" ]; then
  $E python train.py --ptbxl "$PTBXL" --out "$OUT" --arch $ARCH --tasks $TASKS --lead-head $LH --augment $AUG --seed $s $EXTRA ${EPOCHS:+--epochs $EPOCHS} ${MAXTRAIN:+--max-train $MAXTRAIN} || exit 1
fi
NS=""; [ -f "$NSTDB" ] && NS="--nstdb-bank $NSTDB"
if [ -n "$FORCE_EXPORT" ] || [ ! -f "$OUT/lock.json" ] || ! grep -q test_stress "$OUT/lock.json"; then
  $E python export.py --run "$OUT" --ptbxl "$PTBXL" --mc ${MC:-20} --realizations ${REAL:-5} $NS || exit 1
fi
echo DONE $NAME $s
# 16 configs x 3 seeds = 48 tasks -> --array=0-47
