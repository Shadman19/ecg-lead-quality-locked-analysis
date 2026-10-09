#!/bin/bash -l
# Revision 1 (Dr. Wang, Oct 7 2026). Point 1: augmentation-matched diagnosis-only CNN
# (identical corruption recipe to cnn_full_leadwise_cleanfrac20: clean_frac 0.2, sigma_min 0.02,
# min_one_masked on), seeds 0-4. Point 10: seeds 3-4 for the main comparison configs.
# sbatch --array=0-10 wahab_rev1.sh
#SBATCH --job-name=ecg_rev1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
PTBXL=${PTBXL:-"$HOME/cardio ML/ptbxl"}
RUNS=${RUNS:-/scratch/spath004/ecg_lock_runs}
NSTDB=${NSTDB:-$HOME/ecg_lock/nstdb_100hz.npy}
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
JOBS=(
"cnn_diagonly_aug_matched 0 cnn diag pooled 1 --clean-frac 0.2 --sigma-min 0.02 --min-one-masked 1"
"cnn_diagonly_aug_matched 1 cnn diag pooled 1 --clean-frac 0.2 --sigma-min 0.02 --min-one-masked 1"
"cnn_diagonly_aug_matched 2 cnn diag pooled 1 --clean-frac 0.2 --sigma-min 0.02 --min-one-masked 1"
"cnn_diagonly_aug_matched 3 cnn diag pooled 1 --clean-frac 0.2 --sigma-min 0.02 --min-one-masked 1"
"cnn_diagonly_aug_matched 4 cnn diag pooled 1 --clean-frac 0.2 --sigma-min 0.02 --min-one-masked 1"
"cnn_full_leadwise_cleanfrac20 3 cnn diag,lead,rec leadwise 1 --clean-frac 0.2 --sigma-min 0.02"
"cnn_full_leadwise_cleanfrac20 4 cnn diag,lead,rec leadwise 1 --clean-frac 0.2 --sigma-min 0.02"
"cnn_diagonly_aug 3 cnn diag pooled 1"
"cnn_diagonly_aug 4 cnn diag pooled 1"
"cnn_diagonly_clean 3 cnn diag pooled 0"
"cnn_diagonly_clean 4 cnn diag pooled 0"
)
i=${SLURM_ARRAY_TASK_ID:-0}
[ $i -ge ${#JOBS[@]} ] && exit 0
read -r NAME s ARCH TASKS LH AUG EXTRA <<< "${JOBS[$i]}"
OUT=$RUNS/${NAME}_s${s}
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
if [ ! -f "$OUT/train_log.json" ]; then
  $E python train.py --ptbxl "$PTBXL" --out "$OUT" --arch $ARCH --tasks $TASKS --lead-head $LH --augment $AUG --seed $s $EXTRA || exit 1
fi
NS=""; [ -f "$NSTDB" ] && NS="--nstdb-bank $NSTDB"
if [ ! -f "$OUT/lock.json" ] || ! grep -q test_stress "$OUT/lock.json"; then
  $E python export.py --run "$OUT" --ptbxl "$PTBXL" --mc 20 --realizations 5 $NS || exit 1
fi
[ -f "$OUT/nstdb_lock.json" ] || $E python export_nstdb.py --run "$OUT" --ptbxl "$PTBXL" --bank "$NSTDB" || exit 1
[ -f "$OUT/multireal_lock.json" ] || $E python export_multireal.py --run "$OUT" --ptbxl "$PTBXL" || exit 1
echo DONE $NAME $s
