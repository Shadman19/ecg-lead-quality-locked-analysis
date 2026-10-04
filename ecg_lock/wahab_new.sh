#!/bin/bash -l
# Round-3 SECONDARY runs. Tasks 0-2: train + export cnn_fused_gated (seeds 0-2). Tasks 3-20: CinC 2011 inference for 6 locked configs x 3 seeds.
#   sbatch --array=0-20 wahab_new.sh      (needs $CINC bank built first with cinc2011_bank.py)
#SBATCH --job-name=ecg_round3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
PTBXL=${PTBXL:-"$HOME/cardio ML/ptbxl"}
RUNS=${RUNS:-/scratch/spath004/ecg_lock_runs}
CINC=${CINC:-$HOME/data/cinc2011_seta_100hz.npz}
NSTDB=${NSTDB:-$HOME/ecg_lock/nstdb_100hz.npy}
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
i=${SLURM_ARRAY_TASK_ID:-0}
if [ $i -lt 3 ]; then
  s=$i; OUT=$RUNS/cnn_fused_gated_s$s
  if [ ! -f "$OUT/train_log.json" ]; then
    $E python train.py --ptbxl "$PTBXL" --out "$OUT" --arch cnn --tasks diag,lead --lead-head fused --augment 1 --seed $s || exit 1
  fi
  NS=""; [ -f "$NSTDB" ] && NS="--nstdb-bank $NSTDB"
  if [ ! -f "$OUT/lock.json" ] || ! grep -q test_stress "$OUT/lock.json"; then
    $E python export.py --run "$OUT" --ptbxl "$PTBXL" --mc 20 --realizations 5 $NS || exit 1
  fi
  $E python export_multireal.py --run "$OUT" --ptbxl "$PTBXL" || exit 1
  $E python export_cinc2011.py --run "$OUT" --bank "$CINC" || exit 1
else
  CFGS=(cnn_full_leadwise_cleanfrac20 cnn_full_leadwise cnn_full cnn_full_cleanfrac20 cnn_diag_lead tx_full)
  j=$(( i - 3 )); c=$(( j / 3 )); s=$(( j % 3 ))
  $E python export_cinc2011.py --run "$RUNS/${CFGS[$c]}_s$s" --bank "$CINC" || exit 1
fi
echo DONE $i
