#!/bin/bash -l
# Round-6 SECONDARY: BUT QDB curated-label evaluation and utility analyses (PREREG_ANALYSIS_PLAN.md, Round-6 entry).
# One job, every step skips if its output exists:  sbatch wahab_butqdb.sh
#SBATCH --job-name=ecg_butqdb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
module load container_env pytorch-gpu/2.1.0; E="crun -p $HOME/envs/default-pytorch-gpu-2.1.0"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BANK=$HOME/data/butqdb_windows_100hz.npz; OUT=$HOME/ecg_lock_results/butqdb; mkdir -p "$OUT"
[ -f "$BANK" ] || $E python butqdb_bank.py --root $HOME/data/butqdb --out "$BANK" || exit 1
[ -f "$OUT/sqi_butqdb.npz" ] || $E python sqi_butqdb.py --bank "$BANK" --out "$OUT" --procs 16 || exit 1
$E python export_butqdb.py --bank "$BANK" --out "$OUT" || exit 1
$E python analyze_butqdb.py > "$OUT/analyze_butqdb.txt" 2>&1 || exit 1
$E python triage_utility.py --which ptbxl > "$OUT/triage_ptbxl.txt" 2>&1 || exit 1
ECG_CLASSES=NORM,AF,AVB,BBB,STC $E python triage_utility.py --which cpsc > "$OUT/triage_cpsc.txt" 2>&1 || exit 1
echo DONE_BUTQDB
