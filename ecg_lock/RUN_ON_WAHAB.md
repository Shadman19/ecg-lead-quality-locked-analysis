# Locked rerun on Wahab (about 48 GPU jobs, roughly 10 to 25 min each)

This code replaces every number in the paper. It was smoke-tested end to end
on synthetic data (all 16 configurations: train, export, reconcile --strict,
analyze). It has **not** yet been run on PTB-XL.

```bash
# 0) once: copy ecg_lock/ to Wahab, e.g. ~/ecg_lock
# 1) environment: python with torch, wfdb, scikit-learn, pandas, scipy, matplotlib
# 2) data (skip if PTB-XL is already on Wahab):
wget -r -N -c -np -nH --cut-dirs=3 -P ~/data/ptb-xl-1.0.3 https://physionet.org/files/ptb-xl/1.0.3/
python nstdb_bank.py --out ~/data/nstdb_100hz.npy        # MIT-BIH NSTDB noise bank
# 3) cache signals once (first job does it automatically; run once to avoid 48 jobs racing):
python -c "from data import *; from pathlib import Path; r=Path('$HOME/data/ptb-xl-1.0.3'); load_signals(r, load_labels(r), r/'cache_x100.npy')"
# 4) submit (16 configs x 3 seeds):
PTBXL=~/data/ptb-xl-1.0.3 RUNS=~/ecg_lock_runs NSTDB=~/data/nstdb_100hz.npy sbatch --array=0-47 wahab_jobs.sh
# 5) after all jobs finish:
python analyze.py --runs ~/ecg_lock_runs --out ~/ecg_lock_results --bootstrap 1000
# 6) integrity check per headline run:
python ../claude_audit/reconcile_ecg_results_v1_1.py --strict \
  --val ~/ecg_lock_runs/cnn_full_s0/bundles/val_traindist_oracle.npz \
  --test ~/ecg_lock_runs/cnn_full_s0/bundles/test_traindist_oracle.npz \
  --outdir ~/ecg_lock_results/reconcile_cnn_full_s0 --target-coverage 0.9
```

Decision gate (decided in advance, before looking at test results):

* **Go for JBHI** if the learned per-lead head (any variant) beats the combined
  transparent rule on noise-corrupted leads, and either predicted-mask repair
  or the multitask model shows a paired, CI-excluding-zero gain over the
  augmentation-matched diagnosis-only CNN under corruption.
* **Otherwise** re-scope: report it honestly as a controlled benchmark and
  negative result, and target a conference or journal where that is in scope.
