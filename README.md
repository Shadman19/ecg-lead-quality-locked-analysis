# Per-lead ECG quality detection without human quality labels: locked analysis, code and results

Code and decision record for the manuscript "Per-Lead ECG Quality Detection Without Human Quality Labels: Cross-Dataset Validation and Limits for Diagnostic Robustness" (submitted to IEEE Access). Section numbers below (S1, S2, S8, S9) refer to its supplementary material.

## Layout

- `ecg_lock/` training, export and analysis code exactly as run (`train.py`, `models.py`, `corruption.py`, `data.py`, `export*.py`, `analyze*.py`, `final_numbers.py`, the quality-index re-implementations, the PTB-XL annotation parser, the CPSC 2018, CinC 2011 and BUT QDB pipelines), Slurm launch scripts and logs.
- `ecg_lock/PREREG_ANALYSIS_PLAN.md` the internally locked analysis plan with its dated correction log and secondary additions (reproduced in supplement S1; SHA-256 prefix 7160c8ffa8a4).
- `ecg_lock_results/` every result table of the locked analysis (`T_*.csv`, `final_numbers.txt`, `results_macros.tex`, `results_manifest.json` with the prediction-bundle hashes), including `recon_quality/` for the post hoc repair decomposition.
- `ecg_lock_results_rev1/` the revision analyses added after the locked decision (supplement S8): the augmentation-matched control, the five-seed comparisons, clean-input metrics per seed (`T_clean_test_per_seed.csv`), paired contrasts (`T_clean_contrasts.csv`, `T_stress_contrasts.csv`), stress grids (`T_stress_summary.csv`, `T_stress_raw.csv`), recorded noise (`T_nstdb_*.csv`), repair, abstention and per-lead tables, `results_manifest.json`, and `quality/` (PTB-XL annotation AUPRC and operating points, BUT QDB channel-mapping sensitivity).
- `ecg_lock_results_b10k/` the 10,000-resample bootstrap rerun (supplement S2); `ecg_cpsc_results/` the CPSC 2018 replication.
- `scale_check/` the post hoc width-scaling check: plan frozen before any run (`SCALE_PLAN.md`, `SCALE_PLAN.frozen` with timestamp and SHA-256), scripts and outputs.
- `RELEASE_MANIFEST.sha256` SHA-256 of every released file.

Prediction bundles (`*.npz`), checkpoints (`*.pt`) and cached signals (`*.npy`) are not included because of size; their SHA-256 values are listed in `results_manifest.json` and the runs' `lock.json` files, and they are available from the author on request.

## Revision analyses (added after the locked decision)

These analyses strengthen the diagnostic comparison but are not part of the locked decision rule.

- **Augmentation-matched control** (`cnn_diagonly_aug_matched`). The original augmented diagnosis-only CNN (`cnn_diagonly_aug`) used sigma_min = 0, no clean records and no forced corrupted lead. The matched control uses exactly the proposed model's corruption profile (`--sigma-min 0.02 --clean-frac 0.2 --min-one-masked 1`), the same code and the same seeds, so the auxiliary per-lead and reconstruction tasks are the only difference. Seeds 0 to 4.
- **Five seeds.** Seeds 3 and 4 were added for the proposed model (`cnn_full_leadwise_cleanfrac20`), the original augmented CNN and the clean-trained CNN (`cnn_diagonly_clean`).
- **Scripts** (in `ecg_lock/`): `train.py` (adds `--min-one-masked`; the previous version is kept as `train_pre_rev1_backup.py`), `wahab_rev1.sh` (training), `analyze_rev1.py` and `wahab_rev1_analyze.sh` (paired contrasts and stress grids), `rev1_b1_fast.py` and `wahab_rev1_b1.sh` (locked B1 endpoint against the matched control), `rev1_quality_metrics.py` and `wahab_rev1_q.sh` (AUPRC, operating points, BUT QDB mapping).
- **Main results** (macro AUROC difference, patient-clustered bootstrap): proposed minus matched, locked test input, -0.0012 (97.5% CI -0.0040 to +0.0014); over five corruption realizations, +0.0003 (95% CI -0.0020 to +0.0024); matched minus clean-trained CNN over five realizations, +0.028 (95% CI +0.025 to +0.031). The CPSC 2018 replication predates the matched control and compares against the original augmented CNN only; CPSC 2018 has no patient identifiers, so its splits and bootstrap are record-level.

## Data

All data are public: PTB-XL v1.0.3, MIT-BIH Noise Stress Test Database, PhysioNet/CinC Challenge 2011 set a, CPSC 2018 (PhysioNet/CinC Challenge 2021 training set) and BUT QDB, all from PhysioNet. No data are redistributed here.

## Reproducing

Python with PyTorch 2.1 (the runs used the ODU Wahab container `pytorch-gpu/2.1.0`), numpy, pandas, scikit-learn, wfdb. Example: `python ecg_lock/train.py --ptbxl <path-to-ptbxl> --out runs/cnn_full_leadwise_cleanfrac20_s0 --arch cnn --tasks diag,lead,rec --lead-head leadwise --sigma-min 0.02 --clean-frac 0.2 --seed 0`; launch scripts for every configuration are in `ecg_lock/wahab_*.sh` and `ecg_lock/run_*.sh`.
