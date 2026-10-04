# Per-lead ECG quality detector: locked analysis, code and results

Code and decision record for the study "A Per-Lead ECG Quality Detector Trained Without Human Quality Labels: Cross-Dataset Agreement With Expert Quality Annotations and No Diagnostic Gain Under Training-Distribution Corruption".

## Layout

- `ecg_lock/` training, export and analysis code exactly as run (`train.py`, `models.py`, `corruption.py`, `data.py`, `export*.py`, `analyze*.py`, `final_numbers.py`, the quality-index re-implementations, the PTB-XL annotation parser, the CPSC 2018, CinC 2011 and BUT QDB pipelines), Slurm launch scripts and logs.
- `ecg_lock/PREREG_ANALYSIS_PLAN.md` the internally locked analysis plan with its dated correction log and secondary additions (reproduced in supplement S1; SHA-256 prefix 7160c8ffa8a4).
- `ecg_lock_results/` every result table used in the paper (`T_*.csv`, `final_numbers.txt`, `results_macros.tex`, `results_manifest.json` with the prediction-bundle hashes), including `recon_quality/` for the post hoc repair decomposition.
- `ecg_lock_results_b10k/` the 10,000-resample bootstrap rerun (supplement S2); `ecg_cpsc_results/` the CPSC 2018 replication.
- `scale_check/` the post hoc width-scaling check: plan frozen before any run (`SCALE_PLAN.md`, `SCALE_PLAN.frozen` with timestamp and SHA-256), scripts and outputs.
- `RELEASE_MANIFEST.sha256` SHA-256 of every released file.

Prediction bundles (`*.npz`), checkpoints (`*.pt`) and cached signals (`*.npy`) are not included because of size; their SHA-256 values are listed in `results_manifest.json` and the runs' `lock.json` files, and they are available from the author on request.

## Data

All data are public: PTB-XL v1.0.3, MIT-BIH Noise Stress Test Database, PhysioNet/CinC Challenge 2011 set a, CPSC 2018 (PhysioNet/CinC Challenge 2021 training set) and BUT QDB, all from PhysioNet. No data are redistributed here.

## Reproducing

Python with PyTorch 2.1 (the runs used the ODU Wahab container `pytorch-gpu/2.1.0`), numpy, pandas, scikit-learn, wfdb. Example: `python ecg_lock/train.py --ptbxl <path-to-ptbxl> --out runs/cnn_full_leadwise_cleanfrac20_s0 --arch cnn --tasks diag,lead,rec --lead-head leadwise --sigma-min 0.02 --clean-frac 0.2 --seed 0`; launch scripts for every configuration are in `ecg_lock/wahab_*.sh` and `ecg_lock/run_*.sh`.
