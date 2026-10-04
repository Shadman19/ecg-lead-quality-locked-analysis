# Label-free per-lead ECG SQA: analysis plan (fixed 2026-09-28 BEFORE any sqa.py run)

Training data: PTB-XL folds 1-8 records with NO quality annotation (static_noise, burst_noise, baseline_drift, electrodes_problems all empty).
Simulated faults (per lead, labelled): gauss, nstdb (em/ma, first 70% of NSTDB, SNR -6..12 dB, 50% partial), flat/disconnect, clip, hf interference 30-49 Hz, spikes, burst.
Unlabelled nuisance: amplitude scaling 0.7-1.4, baseline wander (30%).
Model selection: best epoch by AUROC on a FIXED synthetic validation set (fold 9 unannotated records, all faults). No human labels, no CinC.
Runs (3 seeds each): lp (lead-preserving), lpx (lp + cross-lead Transformer), ae (label-free autoencoder baseline), sup (same lp network trained on PTB-XL human annotations folds 1-8, early stop on fold 9 annotations; supervised upper bound).
Ablation: lpx with one fault type removed (7 runs, seed 0).

PRIMARY MODEL: whichever of lp / lpx has the higher mean best synthetic-validation AUROC over its 3 seeds (label-free rule).
Scores: mean over 3 seeds of per-lead fault probability. Record score = max over leads for every method.

Endpoints (PTB-XL fold 10, real records, weak per-lead labels from static_noise | burst_noise | electrodes_problems; patient bootstrap 1,000):
E1 (primary): per-lead AUROC, PRIMARY minus the best classical SQI chosen on fold 9 labels. Success: 95% CI lower bound > 0.
E2: non-inferiority to supervised SQI-feature GBM (trained folds 1-9): delta 95% CI lower bound > -0.02.
E3: PRIMARY vs the pilot detector (cnn_full_leadwise_cleanfrac20, locked) and vs ae; per-lead AUROC.
Secondary metrics: AUPRC, sensitivity at 90% specificity, static vs burst subsets, per-lead AUROC by lead.
CinC 2011 set a (record-level; FROZEN PILOT, already seen, reported as secondary): PRIMARY vs template SQI and vs supervised 10-fold CV.
Combination fixed a priori: mean of within-dataset percentile ranks of PRIMARY and template SQI (record level on CinC; per lead on PTB-XL).
Everything else is exploratory and labelled so.
