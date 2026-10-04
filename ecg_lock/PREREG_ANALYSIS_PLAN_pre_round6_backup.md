# [ECG] Pre-registered analysis plan (frozen 2026-09-27, BEFORE any test result was read)

Agreed between Claude and ChatGPT ("[ECG] Lead-Aware ECG, IEEE JBHI").
Only seed-0 cnn_full VALIDATION numbers had been seen (best epoch 23, clean val macro AUROC 0.9317).

## Proposed model (fixed now)
`cnn_full_leadwise_cleanfrac20`: shared CNN encoder, diagnosis + lead-preserving per-lead head + reconstruction,
20% clean training records, noise floor sigma_min = 0.02. `cnn_full` (the original design) is reported as a secondary configuration.
All other configurations are descriptive. No "best variant" is chosen after test.

## Evaluation corruption
Identical for every model: test "train-distribution" corruption with sigma_min = 0 and a fixed seed (export.py).
Stress grids use condition-seeded realizations, identical across models.

## Primary endpoints (three seeds; patients bootstrapped as clusters jointly across seeds and models; 1,000 resamples; per-seed deltas reported)
- **A (required):** macro average of the 12 lead-specific AUROCs (noise-corrupted vs intact leads of that lead; pooled and per-lead reported secondarily), learned head minus the combined transparent rule
  (max percentile rank of low variance and high first-difference energy). Pass if the 95% CI lower bound is > 0.
- **B1:** macro AUROC under test train-distribution corruption, proposed minus `cnn_diagonly_aug`. Pass if the 97.5% CI lower bound is > 0.
- **B2:** predicted-mask repair (tau_mask = max Youden J on validation), macro AUROC post minus pre. Pass if the 97.5% CI lower bound is > 0.
- **Decision:** GO if A passes AND (B1 or B2) passes; B1/B2 use 97.5% intervals (Bonferroni for "or"). Otherwise RE-SCOPE.

## Mandatory sanity report (claim gate)
On clean test input: mean synthetic-intact score and the fraction of clean leads flagged at tau_mask, for the proposed model and cnn_full. If clean-lead behavior is inconsistent with the validation-fixed threshold, the head is NOT described as a usable integrity/quality screen, even if A and B pass.

## Correction log (pre-results)
2026-09-27: export.py used each model's training sigma_min for evaluation corruption; fixed so all models share identical evaluation corruption (sigma_min=0). Affected runs cnn_full_leadwise_cleanfrac20_s{0,1,2} re-exported (validation first, tau_mask re-selected on validation; model selection used clean validation and is unaffected). Old bundles preserved as bundles_superseded_unread/ with their lock.json hashes; never opened. analyze.py asserts identical eval inputs (mask, type, sigma, BW flag, labels, patient IDs, corruption seed) and val/test patient disjointness.
2026-09-27 (correction to the entry above): the re-export (job 6849010) finished at 05:09, BEFORE the "superseded" copy was made at 05:23. The pre-fix bundles were therefore overwritten and are NOT preserved; the "_superseded_unread" copies were duplicates of the corrected bundles and were deleted. The pre-fix bundles were never analyzed. An analysis job with the pre-v3 analyze.py (6849011) was cancelled after writing partial test outputs (figures/, T_clean_test.csv); these were moved unread to ecg_lock_results_partial_v2_UNREAD and are not used. The pre-registered analysis runs only with analyze.py v3 (sha 35dff725006e).

## Abstention (secondary)
Selective risk = label-wise error rate (Hamming) at t=0.5. AURC plus risk at fixed coverage 0.8 and 0.9, comparing
uncertainty-only, intact-score-only, combined (mean rank), and random. The per-lead score adds abstention value only if it beats uncertainty-only.

## Scope
Even a GO is a controlled-corruption PTB-XL result, not evidence of real-artifact or clinical robustness. NSTDB is not claimed unless the bank is built and evaluated.

## Post-hoc analyses (added AFTER the pre-registered decision; labelled post hoc everywhere)
2026-09-27: Decision was RE-SCOPE (A pass; B1, B2 fail). posthoc_rules.py adds stronger transparent baselines for per-lead noise detection (per-lead normalized HF ratio, within-record HF contrast, validation-fitted logistic regression on transparent features), motivated by the observation that the pre-registered combined rule was near chance on noise leads. These do not change the decision.
2026-09-27: SECONDARY recorded-artifact analysis (export_nstdb.py, analyze_nstdb.py, wahab_nstdb.sh). MIT-BIH NSTDB em/ma/bw added to 1/3/6 leads at 12/6/0 dB on the strat test set, one condition-seeded realization shared by all models. Writes new bundles test_nstdb_v1.npz only; locked bundles and lock.json are untouched; best.pt hash verified against lock.json. Reported as secondary, not part of the decision.
2026-09-27: SECONDARY inference-only analyses (export_multireal.py, analyze_extra.py, wahab_extra.sh): five condition-seeded realizations of the test training-distribution corruption for 11 strat configurations, giving paired model contrasts and oracle/predicted repair effects with patient-clustered CIs; plus per-superclass clean AUROC/AUPRC from the locked test_clean bundles. Writes new files only (test_traindist_multireal_v1.npz); locked bundles untouched; checkpoints hash-verified.
Disclosure: an earlier, non-prespecified single-seed analysis of this model family (AIoT 2026 submission) used the GroupShuffleSplit partition that our gss sensitivity arm reproduces; its results were seen before this plan was written.
2026-09-27: analyze_extra.py also writes T_perclass_corrupted.csv (per-superclass AUROC/AUPRC under the five-realization training-distribution corruption), requested in review. SECONDARY. Numerical-precision sensitivity: the locked analyze.py is re-run unchanged with --bootstrap 10000 into a separate output directory; the decision remains the one from the 1,000-resample locked run.

## Round-3 additions (2026-09-28), fixed BEFORE any of these runs or outputs exist. Both are SECONDARY; the original decision stands.
A) Real acquisition quality (PhysioNet/CinC Challenge 2011 set-a; 12-lead mobile-phone ECGs labelled acceptable/unacceptable by annotators; inference only with the locked checkpoints).
   Endpoint: record-level AUROC for 'unacceptable', learned score = max over leads of (1 - intact score), for the proposed model (3 seeds, mean);
   compared with the prespecified combined rule aggregated by max over leads (delta and 95% CI, 1,000 record bootstrap resamples).
   Also reported: variance and HF rules alone, mean-over-leads aggregation, other per-lead-head configurations. No threshold is tuned on CinC data.
B) Quality-gated fusion model cnn_fused_gated (new configuration, 3 seeds): the lead-preserving head's per-lead features, weighted by the predicted
   intact probability, are concatenated with the pooled encoder features as input to the diagnosis head; tasks diag+lead (no decoder), so its
   training corruption is identical to cnn_diagonly_aug and cnn_diag_lead. Same training recipe, model selection and export as the locked runs.
   Primary contrast: five-realization training-distribution corrupted macro AUROC, cnn_fused_gated minus cnn_diagonly_aug, 95% patient-clustered CI.
   An improvement is claimed only if the lower bound is > 0. Secondary: the same contrast vs cnn_diag_lead and vs the proposed model; clean AUROC;
   stress k=8, k=10, noise sigma 0.5, BW 1.0 (realization-averaged predictions); NSTDB em+ma mean over 18 conditions.
   Code: models.py (fused head; existing checkpoints load unchanged, models_v3_locked.py kept), cinc2011_bank.py, export_cinc2011.py, analyze_new.py, wahab_new.sh.
## Round-4 addition (2026-09-28), fixed BEFORE any run or output exists. SECONDARY; the original decision stands.
Detector-gated lead masking (export_gated.py, analyze_gated.py; inference only, no training, no tuning on test data).
For seed s: the frozen lead-preserving detector of cnn_full_leadwise_cleanfrac20_s{s} scores each lead; leads with intact score < tau are set
to zero; the frozen classifier cnn_diagonly_aug_s{s} is applied to the gated input ("gated") and to the ungated input ("plain").
tau = the detector's locked validation tau_mask (primary); tau = 0.5 (secondary). Inputs reuse the exact corruption seeds of the locked
multireal, stress and NSTDB exports.
PRIMARY: five-realization training-distribution corrupted macro AUROC, gated minus plain, cnn_diagonly_aug, locked tau; mean of seed-level
deltas; patient-clustered bootstrap (1,000). Improvement is claimed only if the 97.5% CI lower bound > 0 AND the clean-input delta has a
95% CI lower bound > -0.002. Otherwise no benefit is claimed.
SECONDARY: same contrast for resnet_aug and inception_aug; tau 0.5; stress noise 0.10/0.25/0.50, joint k=2,4 with noise 0.25, k=8 missing,
BW 1.0 (each averaged over 5 realizations); NSTDB em+ma mean over 18 conditions; gate precision/recall against the true corruption mask.

## Round-5 addition (2026-09-28), fixed BEFORE any CPSC run, bank or output exists. SECONDARY; the original decision stands.
External replication of the diagnostic contrasts on a second public 12-lead dataset: CPSC 2018 (the cpsc_2018 folder of the
PhysioNet/CinC 2021 training set; 500 Hz, resampled to 100 Hz, first 10 s; records shorter than 10 s excluded, count reported).
Targets (five binary labels, one per record allowed): NORM = sinus rhythm 426783006; AF = 164889003; AVB = first-degree AV block
270492004; BBB = LBBB 164909002 or RBBB 59118001; STC = ST depression 429622005 or ST elevation 164931005. Records with none of
these (PAC/PVC only) are excluded, as PTB-XL records without a superclass are. No patient identifiers exist in CPSC 2018; each
record is one cluster, so the patient-clustered bootstrap reduces to a record bootstrap (limitation, stated). Folds: sha256 of
the record id mod 10 plus 1; folds 1-8 train, 9 validation (model selection and tau_mask), 10 test. cpsc_bank.py writes a
PTB-XL-shaped directory; train.py, export.py, export_multireal.py and analyze.py run unchanged. The only code change is that
data.py reads the class list from the environment variable ECG_CLASSES (default unchanged: the PTB-XL superclasses).
Configurations (3 seeds each, same recipe, hyperparameters, model selection and export as the locked runs):
cnn_full_leadwise_cleanfrac20 (proposed), cnn_diagonly_aug, cnn_diagonly_clean, cnn_full.
Endpoints (95% record-bootstrap CIs, 1,000 resamples, seed-level deltas averaged), computed by the existing analyze.py and
analyze_extra.py on the CPSC results directory:
- R1 (replicates B1): five-realization training-distribution corrupted macro AUROC, proposed minus cnn_diagonly_aug.
- R2: cnn_diagonly_aug minus cnn_diagonly_clean, on clean test input and under the same corruption.
- R3 (replicates B2): predicted-mask repair, macro AUROC post minus pre, for the proposed model; oracle-mask repair reported.
- R4 (replicates A): macro average of the 12 lead-specific noise-vs-intact AUROCs, learned head minus the combined rule.
Interpretation rule: a PTB-XL finding is called "replicated" only if the CPSC point estimate has the same sign and its CI
excludes the opposite sign; otherwise it is reported as not replicated. No CPSC result changes the pre-registered decision.
