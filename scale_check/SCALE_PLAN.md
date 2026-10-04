# Post hoc scale check (written 2026-10-04, before any scale run was started)

Question raised in internal review: does the negative diagnostic result depend on the small model size (1.16M parameters)?

Models: the locked CNN recipe with all encoder and decoder channel widths multiplied by 4 (encoder 256-1024 channels, about 15M parameters), trained with train.py unchanged (only build() swapped; scale_common.py). Two configurations, seeds 0, 1, 2, official folds, identical hyperparameters and checkpoint rule to the locked runs:
- scale_w4_diagonly_aug: diagnosis only, corruption augmentation (sigma_min 0, no clean fraction), as cnn_diagonly_aug.
- scale_w4_proposed: diagnosis + lead-preserving per-lead head + decoder, sigma_min 0.02, 20% clean records, as cnn_full_leadwise_cleanfrac20.

Endpoint (fixed now): corrupted-input macro AUROC on the prespecified test realization (corrupt_training, rng 2001, sigma_min 0, min_one_masked), scale_w4_proposed minus scale_w4_diagonly_aug, deltas per seed pair averaged, 95% patient-clustered bootstrap CI (1,000 resamples). The negative result is called scale-dependent only if the CI lower bound exceeds zero.

Descriptive (no decision attached): clean-input contrast; oracle-mask and dropped-only repair, post minus pre, for scale_w4_proposed; missing-lead grid k = 4, 8, 10 (five condition-seeded realizations as in the stress grid), proposed minus augmented, before and after true-mask repair.

Everything here is post hoc relative to the locked plan and is reported as such; nothing changes the prespecified decision.
