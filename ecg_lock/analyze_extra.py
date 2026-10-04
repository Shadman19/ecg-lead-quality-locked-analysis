"""SECONDARY (post-decision) analyses on locked outputs.

Part 1: per-superclass clean-test AUROC/AUPRC from the locked test_clean bundles (no new inference).
Part 2: five-realization training-distribution corruption (test_traindist_multireal_v1.npz from export_multireal.py):
  paired model contrasts and repair effects with patient-clustered bootstrap CIs (1,000 resamples).
Statistic: for each seed, macro AUROC averaged over the 5 realizations; deltas per seed, then averaged over seeds;
patients resampled as clusters jointly for both arms and all seeds.
Outputs: T_perclass_clean.csv, T_perclass_corrupted.csv, T_multireal_contrasts.csv, T_multireal_summary.csv (all SECONDARY).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CONTRASTS = [("cnn_full_leadwise_cleanfrac20", "cnn_diagonly_aug"), ("cnn_full", "cnn_diagonly_aug"),
             ("cnn_diag_lead", "cnn_diagonly_aug"), ("cnn_diag_rec", "cnn_diagonly_aug"),
             ("resnet_aug", "cnn_diagonly_aug"), ("inception_aug", "cnn_diagonly_aug"),
             ("cnn_diagonly_aug", "cnn_diagonly_clean")]


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(Y, P):
    return float(np.nanmean([auc(Y[:, c], P[:, c]) for c in range(Y.shape[1])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # ---- Part 1: per-class clean ----
    rows = []
    for d in sorted(a.runs.iterdir()):
        f = d / "bundles" / "test_clean.npz"
        if not f.exists() or "_gss_" in d.name + "_":
            continue
        cfg, s = d.name.rsplit("_s", 1)
        z = np.load(f)
        Y, P = z["y_true"], z["y_score"]
        for c, name in enumerate(CLASSES):
            rows.append({"config": cfg, "seed": int(s), "class": name, "auroc": auc(Y[:, c], P[:, c]),
                         "auprc": average_precision_score(Y[:, c], P[:, c]), "prevalence": float(Y[:, c].mean())})
    C = pd.DataFrame(rows)
    C.groupby(["config", "class"])[["auroc", "auprc", "prevalence"]].agg(["mean", "std"]).to_csv(a.out / "T_perclass_clean.csv")
    print(C.groupby(["config", "class"]).auroc.mean().unstack("class").round(4).to_string(), flush=True)
    # ---- Part 2: multi-realization ----
    runs = {}
    for d in sorted(a.runs.iterdir()):
        f = d / "bundles" / "test_traindist_multireal_v1.npz"
        if f.exists() and "_gss_" not in d.name + "_":
            cfg, s = d.name.rsplit("_s", 1)
            runs.setdefault(cfg, {})[int(s)] = dict(np.load(f))
    if not runs:
        print("no multireal bundles"); return
    ref = next(iter(next(iter(runs.values())).values()))
    Y, pid = ref["y_true"], ref["patient_id"]
    R = len([k for k in ref if k.endswith("|y_score")])
    for cfg, ss in runs.items():
        for s, z in ss.items():
            assert np.array_equal(z["ecg_id"], ref["ecg_id"])
            for r in range(R):
                assert np.array_equal(z[f"r{r}|mask"], ref[f"r{r}|mask"]), (cfg, s, r)
    up = np.unique(pid); rows_of = [np.flatnonzero(pid == p) for p in up]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rows_of[j] for j in rng.choice(len(up), len(up))]) for _ in range(a.bootstrap)]

    def score(z, key, ix):
        return np.mean([macro_auc(Y[ix], z[f"r{r}|{key}"][ix]) for r in range(R)])

    summ = [{"config": cfg, "seed": s, "key": key, "macro_auroc_mean_over_realizations": score(z, key, np.arange(len(Y)))}
            for cfg, ss in runs.items() for s, z in ss.items() for key in ("y_score", "post_oracle", "post_pred") if f"r0|{key}" in z]
    pd.DataFrame(summ).to_csv(a.out / "T_multireal_summary.csv", index=False)
    # per-class AUROC/AUPRC under corruption (mean over realizations), per config and seed
    pc = []
    for cfg, ss in runs.items():
        for s, z in ss.items():
            for c, name in enumerate(CLASSES):
                pc.append({"config": cfg, "seed": s, "class": name,
                           "auroc": np.mean([auc(Y[:, c], z[f"r{r}|y_score"][:, c]) for r in range(R)]),
                           "auprc": np.mean([average_precision_score(Y[:, c], z[f"r{r}|y_score"][:, c]) for r in range(R)])})
    PC = pd.DataFrame(pc)
    PC.groupby(["config", "class"])[["auroc", "auprc"]].agg(["mean", "std"]).to_csv(a.out / "T_perclass_corrupted.csv")
    print(PC.groupby(["config", "class"]).auroc.mean().unstack("class").round(4).to_string(), flush=True)
    out = []

    def run_contrast(name, fn, seeds):
        per = [fn(np.arange(len(Y)), s) for s in seeds]
        bs = [np.mean([fn(ix, s) for s in seeds]) for ix in boots]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"contrast": name, "per_seed": json.dumps([round(v, 4) for v in per]), "delta": float(np.mean(per)),
                    "ci95_lo": lo, "ci95_hi": hi, "realizations": R, "status": "SECONDARY"})
        print(out[-1], flush=True)

    for A, B in CONTRASTS:
        if A in runs and B in runs:
            ss = sorted(set(runs[A]) & set(runs[B]))
            run_contrast(f"{A} minus {B}", lambda ix, s: score(runs[A][s], "y_score", ix) - score(runs[B][s], "y_score", ix), ss)
    for cfg, ss in runs.items():
        z0 = next(iter(ss.values()))
        for key in ("post_oracle", "post_pred"):
            if f"r0|{key}" in z0:
                run_contrast(f"{cfg} repair {key} minus pre",
                             lambda ix, s, cfg=cfg, key=key: score(runs[cfg][s], key, ix) - score(runs[cfg][s], "y_score", ix), sorted(ss))
    pd.DataFrame(out).to_csv(a.out / "T_multireal_contrasts.csv", index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
