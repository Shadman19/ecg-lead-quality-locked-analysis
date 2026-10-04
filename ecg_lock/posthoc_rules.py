"""POST-HOC (not pre-registered) stronger transparent baselines for per-lead noise detection.

Added after the pre-registered RE-SCOPE decision to answer the expected reviewer objection
"the rule baseline is weak". Reported separately and labelled post hoc in the paper.

Rules (all label-free on test; fitted parameters come from VALIDATION only):
  hf_lead_norm   : log HF ratio, standardized per lead with validation intact-lead median/MAD
  hf_within_rec  : hf_lead_norm minus the record's median across its 12 leads (within-ECG contrast)
  logreg_val     : logistic regression fitted on VALIDATION noise-vs-intact leads using
                   [log std, log HF ratio, both lead-normalized and within-record contrasted, lead one-hot]
Metric = pre-registered Endpoint A definition: macro average of 12 lead-specific AUROCs,
noise-corrupted (type 2) vs intact (type 0) leads, test train-distribution corruption.
CI: patients bootstrapped as clusters, per-seed deltas averaged (same scheme as analyze.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

P = "cnn_full_leadwise_cleanfrac20"


def feats(z, ref=None):
    ls = np.log(z["rule_variance"] + 1e-6)          # log std (dropped leads -> very low)
    lh = np.log(-z["rule_hf"] + 1e-9)                # log HF/total energy ratio
    if ref is None:                                  # per-lead reference from intact validation leads
        ok = z["corruption_type"] == 0
        ref = {}
        for k, v in (("ls", ls), ("lh", lh)):
            med = np.array([np.median(v[ok[:, L], L]) for L in range(12)])
            mad = np.array([np.median(np.abs(v[ok[:, L], L] - med[L])) for L in range(12)]) + 1e-6
            ref[k] = (med, mad)
    nls = (ls - ref["ls"][0]) / ref["ls"][1]
    nlh = (lh - ref["lh"][0]) / ref["lh"][1]
    wls = nls - np.median(nls, axis=1, keepdims=True)
    wlh = nlh - np.median(nlh, axis=1, keepdims=True)
    return {"nls": nls, "nlh": nlh, "wls": wls, "wlh": wlh}, ref


def design(f):
    n = f["nls"].shape[0]
    onehot = np.tile(np.eye(12), (n, 1))
    X = np.column_stack([f[k].ravel() for k in ("nls", "nlh", "wls", "wlh")] + [onehot])
    return X


def macro_lead_auc(m, ct, sc):
    keep = (ct == 2) | (ct == 0)
    out = []
    for L in range(12):
        k = keep[:, L]
        if 0 < m[k, L].mean() < 1:
            out.append(roc_auc_score(m[k, L], sc[k, L]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(p.name.rsplit("_s", 1)[1]) for p in a.runs.glob(f"{P}_s*"))
    V = {s: dict(np.load(a.runs / f"{P}_s{s}" / "bundles" / "val_traindist_predicted.npz")) for s in seeds}
    T = {s: dict(np.load(a.runs / f"{P}_s{s}" / "bundles" / "test_traindist_predicted.npz")) for s in seeds}
    # rules depend only on the (identical) corrupted input, not on the seed
    for s in seeds[1:]:
        for fld in ("corruption_mask", "corruption_type", "rule_hf", "patient_id"):
            assert np.array_equal(T[s][fld], T[seeds[0]][fld]), fld
    zv, zt = V[seeds[0]], T[seeds[0]]
    assert not set(zv["patient_id"]) & set(zt["patient_id"]), "val/test patient overlap"
    fv, ref = feats(zv)
    ft, _ = feats(zt, ref)
    keepv = ((zv["corruption_type"] == 2) | (zv["corruption_type"] == 0)).ravel()
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(design(fv)[keepv], zv["corruption_mask"].ravel()[keepv])
    m, ct = zt["corruption_mask"], zt["corruption_type"]
    rv, rh = -zt["rule_variance"], -zt["rule_hf"]
    rules = {
        "rule_combined_prereg": np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape),
        "hf_lead_norm": ft["nlh"],
        "hf_within_rec": ft["wlh"],
        "logreg_val": lr.predict_proba(design(ft))[:, 1].reshape(m.shape),
    }
    learned = {s: 1 - T[s]["reliability_score"] for s in seeds}
    pid = zt["patient_id"]
    up = np.unique(pid)
    rows_of = {p: np.flatnonzero(pid == p) for p in up}
    rng = np.random.default_rng(0)
    res = []
    for name, sc in rules.items():
        def delta(ix):
            r = macro_lead_auc(m[ix], ct[ix], sc[ix])
            return [macro_lead_auc(m[ix], ct[ix], learned[s][ix]) - r for s in seeds], r
        per, rule_pt = delta(np.arange(len(pid)))
        boots = []
        for _ in range(a.bootstrap):
            ix = np.concatenate([rows_of[p] for p in rng.choice(up, len(up))])
            boots.append(np.mean(delta(ix)[0]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.append({"rule": name, "rule_macro_lead_auroc": rule_pt,
                    "learned_macro_lead_auroc_mean": float(np.mean([macro_lead_auc(m, ct, learned[s]) for s in seeds])),
                    "delta_learned_minus_rule": float(np.mean(per)), "ci95_lo": lo, "ci95_hi": hi,
                    "per_seed": json.dumps([round(x, 4) for x in per]), "status": "POST-HOC"})
        print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(a.out / "T_posthoc_rules.csv", index=False)
    print("wrote", a.out / "T_posthoc_rules.csv")


if __name__ == "__main__":
    main()
