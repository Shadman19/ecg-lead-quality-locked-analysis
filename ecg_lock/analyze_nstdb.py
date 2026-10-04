"""SECONDARY analysis (post-decision): diagnosis and per-lead detection under recorded NSTDB artifacts.

Reads bundles/test_nstdb_v1.npz for every run (strat split only). Outputs (all labelled SECONDARY):
  T_nstdb_diag_raw.csv / T_nstdb_diag_summary.csv   macro AUROC per config x seed x condition
  T_nstdb_contrasts.csv                              paired, patient-clustered bootstrap contrasts
  T_nstdb_detect_raw.csv / T_nstdb_detect_summary.csv  macro-over-leads AUROC, artifact vs intact leads (em, ma)
Statistic for contrasts: macro AUROC averaged over conditions of a family (em+ma: 18 conditions; bw: 9),
per-seed deltas averaged; patients resampled as clusters jointly for both models and all seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from posthoc_rules import design, feats

PROPOSED = "cnn_full_leadwise_cleanfrac20"
CONTRASTS = [(PROPOSED, "cnn_diagonly_aug"), ("cnn_full", "cnn_diagonly_aug"),
             ("cnn_diagonly_aug", "cnn_diagonly_clean"), ("resnet_aug", "resnet_clean"),
             ("inception_aug", "inception_clean"), ("resnet_aug", "cnn_diagonly_aug")]


def auc(y, s):
    """Fast binary AUROC via ranks (ties averaged). Returns nan if one class is absent."""
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(Y, P):
    return float(np.nanmean([auc(Y[:, c], P[:, c]) for c in range(Y.shape[1])]))


def macro_lead_auc(m, sc):
    vals = [auc(m[:, L], sc[:, L]) for L in range(m.shape[1])]
    return float(np.nanmean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in sorted(a.runs.iterdir()):
        f = d / "bundles" / "test_nstdb_v1.npz"
        if not f.exists() or "_gss_" in d.name + "_":
            continue
        cfg, s = d.name.rsplit("_s", 1)
        runs.setdefault(cfg, {})[int(s)] = dict(np.load(f))
    print("configs:", {k: sorted(v) for k, v in runs.items()})
    ref = next(iter(next(iter(runs.values())).values()))
    conds = sorted({k.rsplit("|", 1)[0] for k in ref if k.endswith("|y_score")})
    Y, pid = ref["y_true"], ref["patient_id"]
    # identical inputs across all runs
    for cfg, ss in runs.items():
        for s, z in ss.items():
            assert np.array_equal(z["ecg_id"], ref["ecg_id"]) and np.array_equal(z["y_true"], Y)
            for c in conds:
                assert np.array_equal(z[c + "|mask"], ref[c + "|mask"]), (cfg, s, c)
                assert np.allclose(z[c + "|rule_hf"], ref[c + "|rule_hf"]), (cfg, s, c)
    # ---- diagnosis ----
    rows = [{"config": cfg, "seed": s, "condition": c, "family": c.split("|")[0],
             "macro_auroc": macro_auc(Y, z[c + "|y_score"])} for cfg, ss in runs.items() for s, z in ss.items() for c in conds]
    D = pd.DataFrame(rows); D.to_csv(a.out / "T_nstdb_diag_raw.csv", index=False)
    D.groupby(["config", "condition"]).macro_auroc.agg(["mean", "std"]).to_csv(a.out / "T_nstdb_diag_summary.csv")
    fam = {"em+ma": [c for c in conds if c.startswith(("em", "ma"))], "bw": [c for c in conds if c.startswith("bw")],
           "em_ma_0dB_6leads": [c for c in conds if c in ("em|snr=0|leads=6", "ma|snr=0|leads=6")]}
    up = np.unique(pid); rows_of = [np.flatnonzero(pid == p) for p in up]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rows_of[j] for j in rng.choice(len(up), len(up))]) for _ in range(a.bootstrap)]
    crow = []
    for A, B in CONTRASTS:
        if A not in runs or B not in runs:
            continue
        ss = sorted(set(runs[A]) & set(runs[B]))
        for fname, cl in fam.items():
            def stat(ix):
                return [np.mean([macro_auc(Y[ix], runs[A][s][c + "|y_score"][ix]) - macro_auc(Y[ix], runs[B][s][c + "|y_score"][ix]) for c in cl]) for s in ss]
            per = stat(np.arange(len(Y)))
            bs = [np.mean(stat(ix)) for ix in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            crow.append({"A": A, "B": B, "family": fname, "per_seed": json.dumps([round(v, 4) for v in per]),
                         "delta": float(np.mean(per)), "ci95_lo": lo, "ci95_hi": hi, "status": "SECONDARY"})
            print(crow[-1], flush=True)
    pd.DataFrame(crow).to_csv(a.out / "T_nstdb_contrasts.csv", index=False)
    # ---- per-lead detection (em, ma only; bw is excluded from the label by protocol) ----
    zv = dict(np.load(a.runs / f"{PROPOSED}_s0" / "bundles" / "val_traindist_predicted.npz"))
    fv, fref = feats(zv)
    keepv = ((zv["corruption_type"] == 2) | (zv["corruption_type"] == 0)).ravel()
    lr = LogisticRegression(max_iter=2000).fit(design(fv)[keepv], zv["corruption_mask"].ravel()[keepv])
    drow = []
    for c in fam["em+ma"]:
        m = ref[c + "|mask"]
        zc = {"rule_variance": ref[c + "|rule_variance"], "rule_hf": ref[c + "|rule_hf"], "corruption_type": 2 * m}
        ft, _ = feats(zc, fref)
        rv, rh = -zc["rule_variance"], -zc["rule_hf"]
        rules = {"rule_combined_prereg": np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(m.shape),
                 "hf_within_rec": ft["wlh"], "logreg_val": lr.predict_proba(design(ft))[:, 1].reshape(m.shape)}
        for name, sc in rules.items():
            drow.append({"config": "rule", "seed": -1, "condition": c, "detector": name, "macro_lead_auroc": macro_lead_auc(m, sc)})
        for cfg, ss in runs.items():
            for s, z in ss.items():
                if c + "|lead" in z:
                    drow.append({"config": cfg, "seed": s, "condition": c, "detector": "learned_head",
                                 "macro_lead_auroc": macro_lead_auc(m, 1 - z[c + "|lead"])})
    # CI: proposed learned head minus validation-fitted logistic rule, averaged over em+ma conditions
    if PROPOSED in runs:
        LRS = {c: None for c in fam["em+ma"]}
        for c in LRS:
            zc = {"rule_variance": ref[c + "|rule_variance"], "rule_hf": ref[c + "|rule_hf"]}
            ft, _ = feats(zc, fref)
            LRS[c] = lr.predict_proba(design(ft))[:, 1].reshape(ref[c + "|mask"].shape)
        ssP = sorted(runs[PROPOSED])
        def dstat(ix):
            return [np.mean([macro_lead_auc(ref[c + "|mask"][ix], 1 - runs[PROPOSED][s][c + "|lead"][ix])
                             - macro_lead_auc(ref[c + "|mask"][ix], LRS[c][ix]) for c in fam["em+ma"]]) for s in ssP]
        per = dstat(np.arange(len(Y)))
        bs = [np.mean(dstat(ix)) for ix in boots]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        crow.append({"A": PROPOSED + ":learned_head", "B": "logreg_val_rule", "family": "detect_em+ma",
                     "per_seed": json.dumps([round(v, 4) for v in per]), "delta": float(np.mean(per)),
                     "ci95_lo": lo, "ci95_hi": hi, "status": "SECONDARY"})
        print(crow[-1], flush=True)
        pd.DataFrame(crow).to_csv(a.out / "T_nstdb_contrasts.csv", index=False)
    T = pd.DataFrame(drow); T.to_csv(a.out / "T_nstdb_detect_raw.csv", index=False)
    T.groupby(["config", "detector", "condition"]).macro_lead_auroc.mean().unstack("condition").to_csv(a.out / "T_nstdb_detect_summary.csv")
    T["snr"] = T.condition.str.extract(r"snr=(\d+)")[0]
    T.groupby(["config", "detector", "snr"]).macro_lead_auroc.mean().unstack("snr").to_csv(a.out / "T_nstdb_detect_by_snr.csv")
    print(T.groupby(["config", "detector", "snr"]).macro_lead_auroc.mean().unstack("snr").round(3).to_string())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
