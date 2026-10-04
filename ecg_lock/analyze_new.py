"""SECONDARY analyses added after the RE-SCOPE decision (endpoints fixed in PREREG_ANALYSIS_PLAN.md before
any of these outputs existed).

Part 1 (CinC 2011, real acquisition quality): record-level AUROC for human-labelled 'unacceptable'
recordings. Learned score = max over leads of (1 - intact score); rules = the transparent per-lead rules
aggregated by max over leads. Records resampled jointly (1,000 bootstrap resamples).

Part 2 (quality-gated fusion model, cnn_fused_gated): paired patient-clustered contrasts against
cnn_diagonly_aug (identical training corruption), cnn_diag_lead, and the proposed model, on clean test data,
five-realization training-distribution corruption, stress grids, and NSTDB recorded noise.
Outputs: T_cinc2011.csv, T_fused_contrasts.csv (all SECONDARY).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CINC_CFGS = ["cnn_full_leadwise_cleanfrac20", "cnn_full_leadwise", "cnn_fused_gated", "cnn_full",
             "cnn_full_cleanfrac20", "cnn_diag_lead", "tx_full"]
FUSED = "cnn_fused_gated"
COMPARATORS = ["cnn_diagonly_aug", "cnn_diag_lead", "cnn_full_leadwise_cleanfrac20"]


def auc(y, s):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(Y, P):
    return float(np.nanmean([auc(Y[:, c], P[:, c]) for c in range(Y.shape[1])]))


def runs_of(root, cfg, fname):
    out = {}
    for s in (0, 1, 2):
        f = root / f"{cfg}_s{s}" / "bundles" / fname
        if f.exists():
            out[s] = f
    return out


def ci(vals):
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def part_cinc(root, out, B, rng):
    rows, learned = [], {}
    rules = None
    for cfg in CINC_CFGS:
        rs = runs_of(root, cfg, "cinc2011_v1.npz")
        if not rs:
            continue
        per = {}
        for s, f in rs.items():
            z = np.load(f)
            y = z["unacceptable"].astype(int)
            if rules is None:
                rv, rh = -z["rule_variance"], -z["rule_hf"]          # higher = more likely corrupted
                comb = np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape)
                rules = {"rule_variance": rv.max(1), "rule_hf": rh.max(1), "rule_combined": comb.max(1)}
                y_ref = y
            assert np.array_equal(y, y_ref)
            if "lead" in z:
                per[s] = {"max": (1 - z["lead"]).max(1), "mean": (1 - z["lead"]).mean(1)}
        if per:
            learned[cfg] = per
    if rules is None:
        print("no CinC bundles"); return
    y = y_ref
    n = len(y)
    boots = [rng.integers(0, n, n) for _ in range(B)]
    for name, sc in rules.items():
        bs = [auc(y[ix], sc[ix]) for ix in boots]
        rows.append({"scorer": name, "aggregation": "max over leads", "auroc": auc(y, sc), "ci_lo": ci(bs)[0],
                     "ci_hi": ci(bs)[1], "per_seed": "", "n_unacceptable": int(y.sum()), "n": n})
    for cfg, per in learned.items():
        for agg in ("max", "mean"):
            ss = sorted(per)
            pt = [auc(y, per[s][agg]) for s in ss]
            bs = [np.mean([auc(y[ix], per[s][agg][ix]) for s in ss]) for ix in boots]
            rows.append({"scorer": cfg, "aggregation": f"{agg} over leads of (1 - intact)", "auroc": float(np.mean(pt)),
                         "ci_lo": ci(bs)[0], "ci_hi": ci(bs)[1], "per_seed": json.dumps([round(v, 4) for v in pt]),
                         "n_unacceptable": int(y.sum()), "n": n})
            if agg == "max":
                for rname, rsc in rules.items():
                    d = [np.mean([auc(y[ix], per[s]["max"][ix]) for s in ss]) - auc(y[ix], rsc[ix]) for ix in boots]
                    rows.append({"scorer": f"{cfg} minus {rname}", "aggregation": "delta AUROC (max)",
                                 "auroc": float(np.mean(pt) - auc(y, rsc)), "ci_lo": ci(d)[0], "ci_hi": ci(d)[1],
                                 "per_seed": "", "n_unacceptable": int(y.sum()), "n": n})
    T = pd.DataFrame(rows); T["status"] = "SECONDARY"
    T.to_csv(out / "T_cinc2011.csv", index=False)
    print(T.round(4).to_string(), flush=True)


def part_fused(root, out, B, rng):
    if not runs_of(root, FUSED, "test_clean.npz"):
        print("no fused runs"); return
    ref = np.load(next(iter(runs_of(root, FUSED, "test_clean.npz").values())))
    Y, pid = ref["y_true"], ref["patient_id"]
    up = np.unique(pid); rows_of = [np.flatnonzero(pid == p) for p in up]
    boots = [np.concatenate([rows_of[j] for j in rng.integers(0, len(up), len(up))]) for _ in range(B)]
    full = np.arange(len(Y))

    def loader(cfg):
        cache = {}
        for s in (0, 1, 2):
            d = root / f"{cfg}_s{s}" / "bundles"
            if not (d / "test_clean.npz").exists():
                continue
            c = {"clean": np.load(d / "test_clean.npz")["y_score"]}
            assert np.array_equal(np.load(d / "test_clean.npz")["patient_id"], pid)
            if (d / "test_traindist_multireal_v1.npz").exists():
                z = np.load(d / "test_traindist_multireal_v1.npz")
                c["multireal"] = [z[f"r{r}|y_score"] for r in range(5)]
            if (d / "test_stress.npz").exists():
                z = np.load(d / "test_stress.npz")
                for tag, key in (("k8", "missing|k=8|s=0.0|bw=0.0"), ("k10", "missing|k=10|s=0.0|bw=0.0"),
                                 ("noise0.5", "noise|k=0|s=0.5|bw=0.0"), ("bw1.0", "bw|k=0|s=0.0|bw=1.0")):
                    c[tag] = np.mean([z[f"{key}|r={r}|y_score"] for r in range(5)], 0)
            if (d / "test_nstdb.npz").exists():
                z = np.load(d / "test_nstdb.npz")
                c["nstdb_em_ma"] = [z[f"nstdb|{k}|snr={snr}|leads={nl}|y_score"] for k in ("em", "ma")
                                    for snr in (12, 6, 0) for nl in (1, 3, 6)]
            cache[s] = c
        return cache

    def metric(c, key, ix):
        v = c[key]
        if isinstance(v, list):
            return np.mean([macro_auc(Y[ix], p[ix]) for p in v])
        return macro_auc(Y[ix], v[ix])

    F = loader(FUSED)
    rows = []
    for key in ("clean", "multireal", "k8", "k10", "noise0.5", "bw1.0", "nstdb_em_ma"):
        ss = [s for s in F if key in F[s]]
        if ss:
            rows.append({"contrast": f"{FUSED} (absolute)", "condition": key,
                         "value": float(np.mean([metric(F[s], key, full) for s in ss])), "ci_lo": np.nan, "ci_hi": np.nan,
                         "per_seed": json.dumps([round(metric(F[s], key, full), 4) for s in ss])})
    for comp in COMPARATORS:
        C = loader(comp)
        for key in ("clean", "multireal", "k8", "k10", "noise0.5", "bw1.0", "nstdb_em_ma"):
            ss = sorted(s for s in set(F) & set(C) if key in F[s] and key in C[s])
            if not ss:
                continue
            fn = lambda ix: np.mean([metric(F[s], key, ix) - metric(C[s], key, ix) for s in ss])
            bs = [fn(ix) for ix in boots]
            rows.append({"contrast": f"{FUSED} minus {comp}", "condition": key, "value": float(fn(full)),
                         "ci_lo": ci(bs)[0], "ci_hi": ci(bs)[1],
                         "per_seed": json.dumps([round(metric(F[s], key, full) - metric(C[s], key, full), 4) for s in ss])})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T["status"] = "SECONDARY"
    T.to_csv(out / "T_fused_contrasts.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--part", choices=["cinc", "fused", "both"], default="both")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.part in ("cinc", "both"):
        part_cinc(a.runs, a.out, a.bootstrap, np.random.default_rng(2011))
    if a.part in ("fused", "both"):
        part_fused(a.runs, a.out, a.bootstrap, np.random.default_rng(20260928))


if __name__ == "__main__":
    main()
