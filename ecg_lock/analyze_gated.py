"""SECONDARY round 4 analysis: detector-gated lead masking, gated minus plain (same frozen classifier, same inputs).
Primary (fixed before the run): cnn_diagonly_aug, locked tau, five-realization training-distribution corruption,
macro AUROC gated minus plain; improvement claimed only if the 97.5% patient-clustered CI lower bound > 0 AND the
clean-input difference has a 95% CI lower bound > -0.002 (no meaningful clean loss).
Output: T_gated_contrasts.csv, T_gate_stats.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CLASSIFIERS = ["cnn_diagonly_aug", "resnet_aug", "inception_aug"]
TAUS = ["tau", "t05"]


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, P, ix):
    return np.nanmean([auc(Y[ix, c], P[ix, c]) for c in range(Y.shape[1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    Z = {s: np.load(a.dir / f"gated_s{s}.npz") for s in (0, 1, 2) if (a.dir / f"gated_s{s}.npz").exists()}
    seeds = sorted(Z)
    z0 = Z[seeds[0]]
    Y, pid = z0["y_true"], z0["patient_id"]
    for s in seeds:
        assert np.array_equal(Z[s]["patient_id"], pid) and np.array_equal(Z[s]["y_true"], Y)
    keys = sorted({k.split("|cnn_diagonly_aug|")[0] for k in z0.files if "|cnn_diagonly_aug|plain" in k})
    groups = {"clean": ["clean"], "multireal": [k for k in keys if k.startswith("multireal")],
              "nstdb_em_ma": [k for k in keys if k.startswith("nstdb")]}
    for k in keys:
        if k.split("|")[0] in ("noise", "joint", "missing", "bw"):
            groups.setdefault(k.rsplit("|r=", 1)[0], []).append(k)
    rng = np.random.default_rng(20260928)
    up = np.unique(pid); rows_of = [np.flatnonzero(pid == p) for p in up]
    boots = [np.concatenate([rows_of[j] for j in rng.integers(0, len(up), len(up))]) for _ in range(a.bootstrap)]
    full = np.arange(len(Y))
    rows = []
    for c in CLASSIFIERS:
        if f"clean|{c}|plain" not in z0.files:
            continue
        for tn in TAUS:
            for g, ks in groups.items():
                def d(ix):
                    return np.mean([np.mean([macro(Y, Z[s][f"{k}|{c}|gated_{tn}"], ix) - macro(Y, Z[s][f"{k}|{c}|plain"], ix)
                                             for k in ks]) for s in seeds])
                pt = d(full)
                bs = np.array([d(ix) for ix in boots])
                plain = np.mean([np.mean([macro(Y, Z[s][f"{k}|{c}|plain"], full) for k in ks]) for s in seeds])
                row = dict(classifier=c, tau=tn, condition=g, plain_auroc=plain, delta=pt,
                           ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5),
                           ci975_lo=np.percentile(bs, 1.25), ci975_hi=np.percentile(bs, 98.75), n_keys=len(ks))
                rows.append(row); print(row, flush=True)
    T = pd.DataFrame(rows); T["status"] = "SECONDARY round 4"
    T.to_csv(a.dir / "T_gated_contrasts.csv", index=False)
    st = []
    for g, ks in groups.items():
        for tn in TAUS:
            gate = np.concatenate([Z[s][f"{k}|gate_{tn}"].ravel() for s in seeds for k in ks])
            tm = np.concatenate([np.broadcast_to(Z[s][f"{k}|true_mask"], Z[s][f"{k}|gate_{tn}"].shape).ravel()
                                 for s in seeds for k in ks])
            st.append(dict(condition=g, tau=tn, frac_gated=gate.mean(), frac_corrupted=tm.mean(),
                           precision=(gate & tm).sum() / max(gate.sum(), 1), recall=(gate & tm).sum() / max(tm.sum(), 1)))
    pd.DataFrame(st).to_csv(a.dir / "T_gate_stats.csv", index=False)
    print(pd.DataFrame(st).round(3).to_string())


if __name__ == "__main__":
    main()
