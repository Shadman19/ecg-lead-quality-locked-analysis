"""Round 6 utility analyses U1 and U2 on EXISTING outputs (no new training), PREREG_ANALYSIS_PLAN.md Round-6 entry.
U1 (PTB-XL fold 10, real records): recall of annotated leads / annotated records when the top 1, 2, 5, 10% of leads / records
ranked by a score are reviewed; detector score average and seed mean, template index, kSQI, PTB-XL-trained GBM; patient bootstrap.
U2 (unmodified PTB-XL test fold and CPSC test fold): per-record Brier score (mean over the five labels) of cnn_diagonly_aug and of
the proposed model for records in the top 5% / 10% of the detector record score (max over leads, score average) versus the rest;
difference (flagged minus rest) averaged over the three classifier seeds, patient/record bootstrap CI; same split by the template index.
Usage: python triage_utility.py --which ptbxl   |   ECG_CLASSES=NORM,AF,AVB,BBB,STC python triage_utility.py --which cpsc
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

from eval_ptbxl_quality import parse
from sqi_baselines import COLS, record_sqis

H = Path.home(); Q = H / "ecg_lock_results/quality"; O = H / "ecg_lock_results/butqdb"; B = 1000
ROOTS = {"ptbxl": (Path("/scratch/spath004/ecg_lock_runs"), H / "cardio ML/ptbxl"), "cpsc": (Path("/scratch/spath004/ecg_cpsc_runs"), H / "data/cpsc2018_shaped")}
FRACS = (0.01, 0.02, 0.05, 0.10)


def boots(groups, seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])


def p(*a):
    print(">>", *a, flush=True)


def recall_at(y, s, frac):
    k = max(1, int(round(frac * len(s)))); top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def u1(RUNS):
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = np.nan_to_num(z["S"]), z["ids"], list(z["cols"])
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    lab = np.stack([parse(v) for v in d.static_noise]) | np.stack([parse(v) for v in d.burst_noise]) | np.stack([parse(v) for v in d.electrodes_problems])
    alles = np.zeros(len(d), bool)
    for f in ("static_noise", "burst_noise", "electrodes_problems"):
        alles |= d[f].astype(str).str.upper().str.contains("ALLES").values
    fold = d.strat_fold.values; pos = {e: i for i, e in enumerate(ids)}
    tid = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz")["ecg_id"]; ti = np.array([pos[e] for e in tid])
    tr = np.flatnonzero(fold <= 9)
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(S[tr].reshape(-1, len(cols)), lab[tr].ravel())
    seeds = [(1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/test_clean.npz")["lead_score_clean"]) for s in (0, 1, 2)]
    L = {"detector_ens": np.mean(seeds, 0), "SQI_tsqi": S[ti, :, cols.index("tsqi_bad")], "SQI_ksqi": S[ti, :, cols.index("ksqi_bad")],
         "SUP_gbm": g.predict_proba(S[ti].reshape(-1, len(cols)))[:, 1].reshape(-1, 12)}
    Y = lab[ti]; pid = d.patient_id.values[ti]; al = alles[ti]; rows = []
    for level in ("lead", "record", "record_noalles"):
        if level == "lead":
            y = Y.ravel(); grp = np.repeat(pid, 12); sc = {k: v.ravel() for k, v in L.items()}; sseeds = [s_.ravel() for s_ in seeds]
        else:
            m = ~al if level == "record_noalles" else np.ones(len(Y), bool)
            y = Y[m].any(1).astype(int); grp = pid[m]; sc = {k: v[m].max(1) for k, v in L.items()}; sseeds = [s_[m].max(1) for s_ in seeds]
        bs = boots(grp, 21)
        for frac in FRACS:
            for k, s_ in list(sc.items()) + [("detector_seedmean", None)]:
                if s_ is None:
                    f0 = np.mean([recall_at(y, q, frac) for q in sseeds]); fb = [np.mean([recall_at(y[ix], q[ix], frac) for q in sseeds]) for ix in bs]
                else:
                    f0 = recall_at(y, s_, frac); fb = [recall_at(y[ix], s_[ix], frac) for ix in bs]
                rows.append(dict(level=level, frac=frac, scorer=k, recall=f0, lo=ci(fb)[0], hi=ci(fb)[1], n=len(y), n_pos=int(y.sum())))
                p("U1", level, frac, k, f"{f0:.3f}[{ci(fb)[0]:.3f},{ci(fb)[1]:.3f}]", "n", len(y), "pos", int(y.sum()))
    pd.DataFrame(rows).to_csv(O / "T_U1_triage_ptbxl.csv", index=False)


def u2(which, RUNS, root):
    from data import load_labels, load_signals, split_indices
    z = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz"); ids = z["ecg_id"]; pid = z["patient_id"]
    det = np.mean([(1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/test_clean.npz")["lead_score_clean"]).max(1) for s in (0, 1, 2)], 0)
    if which == "ptbxl":
        zq = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); pos = {e: i for i, e in enumerate(zq["ids"])}
        tsqi = np.nan_to_num(zq["S"])[np.array([pos[e] for e in ids]), :, list(zq["cols"]).index("tsqi_bad")].max(1)
    else:
        df = load_labels(root); X = load_signals(root, df, root / "cache_x100.npy"); ii = split_indices(df, "strat")["test"]
        assert np.array_equal(df.index.values[ii], ids)
        with Pool(8) as pool:
            Sc = np.stack(pool.map(record_sqis, list(X[ii]), chunksize=16))
        tsqi = np.nan_to_num(Sc[:, :, COLS.index("tsqi_bad")]).max(1)
    rows = []; bs = boots(pid, 31)
    for clf in ("cnn_diagonly_aug", "cnn_full_leadwise_cleanfrac20"):
        br = []
        for s in (0, 1, 2):
            zz = np.load(RUNS / f"{clf}_s{s}/bundles/test_clean.npz"); assert np.array_equal(zz["ecg_id"], ids)
            br.append(((zz["y_score"] - zz["y_true"]) ** 2).mean(1))
        br = np.stack(br)                                                    # [3, n] per-record Brier per seed
        for sname, sc in (("detector_ens", det), ("SQI_tsqi", tsqi)):
            for frac in (0.05, 0.10):
                thr = np.quantile(sc, 1 - frac); fl = sc > thr
                dfun = lambda ix: float(np.mean([b_[ix][fl[ix]].mean() - b_[ix][~fl[ix]].mean() for b_ in br]))
                d0 = dfun(np.arange(len(sc))); db = [dfun(ix) for ix in bs]
                rows.append(dict(dataset=which, classifier=clf, split_by=sname, frac=frac, n_flagged=int(fl.sum()),
                                 brier_flagged=float(np.mean([b_[fl].mean() for b_ in br])), brier_rest=float(np.mean([b_[~fl].mean() for b_ in br])),
                                 diff=d0, lo=ci(db)[0], hi=ci(db)[1]))
                p("U2", which, clf, sname, frac, "nflag", int(fl.sum()), f"flagged {rows[-1]['brier_flagged']:.4f} rest {rows[-1]['brier_rest']:.4f}",
                  f"diff {d0:.4f}[{ci(db)[0]:.4f},{ci(db)[1]:.4f}]")
    pd.DataFrame(rows).to_csv(O / f"T_U2_reliability_{which}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", choices=["ptbxl", "cpsc"], required=True); a = ap.parse_args()
    O.mkdir(parents=True, exist_ok=True); RUNS, root = ROOTS[a.which]
    if a.which == "ptbxl":
        u1(RUNS)
    u2(a.which, RUNS, root)


if __name__ == "__main__":
    main()
