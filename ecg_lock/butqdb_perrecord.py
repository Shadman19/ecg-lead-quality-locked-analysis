"""Round 6, POST HOC sensitivity for BUT QDB (added after the Round-6 results were read; descriptive only):
(a) per-recording AUROC for every recording that contains both classes (>= 5 windows of each), for the detector score
average, the best published index of that task, the transferred logistic index model and the within-dataset ceiling;
(b) the mean of per-recording AUROCs with a recording-bootstrap CI (each recording weighted equally);
(c) a capped pooled analysis: at most 250 windows per class per recording (seed 0), subject-clustered bootstrap;
(d) pooled AUROC with recording 105001 (which holds 91% of the class-3 windows) removed.
Tasks: Q1 (class 3 vs 1) and Q3 (class 2 or 3 vs 1). Writes T_butqdb_posthoc_perrecord.csv and prints '>>' lines.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_ptbxl_quality import parse

H = Path.home(); Q = H / "ecg_lock_results/quality"; O = H / "ecg_lock_results/butqdb"; RUNS = Path("/scratch/spath004/ecg_lock_runs")
BANK = H / "data/butqdb_windows_100hz.npz"; B = 1000


def auc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boots(groups, seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])


def p(*a):
    print(">>", *a, flush=True)


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def main():
    b = np.load(BANK); cls, rec, subj = b["cls"], b["record"], b["subject"]
    zs = np.load(O / "sqi_butqdb.npz"); S, cols = np.nan_to_num(zs["S"]), list(zs["cols"])
    leads = [np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/butqdb_v1.npz")["lead"] for s in (0, 1, 2)]
    det = np.mean([(1 - L).mean(1) for L in leads], 0)
    zp = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); Sp, ids = np.nan_to_num(zp["S"]), zp["ids"]
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    lab = np.stack([parse(v) for v in d.static_noise]) | np.stack([parse(v) for v in d.burst_noise]) | np.stack([parse(v) for v in d.electrodes_problems])
    tr = np.flatnonzero(d.strat_fold.values <= 9); Xtr, ytr = Sp[tr].reshape(-1, len(cols)), lab[tr].ravel()
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(S)[:, 1]
    rows = []
    for tname, posset, best in (("Q1_c3_vs_c1", {3}, "tsqi_bad"), ("Q3_c23_vs_c1", {2, 3}, "ssqi_bad")):
        keep = np.isin(cls, list(posset | {1})); y = np.isin(cls[keep], list(posset)).astype(int)
        g, r = subj[keep], rec[keep]; Sk = S[keep]
        ceil = np.zeros(len(y))
        for u in np.unique(g):
            te = g == u
            ceil[te] = 0.5 if y[~te].min() == y[~te].max() else gbm().fit(Sk[~te], y[~te]).predict_proba(Sk[te])[:, 1]
        sc = {"detector": det[keep], "best_index": Sk[:, cols.index(best)], "logreg_ptbxl": logreg[keep], "ceiling_loso": ceil}
        # (a) per-recording AUROCs
        per = {k: [] for k in sc}; recs = []
        for rr in np.unique(r):
            m = r == rr
            if y[m].sum() >= 5 and (1 - y[m]).sum() >= 5:
                recs.append(rr)
                for k, s_ in sc.items():
                    per[k].append(auc(y[m], s_[m]))
                rows.append(dict(task=tname, record=rr, n=int(m.sum()), n_pos=int(y[m].sum()), **{k: per[k][-1] for k in sc}))
        per = {k: np.array(v) for k, v in per.items()}
        rng = np.random.default_rng(5); bs_r = [rng.integers(0, len(recs), len(recs)) for _ in range(B)]
        for k in sc:
            v = per[k]; mb = [v[ix].mean() for ix in bs_r]
            p(tname, "per-recording", k, "n_rec", len(recs), "median", round(float(np.median(v)), 3), "min", round(float(v.min()), 3),
              "max", round(float(v.max()), 3), "mean", round(float(v.mean()), 3), "recCI", np.round(ci(mb), 3).tolist())
        dv = per["detector"] - per["best_index"]; mb = [dv[ix].mean() for ix in bs_r]
        p(tname, "per-recording detector minus best index: mean", round(float(dv.mean()), 3), "recCI", np.round(ci(mb), 3).tolist(),
          "recordings detector >= index", int((dv >= 0).sum()), "of", len(recs),
          "| detector minus ceiling mean", round(float((per["detector"] - per["ceiling_loso"]).mean()), 3))
        # (c) capped pooled analysis (<= 250 windows per class per recording)
        rng = np.random.default_rng(0); sel = []
        for rr in np.unique(r):
            for c in (0, 1):
                ix = np.flatnonzero((r == rr) & (y == c)); sel.append(rng.choice(ix, min(250, len(ix)), replace=False))
        sel = np.sort(np.concatenate(sel)); yc, gc = y[sel], g[sel]; bs = boots(gc, 7)
        for k, s_ in sc.items():
            a0 = auc(yc, s_[sel]); ab = np.array([auc(yc[ix], s_[sel][ix]) for ix in bs])
            p(tname, "capped250", k, "n", len(yc), "pos", int(yc.sum()), "auroc", round(a0, 3), "subjCI", np.round(ci(ab), 3).tolist())
        db = np.array([auc(yc[ix], sc["detector"][sel][ix]) - auc(yc[ix], sc["best_index"][sel][ix]) for ix in bs])
        d0 = auc(yc, sc["detector"][sel]) - auc(yc, sc["best_index"][sel])
        p(tname, "capped250 detector minus best index", round(d0, 3), "subjCI", np.round(ci(db), 3).tolist())
        # (d) without recording 105001
        m = r != "105001"
        p(tname, "without 105001: n", int(m.sum()), "pos", int(y[m].sum()), " ".join(f"{k}={auc(y[m], s_[m]):.3f}" for k, s_ in sc.items()))
    pd.DataFrame(rows).to_csv(O / "T_butqdb_posthoc_perrecord.csv", index=False)


if __name__ == "__main__":
    main()
