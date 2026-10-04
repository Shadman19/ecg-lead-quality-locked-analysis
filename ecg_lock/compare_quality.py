"""Head-to-head comparison: label-free per-lead detector vs published SQIs vs supervised SQI models.
PTB-XL (per-lead, human annotations static/burst/electrodes; test = fold 10; supervised models trained on folds 1-9).
CinC 2011 set a (record-level acceptable/unacceptable; score = max over leads; supervised = 10-fold CV on CinC).
Deltas: detector minus comparator, bootstrap 1,000 (PTB-XL: patients; CinC: records). Output T_compare_ptbxl.csv, T_compare_cinc.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_ptbxl_quality import parse

RUNS = Path("/scratch/spath004/ecg_lock_runs")
Q = Path.home() / "ecg_lock_results/quality"
DET = ["cnn_full_leadwise_cleanfrac20", "cnn_full_leadwise", "tx_full"]


def auc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_delta(y, a, b, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    out = []
    for _ in range(B):
        ix = np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))])
        out.append(auc(y[ix], a[ix]) - auc(y[ix], b[ix]))
    return np.percentile(out, [2.5, 97.5])


def models():
    return {"sup_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
            "sup_gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)}


def ptbxl():
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = z["S"], z["ids"], list(z["cols"])
    d = pd.read_csv(Path.home() / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    lab = np.stack([parse(a) | parse(b) | parse(c) for a, b, c in zip(d.static_noise, d.burst_noise, d.electrodes_problems)])
    test = d.strat_fold.values == 10
    ref = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz")
    tid = ref["ecg_id"]; pos = {e: i for i, e in enumerate(ids)}; ti = np.array([pos[e] for e in tid])
    assert test.sum() == len(tid) and set(ids[test]) == set(tid)
    y = lab[ti].ravel(); grp = np.repeat(d.patient_id.values[ti], 12)
    scores = {c: S[ti, :, k].ravel() for k, c in enumerate(cols)}
    tr = np.flatnonzero(~test)
    Xtr, ytr = S[tr].reshape(-1, len(cols)), lab[tr].ravel()
    Xte = S[ti].reshape(-1, len(cols))
    for n, m in models().items():
        m.fit(np.nan_to_num(Xtr), ytr); scores[n] = m.predict_proba(np.nan_to_num(Xte))[:, 1]
    for c in DET:
        ss = [1 - np.load(RUNS / f"{c}_s{s}/bundles/test_clean.npz")["lead_score_clean"] for s in (0, 1, 2)
              if (RUNS / f"{c}_s{s}/bundles/test_clean.npz").exists()]
        scores["DET_" + c] = np.mean(ss, 0).ravel()
    return y, grp, scores


def cinc():
    z = np.load(Q / "sqi_cinc.npz", allow_pickle=True); S, rec, cols = z["S"], z["ids"], list(z["cols"])
    b = np.load(Path.home() / "data/cinc2011_seta_100hz.npz"); assert np.array_equal(b["record"], rec)
    y = b["unacceptable"].astype(int)
    scores = {c: S[:, :, k].max(1) for k, c in enumerate(cols)}
    F = np.nan_to_num(np.concatenate([S.max(1), S.mean(1)], 1))
    skf = StratifiedKFold(10, shuffle=True, random_state=0)
    for n, m in models().items():
        p = np.zeros(len(y))
        for a, t in skf.split(F, y):
            m.fit(F[a], y[a]); p[t] = m.predict_proba(F[t])[:, 1]
        scores[n + "_10foldCV"] = p
    for c in DET:
        ss = []
        for s in (0, 1, 2):
            f = RUNS / f"{c}_s{s}/bundles/cinc2011_v1.npz"
            if f.exists():
                zz = np.load(f); assert np.array_equal(zz["record"], rec); ss.append((1 - zz["lead"]).max(1))
        if ss:
            scores["DET_" + c] = np.mean(ss, 0)
    return y, np.arange(len(y)), scores


def table(name, y, grp, scores):
    ref = scores["DET_cnn_full_leadwise_cleanfrac20"]
    rows = []
    for k, s in scores.items():
        lo, hi = (np.nan, np.nan) if k.startswith("DET_cnn_full_leadwise_cleanfrac20") else boot_delta(y, ref, s, grp)
        rows.append(dict(scorer=k, auroc=auc(y, s), det_minus_this=auc(y, ref) - auc(y, s), ci_lo=lo, ci_hi=hi,
                         n_pos=int(np.sum(y)), n=len(y)))
    T = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    T.to_csv(Q / f"T_compare_{name}.csv", index=False)
    print(name); print(T.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    table("cinc", *cinc())
    table("ptbxl", *ptbxl())
