"""Round 6 analysis (SECONDARY): BUT QDB endpoints Q1-Q5 of PREREG_ANALYSIS_PLAN.md (Round-6 entry).
Scores are 'higher = worse'. Detector window score = mean over the 12 channel copies of (1 - intact probability); both estimands
(mean of seed-specific AUROCs, three-seed score average). CIs: subject-clustered bootstrap (1,000). Prints '>>' lines and writes
T_butqdb_{task}.csv, T_butqdb_perrecord.csv, T_butqdb_operating.csv to ~/ecg_lock_results/butqdb.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
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


def sens90(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= 0.10]))


def recall_at(y, s, frac):
    k = max(1, int(round(frac * len(s)))); top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(1, y.sum()))


def boots(groups, seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]


def ci(v):
    return np.percentile(v, [2.5, 97.5])


def p(*a):
    print(">>", *a, flush=True)


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)


def main():
    O.mkdir(parents=True, exist_ok=True)
    b = np.load(BANK); cls, maj, rec, subj = b["cls"], b["cls_maj"], b["record"], b["subject"]
    zs = np.load(O / "sqi_butqdb.npz"); S, cols = np.nan_to_num(zs["S"]), list(zs["cols"])
    p("windows", len(cls), "pure", {c: int((cls == c).sum()) for c in (0, 1, 2, 3)}, "majority", {c: int((maj == c).sum()) for c in (1, 2, 3)},
      "records", len(np.unique(rec)), "subjects", len(np.unique(subj)))
    # detector (locked checkpoints) and its sensitivities
    leads, taus = [], []
    for s in (0, 1, 2):
        z = np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/butqdb_v1.npz"); leads.append(z["lead"]); taus.append(float(z["tau_mask"]))
    seeds = [(1 - L).mean(1) for L in leads]; ens = np.mean(seeds, 0)
    sc = {"detector_ens": ens, "detector_ch1_ens": np.mean([(1 - L[:, 0]) for L in leads], 0), "detector_max_ens": np.mean([(1 - L).max(1) for L in leads], 0)}
    for k, c in enumerate(cols):
        sc[f"SQI_{c}"] = S[:, k]
    # transferred supervised index classifiers (PTB-XL folds 1-9 annotations; identical fit to final_numbers.py)
    zp = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); Sp, ids = np.nan_to_num(zp["S"]), zp["ids"]
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    lab = np.stack([parse(v) for v in d.static_noise]) | np.stack([parse(v) for v in d.burst_noise]) | np.stack([parse(v) for v in d.electrodes_problems])
    tr = np.flatnonzero(d.strat_fold.values <= 9); Xtr, ytr = Sp[tr].reshape(-1, len(cols)), lab[tr].ravel()
    sc["SUP_gbm_ptbxl"] = gbm().fit(Xtr, ytr).predict_proba(S)[:, 1]
    sc["SUP_logreg_ptbxl"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(S)[:, 1]
    for kind in ("sup", "lp", "lpx", "ae"):
        fs = [O / f"sqa_{kind}_s{s}.npy" for s in (0, 1, 2) if (O / f"sqa_{kind}_s{s}.npy").exists()]
        if fs:
            sc[f"sqa_{kind}_ens"] = np.mean([np.load(f).mean(1) for f in fs], 0)
    tasks = {"Q1_c3_vs_c1": (cls, {3}, {1}), "Q3_c23_vs_c1": (cls, {2, 3}, {1}), "Q4_c3_vs_c12": (cls, {3}, {1, 2}), "Q1maj_c3_vs_c1": (maj, {3}, {1})}
    rows_op = []
    for ti, (tname, (lb, posset, negset)) in enumerate(tasks.items()):
        keep = np.isin(lb, list(posset | negset)); y = np.isin(lb[keep], list(posset)).astype(int); g = subj[keep]
        # within-dataset ceiling: leave-one-subject-out GBM on the seven indices (uses BUT QDB labels)
        ceil = np.zeros(len(y))
        for u in np.unique(g):
            te = g == u
            if y[~te].min() == y[~te].max():
                ceil[te] = 0.5; continue
            ceil[te] = gbm().fit(S[keep][~te], y[~te]).predict_proba(S[keep][te])[:, 1]
        scores = {k: v[keep] for k, v in sc.items()}; scores["CEIL_gbm_loso_butqdb"] = ceil
        seed_sc = [s_[keep] for s_ in seeds]
        bs = boots(g, 11 + ti)
        per = [auc(y, s_) for s_ in seed_sc]; smean = float(np.mean(per))
        stat = lambda ix: np.mean([auc(y[ix], s_[ix]) for s_ in seed_sc])
        sm_b = np.array([stat(ix) for ix in bs]); ens_b = np.array([auc(y[ix], scores["detector_ens"][ix]) for ix in bs])
        p(tname, "n", len(y), "pos", int(y.sum()), "detector per-seed", np.round(per, 4), "seed-mean", round(smean, 4), "CI", np.round(ci(sm_b), 4),
          "ens", round(auc(y, scores["detector_ens"]), 4), "CI", np.round(ci(ens_b), 4))
        rows = []
        for k, s_ in scores.items():
            a_b = np.array([auc(y[ix], s_[ix]) for ix in bs]); a0 = auc(y, s_)
            rows.append(dict(task=tname, scorer=k, auroc=a0, ci_lo=ci(a_b)[0], ci_hi=ci(a_b)[1], auprc=average_precision_score(y, s_), sens90=sens90(y, s_),
                             ens_minus=auc(y, scores["detector_ens"]) - a0, ens_minus_lo=ci(ens_b - a_b)[0], ens_minus_hi=ci(ens_b - a_b)[1],
                             seedmean_minus=smean - a0, seedmean_minus_lo=ci(sm_b - a_b)[0], seedmean_minus_hi=ci(sm_b - a_b)[1], n=len(y), n_pos=int(y.sum())))
        T = pd.DataFrame(rows); T.to_csv(O / f"T_butqdb_{tname}.csv", index=False)
        f3 = lambda v: f"{v:.3f}"
        for r in T.itertuples():
            p(tname.split("_")[0], r.scorer, f3(r.auroc) + "[" + f3(r.ci_lo) + "," + f3(r.ci_hi) + "]", "dE", f3(r.ens_minus) + "[" + f3(r.ens_minus_lo) + ","
              + f3(r.ens_minus_hi) + "]", "dS", f3(r.seedmean_minus) + "[" + f3(r.seedmean_minus_lo) + "," + f3(r.seedmean_minus_hi) + "]", "s90", f3(r.sens90))
        # Q5 operating points (score average) and flags at each seed's locked tau_mask (mean channel intact probability < tau)
        e = scores["detector_ens"]
        op = dict(task=tname, sens90=sens90(y, e), recall_top5=recall_at(y, e, 0.05), recall_top10=recall_at(y, e, 0.10))
        for s in (0, 1, 2):
            fl = (1 - seed_sc[s]) < taus[s]
            op[f"flag_pos_s{s}"] = float(fl[y == 1].mean()); op[f"flag_neg_s{s}"] = float(fl[y == 0].mean()); op[f"tau_s{s}"] = taus[s]
        rows_op.append(op); p(tname.split("_")[0], "OP", " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in op.items() if k != "task"))
        if tname == "Q1_c3_vs_c1":
            best_sqi = max([f"SQI_{c}" for c in cols], key=lambda k: auc(y, scores[k]))
            pr = []
            for r_ in np.unique(rec[keep]):
                m = rec[keep] == r_
                if y[m].min() != y[m].max():
                    pr.append(dict(record=r_, n=int(m.sum()), n_pos=int(y[m].sum()), detector_ens=auc(y[m], e[m]), best_sqi=best_sqi,
                                   best_sqi_auroc=auc(y[m], scores[best_sqi][m]), sup_gbm_ptbxl=auc(y[m], scores["SUP_gbm_ptbxl"][m]), ceiling=auc(y[m], ceil[m])))
            P = pd.DataFrame(pr); P.to_csv(O / "T_butqdb_perrecord.csv", index=False)
            p("PER-RECORD (Q1) records", len(P), "detector median/min/max", np.round([P.detector_ens.median(), P.detector_ens.min(), P.detector_ens.max()], 3),
              best_sqi, np.round([P.best_sqi_auroc.median(), P.best_sqi_auroc.min(), P.best_sqi_auroc.max()], 3),
              "sup_gbm", np.round([P.sup_gbm_ptbxl.median(), P.sup_gbm_ptbxl.min(), P.sup_gbm_ptbxl.max()], 3),
              "records where detector > best SQI", int((P.detector_ens > P.best_sqi_auroc).sum()))
    pd.DataFrame(rows_op).to_csv(O / "T_butqdb_operating.csv", index=False)
    meta = json.loads(BANK.with_suffix(".json").read_text()); p("bank sha", meta["bank_sha256"][:12], "counts", meta["total_pure"], meta["total_majority"])


if __name__ == "__main__":
    main()
