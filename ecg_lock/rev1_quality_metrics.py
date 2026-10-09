"""Revision 1 (Oct 2026), existing outputs only, no retraining.
Point 4: PTB-XL recorded lead annotations: per-lead AUROC + AUPRC (patient bootstrap), operating point at each seed's locked tau_mask
(sens, spec, precision, F1), and every comparator at the detector's specificity.
Point 5: BUT QDB channel-aggregation sensitivity (mean, median, max, min, channel I, channel II), seed mean and score average,
subject-clustered bootstrap. Writes ~/ecg_lock_results_rev1/quality/*.csv"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_ptbxl_quality import parse
H = Path.home(); RUNS = Path("/scratch/spath004/ecg_lock_runs"); Q = H / "ecg_lock_results/quality"; BQ = H / "ecg_lock_results/butqdb"
OUT = H / "ecg_lock_results_rev1/quality"; OUT.mkdir(parents=True, exist_ok=True); B = 1000; P = "cnn_full_leadwise_cleanfrac20"
def boots(g, seed):
    rng = np.random.default_rng(seed); u = np.unique(g); rows = [np.flatnonzero(g == x) for x in u]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(B)]
def ci(v): return np.nanpercentile(v, [2.5, 97.5])
def op(y, flag):
    tp = (flag & y).sum(); fp = (flag & ~y).sum(); fn = (~flag & y).sum(); tn = (~flag & ~y).sum()
    se = tp / max(tp + fn, 1); sp = tn / max(tn + fp, 1); pr = tp / max(tp + fp, 1); f1 = 2 * pr * se / max(pr + se, 1e-12)
    return dict(sens=se, spec=sp, prec=pr, f1=f1, flag_rate=flag.mean())
def at_spec(y, s, spec):
    thr = np.quantile(s[~y], spec); return op(y, s > thr)
def ptbxl():
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = np.nan_to_num(z["S"]), z["ids"], list(z["cols"])
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    lab = np.stack([parse(v) for v in d.static_noise]) | np.stack([parse(v) for v in d.burst_noise]) | np.stack([parse(v) for v in d.electrodes_problems])
    pos = {e: i for i, e in enumerate(ids)}; tr = np.flatnonzero(d.strat_fold.values <= 9)
    zz = [np.load(RUNS / f"{P}_s{s}/bundles/test_clean.npz") for s in (0, 1, 2)]; ti = np.array([pos[e] for e in zz[0]["ecg_id"]])
    taus = [json.loads((RUNS / f"{P}_s{s}/lock.json").read_text())["tau_mask"] for s in (0, 1, 2)]
    intact = [q["lead_score_clean"] for q in zz]; Y = lab[ti].astype(bool); y = Y.ravel(); g = np.repeat(d.patient_id.values[ti], 12)
    Xtr, ytr = S[tr].reshape(-1, len(cols)), lab[tr].ravel(); Xte = S[ti].reshape(-1, len(cols))
    sc = {"detector_ens": np.mean([1 - i for i in intact], 0).ravel()}
    for s in (0, 1, 2): sc[f"detector_s{s}"] = (1 - intact[s]).ravel()
    sc["SUP_gbm"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    sc["SUP_logreg"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    for k, c in enumerate(cols): sc[f"SQI_{c}"] = Xte[:, k]
    bs = boots(g, 41); rows = []
    det_ops = [op(y, (intact[s].ravel() < taus[s])) for s in (0, 1, 2)]
    spec_ref = float(np.mean([o["spec"] for o in det_ops]))
    for k, s_ in sc.items():
        ab = [average_precision_score(y[ix], s_[ix]) for ix in bs]
        r = dict(scorer=k, auroc=roc_auc_score(y, s_), auprc=average_precision_score(y, s_), auprc_lo=ci(ab)[0], auprc_hi=ci(ab)[1],
                 prevalence=y.mean(), n_leads=len(y), n_pos=int(y.sum()), ref_spec=spec_ref)
        r.update({f"atspec_{a}": b for a, b in at_spec(y, s_, spec_ref).items()}); rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT / "T_ptbxl_annot_auprc.csv", index=False)
    O = pd.DataFrame([dict(seed=s, tau=taus[s], **det_ops[s]) for s in (0, 1, 2)]); O.to_csv(OUT / "T_ptbxl_annot_locked_threshold.csv", index=False)
    e = sc["detector_ens"]
    for comp in ("SUP_gbm", "SUP_logreg", "SQI_tsqi_bad"):
        if comp not in sc: continue
        db = [average_precision_score(y[ix], e[ix]) - average_precision_score(y[ix], sc[comp][ix]) for ix in bs]
        print(">> dAUPRC ens -", comp, round(average_precision_score(y, e) - average_precision_score(y, sc[comp]), 4), np.round(ci(db), 4))
    print(T[["scorer", "auroc", "auprc", "auprc_lo", "auprc_hi", "atspec_sens", "atspec_prec", "atspec_f1"]].round(3).to_string()); print(O.round(3).to_string())
def butqdb():
    b = np.load(H / "data/butqdb_windows_100hz.npz"); cls, subj = b["cls"], b["subject"]
    zs = np.load(BQ / "sqi_butqdb.npz"); S, cols = np.nan_to_num(zs["S"]), list(zs["cols"])
    leads = [np.load(RUNS / f"{P}_s{s}/bundles/butqdb_v1.npz")["lead"] for s in (0, 1, 2)]
    agg = {"mean12": lambda c: c.mean(1), "median12": lambda c: np.median(c, 1), "max12": lambda c: c.max(1), "min12": lambda c: c.min(1),
           "ch_I": lambda c: c[:, 0], "ch_II": lambda c: c[:, 1], "ch_V1": lambda c: c[:, 6], "ch_V5": lambda c: c[:, 10]}
    tasks = {"c3_vs_c1": ({3}, {1}), "c23_vs_c1": ({2, 3}, {1}), "c3_vs_c12": ({3}, {1, 2})}; rows = []
    for ti_, (tn, (ps, ns)) in enumerate(tasks.items()):
        keep = np.isin(cls, list(ps | ns)); y = np.isin(cls[keep], list(ps)); g = subj[keep]; bs = boots(g, 51 + ti_)
        best = max(cols, key=lambda c: roc_auc_score(y, S[keep][:, cols.index(c)])); bsq = S[keep][:, cols.index(best)]
        for an, f in agg.items():
            per = [f(1 - L[keep]) for L in leads]; ens = np.mean(per, 0)
            sm_b = [np.mean([roc_auc_score(y[ix], q[ix]) for q in per]) for ix in bs]; en_b = [roc_auc_score(y[ix], ens[ix]) for ix in bs]
            d_b = [roc_auc_score(y[ix], ens[ix]) - roc_auc_score(y[ix], bsq[ix]) for ix in bs]
            rows.append(dict(task=tn, aggregation=an, seed_aucs=[round(roc_auc_score(y, q), 4) for q in per],
                             seedmean=np.mean([roc_auc_score(y, q) for q in per]), seedmean_lo=ci(sm_b)[0], seedmean_hi=ci(sm_b)[1],
                             ens=roc_auc_score(y, ens), ens_lo=ci(en_b)[0], ens_hi=ci(en_b)[1], ens_auprc=average_precision_score(y, ens),
                             best_sqi=best, best_sqi_auroc=roc_auc_score(y, bsq), ens_minus_best=roc_auc_score(y, ens) - roc_auc_score(y, bsq),
                             d_lo=ci(d_b)[0], d_hi=ci(d_b)[1]))
    T = pd.DataFrame(rows); T.to_csv(OUT / "T_butqdb_channel_sensitivity.csv", index=False)
    print(T.drop(columns=["seed_aucs"]).round(3).to_string())
if __name__ == "__main__":
    ptbxl(); butqdb()
