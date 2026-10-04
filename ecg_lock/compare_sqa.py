"""Final comparison for the label-free SQA paper, following SQA_PLAN.md.
Outputs (in ~/ecg_lock_results/sqa): T_sqa_ptbxl.csv, T_sqa_cinc.csv, T_sqa_ablation.csv, T_sqa_perlead.csv, T_sqa_select.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sqa import FAULTS, LEADS, parse

H = Path.home(); R = H / "ecg_lock_results/sqa"; Q = H / "ecg_lock_results/quality"
RUNS = Path("/scratch/spath004/ecg_lock_runs")


def auc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sens_at_spec(y, s, spec=0.90):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= 1 - spec])) if np.any(fpr <= 1 - spec) else 0.0


def boot(y, a, b, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    d = []
    for _ in range(B):
        ix = np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))])
        d.append(auc(y[ix], a[ix]) - auc(y[ix], b[ix]))
    return np.percentile(d, [2.5, 97.5])


def prank(s):
    return rankdata(s) / len(s)


def run_scores(prefix, key, seeds=(0, 1, 2)):
    fs = [R / f"{prefix}_s{s}.npz" for s in seeds if (R / f"{prefix}_s{s}.npz").exists()]
    return (np.mean([np.load(f)[key] for f in fs], 0), len(fs)) if fs else (None, 0)


def table(y, grp, scores, primary, name):
    rows = []
    for k, s in scores.items():
        lo, hi = (np.nan, np.nan) if k == primary else boot(y, scores[primary], s, grp)
        rows.append(dict(scorer=k, auroc=auc(y, s), auprc=average_precision_score(y, s), sens_at_90spec=sens_at_spec(y, s),
                         primary_minus_this=auc(y, scores[primary]) - auc(y, s), ci_lo=lo, ci_hi=hi, n_pos=int(np.sum(y)), n=len(y)))
    T = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    T.to_csv(R / f"T_sqa_{name}.csv", index=False); print("\n==", name); print(T.round(3).to_string(index=False), flush=True)
    return T


def main():
    # ---- primary selection (label-free rule)
    sel = {}
    for k in ("lp", "lpx"):
        v = [json.loads((R / f"{k}_s{s}.json").read_text())["best_val_auroc"] for s in (0, 1, 2) if (R / f"{k}_s{s}.json").exists()]
        sel[k] = float(np.mean(v))
    primary = max(sel, key=sel.get)
    pd.DataFrame([dict(kind=k, mean_synthetic_val_auroc=v, chosen=(k == primary)) for k, v in sel.items()]).to_csv(R / "T_sqa_select.csv", index=False)
    print("selection", sel, "PRIMARY =", primary)
    P = "LF_" + primary

    # ---- PTB-XL
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = z["S"], z["ids"], list(z["cols"])
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    stat = np.stack([parse(v) for v in d.static_noise]); burst = np.stack([parse(v) for v in d.burst_noise])
    elec = np.stack([parse(v) for v in d.electrodes_problems]); lab = stat | burst | elec
    fold = d.strat_fold.values; pos = {e: i for i, e in enumerate(ids)}
    tid = np.load(R / "lp_s0.npz")["ptb_test_ids"]; ti = np.array([pos[e] for e in tid])
    vid = np.load(R / "lp_s0.npz")["ptb_val_ids"]; vi = np.array([pos[e] for e in vid])
    y = lab[ti].ravel(); grp = np.repeat(d.patient_id.values[ti], 12)
    # best classical SQI chosen on fold 9 labels
    val_auc = {c: auc(lab[vi].ravel(), S[vi, :, k].ravel()) for k, c in enumerate(cols)}
    best_sqi = max(val_auc, key=val_auc.get); print("fold-9 SQI AUROCs", val_auc, "best", best_sqi)
    sc = {f"SQI_{c}": S[ti, :, k].ravel() for k, c in enumerate(cols)}
    tr = np.flatnonzero(fold <= 9)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    gbm.fit(np.nan_to_num(S[tr].reshape(-1, len(cols))), lab[tr].ravel())
    sc["SUP_SQI_GBM"] = gbm.predict_proba(np.nan_to_num(S[ti].reshape(-1, len(cols))))[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.nan_to_num(S[tr].reshape(-1, len(cols))), lab[tr].ravel())
    sc["SUP_SQI_LOGREG"] = lr.predict_proba(np.nan_to_num(S[ti].reshape(-1, len(cols))))[:, 1]
    for k, tag in (("lp", "LF_lp"), ("lpx", "LF_lpx"), ("ae", "LF_ae"), ("sup", "SUP_deep_lp")):
        s, n = run_scores(k, "ptb_test")
        if s is not None:
            sc[tag] = s.ravel()
    pil = [1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/test_clean.npz")["lead_score_clean"] for s in (0, 1, 2)]
    ref_ids = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz")["ecg_id"]
    assert np.array_equal(ref_ids, tid)
    sc["PILOT_detector"] = np.mean(pil, 0).ravel()
    sc["COMBO_primary+tSQI"] = (prank(sc[P]) + prank(sc["SQI_tsqi_bad"])) / 2
    T = table(y, grp, sc, P, "ptbxl")
    # subsets and per-lead
    rows = []
    for nm, Ls in (("static", stat), ("burst", burst), ("electrodes", elec)):
        yy = Ls[ti].ravel(); keep = yy | ~lab[ti].ravel()        # positives of this type vs clean leads
        for k in (P, "SUP_deep_lp", "SUP_SQI_GBM", f"SQI_{best_sqi}"):
            if k in sc:
                rows.append(dict(subset=nm, scorer=k, auroc=auc(yy[keep], sc[k][keep]), n_pos=int(yy.sum())))
    Pm = sc[P].reshape(-1, 12); Lm = lab[ti]
    for j, L in enumerate(LEADS):
        rows.append(dict(subset=f"lead_{L}", scorer=P, auroc=auc(Lm[:, j], Pm[:, j]), n_pos=int(Lm[:, j].sum())))
    pd.DataFrame(rows).to_csv(R / "T_sqa_perlead.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- CinC (record level, max over leads)
    zc = np.load(Q / "sqi_cinc.npz", allow_pickle=True); Sc = zc["S"]; rec = zc["ids"]
    b = np.load(H / "data/cinc2011_seta_100hz.npz"); yc = b["unacceptable"].astype(int); assert np.array_equal(b["record"], rec)
    cc = {f"SQI_{c}": Sc[:, :, k].max(1) for k, c in enumerate(cols)}
    F = np.nan_to_num(np.concatenate([Sc.max(1), Sc.mean(1)], 1)); skf = StratifiedKFold(10, shuffle=True, random_state=0)
    p = np.zeros(len(yc))
    for tr_, te_ in skf.split(F, yc):
        p[te_] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(F[tr_], yc[tr_]).predict_proba(F[te_])[:, 1]
    cc["SUP_SQI_GBM_CinC10foldCV"] = p
    for k, tag in (("lp", "LF_lp"), ("lpx", "LF_lpx"), ("ae", "LF_ae"), ("sup", "SUP_deep_lp_PTBXLlabels")):
        s, n = run_scores(k, "cinc")
        if s is not None:
            cc[tag] = s.max(1)
    cc["PILOT_detector"] = np.mean([(1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/cinc2011_v1.npz")["lead"]).max(1)
                                    for s in (0, 1, 2)], 0)
    cc["COMBO_primary+tSQI"] = (prank(cc[P]) + prank(cc["SQI_tsqi_bad"])) / 2
    table(yc, np.arange(len(yc)), cc, P, "cinc")

    # ---- ablation (lpx, seed 0)
    base = np.load(R / "lpx_s0.npz"); rows = []
    for f in [""] + FAULTS:
        fn = R / (f"lpx_no{f}_s0.npz" if f else "lpx_s0.npz")
        if fn.exists():
            zz = np.load(fn)
            rows.append(dict(dropped=f or "none", ptbxl_perlead_auroc=auc(y, zz["ptb_test"].ravel()),
                             cinc_record_auroc=auc(yc, zz["cinc"].max(1)),
                             synth_val=json.loads(fn.with_suffix(".json").read_text())["best_val_auroc"]))
    pd.DataFrame(rows).to_csv(R / "T_sqa_ablation.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))
    params = {k: json.loads((R / f"{k}_s0.json").read_text())["n_params"] for k in ("lp", "lpx", "ae", "sup") if (R / f"{k}_s0.json").exists()}
    print("params", params)


if __name__ == "__main__":
    main()
