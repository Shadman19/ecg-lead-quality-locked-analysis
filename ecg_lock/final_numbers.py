"""Numbers for the v9 manuscript: per-seed-mean convention with patient/record bootstrap CIs, plus ensemble values.
Prints compact lines prefixed with '>>' for easy reading."""
import glob
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

from eval_ptbxl_quality import parse

H = Path.home(); Q = H / "ecg_lock_results/quality"; RUNS = Path("/scratch/spath004/ecg_lock_runs"); B = 1000


def auc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sens90(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= 0.10]))


def boots(groups, seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rows = [np.flatnonzero(groups == g) for g in ug]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(B)]


def ci(vals):
    return np.percentile(vals, [2.5, 97.5])


def p(*a):
    print(">>", *a, flush=True)


def report(name, y, grp, seed_scores, comps, seed):
    """seed_scores: list of per-seed score arrays for the detector; comps: dict name -> score array."""
    bs = boots(grp, seed)
    per = [auc(y, s) for s in seed_scores]; ens = np.mean(seed_scores, 0)
    stat = lambda ix: np.mean([auc(y[ix], s[ix]) for s in seed_scores])
    p(name, "per-seed", np.round(per, 4), "mean", round(np.mean(per), 4), "CI", np.round(ci([stat(ix) for ix in bs]), 4),
      "ensemble", round(auc(y, ens), 4), "ens CI", np.round(ci([auc(y[ix], ens[ix]) for ix in bs]), 4),
      "AUPRC ens", round(average_precision_score(y, ens), 4), "sens@90spec ens", round(sens90(y, ens), 4))
    for k, c in comps.items():
        d_seed = [stat(ix) - auc(y[ix], c[ix]) for ix in bs]
        d_ens = [auc(y[ix], ens[ix]) - auc(y[ix], c[ix]) for ix in bs]
        p(name, "vs", k, "comp AUROC", round(auc(y, c), 4), "comp CI", np.round(ci([auc(y[ix], c[ix]) for ix in bs]), 4),
          "| seedmean delta", round(np.mean(per) - auc(y, c), 4), np.round(ci(d_seed), 4),
          "| ens delta", round(auc(y, ens) - auc(y, c), 4), np.round(ci(d_ens), 4),
          "| comp AUPRC", round(average_precision_score(y, c), 4), "sens@90", round(sens90(y, c), 4))
    return ens


def main():
    # ---------------- PTB-XL annotations
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = z["S"], z["ids"], list(z["cols"])
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    stat_ = np.stack([parse(v) for v in d.static_noise]); burst = np.stack([parse(v) for v in d.burst_noise])
    elec = np.stack([parse(v) for v in d.electrodes_problems]); lab = stat_ | burst | elec
    fold = d.strat_fold.values; pos = {e: i for i, e in enumerate(ids)}
    ref = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz"); tid = ref["ecg_id"]
    ti = np.array([pos[e] for e in tid]); y = lab[ti].ravel(); grp = np.repeat(d.patient_id.values[ti], 12)
    p("PTBXL n leads", len(y), "n pos", int(y.sum()), "records with any", int(lab[ti].any(1).sum()),
      "static leads", int(stat_[ti].sum()), "burst leads", int(burst[ti].sum()), "electrode leads", int(elec[ti].sum()),
      "records annotated static/burst/elec", int(stat_[ti].any(1).sum()), int(burst[ti].any(1).sum()), int(elec[ti].any(1).sum()))
    tr = np.flatnonzero(fold <= 9); Xtr = np.nan_to_num(S[tr].reshape(-1, len(cols))); ytr = lab[tr].ravel(); Xte = np.nan_to_num(S[ti].reshape(-1, len(cols)))
    comps = {c: S[ti, :, k].ravel() for k, c in enumerate(cols)}
    comps["sup_gbm"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    comps["sup_logreg"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    R = H / "ecg_lock_results/sqa"
    for k in ("sup", "ae", "lpx", "lp"):
        fs = sorted(glob.glob(str(R / f"{k}_s?.npz")))
        if fs:
            comps[f"sqa_{k}"] = np.mean([np.load(f)["ptb_test"] for f in fs], 0).ravel()
    seeds = [(1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/test_clean.npz")["lead_score_clean"]).ravel() for s in (0, 1, 2)]
    ens = report("PTBXL pilot", y, grp, seeds, comps, 1)
    seeds2 = [(1 - np.load(RUNS / f"cnn_full_leadwise_s{s}/bundles/test_clean.npz")["lead_score_clean"]).ravel() for s in (0, 1, 2)]
    report("PTBXL leadwise-nocf", y, grp, seeds2, {"sup_gbm": comps["sup_gbm"]}, 2)
    # subsets (positives of one type vs unannotated leads)
    for nm, Ls in (("static", stat_), ("burst", burst)):
        yy = Ls[ti].ravel(); keep = yy | ~y
        p("PTBXL subset", nm, "pilot seedmean", round(np.mean([auc(yy[keep], s[keep]) for s in seeds]), 4), "ens", round(auc(yy[keep], ens[keep]), 4),
          "sup_gbm", round(auc(yy[keep], comps["sup_gbm"][keep]), 4), "ksqi", round(auc(yy[keep], comps["ksqi_bad"][keep]), 4), "tsqi", round(auc(yy[keep], comps["tsqi_bad"][keep]), 4))
    combo = (rankdata(ens) + rankdata(comps["tsqi_bad"])) / 2
    p("PTBXL combo(pilot ens + tsqi) AUROC", round(auc(y, combo), 4))
    # ---------------- CinC
    zc = np.load(Q / "sqi_cinc.npz", allow_pickle=True); Sc = zc["S"]; rec = zc["ids"]
    b = np.load(H / "data/cinc2011_seta_100hz.npz"); yc = b["unacceptable"].astype(int); assert np.array_equal(b["record"], rec)
    cc = {c: Sc[:, :, k].max(1) for k, c in enumerate(cols)}
    F = np.nan_to_num(np.concatenate([Sc.max(1), Sc.mean(1)], 1)); pr = np.zeros(len(yc))
    for a_, t_ in StratifiedKFold(10, shuffle=True, random_state=0).split(F, yc):
        pr[t_] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(F[a_], yc[a_]).predict_proba(F[t_])[:, 1]
    cc["sup_gbm_cv"] = pr
    for k in ("sup", "ae", "lpx", "lp"):
        fs = sorted(glob.glob(str(R / f"{k}_s?.npz")))
        if fs:
            cc[f"sqa_{k}"] = np.mean([np.load(f)["cinc"] for f in fs], 0).max(1)
    cs = []
    for s in (0, 1, 2):
        zz = np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/cinc2011_v1.npz"); assert np.array_equal(zz["record"], rec); cs.append((1 - zz["lead"]).max(1))
    ensc = report("CINC pilot", yc, np.arange(len(yc)), cs, cc, 3)
    combo_c = (rankdata(ensc) + rankdata(cc["tsqi_bad"])) / 2
    bs = boots(np.arange(len(yc)), 4)
    p("CINC combo(pilot ens + tsqi)", round(auc(yc, combo_c), 4), "CI", np.round(ci([auc(yc[ix], combo_c[ix]) for ix in bs]), 4),
      "combo minus tsqi", round(auc(yc, combo_c) - auc(yc, cc["tsqi_bad"]), 4), np.round(ci([auc(yc[ix], combo_c[ix]) - auc(yc[ix], cc["tsqi_bad"][ix]) for ix in bs]), 4))
    # ---------------- gated
    T = pd.read_csv(H / "ecg_lock_results/gated/T_gated_contrasts.csv")
    for _, r in T[(T.tau == "tau")].iterrows():
        p("GATED", r.classifier, r.condition, "plain", round(r.plain_auroc, 4), "delta", round(r.delta, 4), "95CI", round(r.ci95_lo, 4), round(r.ci95_hi, 4), "97.5CI", round(r.ci975_lo, 4), round(r.ci975_hi, 4))
    G = pd.read_csv(H / "ecg_lock_results/gated/T_gate_stats.csv")
    for _, r in G[G.tau == "tau"].iterrows():
        p("GATE", r.condition, "frac_gated", round(r.frac_gated, 4), "frac_corrupted", round(r.frac_corrupted, 4), "prec", round(r.precision, 3), "rec", round(r.recall, 3))
    # ---------------- 10k precision check
    for dname in ("ecg_lock_results", "ecg_lock_results_b10k"):
        for f in sorted(glob.glob(str(H / dname / "PRIMARY*.csv"))) + sorted(glob.glob(str(H / dname / "*primary*.csv"))):
            p(dname, Path(f).name); print(pd.read_csv(f).to_string(index=False))
        p(dname, "files", sorted(Path(H / dname).glob("*.csv"))[:12])


if __name__ == "__main__":
    main()
