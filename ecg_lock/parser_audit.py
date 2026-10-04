"""Parser audit and label-definition sensitivity for the PTB-XL recorded lead annotations (secondary, descriptive).
1. Lists every distinct raw annotation string (static_noise, burst_noise, electrodes_problems) with its count in all folds
   and in the test fold, the leads the fixed rule set assigns, and whether the string yielded any lead (T_parser_audit.csv).
2. Recomputes the per-lead AUROC of the locked detector (seed mean and three-seed score average), the supervised GBM,
   the template index and kSQI on the test fold under alternative label definitions:
   base | static only | burst only | records whose annotation is 'alles' removed | unparsed records removed
   | leads of annotated records only (localization within a flagged record) | record level (any lead vs none, max over leads).
Outputs T_label_sensitivity.csv. Nothing here changes any locked result.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from eval_ptbxl_quality import parse, LEADS
from final_numbers import auc, H, Q, RUNS, sens90, boots, ci

FIELDS = ["static_noise", "burst_noise", "electrodes_problems"]


def parse2(s):
    """Sensitivity variant of the fixed rule set: also accepts 'alle' (typo of 'alles') and shorthand ranges 'V3-4'."""
    if not isinstance(s, str):
        return parse(s)
    u = s.upper().replace(" ", "")
    if "ALLE" in u:
        return parse("alles")
    parts = []
    for part in u.replace(";", ",").split(","):
        part = part.strip("?")
        if "-" in part:
            a, b = part.split("-", 1)
            if a.startswith("V") and b.isdigit():
                b = "V" + b
            part = a + "-" + b
        parts.append(part)
    return parse(",".join(parts))


def main():
    z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); S, ids, cols = z["S"], z["ids"], list(z["cols"])
    d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
    fold = d.strat_fold.values
    # ---- 1. audit
    rows = []
    for f in FIELDS:
        raw = d[f].where(d[f].notna(), None)
        for s in sorted({v for v in raw if v is not None}):
            m = parse(s); sel = (raw == s).values
            rows.append(dict(field=f, raw=repr(s), n_all=int(sel.sum()), n_test=int((sel & (fold == 10)).sum()),
                             leads=",".join(np.array(LEADS)[m]) if m.any() else "", recognized=bool(m.any())))
    A = pd.DataFrame(rows); A.to_csv(Q / "T_parser_audit.csv", index=False)
    unrec = A[~A.recognized]
    print(">> distinct strings", len(A), "unrecognized", len(unrec), "records with unrecognized string (all/test)",
          int(unrec.n_all.sum()), int(unrec.n_test.sum()))
    print(">> unrecognized examples", unrec.sort_values("n_all", ascending=False).head(12)[["field", "raw", "n_all", "n_test"]].values.tolist())
    # ---- 2. sensitivity
    P = {f: np.stack([parse(v) for v in d[f]]) for f in FIELDS}
    lab = P["static_noise"] | P["burst_noise"] | P["electrodes_problems"]
    has_txt = np.zeros(len(d), bool); alles = np.zeros(len(d), bool)
    for f in FIELDS:
        v = d[f].astype(str).str.upper()
        has_txt |= d[f].notna().values; alles |= v.str.contains("ALLES").values
    unparsed = has_txt & ~lab.any(1)
    pos = {e: i for i, e in enumerate(ids)}
    ref = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz"); tid = ref["ecg_id"]
    ti = np.array([pos[e] for e in tid]); n = len(ti)
    seeds = [(1 - np.load(RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}/bundles/test_clean.npz")["lead_score_clean"]) for s in (0, 1, 2)]
    ens = np.mean(seeds, 0)                                                   # [n, 12]
    tr = np.flatnonzero(fold <= 9); Xtr = np.nan_to_num(S[tr].reshape(-1, len(cols))); ytr = lab[tr].ravel()
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    G = gbm.predict_proba(np.nan_to_num(S[ti].reshape(-1, len(cols))))[:, 1].reshape(n, 12)
    T = S[ti, :, cols.index("tsqi_bad")]; K = S[ti, :, cols.index("ksqi_bad")]
    L = lab[ti]; AL = alles[ti]; UN = unparsed[ti]
    R = lambda m: np.repeat(m[:, None], 12, 1)                              # record mask -> lead mask
    ST, BU, EL = P["static_noise"][ti], P["burst_noise"][ti], P["electrodes_problems"][ti]
    P2 = {f: np.stack([parse2(v) for v in d[f]]) for f in FIELDS}
    L2 = (P2["static_noise"] | P2["burst_noise"] | P2["electrodes_problems"])[ti]
    print(">> parse2 changes labels of", int((L2 != L).any(1).sum()), "test records;", int(L2.sum() - L.sum()), "more positive leads")
    pid = d.patient_id.values[ti]; PID = np.repeat(pid[:, None], 12, 1)
    defs = {
        "base (all annotations)": (L, np.ones((n, 12), bool)),
        "parser variant (shorthand ranges, 'alle')": (L2, np.ones((n, 12), bool)),
        "static noise only (other annotated leads excluded)": (ST, ST | ~(BU | EL)),
        "burst noise only (other annotated leads excluded)": (BU, BU | ~(ST | EL)),
        "'alles' records removed": (L, R(~AL)),
        "unparsed records removed": (L, R(~UN)),
        "within annotated records": (L, R(L.any(1))),
    }
    out = []
    for name, (Y, keep) in defs.items():
        y = Y[keep]
        r = dict(definition=name, n_records=int(keep.any(1).sum()), n_leads=int(keep.sum()), n_pos=int(y.sum()),
                 det_seedmean=round(float(np.mean([auc(y, s[keep]) for s in seeds])), 4),
                 det_scoreavg=round(float(auc(y, ens[keep])), 4), gbm=round(float(auc(y, G[keep])), 4),
                 tsqi=round(float(auc(y, np.nan_to_num(T)[keep])), 4), ksqi=round(float(auc(y, np.nan_to_num(K)[keep])), 4))
        g = PID[keep]; bs = boots(g, 0)
        d_ens = [auc(y[ix], ens[keep][ix]) - auc(y[ix], G[keep][ix]) for ix in bs]
        d_seed = [np.mean([auc(y[ix], s_[keep][ix]) for s_ in seeds]) - auc(y[ix], G[keep][ix]) for ix in bs]
        r["d_scoreavg_minus_gbm"] = round(r["det_scoreavg"] - r["gbm"], 4); r["d_scoreavg_ci"] = np.round(ci(d_ens), 4).tolist()
        r["d_seedmean_minus_gbm"] = round(r["det_seedmean"] - r["gbm"], 4); r["d_seedmean_ci"] = np.round(ci(d_seed), 4).tolist()
        out.append(r); print(">>", r)
    # record level: any annotated lead vs none; score = max over leads
    yr = L.any(1)
    r = dict(definition="record level (max over leads)", n_records=n, n_leads=None, n_pos=int(yr.sum()),
             det_seedmean=round(float(np.mean([auc(yr, s.max(1)) for s in seeds])), 4), det_scoreavg=round(float(auc(yr, ens.max(1))), 4),
             gbm=round(float(auc(yr, G.max(1))), 4), tsqi=round(float(auc(yr, np.nan_to_num(T).max(1))), 4), ksqi=round(float(auc(yr, np.nan_to_num(K).max(1))), 4))
    bs = boots(pid, 0)
    r["d_scoreavg_minus_gbm"] = round(r["det_scoreavg"] - r["gbm"], 4)
    r["d_scoreavg_ci"] = np.round(ci([auc(yr[ix], ens.max(1)[ix]) - auc(yr[ix], G.max(1)[ix]) for ix in bs]), 4).tolist()
    r["d_seedmean_minus_gbm"] = round(r["det_seedmean"] - r["gbm"], 4)
    r["d_seedmean_ci"] = np.round(ci([np.mean([auc(yr[ix], s_.max(1)[ix]) for s_ in seeds]) - auc(yr[ix], G.max(1)[ix]) for ix in bs]), 4).tolist()
    out.append(r); print(">>", r)
    # localization: within annotated records, fraction of records whose highest-scoring lead is an annotated lead
    ann = np.flatnonzero(L.any(1) & ~AL)
    for nm, sc in (("det_scoreavg", ens), ("gbm", G), ("tsqi", np.nan_to_num(T))):
        top = sc[ann].argmax(1); hits = L[ann, top]; hit = hits.mean()
        chance = L[ann].mean(); bs = boots(pid[ann], 0)
        print(">> top-1 lead hit rate within annotated non-alles records", nm, round(float(hit), 4), "CI", np.round(ci([hits[ix].mean() for ix in bs]), 4).tolist(),
              "chance", round(float(chance), 4), "n", len(ann))
    pd.DataFrame(out).to_csv(Q / "T_label_sensitivity.csv", index=False)
    print(">> counts: alles records in test", int(AL.sum()), "unparsed records in test", int(UN.sum()),
          "annotated records", int(L.any(1).sum()), "sens@90 det_scoreavg base", round(sens90(L.ravel(), ens.ravel()), 4))


if __name__ == "__main__":
    main()
