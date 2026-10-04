"""EXPLORATORY feasibility: per-lead detection of human-annotated noise in real PTB-XL test records (fold 10).
Labels from ptbxl_database.csv: static_noise, burst_noise, electrodes_problems (lead lists / ranges, 'alles' = all).
baseline_drift is reported separately (training label excludes baseline wander). Scores on the unaltered test records.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from corruption import rule_hf_energy, rule_variance

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RUNS = Path("/scratch/spath004/ecg_lock_runs")
DB = Path.home() / "cardio ML/ptbxl/ptbxl_database.csv"


def parse(s):
    m = np.zeros(12, bool)
    if not isinstance(s, str):
        return m
    s = s.upper().replace(" ", "")
    if "ALLES" in s:
        m[:] = True; return m
    for part in s.replace(";", ",").split(","):
        part = part.strip("?")
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip("?"), b.strip("?")
            if a in LEADS and b in LEADS:
                i, j = LEADS.index(a), LEADS.index(b)
                m[min(i, j):max(i, j) + 1] = True
        elif part in LEADS:
            m[LEADS.index(part)] = True
    return m


def auc(y, s):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    d = pd.read_csv(DB, index_col="ecg_id")
    ref = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz")
    ids = ref["ecg_id"]
    dd = d.loc[ids]
    lab = {c: np.stack([parse(v) for v in dd[c].values]) for c in ["static_noise", "burst_noise", "electrodes_problems", "baseline_drift"]}
    Y = lab["static_noise"] | lab["burst_noise"] | lab["electrodes_problems"]
    print("test records", len(ids), "noisy leads", int(Y.sum()), "records with any", int(Y.any(1).sum()),
          "unparsed static", int((dd.static_noise.notna().values & ~lab["static_noise"].any(1)).sum()))
    # rules on the unaltered, standardized test signals (same as models see)
    sys.path.insert(0, ".")
    from data import load_labels, load_signals, split_indices
    cfg = json.loads((RUNS / "cnn_full_leadwise_cleanfrac20_s0/config.json").read_text())
    ptb = DB.parent
    df = load_labels(ptb); X = load_signals(ptb, df, ptb / "cache_x100.npy")
    ii = split_indices(df, cfg["split"])["test"]
    assert np.array_equal(df.index.values[ii], ids)
    mu, sd = np.array(cfg["norm_mu"], np.float32), np.array(cfg["norm_sd"], np.float32)
    x = ((X[ii] - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32)
    rv, rh = -rule_variance(x), -rule_hf_energy(x)
    comb = np.maximum(rankdata(rv.ravel()), rankdata(rh.ravel())).reshape(rv.shape)
    scores = {"rule_variance": rv, "rule_hf": rh, "rule_combined": comb, "rule_hf_inverted": -rh}
    for cfgname in ["cnn_full_leadwise_cleanfrac20", "cnn_full_leadwise", "tx_full", "cnn_full", "cnn_fused_gated"]:
        ss = []
        for s in (0, 1, 2):
            f = RUNS / f"{cfgname}_s{s}/bundles/test_clean.npz"
            if f.exists():
                z = np.load(f)
                if "lead_score_clean" in z:
                    ss.append(1 - z["lead_score_clean"])
        if ss:
            scores[cfgname] = np.mean(ss, 0)
    rows = []
    for name, sc in scores.items():
        r = dict(scorer=name, perlead_auroc_noise=auc(Y.ravel(), sc.ravel()),
                 record_auroc_max=auc(Y.any(1), sc.max(1)),
                 perlead_auroc_static=auc(lab["static_noise"].ravel(), sc.ravel()),
                 perlead_auroc_burst=auc(lab["burst_noise"].ravel(), sc.ravel()),
                 perlead_auroc_drift=auc(lab["baseline_drift"].ravel(), sc.ravel()))
        rows.append(r)
    T = pd.DataFrame(rows)
    out = Path.home() / "ecg_lock_results/quality"; out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "T_ptbxl_real_noise.csv", index=False)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
