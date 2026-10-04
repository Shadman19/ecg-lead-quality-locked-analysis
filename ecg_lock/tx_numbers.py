import numpy as np, pandas as pd
from final_numbers import auc, parse, H, Q, RUNS
z = np.load(Q / "sqi_ptbxl.npz", allow_pickle=True); ids = z["ids"]
d = pd.read_csv(H / "cardio ML/ptbxl/ptbxl_database.csv", index_col="ecg_id").loc[ids]
lab = np.stack([parse(v) for v in d.static_noise]) | np.stack([parse(v) for v in d.burst_noise]) | np.stack([parse(v) for v in d.electrodes_problems])
pos = {e: i for i, e in enumerate(ids)}
ref = np.load(RUNS / "cnn_full_leadwise_cleanfrac20_s0/bundles/test_clean.npz"); tid = ref["ecg_id"]
ti = np.array([pos[e] for e in tid]); y = lab[ti].ravel()
b = np.load(H / "data/cinc2011_seta_100hz.npz"); yc = b["unacceptable"].astype(int)
for cfg in ("tx_full", "cnn_full_leadwise", "cnn_full_leadwise_cleanfrac20", "cnn_full", "cnn_full_cleanfrac20", "cnn_diag_lead", "cnn_fused_gated"):
    try:
        ss = [(1 - np.load(RUNS / f"{cfg}_s{s}/bundles/test_clean.npz")["lead_score_clean"]).ravel() for s in (0, 1, 2)]
        per = [auc(y, s) for s in ss]
        cs = [(1 - np.load(RUNS / f"{cfg}_s{s}/bundles/cinc2011_v1.npz")["lead"]).max(1) for s in (0, 1, 2)]
        perc = [auc(yc, s) for s in cs]
        print(">>", cfg, "PTBXL per-seed", np.round(per, 4), "mean", round(np.mean(per), 4), "ens", round(auc(y, np.mean(ss, 0)), 4), "| CINC per-seed", np.round(perc, 4), "mean", round(np.mean(perc), 4), "ens", round(auc(yc, np.mean(cs, 0)), 4))
    except Exception as e:
        print(">>", cfg, "ERR", e)
