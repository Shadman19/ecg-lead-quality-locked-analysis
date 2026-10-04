"""External replication dataset: CPSC 2018 (PhysioNet/CinC 2021 training set, folder cpsc_2018), 12-lead, 500 Hz.
Builds a PTB-XL-shaped directory so the locked training/export code runs unchanged with ECG_CLASSES set:
  <out>/ptbxl_database.csv  (index ecg_id; patient_id; strat_fold; scp_codes dict string; filename_lr)
  <out>/scp_statements.csv  (code -> diagnostic=1, diagnostic_class)
  <out>/cache_x100.npy      [N, 12, 1000] float32 mV at 100 Hz (first 10 s)
Five classes: NORM (sinus rhythm), AF, AVB (first-degree AV block), BBB (LBBB or RBBB), STC (ST depression or elevation).
Records whose only labels are PAC or PVC are excluded, as PTB-XL records without a superclass are excluded.
Records shorter than 10 s are excluded (the locked pipeline expects 10 s windows; the count is printed and reported).
No patient identifiers are available; each record is treated as one patient (stated in the manuscript).
Usage: python cpsc_bank.py --src ~/data/cpsc2018 --out ~/data/cpsc2018_shaped
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

CODES = {"426783006": "NORM", "164889003": "AF", "270492004": "AVB", "164909002": "BBB", "59118001": "BBB",
         "429622005": "STC", "164931005": "STC"}
CLASSES = ["NORM", "AF", "AVB", "BBB", "STC"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
T = 1000


def read_hea(p):
    lines = p.read_text().splitlines()
    head = lines[0].split(); fs = int(head[2]); n = int(head[3])
    names = [l.split()[8].upper() for l in lines[1:13]]
    gains, bases = [], []
    for l in lines[1:13]:
        g = l.split()[2].split("/")[0]                                        # e.g. 1000.0(0)
        gains.append(float(g.split("(")[0]))
        bases.append(float(g.split("(")[1].rstrip(")")) if "(" in g else 0.0)
    dx = []
    for l in lines:
        if l.startswith("#Dx:") or l.startswith("# Dx:"):
            dx = [d.strip() for d in l.split(":", 1)[1].split(",") if d.strip()]
    return fs, n, names, gains, bases, dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    heas = sorted(a.src.rglob("*.hea"))
    X, rows = [], []
    n_short, n_nolabel = 0, 0
    for i, h in enumerate(heas):
        fs, n, names, gains, bases, dx = read_hea(h)
        assert names == LEADS, (h, names)
        cls = sorted({CODES[d] for d in dx if d in CODES})
        if not cls:
            n_nolabel += 1
            continue
        if n < 10 * fs:
            n_short += 1
            continue
        m = loadmat(h.with_suffix(".mat"))["val"].astype(np.float64)          # [12, n] ADC units
        sig = (m - np.array(bases)[:, None]) / np.array(gains)[:, None]       # mV
        assert fs % 100 == 0
        sig = resample_poly(sig, 1, fs // 100, axis=1)[:, :T]
        assert sig.shape[1] == T
        X.append(sig.astype(np.float32))
        rid = h.stem
        fold = int(hashlib.sha256(rid.encode()).hexdigest(), 16) % 10 + 1
        rows.append(dict(ecg_id=i + 1, record=rid, patient_id=i + 1, strat_fold=fold,
                         scp_codes=str({c: 100.0 for c in cls}), filename_lr=rid, n_samples_orig=n))
        if i % 500 == 0:
            print(i, rid, flush=True)
    df = pd.DataFrame(rows).set_index("ecg_id")
    df.to_csv(a.out / "ptbxl_database.csv")
    pd.DataFrame([dict(code=c, diagnostic=1, diagnostic_class=c) for c in CLASSES]).set_index("code").to_csv(a.out / "scp_statements.csv")
    np.save(a.out / "cache_x100.npy", np.stack(X))
    print("saved", len(df), "records; excluded: no target label", n_nolabel, "; shorter than 10 s", n_short)
    print("fold counts", df.strat_fold.value_counts().sort_index().to_dict())
    for c in CLASSES:
        print(c, int(df.scp_codes.str.contains(f"'{c}'").sum()))


if __name__ == "__main__":
    main()
