"""Round 6: the seven published per-lead quality indices (sqi_baselines.lead_sqis, unchanged) on each BUT QDB window (single lead,
100 Hz, mV). Output: sqi_butqdb.npz with S [n, 7] (higher = worse), cols, bank_sha256.
Usage: python sqi_butqdb.py --bank ~/data/butqdb_windows_100hz.npz --out ~/ecg_lock_results/butqdb [--procs 16]
"""
import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sqi_baselines import COLS, lead_sqis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path.home() / "data/butqdb_windows_100hz.npz")
    ap.add_argument("--out", type=Path, default=Path.home() / "ecg_lock_results/butqdb")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X = np.load(a.bank)["X"]
    with Pool(a.procs) as pool:
        S = np.array(pool.map(lead_sqis, list(X), chunksize=64), np.float32)      # [n, 7]
    assert S.shape == (len(X), len(COLS))
    np.savez_compressed(a.out / "sqi_butqdb.npz", S=S, cols=np.array(COLS),
                        bank_sha256=np.array(hashlib.sha256(a.bank.read_bytes()).hexdigest()))
    print("saved", S.shape, "nan frac", float(np.isnan(S).mean()))


if __name__ == "__main__":
    main()
