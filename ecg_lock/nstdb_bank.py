"""Build a 100 Hz noise bank from the MIT-BIH Noise Stress Test Database (PhysioNet `nstdb`).

Records: 'em' (electrode motion), 'ma' (muscle artifact), 'bw' (baseline wander);
360 Hz, 2 channels, ~30 min each. Both channels are concatenated per record.
Usage:  python nstdb_bank.py --out nstdb_100hz.npy   (downloads via wfdb from PhysioNet)
"""
import argparse

import numpy as np
from scipy.signal import resample_poly

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--local-dir", default=None, help="use a local copy of nstdb instead of downloading")
a = p.parse_args()
import wfdb

bank = {}
for rec in ("em", "ma", "bw"):
    r = wfdb.rdrecord(f"{a.local_dir}/{rec}" if a.local_dir else rec, pn_dir=None if a.local_dir else "nstdb")
    sig = np.concatenate([resample_poly(r.p_signal[:, c], 5, 18) for c in range(r.p_signal.shape[1])])
    bank[rec] = sig.astype(np.float32)          # 360 Hz * 5/18 = 100 Hz
    print(rec, sig.shape)
np.save(a.out, bank, allow_pickle=True)
