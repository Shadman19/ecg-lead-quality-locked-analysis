"""Build a 100 Hz array from PhysioNet/CinC Challenge 2011 set-a (12-lead ECGs recorded with mobile
phones, each labelled acceptable or unacceptable by human annotators). SECONDARY, inference only.

Records are 10 s at 500 Hz (5000 samples, mV). Leads are checked to be in the PTB-XL order
(I, II, III, aVR, aVL, aVF, V1-V6) and resampled to 100 Hz (1000 samples) with resample_poly.
Usage: python cinc2011_bank.py --out ~/data/cinc2011_seta_100hz.npz   (downloads via wfdb)
Output: X [n,12,1000] float32 (mV), unacceptable [n] int8 (1 = unacceptable), record [n] str.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BASE = "https://physionet.org/files/challenge-2011/1.0.0/set-a/"


def fetch_list(name, local):
    if local:
        return Path(local, name).read_text().split()
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        return r.read().decode().split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--local-dir", default=None, help="local copy of set-a (else download from PhysioNet)")
    a = p.parse_args()
    import wfdb
    acc, unacc = fetch_list("RECORDS-acceptable", a.local_dir), fetch_list("RECORDS-unacceptable", a.local_dir)
    assert not set(acc) & set(unacc)
    recs = sorted(acc + unacc)
    X, lab, names = [], [], []
    for i, rec in enumerate(recs):
        r = wfdb.rdrecord(str(Path(a.local_dir, rec)) if a.local_dir else rec, pn_dir=None if a.local_dir else "challenge-2011/1.0.0/set-a")
        names_up = [s.upper() for s in r.sig_name]
        assert sorted(names_up) == sorted(LEADS), (rec, r.sig_name); idx = [names_up.index(L) for L in LEADS]
        assert r.fs == 500 and r.p_signal.shape == (5000, 12), (rec, r.fs, r.p_signal.shape)
        sig = np.nan_to_num(r.p_signal[:, idx].T.astype(np.float64))           # [12, 5000]
        X.append(resample_poly(sig, 1, 5, axis=1).astype(np.float32))  # [12, 1000]
        lab.append(1 if rec in unacc else 0)
        names.append(rec)
        if i % 100 == 0:
            print(i, rec, flush=True)
    X, lab = np.stack(X), np.array(lab, np.int8)
    np.savez_compressed(a.out, X=X, unacceptable=lab, record=np.array(names))
    print("saved", a.out, X.shape, "unacceptable:", int(lab.sum()), "acceptable:", int((lab == 0).sum()))


if __name__ == "__main__":
    main()
