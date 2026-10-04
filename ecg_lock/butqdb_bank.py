"""Round 6 (SECONDARY, inference only): 10-s windows from BUT QDB v1.0.0 with consensus quality labels.
Windows are aligned to the record start (10,000 samples at 1000 Hz), converted to mV with the WFDB gain and baseline, and
resampled to 100 Hz with resample_poly(1, 10) -> [1000]. A window gets a pure label only if ONE consensus class covers every
sample; the majority label (> 50% of samples) is stored separately. Windows with no pure label are excluded from the primary
analysis; unannotated windows are excluded from both. Consensus = columns 10-12 of *_ANN.csv (1-based inclusive samples).
Usage: python butqdb_bank.py --root ~/data/butqdb --out ~/data/butqdb_windows_100hz.npz
"""
import argparse
import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

W = 10000  # samples per window at 1000 Hz


def consensus(csv):
    a = pd.read_csv(csv, header=None)
    c = a.iloc[:, 9:12].dropna().astype(np.int64).values          # start, end, class (consensus); class 0 = not annotated
    assert c.shape[1] == 3 and np.all(np.isin(c[:, 2], [0, 1, 2, 3])) and np.all(c[:, 1] >= c[:, 0]), csv
    return c[c[:, 2] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "data/butqdb")
    ap.add_argument("--out", type=Path, default=Path.home() / "data/butqdb_windows_100hz.npz")
    a = ap.parse_args()
    import wfdb
    recs = sorted({line.split("/")[0] for line in a.root.joinpath("RECORDS").read_text().split()})   # RECORDS lists 100001/100001_ECG etc.
    assert len(recs) == 18 or os.environ.get("BUTQDB_SMOKE"), recs
    X, cls, maj, rec_l, win_l, counts = [], [], [], [], [], {}
    for rid in recs:
        h = wfdb.rdheader(str(a.root / rid / f"{rid}_ECG"))
        assert h.fs == 1000 and h.n_sig == 1 and h.units[0].lower() == "uv", (rid, h.fs, h.n_sig, h.units)
        n = h.sig_len
        lab = np.zeros(n, np.int8)
        for s, e, c in consensus(a.root / rid / f"{rid}_ANN.csv"):
            lab[max(s - 1, 0):min(e, n)] = c
        nwin = n // W
        L = lab[:nwin * W].reshape(nwin, W)
        cnt = np.stack([(L == c).sum(1) for c in (1, 2, 3)], 1)              # [nwin, 3]
        pure = np.where((cnt == W).any(1), cnt.argmax(1) + 1, 0).astype(np.int8)
        mj = np.where(cnt.max(1) > W // 2, cnt.argmax(1) + 1, 0).astype(np.int8)
        keep = np.flatnonzero(mj > 0)                                         # pure windows are a subset
        d = wfdb.rdrecord(str(a.root / rid / f"{rid}_ECG"), physical=False).d_signal[:, 0].astype(np.float64)
        g, b = float(h.adc_gain[0]), float(h.baseline[0])
        for j0 in range(0, len(keep), 2000):
            ks = keep[j0:j0 + 2000]
            seg = np.stack([d[k * W:(k + 1) * W] for k in ks])
            seg = (seg - b) / g / 1000.0                                      # ADC units -> uV -> mV
            X.append(resample_poly(seg, 1, 10, axis=1).astype(np.float32))
        cls.append(pure[keep]); maj.append(mj[keep]); rec_l += [rid] * len(keep); win_l.append(keep)
        counts[rid] = dict(sig_len=int(n), n_windows=int(nwin), unannotated=int((cnt.sum(1) == 0).sum()),
                           pure={c: int((pure == c).sum()) for c in (1, 2, 3)}, majority={c: int((mj == c).sum()) for c in (1, 2, 3)},
                           mixed_excluded=int(((cnt.sum(1) > 0) & (mj == 0)).sum()))
        print(rid, counts[rid], flush=True)
    X = np.concatenate(X); cls = np.concatenate(cls); maj = np.concatenate(maj); win = np.concatenate(win_l)
    rec = np.array(rec_l); subj = np.array([s[:3] for s in rec_l])
    assert X.shape == (len(cls), 1000) and np.isfinite(X).all()
    np.savez_compressed(a.out, X=X, cls=cls, cls_maj=maj, record=rec, subject=subj, win=win.astype(np.int64))
    tot = {c: int((cls == c).sum()) for c in (0, 1, 2, 3)}
    meta = dict(records=recs, counts=counts, total_pure=tot, total_majority={c: int((maj == c).sum()) for c in (1, 2, 3)},
                bank_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("saved", a.out, X.shape, "pure", tot, "sha", meta["bank_sha256"][:12])


if __name__ == "__main__":
    main()
