"""Classical per-lead ECG signal quality indices (SQIs) as published baselines, computed on raw mV signals at 100 Hz.
Higher value = WORSE quality for every column saved here (orientation fixed a priori from the defining papers):
  bsqi_bad   = 1 - bSQI (agreement of two QRS detectors, xqrs vs gqrs, 150 ms window)      [Li, Mark, Clifford 2008]
  ksqi_bad   = -kurtosis (clean ECG is peaky, kurtosis > 5)                                  [Li et al. 2008; Clifford et al. 2012]
  ssqi_bad   = -|skewness|                                                                   [Li et al. 2008]
  psqi_bad   = |P(5-15 Hz)/P(5-40 Hz) - 0.6|  (deviation from the typical QRS band ratio)    [Li et al. 2008; Clifford et al. 2012]
  bassqi_bad = P(0-1 Hz)/P(0-40 Hz)  (baseline power fraction)                               [Li et al. 2008]
  flat_bad   = fraction of samples with |diff| < 1e-3 mV (flat-line / disconnection)          [Clifford et al. 2012]
  tsqi_bad   = 1 - mean correlation of beats with the average beat template (xqrs peaks)     [Orphanidou et al. 2015]
Usage: python sqi_baselines.py --which ptbxl|cinc --out DIR [--procs 16]
"""
import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

FS = 100
COLS = ["bsqi_bad", "ksqi_bad", "ssqi_bad", "psqi_bad", "bassqi_bad", "flat_bad", "tsqi_bad"]


def _peaks(x):
    import wfdb.processing as wp
    try:
        p1 = np.asarray(wp.xqrs_detect(sig=x.astype(np.float64), fs=FS, verbose=False), int)
    except Exception:
        p1 = np.array([], int)
    try:
        p2 = np.asarray(wp.gqrs_detect(sig=x.astype(np.float64), fs=FS, adc_gain=1000, adc_zero=0), int)
    except Exception:
        p2 = np.array([], int)
    return p1, p2


def bsqi(p1, p2, tol=int(0.15 * FS)):
    if len(p1) == 0 and len(p2) == 0:
        return 0.0
    m = sum(1 for a in p1 if len(p2) and np.min(np.abs(p2 - a)) <= tol)
    return m / (len(p1) + len(p2) - m) if (len(p1) + len(p2) - m) > 0 else 0.0


def tsqi(x, p, half=int(0.25 * FS)):
    beats = [x[i - half:i + half] for i in p if i - half >= 0 and i + half <= len(x)]
    if len(beats) < 3:
        return 0.0
    B = np.stack(beats); tpl = B.mean(0)
    if tpl.std() < 1e-8:
        return 0.0
    c = [np.corrcoef(b, tpl)[0, 1] if b.std() > 1e-8 else 0.0 for b in B]
    return float(np.nanmean(c))


def lead_sqis(x):
    x = np.nan_to_num(x.astype(np.float64))
    p1, p2 = _peaks(x)
    f, P = welch(x, fs=FS, nperseg=min(256, len(x)))
    band = lambda lo, hi: P[(f >= lo) & (f <= hi)].sum()
    tot40 = band(0, 40) + 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = kurtosis(x, fisher=False) if x.std() > 1e-8 else 0.0
        s = skew(x) if x.std() > 1e-8 else 0.0
    return [1 - bsqi(p1, p2), -k, -abs(s), abs(band(5, 15) / (band(5, 40) + 1e-12) - 0.6), band(0, 1) / tot40,
            float(np.mean(np.abs(np.diff(x)) < 1e-3)), 1 - tsqi(x, p1)]


def record_sqis(rec):
    return np.array([lead_sqis(rec[j]) for j in range(rec.shape[0])], np.float32)  # [12, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ptbxl", "cinc"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.which == "cinc":
        b = np.load(Path.home() / "data/cinc2011_seta_100hz.npz")
        X, ids = b["X"], b["record"]
    else:
        from data import load_labels, load_signals
        ptb = Path.home() / "cardio ML/ptbxl"
        df = load_labels(ptb)
        X = load_signals(ptb, df, ptb / "cache_x100.npy")
        ids = df.index.values
    with Pool(a.procs) as pool:
        S = np.stack(pool.map(record_sqis, list(X), chunksize=16))
    np.savez_compressed(a.out / f"sqi_{a.which}.npz", S=S, ids=ids, cols=np.array(COLS))
    print("saved", S.shape)


if __name__ == "__main__":
    main()
