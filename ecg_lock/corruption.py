"""Synthetic lead corruption with explicit, logged parameters.

All functions operate on standardized signals x [B, 12, T] (numpy float32) and a
numpy Generator, so every realization is reproducible from its seed.

Type codes (per lead): 0 intact, 1 dropout (zeroed), 2 additive noise.
Baseline wander (BW) is NEVER written into the mask (paper design); it is
returned separately as a per-record flag.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FS = 100.0


@dataclass(frozen=True)
class CorruptionProfile:
    p_dropout: float = 0.5          # probability a record receives dropout
    max_dropout: int = 6            # k ~ U{1..max_dropout}
    p_noise_lead: float = 0.45      # per-lead probability of additive noise
    sigma_min: float = 0.0          # paper: U[0, 0.25]; see sigma-floor ablation
    sigma_max: float = 0.25
    p_bw: float = 0.3               # probability a record receives baseline wander
    bw_amp: tuple = (0.1, 0.5)      # std units
    bw_freq: tuple = (0.05, 0.5)    # Hz
    min_one_masked: bool = True     # paper: >=1 masked lead when reconstruction is active
    clean_frac: float = 0.0         # fraction of records left fully clean (fix candidate)

    def as_dict(self) -> dict:
        return asdict(self)


def baseline_wander(b: int, t: int, rng: np.random.Generator, prof: CorruptionProfile) -> np.ndarray:
    tt = np.arange(t) / FS
    amp = rng.uniform(*prof.bw_amp, size=(b, 1, 1))
    f = rng.uniform(*prof.bw_freq, size=(b, 1, 1))
    ph = rng.uniform(0, 2 * np.pi, size=(b, 12, 1))
    return (amp * np.sin(2 * np.pi * f * tt[None, None, :] + ph)).astype(np.float32)


def corrupt_training(x: np.ndarray, rng: np.random.Generator, prof: CorruptionProfile):
    """Returns x_tilde, mask [B,12] (1=corrupted), ctype [B,12], sigma [B,12], bw [B]."""
    b, l, t = x.shape
    xt = x.copy()
    ctype = np.zeros((b, l), np.int8)
    sigma = np.zeros((b, l), np.float32)
    clean = rng.random(b) < prof.clean_frac
    # additive noise
    noisy = (rng.random((b, l)) < prof.p_noise_lead) & ~clean[:, None]
    s = rng.uniform(prof.sigma_min, prof.sigma_max, size=(b, l)).astype(np.float32)
    sigma[noisy] = s[noisy]
    xt += rng.standard_normal(x.shape).astype(np.float32) * sigma[:, :, None]
    ctype[noisy] = 2
    # dropout (after noise; dropped leads are zero)
    for i in np.flatnonzero((rng.random(b) < prof.p_dropout) & ~clean):
        k = rng.integers(1, prof.max_dropout + 1)
        leads = rng.choice(l, size=k, replace=False)
        xt[i, leads] = 0.0
        ctype[i, leads] = 1
        sigma[i, leads] = 0.0
    # guarantee one masked lead for non-clean records
    if prof.min_one_masked:
        for i in np.flatnonzero((ctype.sum(1) == 0) & ~clean):
            j = rng.integers(l)
            sg = rng.uniform(max(prof.sigma_min, 0.05), prof.sigma_max)
            xt[i, j] += rng.standard_normal(t).astype(np.float32) * sg
            ctype[i, j] = 2
            sigma[i, j] = sg
    # baseline wander: unlabeled
    bw = (rng.random(b) < prof.p_bw) & ~clean
    if bw.any():
        xt[bw] += baseline_wander(int(bw.sum()), t, rng, prof)
    return xt, (ctype > 0).astype(np.int8), ctype, sigma, bw.astype(np.int8)


def corrupt_stress(x: np.ndarray, rng: np.random.Generator, k: int, sigma: float, bw_amp: float = 0.0,
                   noise_leads: str = "remaining"):
    """Stress condition: exactly k leads dropped; Gaussian noise sigma on the other leads
    (noise_leads='remaining') ; optional BW with fixed amplitude. Mask marks every lead that
    received dropout or noise (paper definition)."""
    b, l, t = x.shape
    xt = x.copy()
    ctype = np.zeros((b, l), np.int8)
    for i in range(b):
        if k:
            leads = rng.choice(l, size=k, replace=False)
            ctype[i, leads] = 1
    if sigma > 0:
        noise_sel = ctype == 0
        xt += rng.standard_normal(x.shape).astype(np.float32) * (sigma * noise_sel)[:, :, None]
        ctype[noise_sel] = 2
    xt[ctype == 1] = 0.0
    if bw_amp > 0:
        tt = np.arange(t) / FS
        f = rng.uniform(0.05, 0.5, size=(b, 1, 1))
        ph = rng.uniform(0, 2 * np.pi, size=(b, l, 1))
        xt += (bw_amp * np.sin(2 * np.pi * f * tt + ph)).astype(np.float32)
    sig = np.where(ctype == 2, sigma, 0.0).astype(np.float32)
    return xt, (ctype > 0).astype(np.int8), ctype, sig


def add_recorded_noise(x: np.ndarray, rng: np.random.Generator, bank: np.ndarray, n_leads: int, snr_db: float):
    """Add real recorded artifact segments (e.g. MIT-BIH NSTDB 'em'/'ma' resampled to 100 Hz,
    bank [M] 1-D) to n_leads random leads per record at a target per-lead SNR."""
    b, l, t = x.shape
    xt = x.copy()
    ctype = np.zeros((b, l), np.int8)
    for i in range(b):
        leads = rng.choice(l, size=n_leads, replace=False)
        for j in leads:
            st = rng.integers(0, len(bank) - t)
            seg = bank[st:st + t].astype(np.float32)
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            p_sig = np.mean(x[i, j] ** 2) + 1e-8
            xt[i, j] += seg * np.sqrt(p_sig / 10 ** (snr_db / 10))
            ctype[i, j] = 2
    return xt, (ctype > 0).astype(np.int8), ctype


# --- transparent per-lead rule baselines (higher = more likely intact) ---

def rule_variance(xt: np.ndarray) -> np.ndarray:
    """Within-lead standard deviation. Zeroed (dropped) leads score 0."""
    return xt.std(axis=2)


def rule_hf_energy(xt: np.ndarray) -> np.ndarray:
    """Negative first-difference energy relative to lead variance: additive white noise
    raises high-frequency energy. Returned as a score where higher = more intact."""
    d = np.diff(xt, axis=2)
    ratio = (d ** 2).mean(axis=2) / ((xt ** 2).mean(axis=2) + 1e-6)
    return -ratio


def to_prob(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Monotone map of a rule score to [0,1] with validation-fixed bounds (AUROC unchanged)."""
    return np.clip((score - lo) / (hi - lo + 1e-12), 0.0, 1.0)
