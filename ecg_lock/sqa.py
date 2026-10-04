"""Label-free per-lead 12-lead ECG signal quality assessment (SQA).

Detector trained ONLY on simulated faults injected into PTB-XL records that carry no quality annotation.
Model selection uses a fixed synthetic validation set (fold 9), never human quality labels and never CinC.
Kinds of run (--kind):
  lp    : lead-preserving grouped-conv detector (each lead processed independently)
  lpx   : lp features + a 2-layer Transformer across the 12 lead tokens (cross-lead context)
  ae    : label-free baseline, per-lead conv autoencoder trained on clean records; score = per-lead reconstruction MSE
  sup   : supervised upper bound, lp architecture trained on PTB-XL human annotations (folds 1-8, early stop on fold 9 labels)
--drop FAULT removes one fault type from the simulator (ablation).
Exports per-lead scores (higher = worse) for PTB-XL folds 9 and 10 (unaltered real records) and CinC 2011 set a.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import load_labels, load_signals

FS, T = 100, 1000
FAULTS = ["gauss", "nstdb", "flat", "clip", "hf", "spikes", "burst"]
QCOLS = ["static_noise", "burst_noise", "baseline_drift", "electrodes_problems"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------------------------------------- simulator
def fault(x, kind, rng, bank):
    """x: one standardized lead [T]; returns corrupted copy."""
    y = x.copy()
    if kind == "gauss":
        y += rng.normal(0, rng.uniform(0.05, 0.5), T)
    elif kind == "nstdb":
        src = bank[rng.choice(["em", "ma"])]
        st = rng.integers(0, len(src) - T); seg = src[st:st + T].astype(np.float32)
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        snr = rng.uniform(-6, 12); p = np.mean(x ** 2) + 1e-6
        noise = seg * np.sqrt(p / 10 ** (snr / 10))
        if rng.random() < 0.5:                                   # partial-duration artifact
            L = rng.integers(100, 400); a = rng.integers(0, T - L); m = np.zeros(T, np.float32); m[a:a + L] = 1; noise *= m
        y += noise
    elif kind == "flat":
        lvl = rng.normal(0, 0.3); flat = lvl + rng.normal(0, 0.005, T)
        if rng.random() < 0.3:                                   # disconnection part-way through
            a = rng.integers(100, T - 100); y[a:] = flat[a:]
        else:
            y = flat
    elif kind == "clip":
        g = rng.uniform(1.5, 4.0); c = np.percentile(np.abs(x * g), rng.uniform(40, 80))
        y = np.clip(x * g, -c, c)
    elif kind == "hf":
        f = rng.uniform(30, 49); t = np.arange(T) / FS
        y += rng.uniform(0.1, 0.6) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    elif kind == "spikes":
        n = rng.integers(3, 20); pos = rng.integers(0, T, n)
        y[pos] += rng.choice([-1, 1], n) * rng.uniform(2, 8, n)
    elif kind == "burst":
        L = rng.integers(50, 300); a = rng.integers(0, T - L); y[a:a + L] += rng.normal(0, rng.uniform(0.3, 1.5), L)
    return y.astype(np.float32)


def nuisance(x, rng, bank):
    """Unlabelled variation the detector must ignore: baseline wander, amplitude scaling."""
    x = x * rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:
        t = np.arange(T) / FS
        x = x + (rng.uniform(0.1, 1.0) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t
                                                + rng.uniform(0, 2 * np.pi, (12, 1)))).astype(np.float32)
    return x.astype(np.float32)


def corrupt(x, rng, bank, faults, p_clean=0.3, p_lead=0.3):
    x = nuisance(x.copy(), rng, bank)
    y = np.zeros(12, np.float32)
    if rng.random() >= p_clean:
        sel = rng.random(12) < p_lead
        if not sel.any():
            sel[rng.integers(12)] = True
        for j in np.flatnonzero(sel):
            x[j] = fault(x[j], faults[rng.integers(len(faults))], rng, bank); y[j] = 1
    return x, y


class SimSet(Dataset):
    def __init__(self, X, bank, faults, seed=0, fixed=False):
        self.X, self.bank, self.faults, self.seed, self.fixed = X, bank, faults, seed, fixed
        self.cache = None
        if fixed:
            out = [corrupt(self.X[i], np.random.default_rng(seed * 100003 + i), bank, faults) for i in range(len(X))]
            self.cache = (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.cache is not None:
            return self.cache[0][i], self.cache[1][i]
        return corrupt(self.X[i], np.random.default_rng(), self.bank, self.faults)


class RealSet(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X.astype(np.float32), Y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ----------------------------------------------------------------------------------------------- models
class LP(nn.Module):
    def __init__(self, width=32, cross=False):
        super().__init__()
        w = width; self.w = w; self.cross = cross
        self.net = nn.Sequential(
            nn.Conv1d(12, 12 * w, 7, 2, 3, groups=12), nn.GELU(),
            nn.Conv1d(12 * w, 12 * w, 7, 2, 3, groups=12), nn.GELU(),
            nn.Conv1d(12 * w, 12 * w, 7, 2, 3, groups=12), nn.GELU(),
            nn.Conv1d(12 * w, 12 * w, 5, 2, 2, groups=12), nn.GELU())
        d = 3 * w
        if cross:
            self.lead_emb = nn.Parameter(torch.zeros(1, 12, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
            self.mix = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        h = self.net(x); b, _, t = h.shape; h = h.view(b, 12, self.w, t)
        f = torch.cat([h.mean(-1), h.std(-1), h.amax(-1)], -1)          # [B,12,3w]
        if self.cross:
            f = f + self.mix(f + self.lead_emb)
        return self.out(f).squeeze(-1)                                   # [B,12] logits, high = faulty


class AE(nn.Module):
    """Single-lead conv autoencoder shared across leads."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(1, 16, 7, 2, 3), nn.GELU(), nn.Conv1d(16, 32, 7, 2, 3), nn.GELU(),
                                 nn.Conv1d(32, 32, 7, 2, 3), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(32, 32, 8, 2, 3), nn.GELU(), nn.ConvTranspose1d(32, 16, 8, 2, 3),
                                 nn.GELU(), nn.ConvTranspose1d(16, 1, 8, 2, 3))

    def forward(self, x):
        b = x.shape[0]; z = x.reshape(b * 12, 1, -1)
        r = self.dec(self.enc(z))[..., :x.shape[-1]]
        return r.reshape(b, 12, -1)


# ----------------------------------------------------------------------------------------------- utils
def parse(s):
    m = np.zeros(12, bool)
    if not isinstance(s, str):
        return m
    s = s.upper().replace(" ", "")
    if "ALLES" in s:
        m[:] = True; return m
    for part in s.replace(";", ",").split(","):
        part = part.strip("?")
        if "-" in part:
            a, b = [p.strip("?") for p in part.split("-", 1)]
            if a in LEADS and b in LEADS:
                i, j = LEADS.index(a), LEADS.index(b); m[min(i, j):max(i, j) + 1] = True
        elif part in LEADS:
            m[LEADS.index(part)] = True
    return m


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def score(model, kind, X, dev, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        if kind == "ae":
            out.append(((model(x) - x) ** 2).mean(-1).cpu().numpy())
        else:
            out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["lp", "lpx", "ae", "sup"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path.home() / "ecg_lock_results/sqa")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{a.kind}{'_no' + a.drop if a.drop else ''}_s{a.seed}"
    a.out.mkdir(parents=True, exist_ok=True)
    ptb = Path.home() / "cardio ML/ptbxl"
    df = load_labels(ptb); X = load_signals(ptb, df, ptb / "cache_x100.npy").astype(np.float32)
    db = pd.read_csv(ptb / "ptbxl_database.csv", index_col="ecg_id").loc[df.index]
    flagged = db[QCOLS].notna().any(axis=1).values
    fold = df.strat_fold.values
    tr_clean = (fold <= 8) & ~flagged
    mu = X[tr_clean].mean(axis=(0, 2)); sd = X[tr_clean].std(axis=(0, 2)) + 1e-6
    Xn = ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    bank = {k: np.asarray(v, np.float32) for k, v in np.load(Path.home() / "ecg_lock/nstdb_100hz.npy", allow_pickle=True).item().items()}
    bank = {k: v[:int(0.7 * len(v))] for k, v in bank.items()}          # first 70% only for training noise
    faults = [f for f in FAULTS if f != a.drop]
    Yreal = np.stack([parse(r.static_noise) | parse(r.burst_noise) | parse(r.electrodes_problems) for r in db.itertuples()])

    if a.kind == "sup":
        tr = RealSet(Xn[fold <= 8], Yreal[fold <= 8]); va_X, va_Y = Xn[fold == 9], Yreal[fold == 9]
    else:
        tr = SimSet(Xn[tr_clean], bank, faults)
        va_clean = (fold == 9) & ~flagged
        vs = SimSet(Xn[va_clean], bank, FAULTS, seed=9, fixed=True)    # fixed synthetic validation, all faults
        va_X, va_Y = vs.cache
    model = (AE() if a.kind == "ae" else LP(32, cross=(a.kind == "lpx"))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    if a.kind == "ae":
        tr = RealSet(Xn[tr_clean], np.zeros((tr_clean.sum(), 12)))
    dl = DataLoader(tr, batch_size=256, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    best, best_state, log = -1, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            if a.kind == "ae":
                loss = ((model(xb) - xb) ** 2).mean()
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = auc(va_Y.ravel(), score(model, a.kind, va_X, dev).ravel())
        log.append(dict(epoch=ep, val_auroc=v, loss=float(loss), sec=time.time() - t0)); print(log[-1], flush=True)
        if v > best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, a.out / f"{name}.pt")
    cinc = np.load(Path.home() / "data/cinc2011_seta_100hz.npz")
    Xc = ((cinc["X"] - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    np.savez_compressed(a.out / f"{name}.npz",
                        ptb_test=score(model, a.kind, Xn[fold == 10], dev), ptb_test_ids=df.index.values[fold == 10],
                        ptb_val=score(model, a.kind, Xn[fold == 9], dev), ptb_val_ids=df.index.values[fold == 9],
                        cinc=score(model, a.kind, Xc, dev), cinc_rec=cinc["record"], mu=mu, sd=sd)
    (a.out / f"{name}.json").write_text(json.dumps(dict(name=name, kind=a.kind, seed=a.seed, drop=a.drop, faults=faults,
                                                         best_val_auroc=best, n_params=sum(p.numel() for p in model.parameters()),
                                                         log=log), indent=1))
    print("DONE", name, best)


if __name__ == "__main__":
    main()
