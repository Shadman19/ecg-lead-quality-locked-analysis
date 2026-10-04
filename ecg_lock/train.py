"""Train one configuration with a prespecified, validation-only checkpoint rule.

Checkpoint rule (fixed before any test evaluation): the epoch with the highest
macro AUROC on the CLEAN validation set; ties -> earliest epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from corruption import CorruptionProfile, corrupt_training
from data import CLASSES, lead_stats, load_labels, load_signals, split_indices, split_manifest
from models import build


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arch", default="cnn", choices=["cnn", "tx", "resnet_wang", "inception"])
    p.add_argument("--tasks", default="diag,lead,rec")
    p.add_argument("--lead-head", default="pooled", choices=["pooled", "leadwise", "fused"])
    p.add_argument("--augment", type=int, default=1, help="train on corrupted inputs (1) or clean (0)")
    p.add_argument("--split", default="strat", choices=["strat", "gss"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--w-diag", type=float, default=1.0)
    p.add_argument("--w-lead", type=float, default=0.5)
    p.add_argument("--w-rec", type=float, default=1.0)
    p.add_argument("--sigma-min", type=float, default=0.0)
    p.add_argument("--clean-frac", type=float, default=0.0)
    p.add_argument("--max-train", type=int, default=0, help="debug: subsample training set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = hashlib.sha256()
        for f in sorted(Path(__file__).parent.glob("*.py")):
            h.update(f.read_bytes())
        return "nogit-" + h.hexdigest()[:12]


def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def macro_auroc(y, p):
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].mean() < 1]))


@torch.no_grad()
def predict(model, x, device, bs=512):
    model.eval()
    outs = []
    for i in range(0, len(x), bs):
        outs.append(torch.sigmoid(model(torch.from_numpy(x[i:i + bs]).to(device))["diag"]).cpu().numpy())
    return np.concatenate(outs)


def main():
    a = args_()
    set_seed(a.seed)
    tasks = tuple(t for t in a.tasks.split(",") if t)
    if a.arch in ("resnet_wang", "inception"):
        tasks = ("diag",)
    lr = a.lr or (1.5e-3 if a.arch == "tx" else 2e-3)
    a.out.mkdir(parents=True, exist_ok=True)

    df = load_labels(a.ptbxl)
    X = load_signals(a.ptbxl, df, a.ptbxl / "cache_x100.npy")
    Y = df[CLASSES].values.astype(np.float32)
    idx = split_indices(df, a.split)
    mu, sd = lead_stats(X[idx["train"]])
    Xn = (X - mu[None, :, None]) / (sd[None, :, None] + 1e-6)
    tr = idx["train"] if not a.max_train else idx["train"][: a.max_train]
    prof = CorruptionProfile(sigma_min=a.sigma_min, clean_frac=a.clean_frac,
                             min_one_masked=("rec" in tasks))

    model = build(a.arch, tasks, a.lead_head).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    rng = np.random.default_rng(10_000 + a.seed)

    cfg = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "tasks": tasks, "lr": lr,
           "corruption_profile": prof.as_dict(), "code_commit": git_commit(),
           "n_params": sum(p.numel() for p in model.parameters()),
           "checkpoint_rule": "max clean-validation macro AUROC; ties -> earliest epoch",
           "norm_mu": mu.tolist(), "norm_sd": sd.tolist(), "split_manifest": split_manifest(df, idx, a.split)}
    cfg["config_hash"] = hashlib.sha256(json.dumps({k: cfg[k] for k in cfg if k not in ("out", "device")},
                                                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    best, best_ep, log, bad = -1.0, -1, [], 0
    for ep in range(a.epochs):
        model.train()
        t0, perm, tot = time.time(), rng.permutation(tr), 0.0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            xb = Xn[b]
            if a.augment:
                xt, mask, _, _, _ = corrupt_training(xb, rng, prof)
            else:
                xt, mask = xb, np.zeros((len(b), 12), np.int8)
            xt_t = torch.from_numpy(xt).to(a.device)
            out = model(xt_t)
            loss = a.w_diag * F.binary_cross_entropy_with_logits(out["diag"], torch.from_numpy(Y[b]).to(a.device))
            m = torch.from_numpy(mask.astype(np.float32)).to(a.device)
            if "lead" in tasks:
                loss = loss + a.w_lead * F.binary_cross_entropy_with_logits(out["lead"], 1 - m)
            if "rec" in tasks:
                per_lead = ((out["rec"] - torch.from_numpy(xb).to(a.device)) ** 2).mean(-1)   # [B,12]
                n_m = m.sum(1)
                has = n_m > 0
                if has.any():
                    rec = ((per_lead * m).sum(1)[has] / n_m[has]).mean()
                    loss = loss + a.w_rec * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        pv = predict(model, Xn[idx["val"]], a.device)
        auc = macro_auroc(Y[idx["val"]], pv)
        log.append({"epoch": ep, "train_loss": tot / len(tr), "val_clean_macro_auroc": auc, "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            torch.save(model.state_dict(), a.out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    sha = hashlib.sha256((a.out / "best.pt").read_bytes()).hexdigest()
    (a.out / "train_log.json").write_text(json.dumps({"log": log, "best_epoch": best_ep, "best_val_auroc": best,
                                                      "checkpoint_sha256": sha}, indent=2))
    print("BEST", best_ep, best, sha[:12])


if __name__ == "__main__":
    main()
