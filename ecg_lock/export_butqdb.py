"""Round 6 (SECONDARY, inference only): score BUT QDB windows with the locked detector checkpoints and the sqa.py networks.
Single-lead rule (fixed in the plan): the lead is copied into all 12 input channels, each standardized with the run's PTB-XL
training statistics; per-channel intact probabilities are saved so that the analysis can take the mean over channels (primary),
channel I only, or the max (sensitivities). Locked bundles and lock.json are untouched; best.pt hashes are verified.
Writes RUN/bundles/butqdb_v1.npz for each cnn_full_leadwise_cleanfrac20 seed, and OUT/sqa_{kind}_s{seed}.npy for sqa.py runs.
Usage: python export_butqdb.py --bank ~/data/butqdb_windows_100hz.npz --out ~/ecg_lock_results/butqdb
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from corruption import rule_hf_energy, rule_variance
from export import infer
from models import build

RUNS = Path("/scratch/spath004/ecg_lock_runs")
SQA = Path.home() / "ecg_lock_results/sqa"


def x12(X, mu, sd, i, j, eps=1e-6):
    """Copy the single lead into all 12 channels and standardize each channel (eps as in the training code of that model)."""
    x = np.repeat(X[i:j, None, :], 12, axis=1)
    return ((x - mu[None, :, None]) / (sd[None, :, None] + eps)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path.home() / "data/butqdb_windows_100hz.npz")
    ap.add_argument("--out", type=Path, default=Path.home() / "ecg_lock_results/butqdb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X = np.load(a.bank)["X"].astype(np.float32); n = len(X)
    bank_sha = hashlib.sha256(a.bank.read_bytes()).hexdigest()
    CH = 2048
    for s in (0, 1, 2):
        run = RUNS / f"cnn_full_leadwise_cleanfrac20_s{s}"; out_f = run / "bundles" / "butqdb_v1.npz"
        if out_f.exists():
            print("exists", out_f); continue
        cfg = json.loads((run / "config.json").read_text()); lock = json.loads((run / "lock.json").read_text())
        ck = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
        assert ck == lock["checkpoint_sha256"], "best.pt is not the locked checkpoint"
        model = build(cfg["arch"], tuple(cfg["tasks"]), cfg.get("lead_head", "pooled")).to(a.device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=a.device))
        mu, sd = np.array(cfg["norm_mu"], np.float32), np.array(cfg["norm_sd"], np.float32)
        lead, rv, rh = [], [], []
        for i in range(0, n, CH):
            x = x12(X, mu, sd, i, i + CH)
            lead.append(infer(model, x, a.device)["lead"].astype(np.float32))
            rv.append(rule_variance(x).astype(np.float32)); rh.append(rule_hf_energy(x).astype(np.float32))
        arr = dict(lead=np.concatenate(lead), rule_variance=np.concatenate(rv), rule_hf=np.concatenate(rh),
                   tau_mask=np.array(lock["tau_mask"]), checkpoint_id=np.array(ck[:16]), bank_sha256=np.array(bank_sha))
        np.savez_compressed(out_f, **arr)
        (run / "butqdb_lock.json").write_text(json.dumps({"butqdb_v1": hashlib.sha256(out_f.read_bytes()).hexdigest(),
                                                          "checkpoint_sha256": ck, "status": "SECONDARY, post-decision"}, indent=2))
        print("DONE detector seed", s, arr["lead"].shape, "mean intact", float(arr["lead"].mean()), flush=True)
    from sqa import AE, LP, score
    for kind in ("sup", "lp", "lpx", "ae"):
        for s in (0, 1, 2):
            pt, npz, out_f = SQA / f"{kind}_s{s}.pt", SQA / f"{kind}_s{s}.npz", a.out / f"sqa_{kind}_s{s}.npy"
            if out_f.exists() or not pt.exists() or not npz.exists():
                continue
            z = np.load(npz); mu, sd = z["mu"].astype(np.float32), z["sd"].astype(np.float32)
            model = (AE() if kind == "ae" else LP(32, cross=(kind == "lpx"))).to(a.device)
            model.load_state_dict(torch.load(pt, map_location=a.device))
            sc = np.concatenate([score(model, kind, x12(X, mu, sd, i, i + CH, eps=0.0), a.device) for i in range(0, n, CH)])
            np.save(out_f, sc.astype(np.float32)); print("DONE sqa", kind, s, sc.shape, flush=True)


if __name__ == "__main__":
    main()
