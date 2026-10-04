"""SECONDARY (post-decision, inference only): five realizations of the test training-distribution corruption.

Addresses the limitation that endpoints A/B1/B2 used one corruption realization. For one locked run, writes a
NEW file bundles/test_traindist_multireal_v1.npz (locked bundles and lock.json are untouched; best.pt hash is
verified). Realization r uses seed crc32("multireal_v1|<split_id>|r"), identical for every model. Evaluation
profile is identical for all models (sigma_min=0, clean_frac=0, min_one_masked=True), as in export.py.
Saves y_score for every model; for models with a decoder, oracle-mask repair; for models with a per-lead head,
predicted-mask repair with the locked validation tau_mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from corruption import CorruptionProfile, corrupt_training
from data import CLASSES, load_labels, load_signals, split_indices
from export import infer, repaired
from models import build

R = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ptbxl", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    out_f = a.run / "bundles" / "test_traindist_multireal_v1.npz"
    if out_f.exists():
        print("exists, skipping", out_f); return
    cfg = json.loads((a.run / "config.json").read_text())
    lock = json.loads((a.run / "lock.json").read_text())
    ck = hashlib.sha256((a.run / "best.pt").read_bytes()).hexdigest()
    assert ck == lock["checkpoint_sha256"], "best.pt is not the locked checkpoint"
    tasks = tuple(cfg["tasks"])
    model = build(cfg["arch"], tasks, cfg.get("lead_head", "pooled")).to(a.device)
    model.load_state_dict(torch.load(a.run / "best.pt", map_location=a.device))
    df = load_labels(a.ptbxl)
    X = load_signals(a.ptbxl, df, a.ptbxl / "cache_x100.npy")
    ii = split_indices(df, cfg["split"])["test"]
    mu, sd = np.array(cfg["norm_mu"], np.float32), np.array(cfg["norm_sd"], np.float32)
    x = ((X[ii] - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32)
    prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)
    sid = cfg["split_manifest"]["split_id"]
    arr = dict(y_true=df[CLASSES].values.astype(np.int8)[ii], patient_id=df.patient_id.values[ii].astype(np.int64),
               ecg_id=df.index.values[ii].astype(np.int64), checkpoint_id=np.array(ck[:16]), split_id=np.array(sid))
    has_lead, has_rec = "lead" in tasks, "rec" in tasks
    for r in range(R):
        rng = np.random.default_rng(zlib.crc32(f"multireal_v1|{sid}|{r}".encode()))
        xt, mask, ctype, sig, bw = corrupt_training(x, rng, prof)
        o = infer(model, xt, a.device)
        arr[f"r{r}|mask"] = mask.astype(np.int8)
        arr[f"r{r}|ctype"] = ctype.astype(np.int8)
        arr[f"r{r}|y_score"] = o["diag"].astype(np.float32)
        if has_rec:
            arr[f"r{r}|post_oracle"] = infer(model, repaired(xt, o["rec"], mask), a.device)["diag"].astype(np.float32)
        if has_lead:
            arr[f"r{r}|lead"] = o["lead"].astype(np.float32)
            if has_rec:
                pm = (o["lead"] < lock["tau_mask"]).astype(np.int8)
                arr[f"r{r}|post_pred"] = infer(model, repaired(xt, o["rec"], pm), a.device)["diag"].astype(np.float32)
    np.savez_compressed(out_f, **arr)
    h = hashlib.sha256(out_f.read_bytes()).hexdigest()
    (a.run / "multireal_lock.json").write_text(json.dumps({"test_traindist_multireal_v1": h, "checkpoint_sha256": ck,
                                                             "realizations": R, "status": "SECONDARY, post-decision"}, indent=2))
    print("DONE", out_f, h[:12])


if __name__ == "__main__":
    main()
