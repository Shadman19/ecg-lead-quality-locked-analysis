"""SECONDARY (post-decision, inference only): run a locked checkpoint on PhysioNet/CinC 2011 set-a.

Writes a NEW file bundles/cinc2011_v1.npz (locked bundles and lock.json untouched; best.pt hash verified).
Inputs are standardized with the run's own PTB-XL training statistics, exactly as for PTB-XL test data.
Saves diagnosis probabilities, per-lead intact scores (if the model has a per-lead head), and the
transparent rule scores (identical for every model).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    out_f = a.run / "bundles" / "cinc2011_v1.npz"
    if out_f.exists():
        print("exists, skipping", out_f); return
    cfg = json.loads((a.run / "config.json").read_text())
    lock = json.loads((a.run / "lock.json").read_text())
    ck = hashlib.sha256((a.run / "best.pt").read_bytes()).hexdigest()
    assert ck == lock["checkpoint_sha256"], "best.pt is not the locked checkpoint"
    model = build(cfg["arch"], tuple(cfg["tasks"]), cfg.get("lead_head", "pooled")).to(a.device)
    model.load_state_dict(torch.load(a.run / "best.pt", map_location=a.device))
    b = np.load(a.bank)
    mu, sd = np.array(cfg["norm_mu"], np.float32), np.array(cfg["norm_sd"], np.float32)
    x = ((b["X"] - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32)
    o = infer(model, x, a.device)
    arr = dict(unacceptable=b["unacceptable"], record=b["record"], y_score=o["diag"].astype(np.float32),
               rule_variance=rule_variance(x).astype(np.float32), rule_hf=rule_hf_energy(x).astype(np.float32),
               checkpoint_id=np.array(ck[:16]), bank_sha256=np.array(hashlib.sha256(a.bank.read_bytes()).hexdigest()))
    if "lead" in o:
        arr["lead"] = o["lead"].astype(np.float32)
        if "tau_mask" in lock:
            arr["tau_mask"] = np.array(lock["tau_mask"])
    np.savez_compressed(out_f, **arr)
    h = hashlib.sha256(out_f.read_bytes()).hexdigest()
    (a.run / "cinc2011_lock.json").write_text(json.dumps({"cinc2011_v1": h, "checkpoint_sha256": ck,
                                                           "status": "SECONDARY, post-decision"}, indent=2))
    print("DONE", out_f, h[:12])


if __name__ == "__main__":
    main()
