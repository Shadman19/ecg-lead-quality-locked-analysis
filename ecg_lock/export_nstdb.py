"""SECONDARY analysis (added after the pre-registered decision): recorded-artifact evaluation.

Adds real MIT-BIH NSTDB artifacts (em = electrode motion, ma = muscle, bw = baseline wander)
to the TEST set of an already-locked run and writes a NEW bundle, bundles/test_nstdb_v1.npz,
plus nstdb_lock.json. It never modifies lock.json or any existing bundle, and it verifies that
best.pt is the locked checkpoint. Inference is deterministic (no MC dropout).
Realizations are seeded by condition only, so every model sees identical corrupted inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from corruption import add_recorded_noise, rule_hf_energy, rule_variance
from data import CLASSES, load_labels, load_signals, split_indices
from export import infer
from models import build

KINDS, SNRS, NLEADS = ("em", "ma", "bw"), (12, 6, 0), (1, 3, 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ptbxl", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    out_f = a.run / "bundles" / "test_nstdb_v1.npz"
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
    bank = np.load(a.bank, allow_pickle=True).item()
    arr = dict(y_true=df[CLASSES].values.astype(np.int8)[ii], patient_id=df.patient_id.values[ii].astype(np.int64),
               ecg_id=df.index.values[ii].astype(np.int64), checkpoint_id=np.array(ck[:16]),
               split_id=np.array(cfg["split_manifest"]["split_id"]))
    for kind in KINDS:
        for snr in SNRS:
            for nl in NLEADS:
                rng = np.random.default_rng(zlib.crc32(f"nstdb_v1|{cfg['split_manifest']['split_id']}|{kind}|{snr}|{nl}".encode()))
                xs, ms, _ = add_recorded_noise(x, rng, bank[kind], nl, snr)
                o = infer(model, xs, a.device)
                key = f"{kind}|snr={snr}|leads={nl}"
                arr[key + "|y_score"] = o["diag"].astype(np.float32)
                arr[key + "|mask"] = ms
                arr[key + "|rule_variance"] = rule_variance(xs).astype(np.float32)
                arr[key + "|rule_hf"] = rule_hf_energy(xs).astype(np.float32)
                if "lead" in o:
                    arr[key + "|lead"] = o["lead"].astype(np.float32)
    np.savez_compressed(out_f, **arr)
    h = hashlib.sha256(out_f.read_bytes()).hexdigest()
    bank_h = hashlib.sha256(a.bank.read_bytes()).hexdigest()
    (a.run / "nstdb_lock.json").write_text(json.dumps({"test_nstdb_v1": h, "bank_sha256": bank_h,
                                                        "checkpoint_sha256": ck, "status": "SECONDARY, post-decision"}, indent=2))
    print("DONE", out_f, h[:12])


if __name__ == "__main__":
    main()
