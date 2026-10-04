"""SECONDARY (round 4, inference only): detector-gated lead masking.

For seed s, the frozen lead-preserving detector of the proposed model (cnn_full_leadwise_cleanfrac20_s{s}) scores every
lead of the corrupted input; leads with intact score < tau are set to zero (the missing-lead form seen in augmentation),
and a frozen augmentation-trained diagnosis-only classifier is applied to the gated input. Plain = same classifier on the
ungated input. tau = the detector's locked validation tau_mask (primary) and 0.5 (secondary). No training, no tuning on
test data. Corruptions reuse the exact seeds of export.py (stress grid), export_multireal.py and export_nstdb.py, so
inputs are identical to the locked bundles. Checkpoints are hash-verified. Writes one new file per seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from corruption import CorruptionProfile, add_recorded_noise, corrupt_stress, corrupt_training
from data import CLASSES, load_labels, load_signals, split_indices
from export import infer
from models import build

DETECTOR = "cnn_full_leadwise_cleanfrac20"
CLASSIFIERS = ["cnn_diagonly_aug", "resnet_aug", "inception_aug"]
STRESS = [("noise", 0, 0.10, 0.0), ("noise", 0, 0.25, 0.0), ("noise", 0, 0.50, 0.0), ("joint", 2, 0.25, 0.0),
          ("joint", 4, 0.25, 0.0), ("missing", 8, 0.0, 0.0), ("bw", 0, 0.0, 1.0)]
R = 5


def load(run, device):
    cfg = json.loads((run / "config.json").read_text())
    lock = json.loads((run / "lock.json").read_text())
    ck = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    assert ck == lock["checkpoint_sha256"], f"{run} best.pt is not the locked checkpoint"
    m = build(cfg["arch"], tuple(cfg["tasks"]), cfg.get("lead_head", "pooled")).to(device)
    m.load_state_dict(torch.load(run / "best.pt", map_location=device))
    return m, cfg, lock, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ptbxl", type=Path, required=True)
    p.add_argument("--nstdb-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out_f = a.out / f"gated_s{a.seed}.npz"
    if out_f.exists():
        print("exists", out_f); return
    det, dcfg, dlock, dck = load(a.runs / f"{DETECTOR}_s{a.seed}", a.device)
    taus = {"tau": float(dlock["tau_mask"]), "t05": 0.5}
    clf = {}
    for c in CLASSIFIERS:
        m, ccfg, _, cck = load(a.runs / f"{c}_s{a.seed}", a.device)
        assert ccfg["split_manifest"]["split_id"] == dcfg["split_manifest"]["split_id"]
        assert np.allclose(ccfg["norm_mu"], dcfg["norm_mu"]) and np.allclose(ccfg["norm_sd"], dcfg["norm_sd"])
        clf[c] = (m, cck)
    df = load_labels(a.ptbxl)
    X = load_signals(a.ptbxl, df, a.ptbxl / "cache_x100.npy")
    ii = split_indices(df, dcfg["split"])["test"]
    mu, sd = np.array(dcfg["norm_mu"], np.float32), np.array(dcfg["norm_sd"], np.float32)
    x = ((X[ii] - mu[None, :, None]) / (sd[None, :, None] + 1e-6)).astype(np.float32)
    sid = dcfg["split_manifest"]["split_id"]
    arr = dict(y_true=df[CLASSES].values.astype(np.int8)[ii], patient_id=df.patient_id.values[ii].astype(np.int64),
               ecg_id=df.index.values[ii].astype(np.int64), tau=np.array(taus["tau"]))

    def run(key, xs, true_mask):
        intact = infer(det, xs, a.device)["lead"]
        arr[key + "|true_mask"] = true_mask.astype(np.int8)
        for tn, tv in taus.items():
            g = intact < tv
            arr[f"{key}|gate_{tn}"] = g.astype(np.int8)
        for c, (m, _) in clf.items():
            arr[f"{key}|{c}|plain"] = infer(m, xs, a.device)["diag"].astype(np.float32)
            for tn, tv in taus.items():
                xg = xs.copy(); xg[intact < tv] = 0.0
                arr[f"{key}|{c}|gated_{tn}"] = infer(m, xg, a.device)["diag"].astype(np.float32)

    run("clean", x, np.zeros(x.shape[:2], np.int8))
    prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)
    for r in range(R):
        rng = np.random.default_rng(zlib.crc32(f"multireal_v1|{sid}|{r}".encode()))
        xt, mask, *_ = corrupt_training(x, rng, prof)
        run(f"multireal|r={r}", xt, mask)
    sseed = 2_000
    for fam, k, s, b in STRESS:
        for r in range(R):
            rng = np.random.default_rng(zlib.crc32(f'{sseed}|{fam}|{k}|{s}|{b}|{r}'.encode()))
            xs, ms, _, _ = corrupt_stress(x, rng, k, s, b)
            run(f"{fam}|k={k}|s={s}|bw={b}|r={r}", xs, ms)
    bank = np.load(a.nstdb_bank, allow_pickle=True).item()
    for kind in ("em", "ma"):
        for snr in (12, 6, 0):
            for nl in (1, 3, 6):
                rng = np.random.default_rng(zlib.crc32(f"nstdb_v1|{sid}|{kind}|{snr}|{nl}".encode()))
                xs, ms, _ = add_recorded_noise(x, rng, bank[kind], nl, snr)
                run(f"nstdb|{kind}|snr={snr}|leads={nl}", xs, ms)
    np.savez_compressed(out_f, **arr)
    h = hashlib.sha256(out_f.read_bytes()).hexdigest()
    (a.out / f"gated_s{a.seed}_lock.json").write_text(json.dumps(
        {"file_sha256": h, "detector_ckpt": dck, "classifier_ckpts": {c: v[1] for c, v in clf.items()},
         "taus": taus, "status": "SECONDARY round 4, endpoints fixed before run"}, indent=2))
    print("DONE", out_f, h[:12])


if __name__ == "__main__":
    main()
