"""Export immutable validation/test prediction bundles for one locked run.

Order of operations (enforced here):
  1. validation bundles are written first;
  2. the predicted-repair-mask threshold tau_mask is chosen on VALIDATION
     (max Youden J of the synthetic-intact score for corrupted-vs-intact leads)
     and written to lock.json;
  3. only then are test bundles computed, using tau_mask unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

from corruption import CorruptionProfile, add_recorded_noise, corrupt_stress, corrupt_training, rule_hf_energy, rule_variance
from data import CLASSES, load_labels, load_signals, split_indices
from models import build

MISSING_K = [0, 1, 2, 3, 4, 6, 8, 10, 12]
NOISE_S = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
BW_AMP = [0.25, 0.5, 1.0]
JOINT_K = [0, 2, 4, 6, 8]
JOINT_S = [0.0, 0.10, 0.25]


@torch.no_grad()
def infer(model, x, device, mc=0, bs=512):
    """Deterministic outputs (+ MC-dropout mean/std of diagnosis probs when mc>0)."""
    model.eval()
    res = {"diag": [], "lead": [], "rec": []}
    for i in range(0, len(x), bs):
        o = model(torch.from_numpy(x[i:i + bs]).to(device))
        res["diag"].append(torch.sigmoid(o["diag"]).cpu().numpy())
        if "lead" in o:
            res["lead"].append(torch.sigmoid(o["lead"]).cpu().numpy())
        if "rec" in o:
            res["rec"].append(o["rec"].cpu().numpy())
    out = {k: np.concatenate(v) for k, v in res.items() if v}
    if mc:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        draws = []
        for _ in range(mc):
            d = []
            for i in range(0, len(x), bs):
                d.append(torch.sigmoid(model(torch.from_numpy(x[i:i + bs]).to(device))["diag"]).cpu().numpy())
            draws.append(np.concatenate(d))
        model.eval()
        draws = np.stack(draws)
        out["diag_mc"] = draws.mean(0)
        out["unc"] = draws.std(0).mean(1)          # mean over classes of MC std
    return out


def repaired(xt, rec, mask):
    xr = xt.copy()
    xr[mask.astype(bool)] = rec[mask.astype(bool)]
    return xr


def save(path, **arrays):
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ptbxl", type=Path, required=True)
    p.add_argument("--mc", type=int, default=20)
    p.add_argument("--realizations", type=int, default=5)
    p.add_argument("--nstdb-bank", type=Path, default=None, help=".npy dict {'em':1-D,'ma':1-D} at 100 Hz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    tl = json.loads((a.run / "train_log.json").read_text())
    tasks = tuple(cfg["tasks"])
    model = build(cfg["arch"], tasks, cfg.get("lead_head", "pooled")).to(a.device)
    model.load_state_dict(torch.load(a.run / "best.pt", map_location=a.device))
    df = load_labels(a.ptbxl)
    X = load_signals(a.ptbxl, df, a.ptbxl / "cache_x100.npy")
    idx = split_indices(df, cfg["split"])
    mu, sd = np.array(cfg["norm_mu"], np.float32), np.array(cfg["norm_sd"], np.float32)
    Xn = (X - mu[None, :, None]) / (sd[None, :, None] + 1e-6)
    Y = df[CLASSES].values.astype(np.int8)
    meta = dict(checkpoint_id=np.array(tl["checkpoint_sha256"][:16]), config_hash=np.array(cfg["config_hash"]),
                split_id=np.array(cfg["split_manifest"]["split_id"]), training_seed=np.array(cfg["seed"]),
                code_commit=np.array(cfg["code_commit"]), class_names=np.array(CLASSES))
    bdir = a.run / "bundles"
    bdir.mkdir(exist_ok=True)
    lock = {"checkpoint_sha256": tl["checkpoint_sha256"], "bundles": {}}
    prof = CorruptionProfile(clean_frac=0.0, min_one_masked=True, sigma_min=0.0)  # identical eval corruption for ALL models (paired)
    has_lead, has_rec = "lead" in tasks, "rec" in tasks

    for split in ("val", "test"):                     # validation strictly first
        ii = idx[split]
        x, y = Xn[ii], Y[ii]
        base = dict(y_true=y, patient_id=df.patient_id.values[ii].astype(np.int64),
                    ecg_id=df.index.values[ii].astype(np.int64), split=np.array(split), **meta)
        sseed = {"val": 1_000, "test": 2_000}[split]
        # 1) clean
        o = infer(model, x, a.device, a.mc)
        arr = dict(base, y_score=o["diag"], y_score_mc=o["diag_mc"], uncertainty=o["unc"],
                   condition=np.array("clean"), inference_mode=np.array("deterministic"))
        if has_lead:
            arr["lead_score_clean"] = o["lead"]
        lock["bundles"][f"{split}_clean"] = save(bdir / f"{split}_clean.npz", **arr)
        # 2) training-distribution corruption (per-lead labels, reconstruction, repair)
        rng = np.random.default_rng(sseed + 1)
        xt, mask, ctype, sig, bw = corrupt_training(x, rng, prof)
        o = infer(model, xt, a.device, a.mc)
        arr = dict(base, y_score=o["diag"], y_score_mc=o["diag_mc"], uncertainty=o["unc"],
                   corruption_mask=mask, corruption_type=ctype, noise_sigma=sig, baseline_wander=bw,
                   rule_variance=rule_variance(xt), rule_hf=rule_hf_energy(xt),
                   condition=np.array("train_distribution"), corruption_seed=np.array(sseed + 1),
                   inference_mode=np.array("deterministic"))
        if has_lead:
            arr["reliability_score"] = o["lead"]
        if has_rec:
            arr["reconstruction_true"] = x.astype(np.float16)
            arr["reconstruction_pred"] = o["rec"].astype(np.float16)
            post = infer(model, repaired(xt, o["rec"], mask), a.device)["diag"]
            arr_or = dict(arr, pre_repair_score=o["diag"], post_repair_score=post, repair_mask=mask,
                          repair_mask_source=np.array("oracle"))
            lock["bundles"][f"{split}_traindist_oracle"] = save(bdir / f"{split}_traindist_oracle.npz", **arr_or)
            if has_lead:
                if split == "val":
                    fpr, tpr, thr = roc_curve(mask.ravel(), 1 - o["lead"].ravel())
                    j = np.argmax(tpr - fpr)
                    lock["tau_mask"] = float(1 - thr[j])      # flag lead when intact score < tau_mask
                    lock["tau_mask_rule"] = "max Youden J on validation train-distribution corruption"
                    (a.run / "lock.json").write_text(json.dumps(lock, indent=2))
                pm = (o["lead"] < lock["tau_mask"]).astype(np.int8)
                postp = infer(model, repaired(xt, o["rec"], pm), a.device)["diag"]
                arr_pr = dict(arr, pre_repair_score=o["diag"], post_repair_score=postp, repair_mask=pm,
                              repair_mask_source=np.array("predicted"), tau_mask=np.array(lock["tau_mask"]))
                lock["bundles"][f"{split}_traindist_predicted"] = save(bdir / f"{split}_traindist_predicted.npz", **arr_pr)
        else:
            lock["bundles"][f"{split}_traindist"] = save(bdir / f"{split}_traindist.npz", **arr)
        # 3) stress grids (compact: scores only)
        grid = [("missing", k, 0.0, 0.0) for k in MISSING_K] + [("noise", 0, s, 0.0) for s in NOISE_S] \
            + [("bw", 0, 0.0, b) for b in BW_AMP] + [("joint", k, s, 0.0) for k in JOINT_K for s in JOINT_S]
        stress = {}
        for fam, k, s, b in grid:
            for r in range(a.realizations):
                rng = np.random.default_rng(zlib.crc32(f'{sseed}|{fam}|{k}|{s}|{b}|{r}'.encode()))
                xs, ms, _, _ = corrupt_stress(x, rng, k, s, b)
                o = infer(model, xs, a.device)
                key = f"{fam}|k={k}|s={s}|bw={b}|r={r}"
                stress[key + "|y_score"] = o["diag"].astype(np.float32)
                if has_lead:
                    stress[key + "|lead"] = o["lead"].astype(np.float16)
                    stress[key + "|mask"] = ms
        lock["bundles"][f"{split}_stress"] = save(bdir / f"{split}_stress.npz", **base, **stress)
        # 4) real recorded artifacts (optional)
        if a.nstdb_bank is not None:
            bank = np.load(a.nstdb_bank, allow_pickle=True).item()
            rec_n = {}
            for kind in ("em", "ma", "bw"):
                if kind not in bank:
                    continue
                for snr in (12, 6, 0):
                    for nl in (1, 3, 6):
                        rng = np.random.default_rng(zlib.crc32(f'{sseed}|{kind}|{snr}|{nl}'.encode()))
                        xs, ms, _ = add_recorded_noise(x, rng, bank[kind], nl, snr)
                        o = infer(model, xs, a.device)
                        key = f"nstdb|{kind}|snr={snr}|leads={nl}"
                        rec_n[key + "|y_score"] = o["diag"].astype(np.float32)
                        rec_n[key + "|mask"] = ms
                        rec_n[key + "|rule_variance"] = rule_variance(xs).astype(np.float32)
                        rec_n[key + "|rule_hf"] = rule_hf_energy(xs).astype(np.float32)
                        if has_lead:
                            rec_n[key + "|lead"] = o["lead"].astype(np.float16)
            lock["bundles"][f"{split}_nstdb"] = save(bdir / f"{split}_nstdb.npz", **base, **rec_n)
        (a.run / "lock.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
