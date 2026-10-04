"""PTB-XL loading for the locked JBHI study.

Superclass labels follow the PTB-XL benchmark convention (Strodthoff et al.):
diagnostic SCP statements are mapped to `diagnostic_class`; records without any
superclass label are excluded. Split "strat" uses the official strat_fold
(1-8 train, 9 validation, 10 test; patient-disjoint by construction). Split
"gss" reproduces the manuscript's GroupShuffleSplit by patient for sensitivity.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_labels(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]

    def superclasses(codes: dict) -> list[str]:
        return sorted({scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})

    df["superclass"] = df.scp_codes.apply(superclasses)
    df = df[df.superclass.map(len) > 0].copy()
    for c in CLASSES:
        df[c] = df.superclass.apply(lambda s, c=c: int(c in s))
    return df


def load_signals(root: Path, df: pd.DataFrame, cache: Path) -> np.ndarray:
    """[N, 12, 1000] float32 at 100 Hz, in df order. Cached to .npy."""
    if cache.exists():
        x = np.load(cache)
        if x.shape[0] == len(df):
            return x
    import wfdb

    x = np.stack(
        [wfdb.rdsamp(str(root / f))[0].T.astype(np.float32) for f in df.filename_lr]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return x


def split_indices(df: pd.DataFrame, mode: str, seed: int = 7) -> dict[str, np.ndarray]:
    if mode == "strat":
        f = df.strat_fold.values
        idx = {"train": np.flatnonzero(f <= 8), "val": np.flatnonzero(f == 9), "test": np.flatnonzero(f == 10)}
    elif mode == "gss":
        from sklearn.model_selection import GroupShuffleSplit

        g = df.patient_id.values
        a = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, rest = next(a.split(df, groups=g))
        b = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        va, te = next(b.split(rest, groups=g[rest]))
        idx = {"train": tr, "val": rest[va], "test": rest[te]}
    else:
        raise ValueError(mode)
    pid = df.patient_id.values
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(pid[idx[a_]]) & set(pid[idx[b_]]), f"patient overlap {a_}/{b_}"
    return idx


def split_manifest(df: pd.DataFrame, idx: dict[str, np.ndarray], mode: str) -> dict:
    out = {"mode": mode, "n_records_total": int(len(df))}
    for k, v in idx.items():
        ids = df.index.values[v]
        out[k] = {
            "n_records": int(len(v)),
            "n_patients": int(len(set(df.patient_id.values[v]))),
            "ecg_id_sha256": hashlib.sha256(np.sort(ids).astype(np.int64).tobytes()).hexdigest(),
            "label_prevalence": {c: float(df[c].values[v].mean()) for c in CLASSES},
        }
    out["split_id"] = hashlib.sha256(json.dumps({k: out[k]["ecg_id_sha256"] for k in idx}).encode()).hexdigest()[:16]
    return out


def lead_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean/std over ALL training samples and time points (deterministic)."""
    mu = x_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    sd = x_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    return mu, sd
