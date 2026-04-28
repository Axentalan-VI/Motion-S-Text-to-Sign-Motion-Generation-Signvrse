"""Stratified, deterministic 90/10 train/val split.

Stratification key = length-bin of the base-token sequence, so each bucket is
represented in val. The split is written to `data/split_90_10.json` and must
not be changed mid-competition (every model uses the same val set so the local
proxy + ensemble selection are comparable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import SEQ_LEN_MAX, SEQ_LEN_MIN, SPLIT_FILE


LEN_BIN_EDGES = (0, 80, 160, 240, 320, 480, SEQ_LEN_MAX + 1)
SEED = 1234
VAL_FRACTION = 0.10


def _bin_lengths(lengths: np.ndarray) -> np.ndarray:
    return np.digitize(lengths, LEN_BIN_EDGES[1:-1])


def make_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> dict[str, list[int]]:
    """Build a stratified split. Requires `df['id']` and `df['seq_len']`."""
    if "seq_len" not in df.columns:
        raise ValueError("df must have a 'seq_len' column (call load_train first).")
    rng = np.random.default_rng(seed)
    bins = _bin_lengths(df["seq_len"].to_numpy())

    val_ids: list[int] = []
    train_ids: list[int] = []
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_ids.extend(df.iloc[val_idx]["id"].astype(int).tolist())
        train_ids.extend(df.iloc[train_idx]["id"].astype(int).tolist())

    return {"train": sorted(train_ids), "val": sorted(val_ids), "seed": seed,
            "val_fraction": val_fraction, "len_bin_edges": list(LEN_BIN_EDGES)}


def save_split(split: dict, path: str | Path = SPLIT_FILE) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2))
    return p


def load_split(path: str | Path = SPLIT_FILE) -> dict:
    return json.loads(Path(path).read_text())
