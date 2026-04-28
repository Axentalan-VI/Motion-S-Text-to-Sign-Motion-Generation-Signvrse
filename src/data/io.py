"""Data loading helpers for the Motion-S competition.

The training CSV is expected to have these columns (token columns are
space-separated integer strings, same convention as the submission):

    id, sentence, gloss, base_tokens, residual_1, ..., residual_5

If your local copy uses a slightly different column name for the token layers,
extend `TOKEN_COLUMN_ALIASES` below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.constants import (
    NUM_LAYERS,
    TEST_CSV,
    TOKEN_COLUMNS,
    TRAIN_CSV,
)


# In case organizers shipped slightly different naming, try these in order.
TOKEN_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "base_tokens":  ("base_tokens", "base", "layer_0", "tokens_0"),
    "residual_1":   ("residual_1", "layer_1", "tokens_1"),
    "residual_2":   ("residual_2", "layer_2", "tokens_2"),
    "residual_3":   ("residual_3", "layer_3", "tokens_3"),
    "residual_4":   ("residual_4", "layer_4", "tokens_4"),
    "residual_5":   ("residual_5", "layer_5", "tokens_5"),
}


def _resolve_token_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical column names → actual column names present in `df`."""
    resolved: dict[str, str] = {}
    for canonical, candidates in TOKEN_COLUMN_ALIASES.items():
        for cand in candidates:
            if cand in df.columns:
                resolved[canonical] = cand
                break
    return resolved


def parse_token_cell(cell: object) -> np.ndarray:
    """Parse a single 'a b c d' string into an int32 array. Empty/NaN -> empty array."""
    if not isinstance(cell, str) or not cell.strip():
        return np.zeros(0, dtype=np.int32)
    return np.fromstring(cell, dtype=np.int32, sep=" ")


def load_train(path: str | Path = TRAIN_CSV) -> pd.DataFrame:
    """Load training data and parse all 6 token-layer columns into ndarrays."""
    df = pd.read_csv(path)
    resolved = _resolve_token_columns(df)
    if len(resolved) != NUM_LAYERS:
        missing = [c for c in TOKEN_COLUMNS if c not in resolved]
        raise ValueError(f"Could not find token columns for layers: {missing}. Got cols: {df.columns.tolist()}")
    for canonical, actual in resolved.items():
        df[canonical] = df[actual].map(parse_token_cell)
    df["seq_len"] = df["base_tokens"].map(len).astype(np.int32)
    return df


def load_test(path: str | Path = TEST_CSV) -> pd.DataFrame:
    """Load test prompts (id, sentence, gloss)."""
    return pd.read_csv(path)


def stack_layers(row: pd.Series) -> np.ndarray:
    """Stack the 6 layer arrays of a single row into shape (NUM_LAYERS, T)."""
    arrs = [row[c] for c in TOKEN_COLUMNS]
    lens = {len(a) for a in arrs}
    if len(lens) != 1:
        raise ValueError(f"Row {row.get('id', '?')} has mismatched layer lengths: {sorted(lens)}")
    return np.stack(arrs, axis=0)
