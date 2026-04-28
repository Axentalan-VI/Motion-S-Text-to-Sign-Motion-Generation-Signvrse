"""Dataset utilities for MoMask training.

Provides a Dataset that yields:
    {
      "tokens":     (NUM_LAYERS, T)    int64, padded with PAD_ID to T_max
      "length":     ()                 int32, true seq length (in tokens)
      "length_bin": ()                 int32, length-bin index (0..num_bins-1)
      "sentence":   str
    }

Token columns in train.csv are space-separated integer strings; we use
src.data.io.load_train to parse them into ndarrays. Sequences longer than
SEQ_LEN_MAX are truncated; shorter than SEQ_LEN_MIN are dropped at the
caller's discretion.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants import NUM_LAYERS, SEQ_LEN_MAX, SEQ_LEN_MIN, TOKEN_COLUMNS
from src.length import seq_len_to_bin
from src.models.momask import PAD_ID


class TokenSeqDataset(Dataset):
    """Pre-parsed dataframe rows → padded token grids."""

    def __init__(self, df, num_length_bins: int = 32, max_len: int | None = None):
        self.max_len = max_len or SEQ_LEN_MAX
        self.num_bins = num_length_bins

        # Pre-extract numpy arrays once.
        sentences = df["sentence"].fillna("").astype(str).tolist()
        # token columns: each cell is an np.ndarray
        layers: list[list[np.ndarray]] = [df[c].tolist() for c in TOKEN_COLUMNS]
        # Rotate to per-row: list of (NUM_LAYERS, T) arrays.
        rows: list[np.ndarray] = []
        kept_idx: list[int] = []
        for i in range(len(df)):
            arrs = [layers[k][i] for k in range(NUM_LAYERS)]
            T = len(arrs[0])
            if any(len(a) != T for a in arrs):
                continue                      # drop malformed rows
            if T < SEQ_LEN_MIN or T < 1:
                continue
            if T > self.max_len:
                arrs = [a[: self.max_len] for a in arrs]
                T = self.max_len
            rows.append(np.stack(arrs, axis=0).astype(np.int64))
            kept_idx.append(i)

        self.rows = rows
        self.sentences = [sentences[i] for i in kept_idx]
        self.lengths = np.array([r.shape[1] for r in rows], dtype=np.int32)
        self.length_bins = seq_len_to_bin(self.lengths, self.num_bins).astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        toks = self.rows[idx]                          # (NUM_LAYERS, T)
        T = toks.shape[1]
        T_pad = self.max_len
        padded = np.full((NUM_LAYERS, T_pad), PAD_ID, dtype=np.int64)
        padded[:, :T] = toks
        return {
            "tokens": torch.from_numpy(padded),
            "length": torch.tensor(T, dtype=torch.long),
            "length_bin": torch.tensor(int(self.length_bins[idx]), dtype=torch.long),
            "sentence": self.sentences[idx],
        }


def collate_token_batch(batch: Sequence[dict]) -> dict:
    """Stack tensors and pass `sentence` through as a list of strs."""
    out = {
        "tokens":     torch.stack([b["tokens"] for b in batch], dim=0),
        "length":     torch.stack([b["length"] for b in batch], dim=0),
        "length_bin": torch.stack([b["length_bin"] for b in batch], dim=0),
        "sentence":   [b["sentence"] for b in batch],
    }
    # Trim to longest in batch to save compute.
    max_T = int(out["length"].max().item())
    out["tokens"] = out["tokens"][:, :, :max_T].contiguous()
    return out
