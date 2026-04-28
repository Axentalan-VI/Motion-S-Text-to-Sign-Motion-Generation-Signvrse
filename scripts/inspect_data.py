"""CLI: Phase-0 dataset reconnaissance.

Prints summary statistics needed to make modeling decisions:
  - row counts (train/test)
  - sentence + gloss length distributions
  - sequence length distribution per layer
  - codebook usage histogram per layer (hot/cold tokens)
  - basic sanity checks (layer-length consistency, token range)
  - structure of `rvq_vae_best.pth`

Usage:
    python -m scripts.inspect_data            # read from data/
    python -m scripts.inspect_data --train other.csv --test other.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    CODEBOOK_SIZE,
    DATA_DIR,
    RVQ_VAE_CKPT,
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    TEST_CSV,
    TOKEN_COLUMNS,
    TRAIN_CSV,
)
from src.data.io import load_test, load_train
from src.rvq import peek_checkpoint


def _percentiles(arr: np.ndarray, ps=(0, 25, 50, 75, 90, 95, 99, 100)) -> dict:
    if len(arr) == 0:
        return {}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def inspect_train(path: Path) -> dict:
    if not path.exists():
        return {"error": f"missing: {path}"}
    df = load_train(path)
    out: dict = {"path": str(path), "n_rows": int(len(df))}

    # text stats
    if "sentence" in df.columns:
        sent_len = df["sentence"].fillna("").str.split().map(len).to_numpy()
        out["sentence_words"] = _percentiles(sent_len)
    if "gloss" in df.columns:
        gloss_len = df["gloss"].fillna("").str.split().map(len).to_numpy()
        out["gloss_words"] = _percentiles(gloss_len)
        # vocab
        glosses = (
            df["gloss"].fillna("").str.split().explode().dropna().astype(str)
        )
        out["gloss_vocab_size"] = int(glosses.nunique())
        out["gloss_top20"] = glosses.value_counts().head(20).to_dict()

    # sequence-length stats
    out["seq_len"] = _percentiles(df["seq_len"].to_numpy())
    out["seq_len_below_min"] = int((df["seq_len"] < SEQ_LEN_MIN).sum())
    out["seq_len_above_max"] = int((df["seq_len"] > SEQ_LEN_MAX).sum())

    # layer consistency
    mismatches = 0
    for _, row in df.iterrows():
        lens = {len(row[c]) for c in TOKEN_COLUMNS}
        if len(lens) != 1:
            mismatches += 1
    out["rows_with_layer_length_mismatch"] = mismatches

    # codebook usage per layer
    usage: dict = {}
    for col in TOKEN_COLUMNS:
        ctr = Counter()
        for arr in df[col]:
            if len(arr):
                ctr.update(arr.tolist())
        used = len(ctr)
        most = ctr.most_common(5)
        # Check value range
        if ctr:
            mn, mx = min(ctr.keys()), max(ctr.keys())
        else:
            mn = mx = None
        usage[col] = {
            "unique_codes_used": used,
            "codebook_size": CODEBOOK_SIZE,
            "coverage_pct": round(100.0 * used / CODEBOOK_SIZE, 2),
            "min_token": mn,
            "max_token": mx,
            "top5": most,
        }
    out["codebook_usage"] = usage

    # signer/metadata columns we might find
    for col in ("signer", "signer_id", "complexity"):
        if col in df.columns:
            out[f"{col}_value_counts"] = df[col].value_counts().head(10).to_dict()

    return out


def inspect_test(path: Path) -> dict:
    if not path.exists():
        return {"error": f"missing: {path}"}
    df = load_test(path)
    out = {"path": str(path), "n_rows": int(len(df)), "columns": list(df.columns)}
    if "sentence" in df.columns:
        out["sentence_words"] = _percentiles(
            df["sentence"].fillna("").str.split().map(len).to_numpy()
        )
    if "gloss" in df.columns:
        out["gloss_words"] = _percentiles(
            df["gloss"].fillna("").str.split().map(len).to_numpy()
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN_CSV))
    ap.add_argument("--test", default=str(TEST_CSV))
    ap.add_argument("--rvq-ckpt", default=str(RVQ_VAE_CKPT))
    ap.add_argument("--out", default=str(DATA_DIR / "data_recon.json"))
    args = ap.parse_args()

    report: dict = {
        "train": inspect_train(Path(args.train)),
        "test": inspect_test(Path(args.test)),
        "rvq_vae_checkpoint": peek_checkpoint(Path(args.rvq_ckpt)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\n[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
