"""CLI: build the frozen 90/10 stratified train/val split.

Usage:
    python -m scripts.make_split
"""
from __future__ import annotations

from src.constants import SPLIT_FILE, TRAIN_CSV
from src.data.io import load_train
from src.data.split import make_split, save_split


def main() -> int:
    print(f"loading {TRAIN_CSV} ...")
    df = load_train(TRAIN_CSV)
    print(f"  rows: {len(df)}")
    split = make_split(df)
    path = save_split(split, SPLIT_FILE)
    print(f"train: {len(split['train'])}  val: {len(split['val'])}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
