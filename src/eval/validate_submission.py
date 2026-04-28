r"""Submission CSV validator.

Mirrors the organizer's validation rules so we catch problems locally before
spending a Kaggle submission slot. Run as:

    python -m scripts.validate_submission path\to\submission.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.constants import (
    NUM_LAYERS,
    NUM_TEST_ROWS,
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    SUBMISSION_COLUMNS,
    TOKEN_COLUMNS,
    TOKEN_MAX,
    TOKEN_MIN,
)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"[{head}] rows={self.n_rows}"]
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        for e in self.errors:
            lines.append(f"  err : {e}")
        return "\n".join(lines)


def _parse_tokens(cell: object) -> list[int] | None:
    """Parse a space-separated cell into a list of ints. Returns None if invalid."""
    if not isinstance(cell, str):
        return None
    parts = cell.strip().split()
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        # Reject anything that isn't a clean signed/unsigned integer.
        if not (p.lstrip("-").isdigit()):
            return None
        out.append(int(p))
    return out


def validate_submission(
    df: pd.DataFrame,
    *,
    expected_rows: int = NUM_TEST_ROWS,
    expected_ids: Iterable[int] | None = None,
) -> ValidationReport:
    """Validate a submission DataFrame against the competition spec.

    Args:
        df: parsed submission as a DataFrame.
        expected_rows: number of rows required (default 3000).
        expected_ids: optional iterable of ids to require; if given, must match
            the set of `df["id"]` exactly.
    """
    rep = ValidationReport(ok=True, n_rows=len(df))

    # 1. Columns
    missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing:
        rep.add_error(f"missing columns: {missing}")
        return rep  # cannot continue meaningfully
    extra = [c for c in df.columns if c not in SUBMISSION_COLUMNS]
    if extra:
        rep.add_warning(f"extra columns will be ignored: {extra}")

    # 2. Row count
    if len(df) != expected_rows:
        rep.add_error(f"row count {len(df)} != expected {expected_rows}")

    # 3. ID set
    if expected_ids is not None:
        want = set(int(x) for x in expected_ids)
        got = set(int(x) for x in df["id"].tolist())
        miss = want - got
        unexpected = got - want
        if miss:
            rep.add_error(f"{len(miss)} ids missing from submission (e.g. {sorted(list(miss))[:5]})")
        if unexpected:
            rep.add_error(f"{len(unexpected)} unexpected ids (e.g. {sorted(list(unexpected))[:5]})")
    if df["id"].duplicated().any():
        rep.add_error("duplicate ids in submission")

    # 4. Per-row token checks
    bad_rows = 0
    for row_ix, row in df.iterrows():
        layer_seqs: list[list[int]] = []
        for col in TOKEN_COLUMNS:
            tokens = _parse_tokens(row[col])
            if tokens is None:
                rep.add_error(f"row {row_ix} ({col}): unparseable / empty cell -> {row[col]!r}")
                bad_rows += 1
                layer_seqs = []
                break
            # length range
            if not (SEQ_LEN_MIN <= len(tokens) <= SEQ_LEN_MAX):
                rep.add_error(
                    f"row {row_ix} ({col}): length {len(tokens)} outside [{SEQ_LEN_MIN}, {SEQ_LEN_MAX}]"
                )
                bad_rows += 1
            # value range
            mn, mx = min(tokens), max(tokens)
            if mn < TOKEN_MIN or mx > TOKEN_MAX:
                rep.add_error(
                    f"row {row_ix} ({col}): token range [{mn},{mx}] outside [{TOKEN_MIN},{TOKEN_MAX}]"
                )
                bad_rows += 1
            layer_seqs.append(tokens)

        # 4b. Layer-length consistency
        if len(layer_seqs) == NUM_LAYERS:
            lens = {len(s) for s in layer_seqs}
            if len(lens) != 1:
                rep.add_error(f"row {row_ix}: layer lengths differ -> {sorted(lens)}")
                bad_rows += 1

        if bad_rows > 50:
            rep.add_warning("more than 50 row-level errors; stopping per-row scan early")
            break

    return rep


def validate_file(path: str | Path, **kwargs) -> ValidationReport:
    df = pd.read_csv(path)
    return validate_submission(df, **kwargs)
