"""Smoke tests for the submission validator (runnable without competition data).

Usage:
    python -m scripts.test_validator
"""
from __future__ import annotations

import io
import sys

import pandas as pd

from src.constants import NUM_TEST_ROWS, SEQ_LEN_MIN, SUBMISSION_COLUMNS
from src.eval.validate_submission import validate_submission


def _make_row(rid: int, length: int = SEQ_LEN_MIN, val: int = 0) -> dict:
    cell = " ".join(str(val) for _ in range(length))
    row = {"id": rid}
    for c in SUBMISSION_COLUMNS[1:]:
        row[c] = cell
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))


def main() -> int:
    failures = 0

    # 1. Valid 3000-row submission of all-zero tokens at min length.
    df = _frame([_make_row(i) for i in range(NUM_TEST_ROWS)])
    rep = validate_submission(df)
    assert rep.ok, f"expected pass, got: {rep.render()}"
    print("[ok] valid all-zero submission passes")

    # 2. Wrong row count
    rep = validate_submission(_frame([_make_row(i) for i in range(10)]))
    assert not rep.ok and any("row count" in e for e in rep.errors)
    print("[ok] wrong row count detected")

    # 3. Token out of range
    bad = [_make_row(i) for i in range(NUM_TEST_ROWS)]
    bad[0]["base_tokens"] = " ".join(["999"] * SEQ_LEN_MIN)
    rep = validate_submission(_frame(bad))
    assert not rep.ok and any("token range" in e for e in rep.errors)
    print("[ok] out-of-range token detected")

    # 4. Length below min
    bad = [_make_row(i) for i in range(NUM_TEST_ROWS)]
    bad[0]["base_tokens"] = "1 2 3"
    rep = validate_submission(_frame(bad))
    assert not rep.ok and any("length 3" in e for e in rep.errors)
    print("[ok] short-sequence detected")

    # 5. Mismatched layer lengths
    bad = [_make_row(i) for i in range(NUM_TEST_ROWS)]
    bad[0]["residual_1"] = " ".join(["0"] * (SEQ_LEN_MIN + 1))
    rep = validate_submission(_frame(bad))
    assert not rep.ok and any("layer lengths differ" in e for e in rep.errors)
    print("[ok] layer-length mismatch detected")

    # 6. Missing column
    bad_df = _frame([_make_row(i) for i in range(NUM_TEST_ROWS)]).drop(columns=["residual_5"])
    rep = validate_submission(bad_df)
    assert not rep.ok and any("missing columns" in e for e in rep.errors)
    print("[ok] missing column detected")

    print("\nAll validator smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
