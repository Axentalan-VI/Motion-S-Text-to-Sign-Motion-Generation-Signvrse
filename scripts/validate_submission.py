"""CLI: validate a submission CSV against the competition spec.

Usage:
    python -m scripts.validate_submission path\to\submission.csv
"""
from __future__ import annotations

import sys

from src.eval.validate_submission import validate_file


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m scripts.validate_submission <submission.csv>")
        return 2
    report = validate_file(argv[1])
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
