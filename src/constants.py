"""Project-wide constants pulled from the competition spec.

Single source of truth — import these everywhere instead of hard-coding numbers.
"""
from __future__ import annotations

from pathlib import Path

# ── Token grid ────────────────────────────────────────────────────────────────
NUM_LAYERS: int = 6                 # base + residual_1..residual_5
CODEBOOK_SIZE: int = 512            # tokens are integers in [0, 511]
TOKEN_MIN: int = 0
TOKEN_MAX: int = CODEBOOK_SIZE - 1

# ── Sequence length ───────────────────────────────────────────────────────────
SEQ_LEN_MIN: int = 40               # tokens
SEQ_LEN_MAX: int = 800              # tokens
TOKENS_PER_SECOND: float = 7.5      # 30 fps source ÷ 4× temporal compression
FPS: int = 30

# ── Test set / submission ────────────────────────────────────────────────────
NUM_TEST_ROWS: int = 3000
SUBMISSION_COLUMNS: tuple[str, ...] = (
    "id",
    "base_tokens",
    "residual_1",
    "residual_2",
    "residual_3",
    "residual_4",
    "residual_5",
)
TOKEN_COLUMNS: tuple[str, ...] = SUBMISSION_COLUMNS[1:]  # drop "id"

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
CHECKPOINT_DIR: Path = REPO_ROOT / "checkpoints"
SUBMISSION_DIR: Path = REPO_ROOT / "submissions"
RUNS_DIR: Path = REPO_ROOT / "runs"

TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Path = DATA_DIR / "sample_submission.csv"
RVQ_VAE_CKPT: Path = DATA_DIR / "rvq_vae_best.pth"
LENGTH_ESTIMATOR_CKPT: Path = DATA_DIR / "length_estimator.pth"

SPLIT_FILE: Path = DATA_DIR / "split_90_10.json"   # written by scripts/make_split.py

# ── Scoring weights (organizers) ──────────────────────────────────────────────
SCORE_WEIGHTS = {"fid": 0.30, "r_precision": 0.50, "diversity": 0.20}
R_PRECISION_TOP_K: int = 3          # top-3 retrieval over 32 candidates
R_PRECISION_NUM_CANDIDATES: int = 32
