"""Local scorer: R-Precision (32-way), FID, Diversity, combined score.

Mirrors the competition spec:
    final = 0.30 * FID_norm + 0.50 * (1 - R-Precision) + 0.20 * Diversity_term
(We just report each component separately; users can weight as they see fit.)

Inputs are token grids (B, NUM_LAYERS, T_var) and matching texts/lengths.
Embeddings come from the public evaluator (`Evaluator.tokens_embed`,
`Evaluator.text_embed`).

R-Precision (32-way): for each text, build a candidate pool of 32 motions
where index 0 is the GT match for that text, and indices 1..31 are random
motions from the rest of the batch (or the full pool). Rank by cosine with
the text and report top-1, top-2, top-3 retrieval rates.

FID: Frechet Inception Distance between two sets of motion embeddings
(generated vs ground-truth) using mean and covariance, computed in the
same 256-d evaluator space.

Diversity: mean pairwise L2 distance between random pairs of generated
motion embeddings. Higher is more diverse, but only meaningful relative
to GT diversity (so we report both).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.constants import R_PRECISION_NUM_CANDIDATES, R_PRECISION_TOP_K
from src.eval.evaluator import Evaluator


# ---------------------------------------------------------------------------
# R-Precision
# ---------------------------------------------------------------------------

def r_precision_topk(text_emb: torch.Tensor, motion_emb: torch.Tensor,
                     n_candidates: int = R_PRECISION_NUM_CANDIDATES,
                     top_k: int = R_PRECISION_TOP_K,
                     n_repeats: int = 4,
                     seed: int = 0) -> dict:
    """Standard 32-way R-Precision.

    Args:
        text_emb:   (N, D) normalized
        motion_emb: (N, D) normalized   (paired index-by-index with text_emb)
        n_candidates: pool size per query (default 32)
        top_k: report R@1..R@top_k
        n_repeats: average over this many random candidate samplings.

    Returns dict {R@1, R@2, ..., R@top_k}.
    """
    assert text_emb.shape == motion_emb.shape
    N, D = text_emb.shape
    if N < n_candidates:
        raise ValueError(f"need at least {n_candidates} pairs, got {N}")

    rng = np.random.default_rng(seed)
    sums = np.zeros(top_k, dtype=np.float64)
    n_total = 0

    text_emb = text_emb.detach().cpu()
    motion_emb = motion_emb.detach().cpu()

    for rep in range(n_repeats):
        # Process in non-overlapping chunks of n_candidates.
        perm = rng.permutation(N)
        n_chunks = N // n_candidates
        for c in range(n_chunks):
            idx = perm[c * n_candidates : (c + 1) * n_candidates]
            te = text_emb[idx]                    # (32, D)
            me = motion_emb[idx]                  # (32, D)
            sims = te @ me.T                      # (32, 32) — diagonal is the true match
            ranks = sims.argsort(dim=-1, descending=True)  # (32, 32)
            # The true match for query i sits at column i.
            for k in range(1, top_k + 1):
                hits = (ranks[:, :k] == torch.arange(n_candidates)[:, None]).any(dim=-1)
                sums[k - 1] += hits.float().sum().item()
            n_total += n_candidates

    return {f"R@{k+1}": float(sums[k] / max(1, n_total)) for k in range(top_k)}


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """PSD matrix sqrt via eigendecomposition (covariance matrices are PSD)."""
    # Symmetrize defensively.
    mat = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def fid(emb_a: torch.Tensor, emb_b: torch.Tensor, eps: float = 1e-6) -> float:
    """FID between two sets of embeddings.
    emb_a, emb_b: (Na, D), (Nb, D) — typically a=generated, b=GT.
    """
    a = emb_a.detach().cpu().numpy().astype(np.float64)
    b = emb_b.detach().cpu().numpy().astype(np.float64)
    mu_a = a.mean(0); mu_b = b.mean(0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b

    # Tr(A + B - 2 sqrt(AB)). AB isn't symmetric, but sqrt(sqrt(A) B sqrt(A)) is the
    # canonical formulation; we use it directly.
    sa = _matrix_sqrt(cov_a + eps * np.eye(cov_a.shape[0]))
    inner = sa @ cov_b @ sa
    sqrt_inner = _matrix_sqrt(inner + eps * np.eye(inner.shape[0]))
    tr_term = float(np.trace(sqrt_inner))

    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2.0 * tr_term)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def diversity(emb: torch.Tensor, n_pairs: int = 300, seed: int = 0) -> float:
    """Mean L2 distance between `n_pairs` random pairs."""
    e = emb.detach().cpu().numpy()
    N = e.shape[0]
    if N < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N, size=n_pairs)
    b = rng.integers(0, N, size=n_pairs)
    same = a == b
    if same.any():
        b[same] = (b[same] + 1) % N
    diffs = e[a] - e[b]
    return float(np.linalg.norm(diffs, axis=1).mean())


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

@dataclass
class ScoreReport:
    r_precision: dict[str, float]
    fid_gen_vs_gt: float
    diversity_gen: float
    diversity_gt: float

    def pretty(self) -> str:
        rp = "  ".join(f"{k}={v:.3f}" for k, v in self.r_precision.items())
        return (f"[score] {rp}\n"
                f"        FID(gen vs gt) = {self.fid_gen_vs_gt:.3f}\n"
                f"        Div(gen)       = {self.diversity_gen:.3f}\n"
                f"        Div(gt)        = {self.diversity_gt:.3f}")


@torch.no_grad()
def score_predictions(evaluator: Evaluator,
                      sentences: Sequence[str],
                      gen_tokens: list[torch.Tensor],
                      gt_tokens: list[torch.Tensor],
                      n_candidates: int = R_PRECISION_NUM_CANDIDATES,
                      top_k: int = R_PRECISION_TOP_K,
                      n_repeats: int = 4) -> ScoreReport:
    """Compute R-Precision, FID, Diversity given parallel lists of tokens.

    Args:
        sentences: text prompts (length N)
        gen_tokens: list of (NUM_LAYERS, T_i) generated token grids
        gt_tokens:  list of (NUM_LAYERS, T_i) ground-truth token grids (same N)
    """
    assert len(sentences) == len(gen_tokens) == len(gt_tokens)
    text_emb = evaluator.text_embed(sentences)            # (N, 256)
    gen_emb = evaluator.tokens_embed(gen_tokens)          # (N, 256)
    gt_emb = evaluator.tokens_embed(gt_tokens)            # (N, 256)

    rp = r_precision_topk(text_emb, gen_emb,
                          n_candidates=n_candidates,
                          top_k=top_k,
                          n_repeats=n_repeats)
    f = fid(gen_emb, gt_emb)
    d_gen = diversity(gen_emb)
    d_gt = diversity(gt_emb)

    return ScoreReport(
        r_precision=rp,
        fid_gen_vs_gt=f,
        diversity_gen=d_gen,
        diversity_gt=d_gt,
    )
