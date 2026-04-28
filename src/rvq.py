"""Frozen RVQ-VAE wrapper.

The competition forbids modifying or replacing `rvq_vae_best.pth`. This wrapper
provides a strict load-and-forward interface; we only need `decode` to score
generated tokens locally and (optionally) `encode` to verify round-trip.

The exact `state_dict` keys depend on the organizer-provided checkpoint. We
defer building the architecture until we can inspect the keys. To keep the
rest of the project unblocked, this module exposes the public API as stubs
and raises `NotImplementedError` if called before the architecture is wired up.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.constants import RVQ_VAE_CKPT


class FrozenRVQVAE(nn.Module):
    """Adapter around the provided checkpoint. Implementation TBD after we
    inspect the .pth file structure (see scripts/inspect_data.py).
    """

    def __init__(self, ckpt_path: str | Path = RVQ_VAE_CKPT):
        super().__init__()
        self.ckpt_path = Path(ckpt_path)
        self._loaded = False
        # Set after loading:
        self.codebook_size: int | None = None
        self.num_layers: int | None = None
        self.feature_dim: int | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def load(self, device: str | torch.device = "cpu") -> "FrozenRVQVAE":
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"RVQ-VAE checkpoint not found: {self.ckpt_path}. "
                "Download it from the Kaggle competition data tab."
            )
        # We intentionally do NOT instantiate the network here yet — we need
        # to inspect the checkpoint keys first. See `peek_checkpoint`.
        raise NotImplementedError(
            "Architecture wiring deferred. Run `python -m scripts.inspect_data` "
            "to dump the checkpoint structure, then implement load() against it."
        )

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, NUM_LAYERS, T) int -> motion features (B, T*4, feature_dim)."""
        raise NotImplementedError("decode() not yet wired up. See load().")

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        """motion: (B, T_motion, feature_dim) -> tokens (B, NUM_LAYERS, T)."""
        raise NotImplementedError("encode() not yet wired up. See load().")


def peek_checkpoint(ckpt_path: str | Path = RVQ_VAE_CKPT) -> dict:
    """Load the checkpoint and return a summary suitable for printing.

    Useful first step before implementing FrozenRVQVAE.load().
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return {"error": f"missing: {ckpt_path}"}
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary: dict = {"top_level_type": type(obj).__name__}
    if isinstance(obj, dict):
        summary["top_level_keys"] = list(obj.keys())
        for k, v in obj.items():
            if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in v.values()):
                summary[f"{k}__shapes"] = {kk: tuple(vv.shape) for kk, vv in v.items()}
            elif hasattr(v, "shape"):
                summary[f"{k}__shape"] = tuple(v.shape)
    return summary
