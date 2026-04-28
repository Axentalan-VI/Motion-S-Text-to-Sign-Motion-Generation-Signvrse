"""Frozen text-conditioning encoder for MoMask.

Reuses the OpenAI CLIP ViT-B/32 model that the public evaluator uses.
This way text embeddings live in the SAME space the evaluator scores in,
which gives the generator a cleaner signal during training.

Outputs:
    encode_texts(texts) -> (B, 512) on the requested device
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.eval.evaluator import _load_clip


class FrozenTextEncoder(nn.Module):
    """Wraps OpenAI CLIP. Frozen, eval-only."""

    def __init__(self, name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        clip_model, tokenize_fn = _load_clip(name, device=device)
        self.clip_model = clip_model
        self.tokenize_fn = tokenize_fn
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.clip_model.eval()
        self._device = device

    @property
    def out_dim(self) -> int:
        return 512    # ViT-B/32 text width

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], device: str | None = None) -> torch.Tensor:
        device = device or self._device
        tokens = self.tokenize_fn(list(texts), truncate=True).to(device)
        feats = self.clip_model.encode_text(tokens).float()
        return feats   # (B, 512)

    def train(self, mode: bool = True):
        # Keep CLIP in eval mode no matter what.
        super().train(mode)
        self.clip_model.eval()
        return self
