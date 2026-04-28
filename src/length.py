"""Length predictor: text -> seq_len ∈ [SEQ_LEN_MIN, SEQ_LEN_MAX].

Approach
--------
Classification over `num_bins` evenly spaced bins on `[SEQ_LEN_MIN, SEQ_LEN_MAX]`.
Backbone: frozen `distilbert-base-uncased` mean-pooled embedding (768-d).
Head: MLP -> num_bins logits.

We keep this small and self-contained so it can be retrained in a couple of
minutes on a T4 and re-used at inference time as a "length oracle" for the
generative models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.constants import LENGTH_ESTIMATOR_CKPT, SEQ_LEN_MAX, SEQ_LEN_MIN


# ─── bin helpers ──────────────────────────────────────────────────────────────
def bin_edges(num_bins: int) -> np.ndarray:
    """Return `num_bins + 1` edges spanning [SEQ_LEN_MIN, SEQ_LEN_MAX]."""
    return np.linspace(SEQ_LEN_MIN, SEQ_LEN_MAX, num_bins + 1)


def bin_centers(num_bins: int) -> np.ndarray:
    e = bin_edges(num_bins)
    return ((e[:-1] + e[1:]) / 2.0).astype(np.int32)


def seq_len_to_bin(lengths: np.ndarray, num_bins: int) -> np.ndarray:
    """Clip to [MIN, MAX] then map to bin index in [0, num_bins-1]."""
    L = np.clip(lengths, SEQ_LEN_MIN, SEQ_LEN_MAX).astype(np.float32)
    edges = bin_edges(num_bins)
    # right=False so SEQ_LEN_MIN -> bin 0 and SEQ_LEN_MAX -> bin num_bins-1
    idx = np.searchsorted(edges[1:-1], L, side="right")
    return idx.astype(np.int64)


def bin_to_seq_len(bins: np.ndarray, num_bins: int) -> np.ndarray:
    centers = bin_centers(num_bins)
    return centers[np.clip(bins, 0, num_bins - 1)]


# ─── dataset ──────────────────────────────────────────────────────────────────
@dataclass
class LengthExample:
    text: str
    seq_len: int


class LengthDataset(Dataset):
    """Combines sentence + gloss into a single text for tokenization."""

    def __init__(self, sentences: list[str], glosses: list[str], seq_lens: list[int],
                 tokenizer, max_text_tokens: int = 64):
        assert len(sentences) == len(glosses) == len(seq_lens)
        self.sentences = sentences
        self.glosses = glosses
        self.seq_lens = np.asarray(seq_lens, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __len__(self) -> int:
        return len(self.seq_lens)

    def __getitem__(self, i: int):
        # "[sentence] [SEP] [gloss]" — DistilBERT auto-inserts CLS/SEP for pairs.
        enc = self.tokenizer(
            self.sentences[i],
            self.glosses[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "seq_len": int(self.seq_lens[i]),
        }


# ─── model ────────────────────────────────────────────────────────────────────
class LengthPredictor(nn.Module):
    def __init__(self, num_bins: int = 32, hidden: int = 512, dropout: float = 0.1,
                 backbone_name: str = "distilbert-base-uncased", freeze_backbone: bool = True):
        super().__init__()
        from transformers import AutoModel  # local import keeps top-level import light
        self.backbone = AutoModel.from_pretrained(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.num_bins = num_bins
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_bins),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not self.backbone.training else torch.enable_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # mean pool over non-pad tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        return self.head(pooled)

    @torch.no_grad()
    def predict_seq_len(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        mode: str = "argmax") -> torch.Tensor:
        """Return predicted seq_len (int) per row."""
        logits = self(input_ids, attention_mask)
        if mode == "argmax":
            bins = logits.argmax(dim=-1).cpu().numpy()
        elif mode == "expected":
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            centers = bin_centers(self.num_bins)
            return torch.tensor((probs * centers[None, :]).sum(axis=-1).round().astype(np.int64),
                                dtype=torch.long)
        else:
            raise ValueError(mode)
        return torch.tensor(bin_to_seq_len(bins, self.num_bins), dtype=torch.long)


def save(model: LengthPredictor, path: Path | str = LENGTH_ESTIMATOR_CKPT) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": model.head.state_dict(), "num_bins": model.num_bins}, path)


def load_head(model: LengthPredictor, path: Path | str = LENGTH_ESTIMATOR_CKPT) -> LengthPredictor:
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["num_bins"] == model.num_bins, "num_bins mismatch"
    model.head.load_state_dict(ckpt["head"])
    return model
