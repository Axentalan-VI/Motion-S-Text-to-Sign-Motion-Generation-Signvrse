"""MoMask: masked-token base transformer + residual transformer.

Two models share the same Transformer backbone:

  BaseMaskTransformer
    - Inputs: noisy base_tokens with [MASK] placeholders, text cond, length cond
    - Output: per-position logits over codebook (512)
    - Training: random-mask cosine schedule, predict masked positions only
    - Inference: confidence-based unmasking, ~10 steps, CFG

  ResidualTransformer
    - Inputs: token grid up to layer L-1 stacked + summed (256-d each via codebook),
              plus text/length cond and layer-index embedding
    - Output: per-position logits over codebook (512) for layer L
    - Training: pick a random layer L >= 1, predict it given layers 0..L-1
    - Inference: layer-by-layer greedy/sample, optionally with CFG

Vocabulary layout (base):
    0..511   real codebook tokens
    512      [MASK]
    513      [PAD]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import CODEBOOK_SIZE, NUM_LAYERS, SEQ_LEN_MAX

MASK_ID: int = CODEBOOK_SIZE          # 512
PAD_ID: int = CODEBOOK_SIZE + 1       # 513
VOCAB_SIZE: int = CODEBOOK_SIZE + 2   # 514


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MoMaskConfig:
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = SEQ_LEN_MAX        # 800
    vocab_size: int = VOCAB_SIZE
    cond_dim: int = 512               # CLIP text dim
    n_residual_layers: int = NUM_LAYERS - 1   # 5
    num_length_bins: int = 32

    @classmethod
    def from_dict(cls, d: dict) -> "MoMaskConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in keys})


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _PosEmb(nn.Module):
    """Learned absolute positional embedding (simpler than sinusoidal here, and
    the corpus is bounded by SEQ_LEN_MAX)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return x + self.emb(pos)[None]


class _CrossCondBlock(nn.Module):
    """Pre-LN transformer block with cross-attention to a global cond vector.
    The cond is broadcast to a length-1 memory, so cross-attn becomes a soft
    gating over the cond. Cheap and effective for short cond sequences."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x:    (B, T, d)
        # cond: (B, S, d)   (S=1 or more)
        h = self.ln1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask,
                              need_weights=False)
        x = x + h
        h = self.ln2(x)
        h, _ = self.cross_attn(h, cond, cond, need_weights=False)
        x = x + h
        h = self.ln3(x)
        x = x + self.ff(h)
        return x


# ---------------------------------------------------------------------------
# Conditioning encoder: text + length -> cond memory
# ---------------------------------------------------------------------------

class _CondEncoder(nn.Module):
    """Map (text_emb [B, cond_dim], length_bin [B]) -> (B, S, d_model) memory.
    We use S=2 tokens: [text, length].
    """

    def __init__(self, cfg: MoMaskConfig):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.len_emb = nn.Embedding(cfg.num_length_bins, cfg.d_model)

    def forward(self, text_emb: torch.Tensor,
                length_bin: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)               # (B, d)
        l = self.len_emb(length_bin)               # (B, d)
        return torch.stack([t, l], dim=1)          # (B, 2, d)


# ---------------------------------------------------------------------------
# Base mask transformer
# ---------------------------------------------------------------------------

class BaseMaskTransformer(nn.Module):
    """Predicts the base layer (layer 0) tokens via masked-token modeling.

    Forward: tokens (B,T) ints → logits (B,T,V).
    Use `mask_schedule` for training noise and `generate` for inference.
    """

    def __init__(self, cfg: MoMaskConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                    padding_idx=PAD_ID)
        self.pos = _PosEmb(cfg.max_len, cfg.d_model)
        self.cond_enc = _CondEncoder(cfg)
        self.blocks = nn.ModuleList([
            _CrossCondBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Output head only over real codebook (no [MASK]/[PAD] predictions).
        self.head = nn.Linear(cfg.d_model, CODEBOOK_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor, text_emb: torch.Tensor,
                length_bin: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, T) longs in [0, VOCAB_SIZE)
        x = self.tok_emb(tokens)
        x = self.pos(x)
        cond = self.cond_enc(text_emb, length_bin)
        for blk in self.blocks:
            x = blk(x, cond, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)              # (B, T, CODEBOOK_SIZE)
        return logits

    @torch.no_grad()
    def generate(self, text_emb: torch.Tensor, length_bin: torch.Tensor,
                 lengths: torch.Tensor, n_steps: int = 10,
                 cfg_scale: float = 4.0,
                 uncond_text_emb: torch.Tensor | None = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """Iterative confidence-based decoding.

        Args:
            text_emb:    (B, cond_dim) CLIP-ish text embedding
            length_bin:  (B,)          length-bin index
            lengths:     (B,)          target sequence length per sample (in tokens)
            n_steps:     number of unmasking iterations
            cfg_scale:   classifier-free guidance scale (1.0 = off)
            uncond_text_emb: same shape as text_emb; required if cfg_scale != 1
            temperature: sampling temperature

        Returns:
            tokens: (B, T_max) longs in [0, CODEBOOK_SIZE); positions beyond
                    the per-sample length are PAD_ID.
        """
        B = text_emb.size(0)
        T = int(lengths.max().item())
        device = text_emb.device

        tokens = torch.full((B, T), MASK_ID, dtype=torch.long, device=device)
        # mark PAD beyond per-sample length
        pos_idx = torch.arange(T, device=device)[None].expand(B, -1)
        is_pad = pos_idx >= lengths[:, None]
        tokens[is_pad] = PAD_ID
        # only positions in [0, length) are decodable
        decodable = ~is_pad

        # cosine schedule for fraction-still-masked
        for step in range(n_steps):
            t_frac = (step + 1) / n_steps
            # fraction to KEEP masked AFTER this step
            keep_masked_frac = math.cos(math.pi / 2 * t_frac)

            logits = self.forward(tokens, text_emb, length_bin)
            if cfg_scale != 1.0 and uncond_text_emb is not None:
                logits_u = self.forward(tokens, uncond_text_emb, length_bin)
                logits = logits_u + cfg_scale * (logits - logits_u)
            logits = logits / max(temperature, 1e-4)

            # sample at currently-masked positions
            is_masked = (tokens == MASK_ID) & decodable
            probs = F.softmax(logits, dim=-1)             # (B, T, V)
            # categorical sample per position
            flat = probs.view(-1, probs.size(-1))
            sampled = torch.multinomial(flat, 1).view(B, T)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            # write samples ONLY at masked positions
            new_tokens = torch.where(is_masked, sampled, tokens)

            # decide which to RE-MASK based on confidence
            # We want `keep_masked_frac * length` positions still masked per row.
            for b in range(B):
                L = int(lengths[b].item())
                if L == 0:
                    continue
                k_mask = int(round(keep_masked_frac * L))
                k_mask = max(0, min(L - 1, k_mask))  # always keep at least 1 unmasked at last step
                if step == n_steps - 1:
                    k_mask = 0
                if k_mask == 0:
                    continue
                conf_b = confidence[b, :L].clone()
                # only re-mask positions that we just sampled (originally masked)
                was_masked = is_masked[b, :L]
                conf_b[~was_masked] = float("inf")  # don't re-mask fixed tokens
                # lowest-confidence k_mask positions get re-masked
                _, idx = conf_b.topk(k_mask, largest=False)
                new_tokens[b, idx] = MASK_ID

            tokens = new_tokens

        # final cleanup: any leftover MASKs → argmax
        leftover = (tokens == MASK_ID) & decodable
        if leftover.any():
            logits = self.forward(tokens, text_emb, length_bin)
            argmax = logits.argmax(dim=-1)
            tokens = torch.where(leftover, argmax, tokens)
        return tokens


# ---------------------------------------------------------------------------
# Residual transformer
# ---------------------------------------------------------------------------

class ResidualTransformer(nn.Module):
    """Predicts residual layer L (1..5) given layers 0..L-1 + cond.

    Input is the SUM of token embeddings of layers 0..L-1 (per position),
    plus a learned per-layer query embedding. The model's job is to output
    per-position logits for layer L's tokens.
    """

    def __init__(self, cfg: MoMaskConfig):
        super().__init__()
        self.cfg = cfg
        # Separate codebook embedding per residual layer index 0..NUM_LAYERS-1.
        # Real tokens only (no [MASK]); residual training/inference doesn't mask.
        self.tok_emb = nn.ModuleList([
            nn.Embedding(CODEBOOK_SIZE, cfg.d_model)
            for _ in range(NUM_LAYERS)
        ])
        # Per-target-layer query token (broadcast as a fingerprint).
        self.layer_query = nn.Embedding(cfg.n_residual_layers, cfg.d_model)
        self.pos = _PosEmb(cfg.max_len, cfg.d_model)
        self.cond_enc = _CondEncoder(cfg)
        self.blocks = nn.ModuleList([
            _CrossCondBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Single shared head; layer-specific bias term.
        self.head = nn.Linear(cfg.d_model, CODEBOOK_SIZE, bias=False)

    def _embed_prev_layers(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """prev_tokens: (B, L_prev, T) ints. Returns (B, T, d).
        Sums per-layer embeddings so the model sees the full state so far.
        Out-of-vocab ids (e.g. PAD_ID=513 in padded positions, or MASK_ID=512
        leftover from the base decoder) are clamped to 0 — the loss/logits
        on padded positions are masked out by the caller anyway.
        """
        B, Lp, T = prev_tokens.shape
        out = torch.zeros(B, T, self.cfg.d_model, device=prev_tokens.device,
                          dtype=self.tok_emb[0].weight.dtype)
        safe = prev_tokens.clamp(min=0, max=CODEBOOK_SIZE - 1)
        for li in range(Lp):
            out = out + self.tok_emb[li](safe[:, li])
        return out

    def forward(self, prev_tokens: torch.Tensor, target_layer: torch.Tensor,
                text_emb: torch.Tensor, length_bin: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            prev_tokens:   (B, L_prev, T) where L_prev = target_layer
            target_layer:  (B,) ints in [1, NUM_LAYERS-1]
            text_emb:      (B, cond_dim)
            length_bin:    (B,)
        Returns:
            logits: (B, T, CODEBOOK_SIZE)
        """
        x = self._embed_prev_layers(prev_tokens)
        # broadcast layer query: target_layer is 1-indexed (layer 1..5),
        # layer_query is indexed 0..n_residual_layers-1 = 0..4.
        q = self.layer_query(target_layer - 1)[:, None]      # (B, 1, d)
        x = x + q                                            # broadcast add
        x = self.pos(x)
        cond = self.cond_enc(text_emb, length_bin)
        for blk in self.blocks:
            x = blk(x, cond, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, base_tokens: torch.Tensor, text_emb: torch.Tensor,
                 length_bin: torch.Tensor, lengths: torch.Tensor,
                 cfg_scale: float = 1.0,
                 uncond_text_emb: torch.Tensor | None = None,
                 temperature: float = 1.0,
                 sample: bool = False) -> torch.Tensor:
        """Decode residual layers 1..NUM_LAYERS-1 given base layer.

        Returns:
            tokens: (B, NUM_LAYERS, T) — layer 0 is the input base; rest are filled.
        """
        B, T = base_tokens.shape
        device = base_tokens.device
        all_layers = torch.zeros(B, NUM_LAYERS, T, dtype=torch.long, device=device)
        all_layers[:, 0] = base_tokens

        for L in range(1, NUM_LAYERS):
            tgt = torch.full((B,), L, dtype=torch.long, device=device)
            prev = all_layers[:, :L]
            logits = self.forward(prev, tgt, text_emb, length_bin)
            if cfg_scale != 1.0 and uncond_text_emb is not None:
                logits_u = self.forward(prev, tgt, uncond_text_emb, length_bin)
                logits = logits_u + cfg_scale * (logits - logits_u)
            logits = logits / max(temperature, 1e-4)
            if sample:
                probs = F.softmax(logits, dim=-1).view(-1, CODEBOOK_SIZE)
                pred = torch.multinomial(probs, 1).view(B, T)
            else:
                pred = logits.argmax(dim=-1)
            all_layers[:, L] = pred
        return all_layers


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def cosine_mask_schedule(prog: torch.Tensor) -> torch.Tensor:
    """Mass to mask given progress in [0, 1]. prog=0 → 1.0 mask, prog=1 → 0."""
    return torch.cos(prog * math.pi / 2)


def random_mask(tokens: torch.Tensor, lengths: torch.Tensor,
                rng: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random-mass masking with cosine schedule per sample.

    Returns (noisy_tokens, mask_bool) where mask_bool is True at masked positions.
    Padded positions are never masked.
    """
    B, T = tokens.shape
    device = tokens.device
    # Sample a per-sample progress and convert to mask probability.
    prog = torch.rand(B, device=device, generator=rng)
    p_mask = cosine_mask_schedule(prog).clamp_(0.05, 0.95)   # (B,)

    pos = torch.arange(T, device=device)[None]
    is_pad = pos >= lengths[:, None]                          # (B, T)

    rand_u = torch.rand(B, T, device=device, generator=rng)
    mask = (rand_u < p_mask[:, None]) & ~is_pad

    # ensure at least one position is masked per row (when length > 0)
    no_mask_rows = ~mask.any(dim=1) & (lengths > 0)
    if no_mask_rows.any():
        for b in torch.where(no_mask_rows)[0]:
            j = torch.randint(int(lengths[b].item()), (1,), device=device).item()
            mask[b, j] = True

    noisy = tokens.clone()
    noisy[mask] = MASK_ID
    noisy[is_pad] = PAD_ID
    return noisy, mask
