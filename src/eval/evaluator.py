"""Wrapper for the public text-motion alignment evaluator
(`Public_t2m_align.pth`).

Architecture (inferred from `model_state_dict` of the checkpoint):

  evaluator
  ├── rvq_vae                # same FrozenRVQVAE as src/rvq.py (prefix stripped)
  │   ├── encoder            # Conv1d * 4 + BN + (act); last block has BN
  │   ├── rvq                # 6 quantizers, 512x256 each
  │   └── decoder            # ConvT * 4; last block bare
  ├── clip_encoder.clip_model        # OpenAI CLIP (ViT-B/32 — embed_dim=512)
  ├── text_projector         # Sequential
  │     0: Linear(512, 512)
  │     1: LayerNorm(512)
  │     2: GELU                    (no params)
  │     3: Dropout                 (no params)
  │     4: Linear(512, 256)
  │     5: LayerNorm(256)
  │     6: GELU                    (no params)
  │     7: Dropout                 (no params)
  │     8: Linear(256, 256)
  └── motion_projector       # ResMLP-style
        fc1:  Linear(256, 512)
        fc2:  Linear(512, 512)
        fc3:  Linear(512, 256)
        skip: Linear(256, 512)            # residual on input

Forward (verified empirically via 24-combo sweep — arch=v2, act=relu wins
with R@1=0.297, gap=+0.310 on N=64):
    h = act(fc1(x))
    h = act(fc2(h)) + skip(x)
    z = fc3(h)

R-Precision proxy: cosine similarity in the shared 256-d normalized space
between text_embed(text) and motion_embed(motion).

This module exposes:
    Evaluator.text_embed(texts)      -> (B, 256) normalized
    Evaluator.motion_embed(motion)   -> (B, 256) normalized   (raw motion, not tokens)
    Evaluator.r_precision(text, motion) -> cosine sim
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rvq import _Encoder, _Decoder, _RVQ, RVQVAEConfig


# ---------------------------------------------------------------------------
# RVQ-VAE wrapper used inside the evaluator (state-dict prefix `rvq_vae.`).
# ---------------------------------------------------------------------------

class _EvalRVQVAE(nn.Module):
    """The RVQ-VAE module embedded inside the evaluator. Exposes `encode` and
    `decode` like FrozenRVQVAE but is a plain nn.Module so it strict-loads from
    the evaluator's state-dict.
    """

    def __init__(self, cfg: RVQVAEConfig, activation: str = "relu",
                 strides: tuple[int, int, int, int] = (1, 1, 2, 2)):
        super().__init__()
        self.encoder = _Encoder(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim,
                                strides=strides, activation=activation)
        self.rvq = _RVQ(num_quantizers=cfg.num_quantizers,
                        latent_dim=cfg.latent_dim,
                        codebook_size=cfg.num_embeddings)
        self.decoder = _Decoder(cfg.latent_dim, cfg.hidden_dim, cfg.output_dim,
                                strides=strides, activation=activation)


# ---------------------------------------------------------------------------
# Motion projector (ResMLP-style).
# ---------------------------------------------------------------------------

class _MotionProjector(nn.Module):
    """256 -> 512 -> 512 -> 256 with skip on the input.

    Shapes from state-dict:
        fc1.weight  (512, 256)   fc1.bias  (512,)
        fc2.weight  (512, 512)   fc2.bias  (512,)
        fc3.weight  (256, 512)   fc3.bias  (256,)
        skip.weight (512, 256)   skip.bias (512,)

    Several plausible residual layouts; selectable via `arch`:
        "v1": h = act(fc1(x)) + skip(x);  h = act(fc2(h));         z = fc3(h)
        "v2": h = act(fc1(x));            h = act(fc2(h)) + skip(x); z = fc3(h)
        "v3": h = fc1(x) + skip(x);       h = act(h); h = act(fc2(h)); z = fc3(h)
        "v4": h = act(fc1(x));            h = fc2(h) + skip(x); h = act(h); z = fc3(h)
    """

    def __init__(self, in_dim: int = 256, hidden: int = 512, out_dim: int = 256,
                 activation: str = "relu", arch: str = "v2"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, hidden)
        self._act_name = activation
        self._arch = arch

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self._act_name == "relu":
            return F.relu(x)
        if self._act_name == "gelu":
            return F.gelu(x)
        if self._act_name == "leaky_relu":
            return F.leaky_relu(x, 0.2)
        if self._act_name == "silu":
            return F.silu(x)
        if self._act_name == "elu":
            return F.elu(x)
        if self._act_name == "identity":
            return x
        raise ValueError(self._act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._arch == "v1":
            h = self._act(self.fc1(x)) + self.skip(x)
            h = self._act(self.fc2(h))
        elif self._arch == "v2":
            h = self._act(self.fc1(x))
            h = self._act(self.fc2(h)) + self.skip(x)
        elif self._arch == "v3":
            h = self.fc1(x) + self.skip(x)
            h = self._act(h)
            h = self._act(self.fc2(h))
        elif self._arch == "v4":
            h = self._act(self.fc1(x))
            h = self.fc2(h) + self.skip(x)
            h = self._act(h)
        else:
            raise ValueError(f"unknown arch: {self._arch}")
        return self.fc3(h)


# ---------------------------------------------------------------------------
# CLIP loader.
# ---------------------------------------------------------------------------

def _load_clip(name: str = "ViT-B/32", device: str = "cpu"):
    """Load OpenAI CLIP. Requires the `clip` package
    (pip install git+https://github.com/openai/CLIP.git).
    Returns (clip_model, tokenize_fn).
    """
    try:
        import clip  # type: ignore
    except ImportError as e:
        raise ImportError(
            "OpenAI CLIP not installed. Install with:\n"
            "  pip install ftfy regex\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        ) from e
    model, _ = clip.load(name, device=device, jit=False)
    model = model.float()  # disable fp16 for stability on CPU
    return model, clip.tokenize


# ---------------------------------------------------------------------------
# Top-level evaluator module: matches the checkpoint key structure.
# ---------------------------------------------------------------------------

class _ClipWrap(nn.Module):
    """Wrapper that stores a CLIP model under `clip_model` so its state_dict
    keys look like `clip_model.<...>`."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model


class _InfoNCELoss(nn.Module):
    """Just holds the learnable log-temperature scalar from training. Not used
    at eval, but registered so strict-loading the checkpoint succeeds.
    """

    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(()))


class _PublicEvaluator(nn.Module):
    """The full evaluator module, mirroring the checkpoint key structure
    so we can `load_state_dict(strict=True)`.
    """

    def __init__(self, cfg: RVQVAEConfig, clip_name: str = "ViT-B/32",
                 device: str = "cpu", proj_act: str = "relu",
                 proj_arch: str = "v2"):
        super().__init__()
        self.rvq_vae = _EvalRVQVAE(cfg)
        clip_model, _tok = _load_clip(clip_name, device=device)
        self.clip_encoder = _ClipWrap(clip_model)
        self.text_projector = nn.Sequential(
            nn.Linear(512, 512),     # 0
            nn.LayerNorm(512),       # 1
            nn.GELU(),               # 2 (no params)
            nn.Dropout(0.1),         # 3 (no params)
            nn.Linear(512, 256),     # 4
            nn.LayerNorm(256),       # 5
            nn.GELU(),               # 6
            nn.Dropout(0.1),         # 7
            nn.Linear(256, 256),     # 8
        )
        self.motion_projector = _MotionProjector(activation=proj_act, arch=proj_arch)
        self.infonce_loss = _InfoNCELoss()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

class Evaluator:
    """User-facing wrapper around the public evaluator checkpoint.

    Provides text/motion embedding in the shared 256-d cosine space.
    """

    def __init__(self, ckpt_path: str | Path, cfg: RVQVAEConfig | None = None,
                 clip_name: str = "ViT-B/32", device: str = "cpu",
                 proj_act: str = "relu", proj_arch: str = "v2"):
        self.device = torch.device(device)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        # The evaluator checkpoint also stores a `config` key, but that is the
        # *evaluator's* training config (loss weights, lr, etc.), NOT the
        # RVQ-VAE architecture config. Always use the known-good RVQ defaults
        # unless the caller explicitly passes one.
        if cfg is None:
            cfg = RVQVAEConfig()
        self.cfg = cfg
        self.module = _PublicEvaluator(cfg, clip_name=clip_name, device="cpu",
                                       proj_act=proj_act, proj_arch=proj_arch)
        sd = ckpt["model_state_dict"]
        # Try strict; fall back to non-strict on diagnostic load.
        missing, unexpected = self.module.load_state_dict(sd, strict=False)
        self._missing = list(missing)
        self._unexpected = list(unexpected)
        self.module.eval().to(self.device)
        for p in self.module.parameters():
            p.requires_grad_(False)
        # Cache CLIP tokenizer.
        import clip  # type: ignore
        self._tokenize = clip.tokenize

    # -- diagnostics ---------------------------------------------------------
    def report(self) -> dict:
        return {
            "missing": self._missing,
            "unexpected": self._unexpected,
            "missing_count": len(self._missing),
            "unexpected_count": len(self._unexpected),
        }

    # -- text path -----------------------------------------------------------
    @torch.no_grad()
    def text_embed(self, texts: Iterable[str]) -> torch.Tensor:
        toks = self._tokenize(list(texts), truncate=True).to(self.device)
        clip_model = self.module.clip_encoder.clip_model
        x = clip_model.encode_text(toks).float()        # (B, 512)
        z = self.module.text_projector(x)               # (B, 256)
        return F.normalize(z, dim=-1)

    # -- motion path ---------------------------------------------------------
    @torch.no_grad()
    def motion_embed(self, motion: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """motion: (B, T, 668) raw features.
        lengths: optional (B,) valid-length per sample for masked mean-pool.
        Returns (B, 256) normalized.
        """
        if motion.ndim != 3:
            raise ValueError(f"motion must be (B,T,D); got {tuple(motion.shape)}")
        x = motion.to(self.device).transpose(1, 2)               # (B, D, T)
        z = self.module.rvq_vae.encoder(x)                       # (B, 256, T_tok)
        if lengths is not None:
            ds = self.cfg.downsampling_ratio
            t_tok = z.shape[-1]
            mask = torch.zeros(z.shape[0], t_tok, device=z.device)
            for i, L in enumerate(lengths.tolist()):
                Lt = max(1, min(t_tok, int(L) // ds))
                mask[i, :Lt] = 1.0
            mask = mask.unsqueeze(1)                             # (B, 1, T_tok)
            pooled = (z * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        else:
            pooled = z.mean(-1)                                  # (B, 256)
        out = self.module.motion_projector(pooled)               # (B, 256)
        return F.normalize(out, dim=-1)

    # -- motion path from tokens (no raw features needed) -------------------
    @torch.no_grad()
    def tokens_embed(self, tokens_list: list) -> torch.Tensor:
        """tokens_list: list of length B, each item shape (L, T_i) long
        (L = num_quantizers, T_i = sequence length, can vary per sample).
        Reconstructs latents via RVQ codebook lookup, mean-pools over time,
        runs through motion_projector, returns (B, 256) normalized.
        """
        rvq = self.module.rvq_vae.rvq
        embs = []
        for tk in tokens_list:
            tk = tk.to(self.device)
            if tk.ndim != 2:
                raise ValueError(f"each tokens entry must be (L, T); got {tuple(tk.shape)}")
            tk = tk.unsqueeze(0)                                  # (1, L, T)
            z = rvq.decode(tk)                                    # (1, 256, T)
            pooled = z.mean(-1)                                   # (1, 256)
            embs.append(pooled)
        pooled = torch.cat(embs, dim=0)                           # (B, 256)
        out = self.module.motion_projector(pooled)
        return F.normalize(out, dim=-1)

    # -- scoring -------------------------------------------------------------
    @torch.no_grad()
    def cosine(self, text_emb: torch.Tensor, motion_emb: torch.Tensor) -> torch.Tensor:
        """Returns full cosine matrix (Nt, Nm)."""
        return text_emb @ motion_emb.T


__all__ = ["Evaluator", "_PublicEvaluator"]
