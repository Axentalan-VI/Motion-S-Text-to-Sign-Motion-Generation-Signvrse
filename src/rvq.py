"""Frozen RVQ-VAE wrapper for the Motion-S checkpoint.

Architecture (inferred from `rvq_vae_best.pth` `model_state_dict` keys + saved
`config`):

    config = {
        'input_dim': 668, 'output_dim': 668,
        'latent_dim': 256, 'hidden_dim': 512,
        'downsampling_ratio': 4, 'num_layers': 4,
        'num_quantizers': 6, 'num_embeddings': 512, ...
    }

State-dict layout per branch (Sequential):
    encoder.encoder.{0,3,6,9}.{weight,bias}    -> Conv1d
    encoder.encoder.{2,5,8,11}.*               -> BatchNorm1d
    rvq.quantizers.{i}.{embedding,cluster_size,embedding_avg}  i=0..5
    decoder.decoder.{0,3,6,9}.{weight,bias}    -> ConvTranspose1d
    decoder.decoder.{2,5,8,11}.*               -> BatchNorm1d

Slot pattern `Conv -> ? -> BN` -> `?` has no params (an activation; we use
`LeakyReLU(0.2)` which is the common motion-VAE choice). The activation does
not affect state_dict loading (only forward output).

Stride pattern: total downsample = 4 over 4 conv blocks. We use [1, 2, 1, 2]
which matches MoMask convention. If decoded length is wrong, swap to
[2, 1, 2, 1] and retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from src.constants import RVQ_VAE_CKPT


def _act() -> nn.Module:
    # LeakyReLU(0.2) is the common motion-VAE choice; no params either way.
    return nn.LeakyReLU(0.2, inplace=True)


# ─── encoder / decoder ───────────────────────────────────────────────────────
class _Encoder(nn.Module):
    """4-block stack matching `encoder.encoder.{0..11}` indices.

    Sequential layout:
        [Conv0, Act1, BN2, Conv3, Act4, BN5, Conv6, Act7, BN8, Conv9, Act10, BN11]
    """

    def __init__(self, input_dim: int, hidden: int, latent_dim: int,
                 strides: tuple[int, int, int, int] = (1, 2, 1, 2),
                 kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        c_in_out = [
            (input_dim, hidden),
            (hidden, hidden),
            (hidden, hidden),
            (hidden, latent_dim),
        ]
        layers: list[nn.Module] = []
        for (cin, cout), s in zip(c_in_out, strides):
            layers += [
                nn.Conv1d(cin, cout, kernel_size=kernel, stride=s, padding=pad),
                _act(),
                nn.BatchNorm1d(cout),
            ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim, T_motion) -> (B, latent_dim, T_tok)
        return self.encoder(x)


class _Decoder(nn.Module):
    """Mirror of the encoder using ConvTranspose1d. Indices match
    `decoder.decoder.{0..11}`.
    """

    def __init__(self, latent_dim: int, hidden: int, output_dim: int,
                 strides: tuple[int, int, int, int] = (1, 2, 1, 2),
                 kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        c_in_out = [
            (latent_dim, hidden),
            (hidden, hidden),
            (hidden, hidden),
            (hidden, output_dim),
        ]
        # Reverse strides so upsampling happens at mirrored positions.
        strides_rev = tuple(reversed(strides))
        layers: list[nn.Module] = []
        for (cin, cout), s in zip(c_in_out, strides_rev):
            layers += [
                nn.ConvTranspose1d(
                    cin, cout, kernel_size=kernel, stride=s,
                    padding=pad, output_padding=(s - 1),
                ),
                _act(),
                nn.BatchNorm1d(cout),
            ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim, T_tok) -> (B, output_dim, T_motion)
        return self.decoder(z)


# ─── quantizer ───────────────────────────────────────────────────────────────
class _ResidualVQLayer(nn.Module):
    """Single EMA VQ codebook. Stores `embedding` of shape (latent_dim, codebook_size)
    plus `cluster_size` and `embedding_avg` (loaded but unused at inference).
    """

    def __init__(self, latent_dim: int, codebook_size: int):
        super().__init__()
        self.register_buffer("embedding", torch.zeros(latent_dim, codebook_size))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_avg", torch.zeros(latent_dim, codebook_size))

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) long -> embeddings (B, D, T)."""
        # embedding has shape (D, K). Index_select on dim=1 by flattened idx, then reshape.
        if idx.dim() != 2:
            raise ValueError(f"Unsupported idx shape: {tuple(idx.shape)}")
        B, T = idx.shape
        flat = idx.reshape(-1)                                 # (B*T,)
        sel = self.embedding.index_select(1, flat)             # (D, B*T)
        D = self.embedding.shape[0]
        out = sel.view(D, B, T).permute(1, 0, 2).contiguous()  # (B, D, T)
        return out

    @torch.no_grad()
    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, D, T) -> (z_q (B, D, T), idx (B, T))."""
        B, D, T = z.shape
        flat = z.permute(0, 2, 1).reshape(-1, D)               # (BT, D)
        emb = self.embedding.t()                               # (K, D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ emb.t()
             + emb.pow(2).sum(1)[None, :])
        idx = d.argmin(dim=1).view(B, T)                       # (B, T)
        z_q = self.lookup(idx)                                 # (B, D, T)
        return z_q, idx


class _RVQ(nn.Module):
    def __init__(self, num_quantizers: int, latent_dim: int, codebook_size: int):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [_ResidualVQLayer(latent_dim, codebook_size) for _ in range(num_quantizers)]
        )

    def __len__(self) -> int:
        return len(self.quantizers)

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D, T) -> tokens (B, num_quantizers, T)."""
        residual = z
        out: list[torch.Tensor] = []
        for q in self.quantizers:
            z_q, idx = q.quantize(residual)
            out.append(idx)
            residual = residual - z_q
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, T) long -> z_hat (B, D, T)."""
        B, L, T = tokens.shape
        if L != len(self.quantizers):
            raise ValueError(f"tokens has {L} layers but RVQ has {len(self.quantizers)}")
        z = None
        for i, q in enumerate(self.quantizers):
            z_q = q.lookup(tokens[:, i, :])                    # (B, D, T)
            z = z_q if z is None else z + z_q
        assert z is not None
        return z


# ─── full VAE wrapper ────────────────────────────────────────────────────────
@dataclass
class RVQVAEConfig:
    input_dim: int = 668
    output_dim: int = 668
    latent_dim: int = 256
    hidden_dim: int = 512
    num_quantizers: int = 6
    num_embeddings: int = 512
    downsampling_ratio: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "RVQVAEConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FrozenRVQVAE(nn.Module):
    """Frozen RVQ-VAE matching `rvq_vae_best.pth`.

    Usage:
        model = FrozenRVQVAE().load(device='cuda').eval()
        with torch.no_grad():
            motion = model.decode(tokens)        # (B, T_motion, 668)
            tokens = model.encode(motion)        # (B, 6, T_tok)
    """

    def __init__(self, ckpt_path: str | Path = RVQ_VAE_CKPT,
                 cfg: RVQVAEConfig | None = None):
        super().__init__()
        self.ckpt_path = Path(ckpt_path)
        self.cfg = cfg or RVQVAEConfig()
        self._loaded = False
        self._build()

    def _build(self) -> None:
        self.encoder = _Encoder(
            input_dim=self.cfg.input_dim,
            hidden=self.cfg.hidden_dim,
            latent_dim=self.cfg.latent_dim,
        )
        self.rvq = _RVQ(
            num_quantizers=self.cfg.num_quantizers,
            latent_dim=self.cfg.latent_dim,
            codebook_size=self.cfg.num_embeddings,
        )
        self.decoder = _Decoder(
            latent_dim=self.cfg.latent_dim,
            hidden=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
        )

    # convenience ------------------------------------------------------------
    @property
    def codebook_size(self) -> int: return self.cfg.num_embeddings
    @property
    def num_layers(self) -> int: return self.cfg.num_quantizers
    @property
    def feature_dim(self) -> int: return self.cfg.input_dim

    # lifecycle --------------------------------------------------------------
    def load(self, device: str | torch.device = "cpu", strict: bool = True) -> "FrozenRVQVAE":
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"RVQ-VAE checkpoint not found: {self.ckpt_path}. "
                "Download `antonygithinji/motion-s-vae-rvq` and untar to data/rvq_vae/."
            )
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if isinstance(ckpt, dict) and "config" in ckpt:
            file_cfg = RVQVAEConfig.from_dict(ckpt["config"])
            if file_cfg != self.cfg:
                self.cfg = file_cfg
                self._build()
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print(f"[FrozenRVQVAE] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"  first missing: {missing[:5]}")
            if unexpected:
                print(f"  first unexpected: {unexpected[:5]}")
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad = False
        self._loaded = True
        return self

    # core ops ---------------------------------------------------------------
    @torch.no_grad()
    def encode_to_latents(self, motion: torch.Tensor) -> torch.Tensor:
        """motion: (B, T_motion, D_in) -> z (B, D_lat, T_tok)."""
        x = motion.transpose(1, 2)
        return self.encoder(x)

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        """motion: (B, T_motion, D_in) -> tokens (B, num_quantizers, T_tok)."""
        z = self.encode_to_latents(motion)
        return self.rvq.encode(z)

    @torch.no_grad()
    def latents_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, T_tok) -> z_hat (B, D_lat, T_tok)."""
        return self.rvq.decode(tokens)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, T_tok) long -> motion (B, T_motion, D_out)."""
        z = self.latents_from_tokens(tokens)
        x = self.decoder(z)
        return x.transpose(1, 2).contiguous()

    # ergonomics for variable-length lists -----------------------------------
    @torch.no_grad()
    def decode_list(self, tokens_per_row: Iterable[torch.Tensor], device: str | torch.device = "cpu",
                    batch_size: int = 16) -> list[torch.Tensor]:
        """Decode rows that may have different T_tok. Returns list of (T_motion, D) tensors.
        Pads to max length per batch then trims. Faster than 1-by-1.
        """
        rows = list(tokens_per_row)
        out: list[torch.Tensor] = []
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            T_max = max(t.shape[-1] for t in chunk)
            padded = torch.zeros(len(chunk), self.num_layers, T_max, dtype=torch.long, device=device)
            for i, t in enumerate(chunk):
                T = t.shape[-1]
                padded[i, :, :T] = t.to(device)
            motion = self.decode(padded)                      # (B, T_motion_max, D)
            for i, t in enumerate(chunk):
                T = t.shape[-1]
                T_motion = T * self.cfg.downsampling_ratio
                out.append(motion[i, :T_motion].cpu())
        return out


# ─── checkpoint inspector (kept for backwards compat) ────────────────────────
def peek_checkpoint(ckpt_path: str | Path = RVQ_VAE_CKPT) -> dict:
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
