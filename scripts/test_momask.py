"""Quick smoke test for MoMask: forward + tiny backward + .generate() shape check.

Doesn't require CLIP weights when --no-clip is set (uses random text emb).

Usage:
    python -m scripts.test_momask --no-clip
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.constants import NUM_LAYERS
from src.models.momask import (
    BaseMaskTransformer, MoMaskConfig, ResidualTransformer, random_mask, MASK_ID,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-clip", action="store_true",
                   help="Use random text embeddings instead of CLIP.")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    B, T = 4, 64
    cfg = MoMaskConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256,
                       max_len=T, num_length_bins=16)

    base_model = BaseMaskTransformer(cfg).to(device)
    res_model = ResidualTransformer(cfg).to(device)

    # Fake batch
    tokens = torch.randint(0, 512, (B, NUM_LAYERS, T), device=device)
    lengths = torch.tensor([T, T - 4, T - 16, T // 2], device=device)
    length_bin = torch.tensor([3, 5, 8, 10], device=device)

    if args.no_clip:
        text_emb = torch.randn(B, 512, device=device)
        uncond = torch.zeros(B, 512, device=device)
    else:
        from src.models.text_cond import FrozenTextEncoder
        enc = FrozenTextEncoder(device=str(device))
        text_emb = enc.encode_texts(["hello world"] * B, device=str(device))
        uncond = torch.zeros_like(text_emb)

    # ── base: forward + masked CE ────────────────────────────────────────
    base = tokens[:, 0]
    noisy, mask = random_mask(base, lengths)
    logits = base_model(noisy, text_emb, length_bin)
    print(f"[base] logits shape = {tuple(logits.shape)}  expected ({B},{T},512)")
    target = base[mask]; pred = logits[mask]
    loss = F.cross_entropy(pred, target)
    loss.backward()
    print(f"[base] loss = {loss.item():.3f}  (random init ≈ ln(512) = 6.24)")

    # ── base: generate ───────────────────────────────────────────────────
    with torch.no_grad():
        out = base_model.generate(text_emb, length_bin, lengths, n_steps=4,
                                   cfg_scale=2.0, uncond_text_emb=uncond)
    print(f"[base] gen shape = {tuple(out.shape)}; "
          f"any MASK left? {(out == MASK_ID).any().item()}")
    for b in range(B):
        L = int(lengths[b].item())
        assert (out[b, :L] < 512).all(), "real-token region has non-codebook ids"
        assert (out[b, L:] >= 512).all() or L == T, "padded region not flagged"
    print("[base] sanity OK")

    # ── residual: forward + CE ───────────────────────────────────────────
    target_layer = torch.full((B,), 3, dtype=torch.long, device=device)
    prev = tokens[:, :3]
    logits = res_model(prev, target_layer, text_emb, length_bin)
    print(f"[res ] logits shape = {tuple(logits.shape)}")
    pos = torch.arange(T, device=device)[None].expand(B, -1)
    valid = pos < lengths[:, None]
    tgt = tokens[:, 3][valid]; pr = logits[valid]
    loss = F.cross_entropy(pr, tgt)
    loss.backward()
    print(f"[res ] loss = {loss.item():.3f}")

    with torch.no_grad():
        out = res_model.generate(tokens[:, 0], text_emb, length_bin, lengths,
                                 cfg_scale=1.0)
    print(f"[res ] gen shape = {tuple(out.shape)}  expected ({B},{NUM_LAYERS},{T})")
    print("[OK] all smoke checks pass")


if __name__ == "__main__":
    main()
