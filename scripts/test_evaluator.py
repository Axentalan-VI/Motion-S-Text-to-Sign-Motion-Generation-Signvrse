"""Diagnostic loader for the public evaluator checkpoint.

Builds our `_PublicEvaluator` and reports missing/unexpected keys when loading
the state-dict. Use this to verify our module structure matches the checkpoint.

Usage:
    python -m scripts.test_evaluator
    python -m scripts.test_evaluator --ckpt path/to/Public_t2m_align.pth
    python -m scripts.test_evaluator --clip ViT-B/16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.constants import EVALUATOR_PUBLIC_CKPT
from src.rvq import RVQVAEConfig
from src.eval.evaluator import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(EVALUATOR_PUBLIC_CKPT))
    p.add_argument("--clip", type=str, default="ViT-B/32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="run text_embed and motion_embed on toy inputs")
    args = p.parse_args()

    print(f"[eval] loading {args.ckpt}")
    print(f"[eval] CLIP variant: {args.clip}")
    ev = Evaluator(args.ckpt, clip_name=args.clip, device=args.device)
    rep = ev.report()
    print(f"[eval] missing keys:    {rep['missing_count']}")
    print(f"[eval] unexpected keys: {rep['unexpected_count']}")
    if rep["missing_count"]:
        print("  -- first 20 missing --")
        for k in rep["missing"][:20]:
            print(f"    MISSING    {k}")
    if rep["unexpected_count"]:
        print("  -- first 20 unexpected --")
        for k in rep["unexpected"][:20]:
            print(f"    UNEXPECTED {k}")

    # Filter out CLIP visual.* — we don't need it for text embedding and the
    # checkpoint may strip it. Also report the non-CLIP delta separately.
    non_clip_missing = [k for k in rep["missing"] if not k.startswith("clip_encoder.")]
    non_clip_unexpected = [k for k in rep["unexpected"] if not k.startswith("clip_encoder.")]
    print(f"[eval] non-CLIP missing:    {len(non_clip_missing)}")
    print(f"[eval] non-CLIP unexpected: {len(non_clip_unexpected)}")
    for k in non_clip_missing:
        print(f"    NONCLIP MISSING    {k}")
    for k in non_clip_unexpected:
        print(f"    NONCLIP UNEXPECTED {k}")

    if args.smoke:
        print("[smoke] text_embed of 3 phrases ...")
        te = ev.text_embed(["a person waves hello", "the man walks forward", "she signs thank you"])
        print(f"[smoke] text shape: {tuple(te.shape)}, norm: {te.norm(dim=-1).tolist()}")
        print("[smoke] motion_embed of random motion (B=2, T=80, D=668) ...")
        m = torch.randn(2, 80, 668)
        me = ev.motion_embed(m)
        print(f"[smoke] motion shape: {tuple(me.shape)}, norm: {me.norm(dim=-1).tolist()}")
        sim = ev.cosine(te[:2], me)
        print(f"[smoke] cosine sims: {sim.tolist()}")


if __name__ == "__main__":
    main()
