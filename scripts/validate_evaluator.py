"""Validate the public evaluator wrapper against real (text, tokens) pairs
from the training CSV. Confirms the projector activations and CLIP variant
are correct.

Methodology:
  1. Sample N rows from train.csv.
  2. Compute text_embed(sentence) and tokens_embed(tokens).
  3. Build the (N, N) cosine matrix.
  4. Report:
       diag mean       = mean of matched (text_i, motion_i) cosines
       off-diag mean   = mean of mismatched cosines
       gap             = diag - off-diag
       R@1             = fraction of rows where the diagonal is the argmax
       R@5             = fraction of rows where the diagonal is in top-5

If the wrapper is correct, R@1 should be substantially above 1/N and the
gap should be clearly positive (e.g. > 0.2).

Usage:
    python -m scripts.validate_evaluator
    python -m scripts.validate_evaluator --n 64 --device cuda
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.constants import EVALUATOR_PUBLIC_CKPT, TOKEN_COLUMNS
from src.data.io import load_train
from src.eval.evaluator import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(EVALUATOR_PUBLIC_CKPT))
    p.add_argument("--clip", type=str, default="ViT-B/32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=32, help="number of samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proj-act", type=str, default="relu")
    p.add_argument("--proj-arch", type=str, default="v1",
                   choices=["v1", "v2", "v3", "v4"])
    p.add_argument("--sweep", action="store_true",
                   help="try all (proj_arch x proj_act) combinations")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("[val] loading train.csv ...")
    df = load_train()
    idx = rng.choice(len(df), size=args.n, replace=False)
    rows = df.iloc[idx].reset_index(drop=True)
    sentences = rows["sentence"].astype(str).tolist()
    tokens_list = []
    for _, r in rows.iterrows():
        layers = [r[c] for c in TOKEN_COLUMNS]
        T = len(layers[0])
        if T == 0:
            tokens_list.append(torch.zeros(len(layers), 1, dtype=torch.long))
            continue
        tk = torch.tensor(np.stack(layers, axis=0), dtype=torch.long)
        tokens_list.append(tk)

    if args.sweep:
        archs = ["v1", "v2", "v3", "v4"]
        acts = ["relu", "gelu", "silu", "elu", "leaky_relu", "identity"]
        results = []
        for arch in archs:
            for act in acts:
                ev = Evaluator(args.ckpt, clip_name=args.clip, device=args.device,
                               proj_act=act, proj_arch=arch)
                te = ev.text_embed(sentences)
                me = ev.tokens_embed(tokens_list)
                sim = ev.cosine(te, me).cpu().numpy()
                diag = np.diag(sim)
                off = (sim.sum() - diag.sum()) / (sim.size - sim.shape[0])
                ranks = (-sim).argsort(axis=1)
                pos = (ranks == np.arange(args.n)[:, None]).argmax(axis=1)
                r1 = (pos == 0).mean(); r5 = (pos < 5).mean()
                med = float(np.median(pos)) + 1
                gap = diag.mean() - off
                results.append((r1, r5, gap, med, arch, act))
                print(f"  arch={arch} act={act:<10} R@1={r1:.3f} R@5={r5:.3f} gap={gap:+.3f} median={med:.1f}")
                del ev
        print()
        results.sort(key=lambda x: (-x[0], -x[1]))
        print("[sweep] top 10 by R@1:")
        for r1, r5, gap, med, arch, act in results[:10]:
            print(f"  R@1={r1:.3f}  R@5={r5:.3f}  gap={gap:+.3f}  median={med:.1f}  arch={arch}  act={act}")
        return

    print(f"[val] loading evaluator (arch={args.proj_arch} act={args.proj_act}) ...")
    ev = Evaluator(args.ckpt, clip_name=args.clip, device=args.device,
                   proj_act=args.proj_act, proj_arch=args.proj_arch)

    print(f"[val] embedding {args.n} text-motion pairs ...")
    te = ev.text_embed(sentences)              # (N, 256)
    me = ev.tokens_embed(tokens_list)          # (N, 256)
    sim = ev.cosine(te, me).cpu().numpy()      # (N, N)

    diag = np.diag(sim)
    off = sim - np.diag(diag)
    off_mean = off.sum() / (sim.size - sim.shape[0])
    diag_mean = diag.mean()

    # R@k along rows: text_i ranked over all motions
    ranks = (-sim).argsort(axis=1)
    correct_pos = (ranks == np.arange(args.n)[:, None]).argmax(axis=1)
    r_at_1 = (correct_pos == 0).mean()
    r_at_5 = (correct_pos < 5).mean()
    r_at_10 = (correct_pos < 10).mean()
    median_rank = float(np.median(correct_pos)) + 1

    print()
    print(f"[val] N = {args.n}")
    print(f"[val] diag (matched) mean cosine     = {diag_mean:+.4f}")
    print(f"[val] off-diag (mismatched) mean     = {off_mean:+.4f}")
    print(f"[val] gap (diag - off)               = {diag_mean - off_mean:+.4f}")
    print(f"[val] R@1                            = {r_at_1:.3f}   (chance = {1/args.n:.3f})")
    print(f"[val] R@5                            = {r_at_5:.3f}   (chance = {5/args.n:.3f})")
    print(f"[val] R@10                           = {r_at_10:.3f}  (chance = {10/args.n:.3f})")
    print(f"[val] median rank of true match      = {median_rank:.1f} / {args.n}")

    if r_at_1 < 2.0 / args.n:
        print("\n[!] R@1 is at chance level — projector activation or pooling is likely wrong.")
    elif diag_mean - off_mean < 0.05:
        print("\n[?] Gap is small — wrapper may be partially miswired (still functional but suboptimal).")
    else:
        print("\n[OK] Evaluator wrapper looks correct.")


if __name__ == "__main__":
    main()
