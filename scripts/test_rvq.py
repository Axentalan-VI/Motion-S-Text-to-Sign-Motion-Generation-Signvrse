"""Smoke-test FrozenRVQVAE against the real checkpoint, with optional sweep
over (activation x stride) pairs to find the combination that round-trips
tokens correctly.

Usage:
    python -m scripts.test_rvq                            # default config
    python -m scripts.test_rvq --activation gelu
    python -m scripts.test_rvq --strides 2,1,2,1
    python -m scripts.test_rvq --sweep                    # try everything
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch

from src.constants import RVQ_VAE_CKPT, TRAIN_CSV
from src.data.io import load_train, stack_layers
from src.rvq import FrozenRVQVAE


_STRIDE_PATTERNS = [
    (1, 2, 1, 2),
    (2, 1, 2, 1),
    (1, 1, 2, 2),
    (2, 2, 1, 1),
    (1, 2, 2, 1),
    (2, 1, 1, 2),
]
_ACTIVATIONS = ["leaky_relu", "relu", "gelu", "silu", "elu", "selu", "mish",
                "hardswish", "hardtanh", "softplus", "tanh", "identity"]
_CODEBOOK_SOURCES = ["embedding", "embedding_avg"]


def _eval_config(activation: str, strides: tuple[int, int, int, int],
                 rows: list, device: str, ckpt: str,
                 codebook_source: str = "embedding") -> tuple[bool, float, list[float]]:
    """Build model, load, decode+re-encode rows. Returns (shapes_ok, mean_agreement, per_row)."""
    try:
        model = FrozenRVQVAE(ckpt_path=ckpt, activation=activation,
                             strides=strides, codebook_source=codebook_source)
        model.load(device=device, strict=True)
    except Exception as e:
        print(f"  [load FAILED] {type(e).__name__}: {e}")
        return False, 0.0, []

    all_ok = True
    matches: list[float] = []
    with torch.no_grad():
        for tokens in rows:
            T_tok = tokens.shape[-1]
            T_motion_expected = T_tok * model.cfg.downsampling_ratio
            try:
                motion = model.decode(tokens)
            except Exception as e:
                print(f"  [decode FAILED] {type(e).__name__}: {e}")
                return False, 0.0, []
            if motion.shape[1] != T_motion_expected:
                all_ok = False
                continue
            re_tokens = model.encode(motion)
            T_re = re_tokens.shape[-1]
            T_min = min(T_tok, T_re)
            agree = float((tokens[0, 0, :T_min] == re_tokens[0, 0, :T_min]).float().mean())
            matches.append(agree)
    avg = sum(matches) / len(matches) if matches else 0.0
    return all_ok, avg, matches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strides", type=str, default="1,1,2,2")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--codebook", type=str, default="embedding",
                   choices=["embedding", "embedding_avg"])
    p.add_argument("--ckpt", type=str, default=str(RVQ_VAE_CKPT))
    p.add_argument("--sweep", action="store_true",
                   help="try all (activation x stride x codebook) combinations")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[data] loading {TRAIN_CSV}")
    df = load_train()
    df = df[(df["seq_len"] >= 40) & (df["seq_len"] <= 800)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(df), size=args.rows, replace=False)
    rows = []
    for idx in pick:
        layers = stack_layers(df.iloc[idx])
        rows.append(torch.from_numpy(layers).long().unsqueeze(0).to(args.device))
    print(f"[data] picked {list(pick)}  seq_lens: {df.iloc[pick]['seq_len'].tolist()}")

    if args.sweep:
        print("[sweep] trying all (codebook x activation x stride) combinations ...")
        results: list[tuple[float, str, str, tuple, bool]] = []
        for cb, act, strides in itertools.product(_CODEBOOK_SOURCES, _ACTIVATIONS, _STRIDE_PATTERNS):
            t0 = time.time()
            ok, avg, _ = _eval_config(act, strides, rows, args.device, args.ckpt, codebook_source=cb)
            dt = time.time() - t0
            tag = "OK  " if ok else "BAD "
            print(f"  {tag} cb={cb:<14} act={act:<10} strides={strides}  agreement={avg:.3f}  ({dt:.1f}s)")
            results.append((avg, cb, act, strides, ok))
        results.sort(reverse=True)
        print("\n[sweep] top 10:")
        for avg, cb, act, strides, ok in results[:10]:
            print(f"  agreement={avg:.3f}  cb={cb:<14} act={act:<10} strides={strides}  shapes_ok={ok}")
        best = results[0]
        if best[0] >= 0.95:
            print(f"\n[WIN] cb='{best[1]}' act='{best[2]}' strides={best[3]} -> agreement={best[0]:.3f}")
        else:
            print(f"\n[no clear winner] best agreement={best[0]:.3f}; "
                  f"VQ-VAE round-trip may be inherently lossy. Use top config and "
                  f"validate via the public evaluator instead.")
        return

    strides = tuple(int(x) for x in args.strides.split(","))
    assert len(strides) == 4 and int(np.prod(strides)) == 4

    print(f"[load] {args.ckpt}  cb={args.codebook}  activation={args.activation}  strides={strides}")
    t0 = time.time()
    ok, avg, matches = _eval_config(args.activation, strides, rows, args.device,
                                    args.ckpt, codebook_source=args.codebook)
    print(f"[done] {time.time()-t0:.2f}s  shapes_ok={ok}  layer0_agreement={avg:.3f}  per_row={matches}")
    if avg >= 0.95:
        print("  -> architecture is correct.")
    elif avg >= 0.6:
        print("  -> close; try a different activation or stride pattern.")
    else:
        print("  -> wrong combination; run with --sweep to find the right one.")


if __name__ == "__main__":
    main()
