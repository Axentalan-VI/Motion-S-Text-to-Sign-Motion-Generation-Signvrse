"""Peek inside a checkpoint and print every tensor's shape.

Used to reverse-engineer model architectures from saved state_dicts.

    python -m scripts.peek_ckpt path/to/ckpt.pth
    python -m scripts.peek_ckpt --all                # peeks every known ckpt

Why not unpickle and instantiate? Because we don't have the trainer's
class definitions; we have to reconstruct them from the layer shapes alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.constants import (
    EVALUATOR_PUBLIC_CKPT,
    EVALUATOR_INTERNAL_CKPT,
    LENGTH_ESTIMATOR_CKPT,
    RVQ_VAE_CKPT,
)


def _walk(obj, prefix: str = "") -> None:
    """Recursively print shapes of all tensors inside `obj`."""
    if hasattr(obj, "shape"):
        print(f"  {prefix:<60} {tuple(obj.shape)}  {obj.dtype}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{prefix}.{k}" if prefix else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{prefix}[{i}]")
        return
    print(f"  {prefix:<60} <{type(obj).__name__}> {repr(obj)[:80]}")


def peek(path: Path) -> None:
    print(f"\n========== {path} ==========")
    if not path.exists():
        print("  (missing)")
        return
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  top type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"  top-level keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"\n  -- {k} ({type(v).__name__}) --")
            if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in v.values()):
                # state_dict: print one line per tensor
                for kk, vv in v.items():
                    print(f"    {kk:<60} {tuple(vv.shape)}  {vv.dtype}")
            else:
                _walk(v, prefix="  " + k)
    else:
        _walk(obj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", help="path to a .pth file")
    p.add_argument("--all", action="store_true",
                   help="peek every known checkpoint listed in src/constants.py")
    args = p.parse_args()

    if args.all:
        for ck in [RVQ_VAE_CKPT, LENGTH_ESTIMATOR_CKPT,
                   EVALUATOR_PUBLIC_CKPT, EVALUATOR_INTERNAL_CKPT]:
            peek(Path(ck))
        return

    if not args.path:
        p.error("provide a path or --all")
    peek(Path(args.path))


if __name__ == "__main__":
    main()
