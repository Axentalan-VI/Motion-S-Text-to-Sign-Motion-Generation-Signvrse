"""Smoke-test the FrozenRVQVAE wrapper against the real checkpoint.

What it does:
  1. Load rvq_vae_best.pth (strict).
  2. Pick a random training row, decode its 6-layer tokens to motion features.
  3. Verify decoded shape == (T_motion, 668) where T_motion = T_tok * downsampling.
  4. Re-encode the decoded motion -> tokens; compare to the original.
     For a frozen RVQ this is *not* identity in general, but layer-0 agreement
     should be ≥ ~0.95 if the encoder/decoder strides are correct.
  5. Try alternate stride orders if (3) fails.

Usage:
    python -m scripts.test_rvq
    python -m scripts.test_rvq --rows 5 --strides 2,1,2,1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.constants import RVQ_VAE_CKPT, TRAIN_CSV, TOKEN_COLUMNS
from src.data.io import load_train, stack_layers
from src.rvq import FrozenRVQVAE, RVQVAEConfig, _Decoder, _Encoder


def _try_strides(model: FrozenRVQVAE, tokens: torch.Tensor, expected_motion_T: int) -> tuple[bool, str]:
    """Run a forward decode and check shape. Returns (ok, message)."""
    try:
        motion = model.decode(tokens)
        T_motion = motion.shape[1]
        ok = T_motion == expected_motion_T
        return ok, f"decode -> {tuple(motion.shape)}  (expected T_motion={expected_motion_T})"
    except Exception as e:
        return False, f"decode FAILED: {type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3, help="how many train rows to decode")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strides", type=str, default="1,2,1,2",
                   help="encoder stride pattern, e.g. '1,2,1,2' or '2,1,2,1'")
    p.add_argument("--ckpt", type=str, default=str(RVQ_VAE_CKPT))
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    strides = tuple(int(x) for x in args.strides.split(","))
    assert len(strides) == 4, "strides must be 4 ints"
    assert int(np.prod(strides)) == 4, f"product of strides must equal 4, got {int(np.prod(strides))}"

    print(f"[load] {args.ckpt}")
    model = FrozenRVQVAE(ckpt_path=args.ckpt)
    # Inject custom strides into the encoder/decoder *before* loading weights.
    # We re-instantiate the conv stacks with the requested stride pattern.
    cfg: RVQVAEConfig = model.cfg
    model.encoder = _Encoder(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim, strides=strides)
    model.decoder = _Decoder(cfg.latent_dim, cfg.hidden_dim, cfg.output_dim, strides=strides)

    t0 = time.time()
    model.load(device=args.device, strict=True)
    print(f"[load] ok in {time.time()-t0:.2f}s  device={args.device}  strides={strides}")
    print(f"       feature_dim={model.feature_dim}  num_layers={model.num_layers}  codebook={model.codebook_size}")

    print(f"[data] {TRAIN_CSV}")
    df = load_train()
    df = df[(df["seq_len"] >= 40) & (df["seq_len"] <= 800)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(df), size=args.rows, replace=False)
    print(f"[data] picked rows: {list(pick)}  seq_lens: {df.iloc[pick]['seq_len'].tolist()}")

    all_ok = True
    layer0_match: list[float] = []

    for ri, idx in enumerate(pick):
        row = df.iloc[idx]
        layers = stack_layers(row)                            # (6, T)
        T_tok = layers.shape[1]
        T_motion_expected = T_tok * model.cfg.downsampling_ratio

        tokens = torch.from_numpy(layers).long().unsqueeze(0).to(args.device)  # (1, 6, T)

        # 1) decode -----------------------------------------------------------
        ok, msg = _try_strides(model, tokens, T_motion_expected)
        all_ok = all_ok and ok
        print(f"  [row {ri} id={row.get('id')}] T_tok={T_tok}  {msg}")
        if not ok:
            continue

        with torch.no_grad():
            motion = model.decode(tokens)                    # (1, T_motion, 668)

            # 2) re-encode and compare ---------------------------------------
            re_tokens = model.encode(motion)                 # (1, 6, T_re)
            T_re = re_tokens.shape[-1]
            T_min = min(T_tok, T_re)
            orig0 = tokens[0, 0, :T_min].cpu().numpy()
            re0   = re_tokens[0, 0, :T_min].cpu().numpy()
            agree = float((orig0 == re0).mean()) if T_min > 0 else 0.0
            layer0_match.append(agree)
            print(f"            re-encode shape: {tuple(re_tokens.shape)}  "
                  f"layer-0 token agreement: {agree:.3f}")

    print()
    print("=" * 60)
    if all_ok:
        print(f"[OK] all decodes produced expected shapes (down-ratio={model.cfg.downsampling_ratio}).")
    else:
        print(f"[FAIL] some decodes had wrong shape — try a different --strides pattern.")

    if layer0_match:
        avg = sum(layer0_match) / len(layer0_match)
        print(f"[layer-0 agreement] mean={avg:.3f}   per-row={layer0_match}")
        if avg >= 0.95:
            print("  -> looks like the architecture is correct.")
        elif avg >= 0.6:
            print("  -> partial agreement; activation choice may be off (try ReLU instead of LeakyReLU).")
        else:
            print("  -> low agreement; stride pattern is likely wrong. Try --strides 2,1,2,1 or 2,2,1,1.")


if __name__ == "__main__":
    main()
