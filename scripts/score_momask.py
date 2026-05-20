"""End-to-end MoMask inference + local scoring on the val split.

Pipeline:
  1. Predict per-sample length using the trained length predictor
  2. Generate base tokens via BaseMaskTransformer.generate (CFG, n_steps)
  3. Generate residual layers via ResidualTransformer.generate
  4. Stack into (NUM_LAYERS, T) per sample
  5. Score against GT with R-Precision/FID/Diversity using the public evaluator

Usage:
    python -m scripts.score_momask --n 256 --batch-size 16
    python -m scripts.score_momask --cfg-scale 4 --n-steps 10 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.constants import (
    CHECKPOINT_DIR, EVALUATOR_PUBLIC_CKPT, NUM_LAYERS,
    SPLIT_FILE, TRAIN_CSV,
)
from src.data.io import load_train
from src.eval.evaluator import Evaluator
from src.eval.local_score import score_predictions
from src.length import LengthPredictor, bin_to_seq_len, load_head, seq_len_to_bin
from src.models.momask import (
    BaseMaskTransformer, MoMaskConfig, ResidualTransformer,
)
from src.models.text_cond import FrozenTextEncoder


def _load_split() -> tuple[set[int], set[int]]:
    obj = json.loads(SPLIT_FILE.read_text())
    return set(map(int, obj["train"])), set(map(int, obj["val"]))


def _load_momask(ckpt_path: Path, model_cls, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = MoMaskConfig.from_dict(ckpt["config"])
    model = model_cls(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256, help="number of val samples")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=10, help="base decoder steps")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--res-cfg-scale", type=float, default=1.0,
                   help="CFG for residual decoder (1.0 = off)")
    p.add_argument("--use-gt-length", action="store_true",
                   help="Use ground-truth length instead of predicted (oracle test).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-ckpt", type=Path, default=Path(CHECKPOINT_DIR) / "momask_base.pth")
    p.add_argument("--res-ckpt", type=Path, default=Path(CHECKPOINT_DIR) / "momask_residual.pth")
    p.add_argument("--length-ckpt", type=Path,
                   default=Path(CHECKPOINT_DIR) / "length_predictor.pth")
    p.add_argument("--clip-name", type=str, default="ViT-B/32")
    p.add_argument("--evaluator-ckpt", type=Path, default=None,
                   help="Path to evaluator checkpoint. Auto-detected if omitted.")
    args = p.parse_args()

    # Auto-detect evaluator checkpoint — try both known filenames.
    if args.evaluator_ckpt is None:
        from src.constants import EVALUATOR_INTERNAL_CKPT
        for candidate in (EVALUATOR_PUBLIC_CKPT, EVALUATOR_INTERNAL_CKPT):
            if candidate.exists():
                args.evaluator_ckpt = candidate
                break
        if args.evaluator_ckpt is None:
            raise FileNotFoundError(
                f"Evaluator checkpoint not found. Expected one of:\n"
                f"  {EVALUATOR_PUBLIC_CKPT}\n"
                f"  {EVALUATOR_INTERNAL_CKPT}\n"
                "Run the evaluator download cell (fetch from Kaggle models) first."
            )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] val split + {TRAIN_CSV}")
    df = load_train()
    _, val_idx = _load_split()
    df = df[df["id"].astype(int).isin(val_idx)].reset_index(drop=True)
    if args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"[data] using {len(df)} val samples")

    print(f"[load] base ckpt {args.base_ckpt}")
    base_model, base_cfg = _load_momask(args.base_ckpt, BaseMaskTransformer, device)
    print(f"[load] residual ckpt {args.res_ckpt}")
    res_model, _ = _load_momask(args.res_ckpt, ResidualTransformer, device)

    print(f"[load] text encoder ({args.clip_name})")
    text_enc = FrozenTextEncoder(name=args.clip_name, device=str(device))

    # ── Length predictor ────────────────────────────────────────────────
    if args.use_gt_length:
        print("[length] using ground-truth lengths")
        len_model = None
    else:
        print(f"[load] length ckpt {args.length_ckpt}")
        from transformers import AutoTokenizer
        # backbone name persisted in length_predictor.pth args; default safe bet:
        backbone = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(backbone)
        len_model = LengthPredictor(num_bins=base_cfg.num_length_bins,
                                     backbone_name=backbone,
                                     freeze_backbone=True)
        load_head(len_model, str(args.length_ckpt))
        len_model = len_model.to(device)
        len_model.eval()
        len_max_text_tokens = 64

    # ── Predict lengths for all rows ────────────────────────────────────
    sentences = df["sentence"].fillna("").astype(str).tolist()
    gt_lengths = df["base_tokens"].map(len).astype(int).to_numpy()

    if args.use_gt_length:
        pred_lengths = gt_lengths.copy()
    else:
        all_lens = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                chunk = sentences[i : i + 64]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=len_max_text_tokens, return_tensors="pt")
                ids = enc["input_ids"].to(device)
                am = enc["attention_mask"].to(device)
                logits = len_model(ids, am)
                pred_bin = logits.argmax(-1).cpu().numpy()
                all_lens.append(bin_to_seq_len(pred_bin, base_cfg.num_length_bins))
        pred_lengths = np.concatenate(all_lens).astype(int)

    # Cap to model's max_len.
    pred_lengths = np.minimum(pred_lengths, base_cfg.max_len)
    pred_bins = seq_len_to_bin(pred_lengths, base_cfg.num_length_bins).astype(np.int64)
    print(f"[length] pred MAE vs gt = {np.abs(pred_lengths - np.minimum(gt_lengths, base_cfg.max_len)).mean():.1f} frames")

    # ── Generate ────────────────────────────────────────────────────────
    print(f"[gen ] base CFG={args.cfg_scale} steps={args.n_steps} T={args.temperature}; res CFG={args.res_cfg_scale}")
    gen_tokens_per_sample: list[torch.Tensor] = []
    BS = args.batch_size
    n = len(sentences)
    with torch.no_grad():
        for s in range(0, n, BS):
            e = min(n, s + BS)
            chunk_text = sentences[s:e]
            chunk_lens = torch.tensor(pred_lengths[s:e], dtype=torch.long, device=device)
            chunk_bins = torch.tensor(pred_bins[s:e], dtype=torch.long, device=device)

            t_emb = text_enc.encode_texts(chunk_text, device=str(device))
            uncond = torch.zeros_like(t_emb)

            base_tokens = base_model.generate(
                t_emb, chunk_bins, chunk_lens,
                n_steps=args.n_steps,
                cfg_scale=args.cfg_scale,
                uncond_text_emb=uncond,
                temperature=args.temperature,
            )    # (B, T_max)
            all_layers = res_model.generate(
                base_tokens, t_emb, chunk_bins, chunk_lens,
                cfg_scale=args.res_cfg_scale,
                uncond_text_emb=uncond,
                temperature=args.temperature,
                sample=False,
            )   # (B, NUM_LAYERS, T_max)

            # Trim each row to its predicted length.
            for bi in range(all_layers.shape[0]):
                L = int(chunk_lens[bi].item())
                gen_tokens_per_sample.append(all_layers[bi, :, :L].cpu().long())

            if (s // BS) % 5 == 0:
                print(f"  [gen] {e}/{n}")

    # ── Build GT token tensors (same format) ────────────────────────────
    gt_tokens_per_sample: list[torch.Tensor] = []
    from src.constants import TOKEN_COLUMNS
    for i in range(n):
        arrs = [df.iloc[i][c] for c in TOKEN_COLUMNS]
        T = len(arrs[0])
        gt_tokens_per_sample.append(
            torch.from_numpy(np.stack(arrs, axis=0)).long()    # (NUM_LAYERS, T)
        )

    # ── Score ───────────────────────────────────────────────────────────
    print(f"[load] evaluator {args.evaluator_ckpt}")
    evaluator = Evaluator(args.evaluator_ckpt, device=str(device))

    print("[score] computing R-Precision / FID / Diversity ...")
    report = score_predictions(
        evaluator,
        sentences=sentences,
        gen_tokens=gen_tokens_per_sample,
        gt_tokens=gt_tokens_per_sample,
    )
    print(report.pretty())

    # Also score GT vs GT for sanity (R-Precision should be high).
    text_emb = evaluator.text_embed(sentences)
    gt_emb = evaluator.tokens_embed(gt_tokens_per_sample)
    from src.eval.local_score import r_precision_topk
    rp_gt = r_precision_topk(text_emb, gt_emb)
    print(f"[ref ] GT-only R-Precision (upper bound): "
          + "  ".join(f"{k}={v:.3f}" for k, v in rp_gt.items()))


if __name__ == "__main__":
    main()
