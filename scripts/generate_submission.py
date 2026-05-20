"""Generate competition submission CSV from test.csv.

Pipeline:
  1. Load test.csv (id, sentence, gloss — no tokens)
  2. Predict sequence length per row via the trained length predictor
  3. Generate base tokens via BaseMaskTransformer (masked iterative decoding)
  4. Generate residual layers via ResidualTransformer (layer-by-layer)
  5. Write submissions/submission.csv and validate it

Usage:
    python -m scripts.generate_submission
    python -m scripts.generate_submission --cfg-scale 4.0 --n-steps 10 --batch-size 32
    python -m scripts.generate_submission --output submissions/submission_v2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import (
    CHECKPOINT_DIR, NUM_LAYERS, SEQ_LEN_MAX, SEQ_LEN_MIN,
    SUBMISSION_DIR, TEST_CSV, TOKEN_COLUMNS,
)
from src.length import LengthPredictor, bin_to_seq_len, load_head, seq_len_to_bin
from src.models.momask import BaseMaskTransformer, MoMaskConfig, ResidualTransformer
from src.models.text_cond import FrozenTextEncoder


def _load_momask(ckpt_path: Path, model_cls, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = MoMaskConfig.from_dict(ckpt["config"])
    model = model_cls(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--res-cfg-scale", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--clip-name", type=str, default="ViT-B/32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-ckpt", type=Path, default=Path(CHECKPOINT_DIR) / "momask_base.pth")
    p.add_argument("--res-ckpt", type=Path, default=Path(CHECKPOINT_DIR) / "momask_residual.pth")
    p.add_argument("--length-ckpt", type=Path, default=Path(CHECKPOINT_DIR) / "length_predictor.pth")
    p.add_argument("--output", type=Path, default=Path(SUBMISSION_DIR) / "submission.csv")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"[load] {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    print(f"[data] {len(df)} test rows  cols={list(df.columns)}")
    assert "id" in df.columns, "test.csv missing 'id' column"
    for col in ("sentence", "gloss"):
        if col not in df.columns:
            df[col] = ""

    sentences = df["sentence"].fillna("").astype(str).tolist()
    glosses   = df["gloss"].fillna("").astype(str).tolist()
    ids       = df["id"].tolist()
    n         = len(df)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"[load] base ckpt    {args.base_ckpt}")
    base_model, base_cfg = _load_momask(args.base_ckpt, BaseMaskTransformer, device)

    print(f"[load] residual ckpt {args.res_ckpt}")
    res_model, _ = _load_momask(args.res_ckpt, ResidualTransformer, device)

    print(f"[load] text encoder ({args.clip_name})")
    text_enc = FrozenTextEncoder(name=args.clip_name, device=str(device))

    print(f"[load] length ckpt  {args.length_ckpt}")
    from transformers import AutoTokenizer
    backbone = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(backbone)
    len_model = LengthPredictor(num_bins=base_cfg.num_length_bins,
                                backbone_name=backbone, freeze_backbone=True)
    load_head(len_model, str(args.length_ckpt))
    len_model = len_model.to(device).eval()

    # ── Predict lengths ───────────────────────────────────────────────────────
    print("[length] predicting sequence lengths...")
    all_lens = []
    with torch.no_grad():
        for i in range(0, n, 64):
            chunk_s = sentences[i:i + 64]
            chunk_g = glosses[i:i + 64]
            enc = tok(chunk_s, chunk_g, padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            logits = len_model(enc["input_ids"].to(device),
                               enc["attention_mask"].to(device))
            pred_bin = logits.argmax(-1).cpu().numpy()
            all_lens.append(bin_to_seq_len(pred_bin, base_cfg.num_length_bins))
    pred_lengths = np.concatenate(all_lens).astype(int)
    pred_lengths = np.clip(pred_lengths, SEQ_LEN_MIN, min(SEQ_LEN_MAX, base_cfg.max_len))
    pred_bins    = seq_len_to_bin(pred_lengths, base_cfg.num_length_bins).astype(np.int64)
    print(f"[length] p50={int(np.median(pred_lengths))}  p95={int(np.percentile(pred_lengths, 95))}")

    # ── Generate tokens ───────────────────────────────────────────────────────
    print(f"[gen] CFG={args.cfg_scale} steps={args.n_steps} T={args.temperature} | "
          f"res CFG={args.res_cfg_scale}  batch={args.batch_size}")
    all_tokens: list[np.ndarray] = []   # each entry: (NUM_LAYERS, T_i)

    BS = args.batch_size
    with torch.no_grad():
        for s in range(0, n, BS):
            e = min(n, s + BS)
            chunk_text = sentences[s:e]
            chunk_lens = torch.tensor(pred_lengths[s:e], dtype=torch.long, device=device)
            chunk_bins = torch.tensor(pred_bins[s:e],    dtype=torch.long, device=device)

            t_emb   = text_enc.encode_texts(chunk_text, device=str(device))
            uncond  = torch.zeros_like(t_emb)

            base_tokens = base_model.generate(
                t_emb, chunk_bins, chunk_lens,
                n_steps=args.n_steps,
                cfg_scale=args.cfg_scale,
                uncond_text_emb=uncond,
                temperature=args.temperature,
            )                                           # (B, T_max)

            all_layers = res_model.generate(
                base_tokens, t_emb, chunk_bins, chunk_lens,
                cfg_scale=args.res_cfg_scale,
                uncond_text_emb=uncond,
                temperature=args.temperature,
                sample=False,
            )                                           # (B, NUM_LAYERS, T_max)

            for bi in range(all_layers.shape[0]):
                L = int(chunk_lens[bi].item())
                all_tokens.append(all_layers[bi, :, :L].cpu().numpy())  # (6, L)

            print(f"  [gen] {e}/{n}")

    # ── Build submission CSV ──────────────────────────────────────────────────
    print("[csv] building submission...")
    col_names = ["base_tokens", "residual_1", "residual_2",
                 "residual_3", "residual_4", "residual_5"]
    rows = []
    for i, tokens in enumerate(all_tokens):
        row = {"id": ids[i]}
        for j, col in enumerate(col_names):
            row[col] = " ".join(map(str, tokens[j].tolist()))
        rows.append(row)

    sub = pd.DataFrame(rows, columns=["id"] + col_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"[csv] saved {len(sub)} rows -> {args.output}")

    # ── Quick validation ──────────────────────────────────────────────────────
    print("[validate] running submission validator...")
    from src.eval.validate_submission import validate_file
    report = validate_file(args.output)
    if report.errors:
        print(f"[validate] {len(report.errors)} ERRORS:")
        for err in report.errors[:10]:
            print(f"  {err}")
    else:
        print(f"[validate] OK — {report.n_rows} rows, {len(report.warnings)} warnings")
    print(f"\n[done] submit: {args.output}")


if __name__ == "__main__":
    main()
