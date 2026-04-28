"""Train the length predictor.

Usage (Colab):
    !python -m scripts.train_length --epochs 8 --batch-size 64

Outputs: checkpoints/length_predictor.pth + runs/length_predictor.json (val MAE/acc).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.constants import (
    CHECKPOINT_DIR,
    LENGTH_ESTIMATOR_CKPT,
    RUNS_DIR,
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    SPLIT_FILE,
    TRAIN_CSV,
)
from src.length import (
    LengthDataset,
    LengthPredictor,
    bin_to_seq_len,
    save,
    seq_len_to_bin,
)


def _load_split() -> tuple[list[int], list[int]]:
    if not SPLIT_FILE.exists():
        raise SystemExit(f"split file missing: {SPLIT_FILE}. Run `python -m scripts.make_split` first.")
    obj = json.loads(SPLIT_FILE.read_text())
    return obj["train_idx"], obj["val_idx"]


def _filter_in_range(seq_len: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose seq_len is inside [SEQ_LEN_MIN, SEQ_LEN_MAX]."""
    return (seq_len >= SEQ_LEN_MIN) & (seq_len <= SEQ_LEN_MAX)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-text-tokens", type=int, default=64)
    p.add_argument("--backbone", type=str, default="distilbert-base-uncased")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path(CHECKPOINT_DIR) / "length_predictor.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[load] {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    # We only need seq_len from the token columns, not the parsed arrays.
    base_col = next((c for c in ["base_tokens", "base", "layer_0", "tokens_0"] if c in df.columns), None)
    if base_col is None:
        raise SystemExit("Could not find base-layer token column in train.csv")
    df["seq_len"] = df[base_col].fillna("").astype(str).map(lambda s: 0 if not s.strip() else len(s.split()))
    print(f"[data] {len(df)} rows, seq_len p50={df.seq_len.median():.0f} p95={df.seq_len.quantile(0.95):.0f}")

    train_idx, val_idx = _load_split()
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # filter to rows whose target is representable
    tmask = _filter_in_range(train_df["seq_len"].to_numpy())
    vmask = _filter_in_range(val_df["seq_len"].to_numpy())
    print(f"[filter] train kept {tmask.sum()}/{len(tmask)}, val kept {vmask.sum()}/{len(vmask)}")
    train_df, val_df = train_df[tmask].reset_index(drop=True), val_df[vmask].reset_index(drop=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)

    def to_dataset(d: pd.DataFrame) -> LengthDataset:
        return LengthDataset(
            sentences=d["sentence"].fillna("").astype(str).tolist(),
            glosses=d["gloss"].fillna("").astype(str).tolist(),
            seq_lens=d["seq_len"].astype(int).tolist(),
            tokenizer=tok,
            max_text_tokens=args.max_text_tokens,
        )

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"[model] LengthPredictor(num_bins={args.num_bins}, backbone={args.backbone})")
    model = LengthPredictor(num_bins=args.num_bins, hidden=args.hidden,
                            dropout=args.dropout, backbone_name=args.backbone,
                            freeze_backbone=True).to(args.device)

    # Train only the head (backbone frozen).
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, len(train_loader) * args.epochs))

    best = {"val_mae": math.inf, "val_acc": 0.0, "epoch": -1}
    history: list[dict] = []

    for epoch in range(args.epochs):
        model.head.train(); model.backbone.eval()
        train_loss_sum = 0.0; n_seen = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(args.device, non_blocking=True)
            am = batch["attention_mask"].to(args.device, non_blocking=True)
            sl = batch["seq_len"].numpy()
            target = torch.from_numpy(seq_len_to_bin(sl, args.num_bins)).to(args.device)

            logits = model(ids, am)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
            train_loss_sum += loss.item() * ids.size(0); n_seen += ids.size(0)

        # ── validate ──
        model.eval()
        all_pred_bins, all_true_lens = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(args.device)
                am = batch["attention_mask"].to(args.device)
                logits = model(ids, am)
                all_pred_bins.append(logits.argmax(dim=-1).cpu().numpy())
                all_true_lens.append(batch["seq_len"].numpy())
        pred_bins = np.concatenate(all_pred_bins)
        true_lens = np.concatenate(all_true_lens)
        true_bins = seq_len_to_bin(true_lens, args.num_bins)
        pred_lens = bin_to_seq_len(pred_bins, args.num_bins)
        val_acc = float((pred_bins == true_bins).mean())
        val_mae = float(np.abs(pred_lens.astype(np.int64) - true_lens.astype(np.int64)).mean())

        rec = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, n_seen),
            "val_acc": val_acc,
            "val_mae": val_mae,
        }
        history.append(rec)
        print(f"[ep {epoch:02d}] train_loss={rec['train_loss']:.4f}  val_acc={val_acc:.3f}  val_mae={val_mae:.1f}")

        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae, "val_acc": val_acc, "epoch": epoch}
            save(model, args.output)
            print(f"        ↳ new best, saved -> {args.output}")

    # Also write a sibling copy at the canonical data-dir location used by inference.
    try:
        save(model, LENGTH_ESTIMATOR_CKPT)
    except Exception as e:
        print(f"[warn] could not write {LENGTH_ESTIMATOR_CKPT}: {e}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "length_predictor.json"
    out.write_text(json.dumps({"best": best, "history": history, "args": vars(args)}, indent=2, default=str))
    print(f"[done] best epoch {best['epoch']}: val_mae={best['val_mae']:.1f} val_acc={best['val_acc']:.3f}")
    print(f"       wrote {out}")


if __name__ == "__main__":
    main()
