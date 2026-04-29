"""Train MoMask base mask-transformer.

Usage:
    python -m scripts.train_momask_base --epochs 10 --batch-size 32

Outputs:
    checkpoints/momask_base.pth
    runs/momask_base.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class _EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict_for(self, model: torch.nn.Module) -> dict:
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            sd[k] = v.clone()
        return sd

from src.constants import (
    CHECKPOINT_DIR, RUNS_DIR, SPLIT_FILE, TRAIN_CSV,
)
from src.data.io import load_train
from src.drive_sync import mirror_to_drive
from src.models.data import TokenSeqDataset, collate_token_batch
from src.models.momask import (
    BaseMaskTransformer, MoMaskConfig, MASK_ID, PAD_ID, random_mask,
)
from src.models.text_cond import FrozenTextEncoder


def _load_split() -> tuple[set[int], set[int]]:
    if not SPLIT_FILE.exists():
        raise SystemExit(f"split file missing: {SPLIT_FILE}. Run `python -m scripts.make_split` first.")
    obj = json.loads(SPLIT_FILE.read_text())
    return set(map(int, obj["train"])), set(map(int, obj["val"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to existing ckpt to resume from (loads weights only).")
    p.add_argument("--ema", type=float, default=0.999,
                   help="EMA decay (set 0 to disable).")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--max-len", type=int, default=320)         # 99th pct ≈ 240; cap for speed
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)  # CFG: drop text 10% of the time
    p.add_argument("--clip-name", type=str, default="ViT-B/32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path(CHECKPOINT_DIR) / "momask_base.pth")
    p.add_argument("--drive-dir", type=str, default=None,
                   help="If set, mirror checkpoint here after every save (e.g. /content/drive/MyDrive/motion-s-ckpts).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] {TRAIN_CSV}")
    df = load_train()
    print(f"[data] {len(df)} rows; seq_len p50={df.seq_len.median():.0f} p95={df.seq_len.quantile(0.95):.0f}")

    train_idx, val_idx = _load_split()
    ids = df["id"].astype(int).to_numpy()
    train_mask = np.fromiter((i in train_idx for i in ids), dtype=bool, count=len(ids))
    val_mask   = np.fromiter((i in val_idx   for i in ids), dtype=bool, count=len(ids))
    train_df = df[train_mask].reset_index(drop=True)
    val_df   = df[val_mask].reset_index(drop=True)
    print(f"[split] train={len(train_df)}  val={len(val_df)}")

    train_ds = TokenSeqDataset(train_df, num_length_bins=args.num_bins, max_len=args.max_len)
    val_ds   = TokenSeqDataset(val_df,   num_length_bins=args.num_bins, max_len=args.max_len)
    print(f"[dataset] train kept={len(train_ds)}  val kept={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_token_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_token_batch)

    cfg = MoMaskConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, max_len=args.max_len, num_length_bins=args.num_bins,
    )
    model = BaseMaskTransformer(cfg).to(device)
    text_enc = FrozenTextEncoder(name=args.clip_name, device=str(device))

    if args.resume is not None and Path(args.resume).exists():
        ck = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"[resume] loaded weights from {args.resume}")

    ema = _EMA(model, decay=args.ema) if args.ema and args.ema > 0 else None
    if ema is not None:
        print(f"[ema] enabled, decay={args.ema}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BaseMaskTransformer params={n_params/1e6:.1f}M  cfg={cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, len(train_loader) * args.epochs))

    history: list[dict] = []
    best = {"val_loss": math.inf, "val_acc": 0.0, "epoch": -1}

    for epoch in range(args.epochs):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        sum_loss = 0.0; sum_corr = 0.0; sum_tok = 0
        for it, batch in enumerate(train_loader):
            tokens_all = batch["tokens"].to(device, non_blocking=True)         # (B, NUM_LAYERS, T)
            base = tokens_all[:, 0]                                             # (B, T)
            lengths = batch["length"].to(device)
            length_bin = batch["length_bin"].to(device)
            sentences = batch["sentence"]

            # text emb (CLIP frozen, no_grad)
            text_emb = text_enc.encode_texts(sentences, device=str(device))    # (B, 512)

            # CFG: drop with probability cfg_drop_prob → uncond emb (zeros)
            if args.cfg_drop_prob > 0:
                drop = (torch.rand(text_emb.size(0), device=device) < args.cfg_drop_prob)
                text_emb = torch.where(drop[:, None], torch.zeros_like(text_emb), text_emb)

            noisy, mask = random_mask(base, lengths)
            logits = model(noisy, text_emb, length_bin)                         # (B, T, V)

            target = base[mask]                                                 # (Nmask,)
            pred = logits[mask]                                                 # (Nmask, V)
            loss = F.cross_entropy(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                corr = (pred.argmax(-1) == target).float().sum().item()
            sum_loss += loss.item() * target.numel()
            sum_corr += corr
            sum_tok  += target.numel()

            if (it + 1) % args.log_every == 0:
                print(f"  [ep{epoch:02d} it{it+1:04d}] loss={sum_loss/sum_tok:.3f} "
                      f"acc={sum_corr/sum_tok:.3f} lr={sched.get_last_lr()[0]:.2e}")

        train_loss = sum_loss / max(1, sum_tok)
        train_acc  = sum_corr / max(1, sum_tok)

        # ── val (use EMA weights if available) ─────────────────────────────
        model.eval()
        live_sd = None
        if ema is not None:
            live_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict_for(model))
        v_loss = 0.0; v_corr = 0.0; v_tok = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens_all = batch["tokens"].to(device)
                base = tokens_all[:, 0]
                lengths = batch["length"].to(device)
                length_bin = batch["length_bin"].to(device)
                sentences = batch["sentence"]
                text_emb = text_enc.encode_texts(sentences, device=str(device))
                noisy, mask = random_mask(base, lengths)
                logits = model(noisy, text_emb, length_bin)
                target = base[mask]; pred = logits[mask]
                loss = F.cross_entropy(pred, target)
                v_loss += loss.item() * target.numel()
                v_corr += (pred.argmax(-1) == target).float().sum().item()
                v_tok  += target.numel()
        val_loss = v_loss / max(1, v_tok)
        val_acc  = v_corr / max(1, v_tok)

        rec = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(rec)
        print(f"[ep {epoch:02d}] train_loss={train_loss:.3f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f}")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # save whatever weights are currently in `model` — that's EMA if enabled.
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
            }, args.output)
            print(f"        \u21b3 new best, saved -> {args.output}")
            mirror_to_drive(args.output, args.drive_dir)

        # restore live (training) weights for next epoch
        if live_sd is not None:
            model.load_state_dict(live_sd)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "momask_base.json"
    out.write_text(json.dumps({"best": best, "history": history,
                                "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
    print(f"[done] best ep{best['epoch']}: val_loss={best['val_loss']:.3f} val_acc={best['val_acc']:.3f}")
    print(f"       wrote {out}")


if __name__ == "__main__":
    main()
