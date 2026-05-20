"""Train MoMask residual transformer.

Predicts residual layers 1..NUM_LAYERS-1 conditioned on the lower layers + text.
At each step we pick a random target layer L (uniform over {1..NUM_LAYERS-1}),
feed in ground-truth layers 0..L-1 as `prev_tokens`, and predict layer L.

Usage:
    python -m scripts.train_momask_residual --epochs 10 --batch-size 32

Outputs:
    checkpoints/momask_residual.pth
    runs/momask_residual.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.constants import (
    CHECKPOINT_DIR, NUM_LAYERS, RUNS_DIR, SPLIT_FILE, TRAIN_CSV,
)
from src.data.io import load_train
from src.drive_sync import mirror_to_drive
from src.models.data import TokenSeqDataset, collate_token_batch
from src.models.momask import (
    MoMaskConfig, ResidualTransformer, PAD_ID,
)
from src.models.text_cond import FrozenTextEncoder


def _load_split() -> tuple[set[int], set[int]]:
    if not SPLIT_FILE.exists():
        raise SystemExit(f"split file missing: {SPLIT_FILE}.")
    obj = json.loads(SPLIT_FILE.read_text())
    return set(map(int, obj["train"])), set(map(int, obj["val"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)
    p.add_argument("--clip-name", type=str, default="ViT-B/32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path(CHECKPOINT_DIR) / "momask_residual.pth")
    p.add_argument("--drive-dir", type=str, default=None,
                   help="If set, mirror checkpoint here after every save.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to existing checkpoint to resume training from.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] {TRAIN_CSV}")
    df = load_train()
    print(f"[data] {len(df)} rows")

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
    model = ResidualTransformer(cfg).to(device)
    text_enc = FrozenTextEncoder(name=args.clip_name, device=str(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ResidualTransformer params={n_params/1e6:.1f}M  cfg={cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, len(train_loader) * args.epochs))

    history: list[dict] = []
    best = {"val_loss": math.inf, "val_acc": 0.0, "epoch": -1}
    start_epoch = 0

    if args.resume is not None and Path(args.resume).exists():
        ck = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            sched.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck.get("epoch", -1) + 1
        best = ck.get("best", best)
        history = ck.get("history", [])
        print(f"[resume] epoch {start_epoch}, val_loss={best['val_loss']:.3f} from {args.resume}")

    n_residual = NUM_LAYERS - 1                     # 5

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sum_loss = 0.0; sum_corr = 0.0; sum_tok = 0
        per_layer_corr = [0] * n_residual; per_layer_tok = [0] * n_residual

        for it, batch in enumerate(train_loader):
            tokens_all = batch["tokens"].to(device, non_blocking=True)   # (B, 6, T)
            B, _, T = tokens_all.shape
            lengths = batch["length"].to(device)
            length_bin = batch["length_bin"].to(device)
            sentences = batch["sentence"]
            text_emb = text_enc.encode_texts(sentences, device=str(device))

            if args.cfg_drop_prob > 0:
                drop = (torch.rand(B, device=device) < args.cfg_drop_prob)
                text_emb = torch.where(drop[:, None], torch.zeros_like(text_emb), text_emb)

            # Pick a single target layer per batch (simpler + equivalent in expectation).
            target_layer_int = int(torch.randint(1, n_residual + 1, (1,)).item())
            target_layer = torch.full((B,), target_layer_int, dtype=torch.long, device=device)
            prev_tokens = tokens_all[:, :target_layer_int].clone()       # (B, L_prev, T)
            # Mask PAD positions in prev_tokens to a safe id (0) — those positions
            # are excluded from loss anyway.
            prev_safe = prev_tokens.clamp(min=0, max=511)

            target_tokens = tokens_all[:, target_layer_int]              # (B, T)
            pos = torch.arange(T, device=device)[None].expand(B, -1)
            valid = pos < lengths[:, None]                               # (B, T)

            logits = model(prev_safe, target_layer, text_emb, length_bin)  # (B, T, 512)

            target = target_tokens[valid]
            pred = logits[valid]
            loss = F.cross_entropy(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            with torch.no_grad():
                corr = (pred.argmax(-1) == target).float().sum().item()
            sum_loss += loss.item() * target.numel()
            sum_corr += corr
            sum_tok  += target.numel()
            per_layer_corr[target_layer_int - 1] += corr
            per_layer_tok[target_layer_int - 1] += target.numel()

            if (it + 1) % args.log_every == 0:
                print(f"  [ep{epoch:02d} it{it+1:04d}] loss={sum_loss/sum_tok:.3f} "
                      f"acc={sum_corr/sum_tok:.3f} lr={sched.get_last_lr()[0]:.2e}")

        train_loss = sum_loss / max(1, sum_tok)
        train_acc  = sum_corr / max(1, sum_tok)
        layer_acc_train = [
            per_layer_corr[i] / max(1, per_layer_tok[i])
            for i in range(n_residual)
        ]

        # ── val: evaluate each layer in turn ────────────────────────────────
        model.eval()
        v_loss = 0.0; v_corr = 0.0; v_tok = 0
        v_layer_corr = [0] * n_residual; v_layer_tok = [0] * n_residual
        with torch.no_grad():
            for batch in val_loader:
                tokens_all = batch["tokens"].to(device)
                B, _, T = tokens_all.shape
                lengths = batch["length"].to(device)
                length_bin = batch["length_bin"].to(device)
                sentences = batch["sentence"]
                text_emb = text_enc.encode_texts(sentences, device=str(device))
                pos = torch.arange(T, device=device)[None].expand(B, -1)
                valid = pos < lengths[:, None]
                for L in range(1, n_residual + 1):
                    target_layer = torch.full((B,), L, dtype=torch.long, device=device)
                    prev_safe = tokens_all[:, :L].clamp(min=0, max=511)
                    logits = model(prev_safe, target_layer, text_emb, length_bin)
                    target = tokens_all[:, L][valid]
                    pred = logits[valid]
                    loss = F.cross_entropy(pred, target)
                    v_loss += loss.item() * target.numel()
                    c = (pred.argmax(-1) == target).float().sum().item()
                    v_corr += c
                    v_tok  += target.numel()
                    v_layer_corr[L - 1] += c
                    v_layer_tok[L - 1] += target.numel()
        val_loss = v_loss / max(1, v_tok)
        val_acc  = v_corr / max(1, v_tok)
        layer_acc_val = [
            v_layer_corr[i] / max(1, v_layer_tok[i])
            for i in range(n_residual)
        ]

        rec = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "layer_acc_train": layer_acc_train,
            "layer_acc_val": layer_acc_val,
        }
        history.append(rec)
        layer_str = " ".join(f"L{i+1}={a:.2f}" for i, a in enumerate(layer_acc_val))
        print(f"[ep {epoch:02d}] train_loss={train_loss:.3f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f}  {layer_str}")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "config": cfg.__dict__,
                "args": vars(args),
            }, args.output)
            print(f"        ↳ new best, saved -> {args.output}")
            mirror_to_drive(args.output, args.drive_dir)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "momask_residual.json"
    out.write_text(json.dumps({"best": best, "history": history,
                                "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))
    print(f"[done] best ep{best['epoch']}: val_loss={best['val_loss']:.3f} val_acc={best['val_acc']:.3f}")
    print(f"       wrote {out}")


if __name__ == "__main__":
    main()
