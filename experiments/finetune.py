"""Fine-tune Chronos-Bolt on multi-BA hourly load with early stopping.

ML hygiene:
- Seeded RNGs (torch + numpy + python).
- Val-loss-based early stopping with best-checkpoint retention.
- Logs train and val loss each eval step so train/val gap is visible.
- No test-set exposure during training or hyperparameter selection.

Usage:
    python -m experiments.finetune \
        --base amazon/chronos-bolt-base \
        --bas PJM CISO ERCO MISO NYIS ISNE SWPP \
        --context 1024 --horizon 24 --stride 1 \
        --batch 64 --lr 3e-5 --max-steps 2000 \
        --val-every 100 --patience 5 \
        --out /workspace/data/_ft/bolt_base_multi_ba
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_cosine_schedule_with_warmup

from chronos.chronos_bolt import ChronosBoltModelForForecasting

from experiments import eval as eev


class MultiBARollingDataset(Dataset):
    """Concatenates per-BA training series. Each (ba, origin) yields one pair."""

    def __init__(self, split: eev.Split, context: int, horizon: int, stride: int):
        self.series: dict[str, np.ndarray] = {
            ba: bs.train.astype(np.float32) for ba, bs in split.bas.items()
        }
        self.context = context
        self.horizon = horizon
        self.index: list[tuple[str, int]] = []
        for ba, s in self.series.items():
            for o in range(context, len(s) - horizon + 1, stride):
                self.index.append((ba, o))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ba, o = self.index[i]
        s = self.series[ba]
        return {"context": torch.tensor(s[o - self.context:o]),
                "target":  torch.tensor(s[o:o + self.horizon])}


@torch.no_grad()
def _val_loss(model, split: eev.Split, context: int, horizon: int,
              batch_size: int, device: str) -> float:
    """Mean val-split quantile loss, pooled across BAs."""
    model.eval()
    losses: list[float] = []
    for ba, bs in split.bas.items():
        val = bs.val.astype(np.float32)
        n = len(val)
        origins = list(range(context, n - horizon + 1, horizon))
        if not origins:
            continue
        ctx_np = np.stack([val[o - context:o] for o in origins])
        tgt_np = np.stack([val[o:o + horizon] for o in origins])
        ctx = torch.tensor(ctx_np, device=device)
        tgt = torch.tensor(tgt_np, device=device)
        for i in range(0, len(ctx), batch_size):
            out = model(context=ctx[i:i + batch_size], target=tgt[i:i + batch_size])
            losses.append(float(out.loss.item()))
    model.train()
    return float(np.mean(losses))


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="amazon/chronos-bolt-base")
    ap.add_argument("--bas", nargs="+", default=["PJM"])
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    print(f"[args] {vars(args)}", flush=True)

    set_seed(args.seed)

    split = eev.load_split(args.bas)
    train_counts = {ba: len(bs.train) for ba, bs in split.bas.items()}
    print(f"[data] BAs={list(split.bas)} train_rows={train_counts}", flush=True)

    ds = MultiBARollingDataset(split, args.context, args.horizon, args.stride)
    print(f"[data] training pairs: {len(ds):,}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True,
                    generator=torch.Generator().manual_seed(args.seed))

    device = "cuda"
    model = ChronosBoltModelForForecasting.from_pretrained(args.base)
    model = model.to(device).to(torch.bfloat16)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.max_steps)

    best_val = float("inf")
    best_step = 0
    patience_left = args.patience
    history: list[dict] = []
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_dir = out / "best"

    step = 0
    t0 = time.time()
    running_loss: list[float] = []
    dl_iter = iter(dl)
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
            ctx = batch["context"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            out_ = model(context=ctx, target=tgt)
            (out_.loss / args.grad_accum).backward()
            running_loss.append(float(out_.loss.item()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1

        if step % args.val_every == 0 or step == 1 or step == args.max_steps:
            train_avg = sum(running_loss[-args.val_every:]) / max(1, len(running_loss[-args.val_every:]))
            val_avg = _val_loss(model, split, args.context, args.horizon, args.batch, device)
            elapsed = time.time() - t0
            history.append({"step": step, "train_loss": train_avg, "val_loss": val_avg,
                            "lr": sched.get_last_lr()[0], "elapsed_s": elapsed})
            gap = train_avg - val_avg
            improved = val_avg < best_val - 1e-4
            print(f"[step {step}/{args.max_steps}] train={train_avg:.4f} val={val_avg:.4f} "
                  f"gap={gap:+.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"best={best_val:.4f}@{best_step} patience={patience_left} elapsed={elapsed:.0f}s",
                  flush=True)
            if improved:
                best_val = val_avg
                best_step = step
                patience_left = args.patience
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                model.save_pretrained(best_dir)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early-stop] no improvement for {args.patience} evals", flush=True)
                    break

    # Final: always keep a `last` + copy `best` to out root for eval convenience.
    model.save_pretrained(out / "last")
    # Promote best to out root
    if best_dir.exists():
        for f in best_dir.iterdir():
            dest = out / f.name
            if dest.exists(): dest.unlink() if not dest.is_dir() else shutil.rmtree(dest)
            shutil.copy2(f, dest)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print("FINETUNE_DONE:", json.dumps({
        "steps": step, "best_step": best_step, "best_val_loss": best_val,
        "wall_s": round(time.time() - t0, 1), "out": str(out),
    }))


if __name__ == "__main__":
    main()
