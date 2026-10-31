"""Train the torchvision Faster R-CNN (ResNet-50 FPN) baseline on one split.

Usage (from dccnet/):
    python scripts/baselines/frcnn_train.py --split splits/pooled.json --data-root data \
        --runs-dir runs --run-name p3b_pooled_frcnn_s0 --seed 0 [--epochs-max 50] [--patience 10] \
        [--batch 4] [--accum-steps 1] [--device cuda] [--no-pretrained] [--min-size 800] [--max-size 1333]

--split is relative to --data-root. Schedule: SGD (lr 0.005, momentum 0.9, wd 1e-4); as the torchvision
reference, linear warmup from 0.001x in epoch 0 only, over min(1000, optimizer steps in epoch 0 - 1)
steps (factor 1 from epoch 1 on); x0.1 at epochs 32 and 42.
Every epoch: val mAP50-95 (dccnet.eval.coco_eval.coco_map); early stopping with --patience on it
(strictly greater = better). Run dir: metrics.jsonl, best.pt ({"model", "epoch", "args"}), summary.json.
OOM policy: lower --batch and raise --accum-steps to keep the effective batch (e.g. --batch 2 --accum-steps 2).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import frcnn_common as fc  # noqa: E402
from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.coco_eval import coco_map  # noqa: E402
from dccnet.train.loop import EarlyStopping  # noqa: E402

log = logging.getLogger("frcnn_train")
LR, MOMENTUM, WEIGHT_DECAY = 0.005, 0.9, 1e-4
LOSS_NAMES = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _set_lr(opt, factor: float) -> float:
    lr = LR * factor
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def window_size(bi: int, accum: int, n_batches: int) -> int:
    """Number of micro-batches in the accumulation window holding batch ``bi`` (the last one may be short)."""
    start = (bi // accum) * accum
    return min(accum, n_batches - start)


def should_step(n_finite: int) -> bool:
    """Step the optimizer only if the window produced gradients (GradScaler.step asserts otherwise)."""
    return n_finite > 0


def warmup_iters_for(steps_epoch0: int) -> int:
    """torchvision reference: warmup over min(1000, optimizer steps in epoch 0 - 1) steps."""
    return max(0, min(fc.WARMUP_ITERS, int(steps_epoch0) - 1))


def train_one_epoch(model, loader, opt, scaler, device, epoch: int, step: int, warmup_iters: int,
                    accum: int, amp: bool) -> tuple[dict, int]:
    """One epoch with gradient accumulation: each micro-batch loss is divided by its window's real size
    (``accum``, or fewer for the last window); optimizer step at every window end. Non-finite micro-batches
    are skipped; a window with no finite micro-batch skips the optimizer step and does not advance ``step``.
    Returns (mean losses + lr + nonfinite_steps, optimizer steps done so far)."""
    model.train()
    sums = {k: 0.0 for k in ("loss_total", *LOSS_NAMES)}
    n_ok = n_bad = n_win_ok = 0
    lr = opt.param_groups[0]["lr"]
    n_batches = len(loader)
    opt.zero_grad(set_to_none=True)
    for bi, (images, targets) in enumerate(loader):
        n_win = window_size(bi, accum, n_batches)
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type=device.type, enabled=amp):
            loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if torch.isfinite(loss):
            scaled = loss / n_win
            if scaler is not None:
                scaled = scaler.scale(scaled)
            scaled.backward()
            n_ok += 1
            n_win_ok += 1
            sums["loss_total"] += float(loss.detach())
            for k in LOSS_NAMES:
                sums[k] += float(loss_dict[k].detach())
        else:
            n_bad += 1
            log.warning("epoch %d batch %d: non-finite loss, micro-batch skipped", epoch, bi)
        if (bi + 1) % accum == 0 or bi + 1 == n_batches:
            if should_step(n_win_ok):
                lr = _set_lr(opt, fc.lr_factor(step, epoch, warmup_iters))
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                step += 1
            else:
                log.warning("epoch %d batch %d: no finite loss in the window, optimizer step skipped", epoch, bi)
            opt.zero_grad(set_to_none=True)
            n_win_ok = 0
    if n_ok == 0:
        raise RuntimeError(f"epoch {epoch}: every training loss was non-finite")
    out = {k: v / n_ok for k, v in sums.items()}
    out.update(lr=lr, nonfinite_steps=n_bad)
    return out, step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True, help="relative to --data-root")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs-max", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--no-pretrained", action="store_true", help="random init, no weight download")
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1333)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args(argv)
    if args.batch < 1 or args.accum_steps < 1 or args.epochs_max < 1:
        ap.error("--batch, --accum-steps and --epochs-max must be >= 1")

    run_dir = args.runs_dir / args.run_name
    for name in ("best.pt", "summary.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"{run_dir / name} exists; choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    split = read_split(args.data_root / args.split, args.data_root)
    train_samples, val_samples = split["train"], split["val"]

    model = fc.build_frcnn(pretrained=not args.no_pretrained, min_size=args.min_size, max_size=args.max_size)
    n_params = sum(p.numel() for p in model.parameters())
    model.to(device)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(fc.SplitDetDataset(train_samples, train=True), batch_size=args.batch,
                                         shuffle=True, num_workers=args.num_workers, collate_fn=fc.collate,
                                         generator=gen)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    steps_per_epoch = math.ceil(len(loader) / args.accum_steps)
    warmup_iters = warmup_iters_for(steps_per_epoch)
    saved_args = {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()}

    print(f"run dir: {run_dir}  device: {device}  train {len(train_samples)} / val {len(val_samples)}", flush=True)
    es = EarlyStopping(patience=args.patience)
    best_val: dict = {}
    metrics_path = run_dir / "metrics.jsonl"
    step = 0
    epochs_run = 0
    t0 = time.time()
    for epoch in range(args.epochs_max):
        te = time.time()
        row, step = train_one_epoch(model, loader, opt, scaler, device, epoch, step, warmup_iters,
                                    args.accum_steps, amp)
        recs = fc.predict_split(model, val_samples, args.data_root, device, batch=args.batch)
        val = coco_map(val_samples, recs, args.data_root)
        row = {"epoch": epoch, **row, "val_mAP50": val["mAP50"], "val_mAP50_95": val["mAP50_95"],
               "epoch_time_s": time.time() - te}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        epochs_run = epoch + 1
        if es.step(val["mAP50_95"], epoch):
            best_val = {"mAP50": val["mAP50"], "mAP50_95": val["mAP50_95"]}
            _save_atomic({"model": model.state_dict(), "epoch": epoch, "args": saved_args}, run_dir / "best.pt")
        print(f"epoch {epoch}: loss {row['loss_total']:.4f}  val mAP50-95 {val['mAP50_95']}", flush=True)
        if es.should_stop:
            break
    train_seconds = time.time() - t0

    summary = {"family": "frcnn", "best_epoch": es.best_epoch, "select_from_epoch": 0, "epochs_run": epochs_run,
               "val": best_val, "config_stem": "frcnn", "train_seconds": train_seconds, "n_params": n_params}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"best epoch {es.best_epoch}: val {best_val}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
