"""DC-CellNet training loop: data loaders, train/val epochs, early stopping, run-dir writing.

Run dir (``fit``) contains exactly: ``config.yaml`` (resolved config), ``git_hash.txt``,
``metrics.jsonl`` (one JSON line per epoch), ``best.pt`` and ``summary.json``.

* Epochs are 0-based (``epoch`` is what ``compute_losses`` / the λ_cc schedule receive).
* Optimizer AdamW(lr=train.lr, weight_decay=1e-4); CosineAnnealingLR(T_max=train.epochs_max),
  stepped once per epoch. AMP (autocast + GradScaler) only on CUDA with ``train.amp``.
* A step whose ``loss_total`` is non-finite is skipped (no backward / optimizer step), logged as a
  warning and counted in ``nonfinite_steps``; epoch loss means are over finite steps only.
* Validation every epoch: decode with ``score_thr=0.001``, ``max_dets=1000`` (other decode params from
  ``cfg["decode"]``), records in original image px, ``coco_map`` + ``count_errors`` at
  ``decode.score_thr``. Best checkpoint by val mAP50_95 (ties -> earlier epoch); early stop after
  ``train.patience`` epochs without improvement; at most ``train.epochs_max`` epochs.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dccnet.datasets.geometry import circles_to_boxes
from dccnet.datasets.splits import read_split
from dccnet.datasets.torch_dataset import CellDataset, collate_fn, unletterbox
from dccnet.decode import decode
from dccnet.eval.coco_eval import coco_map
from dccnet.eval.counting import count_errors
from dccnet.eval.export_json import image_key
from dccnet.losses import compute_losses
from dccnet.models import DCCellNet
from dccnet.types import Sample

log = logging.getLogger(__name__)

PRED_SCORE_THR = 0.001
PRED_MAX_DETS = 1000
WEIGHT_DECAY = 1e-4
LOSS_KEYS = ("loss_total", "loss_focal", "loss_r", "loss_ciou", "loss_o", "loss_d",
             "loss_cc", "loss_cc_bg", "lambda_cc")
RUN_FILES = ("config.yaml", "git_hash.txt", "metrics.jsonl", "best.pt", "summary.json")


class EarlyStopping:
    """Track the best value (higher is better). ``None`` ranks below every number but above
    "no epoch yet", so epoch 0 is always best and a checkpoint always exists."""

    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best_key = -math.inf
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def step(self, value: float | None, epoch: int) -> bool:
        """Record one epoch; return True if it is the new best (strict improvement)."""
        key = -1.0 if value is None else float(value)
        if key > self.best_key:
            self.best_key, self.best_epoch, self.bad_epochs = key, int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def _use_amp(cfg: dict, device: torch.device) -> bool:
    return device.type == "cuda" and bool(cfg["train"].get("amp", False))


def _eval_loader(samples: list[Sample], cfg: dict, batch: int, num_workers: int, pin: bool) -> DataLoader:
    ds = CellDataset(samples, img_size=int(cfg["data"]["img_size"]), aug=None, stride=int(cfg["data"]["stride"]))
    return DataLoader(ds, batch_size=int(batch), shuffle=False, num_workers=int(num_workers),
                      collate_fn=collate_fn, pin_memory=pin, persistent_workers=False)


def build_loaders(cfg: dict, data_root: Path, num_workers: int = 0,
                  device: torch.device | None = None) -> tuple[DataLoader, DataLoader]:
    """(train_loader, val_loader) from ``data_root / cfg["data"]["split"]`` keys "train" / "val".

    Train: ``CellDataset(aug=cfg["aug"])``, shuffled (seeded by ``train.seed``); val: ``aug=None``.
    ``persistent_workers=False`` so ``train_loader.dataset.set_epoch`` reaches fresh workers each epoch.
    """
    data_root = Path(data_root)
    split = read_split(data_root / cfg["data"]["split"], data_root)
    pin = device is not None and device.type == "cuda"
    batch = int(cfg["train"]["batch"])
    train_ds = CellDataset(split["train"], img_size=int(cfg["data"]["img_size"]), aug=cfg.get("aug"),
                           stride=int(cfg["data"]["stride"]))
    gen = torch.Generator()
    gen.manual_seed(int(cfg["train"]["seed"]))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=int(num_workers),
                              collate_fn=collate_fn, pin_memory=pin, persistent_workers=False, generator=gen)
    val_loader = _eval_loader(split["val"], cfg, batch, num_workers, pin)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, cfg: dict, device: torch.device, epoch: int) -> dict:
    """One pass over ``loader``. Returns mean of each loss key over finite steps + ``nonfinite_steps``."""
    model.train()
    amp = _use_amp(cfg, device)
    sums = {k: 0.0 for k in LOSS_KEYS}
    n_ok = n_bad = 0
    for step, (images, targets, _metas) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
        losses = compute_losses(outputs, targets, cfg, epoch)  # outputs are float32
        total = losses["loss_total"]
        if not torch.isfinite(total):
            n_bad += 1
            log.warning("epoch %d step %d: non-finite loss_total (%s); optimizer step skipped",
                        epoch, step, total.item())
            continue
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()
        n_ok += 1
        for k in LOSS_KEYS:
            sums[k] += float(losses[k].detach().item())
    out = {k: (sums[k] / n_ok if n_ok else float("nan")) for k in LOSS_KEYS}
    out["nonfinite_steps"] = n_bad
    return out


def _decode_kwargs(cfg: dict, score_thr: float) -> dict:
    d = cfg["decode"]
    return {
        "score_thr": float(score_thr),
        "stride": int(cfg["data"]["stride"]),
        "density_fill": bool(d.get("density_fill", False)),
        "window": int(d.get("window", cfg["loss"]["cc"]["window"])),
        "low_thr": float(d.get("low_thr", 0.1)),
        "nms": d.get("nms", "none"),
        "max_dets": PRED_MAX_DETS,
    }


def dets_to_records(dets: np.ndarray, meta: dict, key: str) -> list[dict]:
    """Decode rows [cls, cx, cy, r, score] (letterboxed px) -> evaluator records (original px)."""
    dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
    if not len(dets):
        return []
    circles = unletterbox(dets[:, 1:4], meta)
    boxes = circles_to_boxes(circles)
    recs = []
    for d, c, b in zip(dets, circles, boxes):
        recs.append({
            "image": key,
            "category_id": int(d[0]),
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(d[4]),
            "circle": [float(c[0]), float(c[1]), float(c[2])],
        })
    return recs


@torch.no_grad()
def _predict_loader(model, loader: DataLoader, cfg: dict, device: torch.device, data_root: Path,
                    score_thr: float = PRED_SCORE_THR) -> list[dict]:
    model.eval()
    amp = _use_amp(cfg, device)
    samples = loader.dataset.samples
    kw = _decode_kwargs(cfg, score_thr)
    records: list[dict] = []
    i = 0
    for images, _targets, metas in loader:
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(images.to(device, non_blocking=True))
        dets = decode(out["heatmap"], out["log_r"], out["offset"], out.get("density"), **kw)
        for det, meta in zip(dets, metas):
            sample = samples[i]
            i += 1
            assert Path(meta["image"]) == Path(sample.image_path)
            records.extend(dets_to_records(det, meta, image_key(sample, data_root)))
    return records


def predict_records(model, samples: list[Sample], cfg: dict, device: torch.device, data_root: Path,
                    batch: int = 8, score_thr: float = PRED_SCORE_THR, num_workers: int = 0) -> list[dict]:
    """Evaluator records (unletterboxed, ``score >= score_thr``) for ``samples`` in order."""
    loader = _eval_loader(samples, cfg, batch, num_workers, device.type == "cuda")
    return _predict_loader(model, loader, cfg, device, Path(data_root), score_thr)


def evaluate_epoch(model, val_loader: DataLoader, cfg: dict, device: torch.device, data_root: Path) -> dict:
    """``{"mAP50", "mAP50_95", "rbc_count_mae", "per_class", "per_class_50_95"}`` on ``val_loader``."""
    samples = val_loader.dataset.samples
    recs = _predict_loader(model, val_loader, cfg, device, Path(data_root))
    cm = coco_map(samples, recs, data_root)
    ce = count_errors(samples, recs, float(cfg["decode"]["score_thr"]), data_root)
    return {"mAP50": cm["mAP50"], "mAP50_95": cm["mAP50_95"], "rbc_count_mae": ce["MAE"]["RBC"],
            "per_class": cm["per_class"], "per_class_50_95": cm["per_class_50_95"]}


def git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                           capture_output=True, text=True, timeout=10)
        h = r.stdout.strip()
        return h if r.returncode == 0 and h else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _json_num(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def fit(cfg: dict, run_dir: Path, device: torch.device, data_root: Path, num_workers: int = 0) -> dict:
    """Train with validation + early stopping; write the run dir; return the summary dict.

    The caller seeds RNGs (see ``scripts/train.py``). Refuses to overwrite a run dir that already
    has ``best.pt`` or ``summary.json``.
    """
    run_dir = Path(run_dir)
    for name in ("best.pt", "summary.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"{run_dir / name} exists; choose another --run-name")
    device = torch.device(device)
    data_root = Path(data_root)
    cfg = copy.deepcopy(cfg)
    tcfg = cfg["train"]
    epochs_max, patience = int(tcfg["epochs_max"]), int(tcfg["patience"])

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    ghash = git_hash()
    (run_dir / "git_hash.txt").write_text(ghash + "\n", encoding="utf-8")

    train_loader, val_loader = build_loaders(cfg, data_root, num_workers, device)
    model = DCCellNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max)
    scaler = torch.amp.GradScaler("cuda") if _use_amp(cfg, device) else None
    stopper = EarlyStopping(patience)
    best_val: dict = {}

    with open(run_dir / "metrics.jsonl", "w", encoding="utf-8") as mf:
        for epoch in range(epochs_max):
            t0 = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loader.dataset.set_epoch(epoch)
            tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)
            scheduler.step()
            val = evaluate_epoch(model, val_loader, cfg, device, data_root)
            improved = stopper.step(val["mAP50_95"], epoch)
            if improved:
                best_val = val
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val},
                           run_dir / "best.pt")
            row = {"epoch": epoch, **{k: _json_num(tr[k]) for k in LOSS_KEYS},
                   "val_mAP50": _json_num(val["mAP50"]), "val_mAP50_95": _json_num(val["mAP50_95"]),
                   "val_rbc_count_mae": _json_num(val["rbc_count_mae"]), "lr": lr,
                   "nonfinite_steps": tr["nonfinite_steps"],
                   "epoch_time_s": round(time.perf_counter() - t0, 3)}
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print(f"epoch {epoch:3d}  loss {row['loss_total']}  val mAP50 {row['val_mAP50']}  "
                  f"mAP50-95 {row['val_mAP50_95']}  RBC MAE {row['val_rbc_count_mae']}"
                  f"{'  *best' if improved else ''}  ({row['epoch_time_s']:.1f}s)", flush=True)
            if stopper.should_stop:
                print(f"early stop: no val mAP50-95 improvement for {patience} epochs", flush=True)
                break

    summary = {
        "exp": run_dir.name,
        "seed": int(tcfg["seed"]),
        "git_hash": ghash,
        "best_epoch": stopper.best_epoch,
        "val": {"mAP50": _json_num(best_val.get("mAP50")), "mAP50_95": _json_num(best_val.get("mAP50_95")),
                "rbc_count_mae": _json_num(best_val.get("rbc_count_mae"))},
        "test": None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
