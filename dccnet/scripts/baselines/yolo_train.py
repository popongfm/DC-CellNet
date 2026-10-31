"""Train Ultralytics YOLO26m on one split and write a run-dir contract summary.

Usage:
    python scripts/baselines/yolo_train.py --split splits/pooled.json --data-root data \
        --runs-dir runs --run-name p3b_pooled_yolo26m_s0 --seed 0

--split is relative to --data-root. The split is exported (split-JSON labels, our class order)
to <data-root>/yolo_<split stem>/ (reused if data.yaml exists). Weights: <run>/weights/best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.datasets.yolo_export import write_yolo  # noqa: E402

log = logging.getLogger("yolo_train")
FITNESS_COL = "metrics/mAP50-95(B)"


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, (v.strip() for v in r))) for r in rows[1:] if r]


def count_epochs(csv_path: Path) -> int:
    return len(_read_rows(csv_path))


def best_epoch_from_results(csv_path: Path) -> int:
    """0-based row of max mAP50-95(B); first max on ties."""
    vals = [float(r[FITNESS_COL]) for r in _read_rows(csv_path)]
    if not vals:
        raise ValueError(f"no rows in {csv_path}")
    return max(range(len(vals)), key=lambda i: (vals[i], -i))


def best_epoch_from_ckpt(best_pt: Path, results_csv: Path | None = None) -> int:
    """0-based best epoch for a finished Ultralytics run.

    The final best.pt has been through ``strip_optimizer``, which sets ``epoch = -1``, so a value >= 0 is
    used as-is but -1 (the expected case) falls back silently to results.csv: first max of mAP50-95(B)
    (detection fitness is mAP50-95 only and best.pt is written on the first maximum). A missing key
    also falls back, with a warning. Raises ValueError if the fallback is needed but unusable.
    """
    import torch

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    ep = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    if isinstance(ep, int) and not isinstance(ep, bool) and ep >= 0:
        return ep
    if ep is None:
        log.warning("'epoch' missing in %s; falling back to results.csv", best_pt)
    if results_csv is None or not Path(results_csv).exists():
        raise ValueError(f"best epoch not stored in {best_pt} (epoch={ep!r}) and no results.csv available")
    return best_epoch_from_results(Path(results_csv))


def write_summary(run_dir: Path, best_epoch: int, epochs_run: int, train_seconds: float, n_params: int) -> Path:
    path = Path(run_dir) / "summary.json"
    path.write_text(json.dumps({
        "family": "yolo", "best_epoch": int(best_epoch), "select_from_epoch": 0,
        "epochs_run": int(epochs_run), "config_stem": "yolo26m",
        "train_seconds": float(train_seconds), "n_params": int(n_params)}, indent=2))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True, help="relative to --data-root")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--epochs-max", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    from ultralytics import YOLO  # lazy: not installed everywhere

    split_path = args.data_root / args.split
    export_dir = args.data_root / f"yolo_{split_path.stem}"
    yaml_path = export_dir / "data.yaml"
    if not yaml_path.exists():
        yaml_path = write_yolo(read_split(split_path, args.data_root), export_dir)
    runs_dir = args.runs_dir.resolve()
    model = YOLO(args.model)
    t0 = time.time()
    model.train(data=str(yaml_path.resolve()), epochs=args.epochs_max, patience=args.patience,
                batch=args.batch, imgsz=args.imgsz, seed=args.seed, deterministic=True,
                device=args.device, project=str(runs_dir), name=args.run_name, exist_ok=True)
    train_seconds = time.time() - t0
    run_dir = runs_dir / args.run_name
    results_csv = run_dir / "results.csv"
    n_params = sum(p.numel() for p in model.model.parameters())
    write_summary(run_dir, best_epoch_from_ckpt(run_dir / "weights" / "best.pt", results_csv),
                  count_epochs(results_csv), train_seconds, n_params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
