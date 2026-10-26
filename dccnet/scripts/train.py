"""Train DC-CellNet.

Usage (from dccnet/):
    python scripts/train.py --config configs/exp.yaml [--seed 0] [--device cuda] [--data-root data]
        [--runs-dir runs] [--run-name NAME] [--epochs-max N] [--num-workers 0]

Run dir = <runs-dir>/<run-name> (default run-name: <config stem>_s<seed>) with config.yaml,
git_hash.txt, metrics.jsonl, best.pt, summary.json. The split file is <data-root>/<data.split>.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.config import load_config  # noqa: E402
from dccnet.train.loop import fit  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs-max", type=int, default=None, help="overrides train.epochs_max")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.epochs_max is not None:
        cfg["train"]["epochs_max"] = args.epochs_max
    seed = int(cfg["train"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.run_name or f"{args.config.stem}_s{seed}"
    run_dir = args.runs_dir / run_name
    print(f"run dir: {run_dir}  device: {device}", flush=True)
    summary = fit(cfg, run_dir, device, args.data_root, num_workers=args.num_workers)
    print(f"best epoch {summary['best_epoch']}: val {summary['val']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
