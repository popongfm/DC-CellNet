"""Write evaluator prediction records for one split subset from a trained run.

Usage (from dccnet/):
    python scripts/predict.py --run runs/<exp> --split data/splits/pooled.json --subset test \
        --data-root data --out runs/<exp>/preds_test.json [--device cuda] [--batch 8] [--score-thr 0.001]

Loads <run>/best.pt and the config stored in it. ``--nms`` / ``--nms-center-ratio`` override the stored
``decode.nms`` / ``decode.nms_center_ratio`` (P3b re-decode: ``--nms center`` merges duplicate peaks on large cells). Records are in original image px with
``score >= --score-thr`` (default 0.001, for mAP); operating thresholds are applied by
scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.export_json import save_records  # noqa: E402
from dccnet.models import DCCellNet  # noqa: E402
from dccnet.train.loop import PRED_SCORE_THR, predict_records  # noqa: E402


def apply_decode_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override decode.nms / decode.nms_center_ratio from the CLI (None = keep the checkpoint's)."""
    if getattr(args, "nms", None) is not None:
        cfg["decode"]["nms"] = args.nms
    if getattr(args, "nms_center_ratio", None) is not None:
        cfg["decode"]["nms_center_ratio"] = float(args.nms_center_ratio)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--subset", required=True, choices=("train", "val", "test"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--score-thr", type=float, default=PRED_SCORE_THR)
    ap.add_argument("--nms", choices=("none", "circle", "box", "center"), default=None,
                    help="override decode.nms stored in the checkpoint config")
    ap.add_argument("--nms-center-ratio", type=float, default=None, help="override decode.nms_center_ratio (default 0.2)")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    cfg = apply_decode_overrides(cfg, args)
    model = DCCellNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    samples = read_split(args.split, args.data_root)[args.subset]
    recs = predict_records(model, samples, cfg, device, args.data_root, batch=args.batch, score_thr=args.score_thr)
    save_records(recs, args.out)
    d = cfg["decode"]
    print(f"{len(recs)} records for {len(samples)} images ({args.subset}, best epoch {ckpt['epoch']}, "
          f"nms {d.get('nms', 'none')} ratio {d.get('nms_center_ratio', 0.2)}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
