"""Write evaluator prediction records for one split subset from a trained Faster R-CNN run.

Usage (from dccnet/), same shape as scripts/predict.py:
    python scripts/baselines/frcnn_predict.py --run runs/p3b_pooled_frcnn_s0 --split data/splits/pooled.json \
        --subset test --data-root data --out runs/p3b_pooled_frcnn_s0/preds_test.json [--device cuda] [--batch 4]

--split is used as given (not relative to --data-root). Loads <run>/best.pt and rebuilds the model with the
min/max size stored in it. Records are in original image px with score >= 0.001.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import frcnn_common as fc  # noqa: E402
from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.export_json import save_records  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--subset", required=True, choices=("train", "val", "test"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    a = ckpt["args"]
    model = fc.build_frcnn(pretrained=False, min_size=int(a["min_size"]), max_size=int(a["max_size"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    samples = read_split(args.split, args.data_root)[args.subset]
    recs = fc.predict_split(model, samples, args.data_root, device, batch=args.batch)
    save_records(recs, args.out)
    print(f"{len(recs)} records for {len(samples)} images ({args.subset}, best epoch {ckpt['epoch']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
