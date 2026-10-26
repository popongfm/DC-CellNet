"""Unified evaluator: scores a prediction file against one split subset.

Usage:
    python scripts/evaluate.py --split data/splits/pooled.json --subset test --preds preds.json \
        --score-thr 0.3 --data-root data --out summary.json [--map-ci-n 1000]

    # pick the score threshold on val (micro-F1 at IoU 0.5, thr 0.05..0.95); --score-thr is ignored
    python scripts/evaluate.py --split ... --subset val --preds preds_val.json --data-root data \
        --out sweep_val.json --sweep-thr
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dccnet.datasets.splits import read_split  # noqa: E402
from dccnet.eval.bootstrap import bootstrap_ci  # noqa: E402
from dccnet.eval.coco_eval import coco_map, coco_map_ci  # noqa: E402
from dccnet.eval.common import CLASS_NAMES  # noqa: E402
from dccnet.eval.counting import count_errors  # noqa: E402
from dccnet.eval.export_json import load_records  # noqa: E402
from dccnet.eval.overlap_bins import N_BINS, overlap_counts, overlap_metrics  # noqa: E402


def _clean(o):
    """JSON-safe: NaN -> None, tuples/ndarrays -> lists, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _ratio(v):
    return v[:, 0].sum() / v[:, 1].sum() if v[:, 1].sum() > 0 else float("nan")


def evaluate(samples, preds, score_thr, data_root, map_ci_n=1000, seed=0) -> dict:
    cm = coco_map(samples, preds, data_root)
    ce = count_errors(samples, preds, score_thr, data_root)
    mae_ci = {}
    for c in CLASS_NAMES:
        mae_ci[c] = bootstrap_ci(np.abs(np.asarray(ce["per_image"][c], dtype=np.float64)), seed=seed)
    counts = overlap_counts(samples, preds, score_thr, data_root)  # (images, bins, [TP, FP, n_gt])
    recall_ci = {}
    for b in range(N_BINS):
        per_img = counts[:, b, [0, 2]].astype(np.float64)  # (tp, n_gt) per image
        recall_ci[str(b)] = bootstrap_ci(per_img, stat=_ratio, seed=seed) if per_img[:, 1].sum() > 0 else None
    return {
        "n_images": len(samples),
        "score_thr": score_thr,
        "coco": cm,
        "coco_ci": coco_map_ci(samples, preds, data_root, n=map_ci_n, seed=seed) if map_ci_n > 0 else None,
        "counting": {"MAE": ce["MAE"], "rel": ce["rel"], "MAE_ci": mae_ci},
        "overlap": overlap_metrics(samples, preds, score_thr, data_root),
        "recall_ci_by_bin": recall_ci,
    }


def sweep_thresholds(samples, preds, data_root) -> dict:
    """Micro-F1 (TP/FP/n_gt summed over all bins, IoU 0.5) for thr 0.05..0.95 step 0.05; ties -> lowest thr."""
    thrs = [round(0.05 * k, 2) for k in range(1, 20)]
    sweep = []
    for t in thrs:
        tp, fp, n_gt = overlap_counts(samples, preds, t, data_root).sum(axis=(0, 1)).astype(float)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        sweep.append({"thr": t, "precision": precision, "recall": recall, "f1": f1})
    best = max(sweep, key=lambda e: e["f1"])  # max returns the first maximal element
    return {"sweep": sweep, "best_thr": best["thr"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--score-thr", type=float, help="required unless --sweep-thr")
    ap.add_argument("--sweep-thr", action="store_true",
                    help="ignore --score-thr; write the micro-F1 threshold sweep and best_thr to --out")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map-ci-n", type=int, default=1000, help="bootstrap resamples for the mAP CI; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    split = read_split(args.split, args.data_root)
    if args.subset not in split:
        ap.error(f"subset {args.subset!r} not in split (has {sorted(split)})")
    samples = split[args.subset]
    if args.sweep_thr:
        res = sweep_thresholds(samples, load_records(args.preds), args.data_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_clean(res), indent=2))
        best = next(e for e in res["sweep"] if e["thr"] == res["best_thr"])
        print(f"best_thr={res['best_thr']:.2f} F1={best['f1']:.4f} "
              f"P={best['precision']:.4f} R={best['recall']:.4f}; wrote {args.out}")
        return 0
    if args.score_thr is None:
        ap.error("--score-thr is required unless --sweep-thr is given")
    summary = {"split": str(args.split), "subset": args.subset,
               **evaluate(samples, load_records(args.preds), args.score_thr, args.data_root,
                          args.map_ci_n, args.seed)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_clean(summary), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
