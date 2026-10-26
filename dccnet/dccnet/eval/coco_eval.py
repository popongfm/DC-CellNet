"""COCO-style mAP via pycocotools (bbox), plus an image-resampling bootstrap of the mAP."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dccnet.eval.common import CLASS_NAMES, NUM_CLASSES, gt_arrays, group_preds, sample_keys
from dccnet.types import Sample

MAX_DETS = [1, 100, 1000]
_AREA_ALL = [[0.0, 1e10]]


def _coco_dicts(gts: list[np.ndarray], dts: list[np.ndarray], sizes: list[tuple[int, int]]):
    """Build COCO GT/detection dicts; image id = position in the list (so duplicates stay distinct)."""
    images, anns, dets = [], [], []
    for i, ((w, h), g, d) in enumerate(zip(sizes, gts, dts)):
        images.append({"id": i, "width": int(w), "height": int(h)})
        for c, x, y, bw, bh in g:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c),
                         "bbox": [float(x), float(y), float(bw), float(bh)],
                         "area": float(bw * bh), "iscrowd": 0})
        for c, x, y, bw, bh, sc in d:
            dets.append({"image_id": i, "category_id": int(c),
                         "bbox": [float(x), float(y), float(bw), float(bh)], "score": float(sc)})
    cats = [{"id": k, "name": n} for k, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}, dets


def build_coco_gt(samples: list[Sample], data_root: Path) -> dict:
    """COCO GT dict: category ids = our class ids, area = w*h, iscrowd = 0, bbox = xywh."""
    sample_keys(samples, data_root)  # validates uniqueness
    gt, _ = _coco_dicts(gt_arrays(samples), [np.zeros((0, 6))] * len(samples),
                        [(s.width, s.height) for s in samples])
    for im, s in zip(gt["images"], samples):
        im["file_name"] = Path(s.image_path).name
    return gt


def _eval_lists(gts, dts, sizes) -> tuple[list[float | None], list[float | None]]:
    """(AP50, AP50_95) per class from per-image arrays; None for classes without GT."""
    has_gt = [any(bool((g[:, 0] == k).any()) for g in gts) if gts else False for k in range(NUM_CLASSES)]
    ap50: list[float | None] = [0.0 if h else None for h in has_gt]
    ap5095: list[float | None] = [0.0 if h else None for h in has_gt]
    if not any(len(d) for d in dts) or not any(has_gt):
        return ap50, ap5095
    gt_dict, det_list = _coco_dicts(gts, dts, sizes)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(det_list)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = list(MAX_DETS)
        ev.params.areaRng = list(_AREA_ALL)
        ev.params.areaRngLbl = ["all"]
        ev.params.catIds = list(range(NUM_CLASSES))
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [T, R, K, A, M]; A index 0 = "all", M index 2 = maxDets 1000
    assert ev.params.maxDets[2] == 1000 and ev.params.iouThrs[0] == 0.5
    for k in range(NUM_CLASSES):
        if not has_gt[k]:
            continue
        p = prec[:, :, k, 0, 2]
        p50 = p[0][p[0] > -1]
        pall = p[p > -1]
        ap50[k] = float(p50.mean()) if p50.size else 0.0
        ap5095[k] = float(pall.mean()) if pall.size else 0.0
    return ap50, ap5095


def _mean(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def coco_map(samples: list[Sample], preds: list[dict], data_root: Path) -> dict:
    """``{"mAP50", "mAP50_95", "per_class" (AP50), "per_class_50_95"}``.

    Means are over classes with at least one GT box in ``samples``; others are None.
    """
    dts = group_preds(samples, preds, data_root)
    ap50, ap5095 = _eval_lists(gt_arrays(samples), dts, [(s.width, s.height) for s in samples])
    return {"mAP50": _mean(ap50), "mAP50_95": _mean(ap5095),
            "per_class": dict(zip(CLASS_NAMES, ap50)),
            "per_class_50_95": dict(zip(CLASS_NAMES, ap5095))}


def coco_map_ci(samples: list[Sample], preds: list[dict], data_root: Path, n: int = 1000,
                seed: int = 0) -> dict:
    """Percentile 95% CI of mAP50 / mAP50_95 by resampling images with replacement.

    Duplicated images get fresh ids. Within a resample the mean is over classes with GT in that
    resample.
    """
    gts = gt_arrays(samples)
    dts = group_preds(samples, preds, data_root)
    sizes = [(s.width, s.height) for s in samples]
    rng = np.random.default_rng(seed)
    d50, d5095 = [], []
    for _ in range(n):
        idx = rng.integers(0, len(samples), size=len(samples))
        a50, a5095 = _eval_lists([gts[i] for i in idx], [dts[i] for i in idx], [sizes[i] for i in idx])
        d50.append(_mean(a50))
        d5095.append(_mean(a5095))

    def ci(d):
        x = np.array([np.nan if v is None else v for v in d], dtype=np.float64)
        if np.all(np.isnan(x)):
            return (float("nan"), float("nan"))
        lo, hi = np.nanpercentile(x, [2.5, 97.5])
        return (float(lo), float(hi))

    return {"mAP50": ci(d50), "mAP50_95": ci(d5095)}
