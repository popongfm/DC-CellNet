"""Detection quality stratified by GT overlap level (0..3 overlapping neighbours)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dccnet.datasets.geometry import boxes_to_circles, overlap_levels
from dccnet.eval.common import NUM_CLASSES, group_preds
from dccnet.types import Sample

N_BINS = 4


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def overlap_counts(samples: list[Sample], preds: list[dict], score_thr: float, data_root: Path,
                   iou_thr: float = 0.5) -> np.ndarray:
    """Per-image counts, shape (n_images, 4 bins, 3) = [TP, FP, n_gt]. FN = n_gt - TP."""
    grouped = group_preds(samples, preds, data_root)
    out = np.zeros((len(samples), N_BINS, 3), dtype=np.int64)
    for i, (s, p) in enumerate(zip(samples, grouped)):
        gt = np.asarray(s.boxes, dtype=np.float64).reshape(-1, 4)
        lab = np.asarray(s.labels).reshape(-1)
        if len(gt):
            circles, _ = boxes_to_circles(gt, s.width, s.height)
            levels = overlap_levels(circles)
            centers = np.column_stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2])
            for b in levels:
                out[i, b, 2] += 1
        p = p[p[:, 5] >= score_thr]
        for k in range(NUM_CLASSES):
            pk = p[p[:, 0] == k]
            pk = pk[np.argsort(-pk[:, 5], kind="stable")]
            gidx = np.flatnonzero(lab == k) if len(gt) else np.zeros(0, dtype=np.int64)
            matched = np.zeros(len(gidx), dtype=bool)
            for row in pk:
                box = np.array([row[1], row[2], row[1] + row[3], row[2] + row[4]])
                if len(gidx):
                    iou = iou_xyxy(box, gt[gidx])
                    iou[matched] = -1.0
                    j = int(np.argmax(iou))
                    if iou[j] >= iou_thr:
                        matched[j] = True
                        out[i, levels[gidx[j]], 0] += 1
                        continue
                # false positive -> bin of nearest GT center (same class, else any class, else bin 0)
                if len(gt) == 0:
                    b = 0
                else:
                    cand = gidx if len(gidx) else np.arange(len(gt))
                    c = np.array([row[1] + row[3] / 2, row[2] + row[4] / 2])
                    d = np.hypot(centers[cand, 0] - c[0], centers[cand, 1] - c[1])
                    b = int(levels[cand[int(np.argmin(d))]])
                out[i, b, 1] += 1
    return out


def ratios(tp: float, fp: float, n_gt: float) -> dict:
    recall = tp / n_gt if n_gt > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    if recall is None or precision is None:
        f1 = None
    elif recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"recall": recall, "precision": precision, "f1": f1, "n_gt": int(n_gt)}


def overlap_metrics(samples: list[Sample], preds: list[dict], score_thr: float, data_root: Path,
                    iou_thr: float = 0.5) -> dict:
    """``{"0".."3": {"recall","precision","f1","n_gt"}}``; undefined ratios are None."""
    tot = overlap_counts(samples, preds, score_thr, data_root, iou_thr).sum(axis=0)
    return {str(b): ratios(*(float(v) for v in tot[b])) for b in range(N_BINS)}
