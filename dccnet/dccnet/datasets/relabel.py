"""BCCD relabel helpers: merge GT with confident predictions; YOLO text <-> pixel boxes."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from dccnet.eval.overlap_bins import iou_xyxy

NUM_CLASSES = 3


def merge_labels(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, conf: float = 0.5,
                 dedupe_iou: float = 0.7, add_iou: float = 0.3):
    """Dedupe same-class GT (IoU > dedupe_iou vs an earlier kept box), then add confident predictions.

    A prediction (score >= conf, descending score order) is added iff its IoU with every kept GT box (any class)
    and every already-added prediction is < add_iou. Returns (boxes xyxy float32, labels int64,
    origin int8 [0 GT, 1 added], stats).
    """
    gtb = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gtl = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    keep: list[int] = []
    for i in range(len(gtb)):
        same = [j for j in keep if gtl[j] == gtl[i]]
        if same and iou_xyxy(gtb[i], gtb[same]).max() > dedupe_iou:
            continue
        keep.append(i)
    boxes = [gtb[i] for i in keep]
    labels = [int(gtl[i]) for i in keep]
    n_gt_kept = len(keep)

    pb = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pl = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    ps = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    added = 0
    for i in np.argsort(-ps, kind="stable"):
        if ps[i] < conf:
            break
        if boxes and iou_xyxy(pb[i], np.stack(boxes)).max() >= add_iou:
            continue
        boxes.append(pb[i])
        labels.append(int(pl[i]))
        added += 1
    out_b = np.stack(boxes).astype(np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
    origin = np.concatenate([np.zeros(n_gt_kept), np.ones(added)]).astype(np.int8)
    stats = {"n_gt": int(len(gtb)), "n_dupes_removed": int(len(gtb) - n_gt_kept), "n_added": int(added),
             "n_final": int(len(boxes))}
    return out_b, np.asarray(labels, dtype=np.int64), origin, stats


def to_yolo_lines(boxes, labels, width: int, height: int) -> list[str]:
    out = []
    for (x1, y1, x2, y2), c in zip(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(labels).reshape(-1)):
        out.append(f"{int(c)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                   f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return out


def from_yolo_lines(lines, width: int, height: int):
    """Parse YOLO lines to (boxes xyxy float32, labels int64); ValueError names the offending line."""
    boxes, labels = [], []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError("expected 5 fields")
            if not parts[0].lstrip("-").isdigit():
                raise ValueError("class is not an integer")
            c = int(parts[0])
            if not 0 <= c < NUM_CLASSES:
                raise ValueError(f"class {c} not in 0..{NUM_CLASSES - 1}")
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError("non-finite value")
            if w <= 0 or h <= 0:
                raise ValueError("w and h must be > 0")
            if any(v < -1e-6 or v > 1 + 1e-6 for v in (cx, cy, w, h)):
                raise ValueError("cx/cy/w/h must be within [0, 1]")
        except ValueError as e:
            raise ValueError(f"bad YOLO line {ln!r}: {e}") from None
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
        labels.append(c)
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64))


def normalize_label_text(text: str) -> str:
    """Newline-normalise (CRLF / CR -> LF) and make a non-empty text end with exactly the C1 trailing newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def compare_label_sets(old_boxes, old_labels, new_boxes, new_labels, iou: float = 0.5) -> dict:
    """Greedy same-class matching by IoU descending (IoU >= iou). Returns counts kept / added / removed."""
    ob = np.asarray(old_boxes, dtype=np.float32).reshape(-1, 4)
    ol = np.asarray(old_labels, dtype=np.int64).reshape(-1)
    nb = np.asarray(new_boxes, dtype=np.float32).reshape(-1, 4)
    nl = np.asarray(new_labels, dtype=np.int64).reshape(-1)
    pairs = []
    for i in range(len(ob)):
        if len(nb) == 0:
            break
        ious = iou_xyxy(ob[i], nb)
        for j in range(len(nb)):
            if ol[i] == nl[j] and ious[j] >= iou:
                pairs.append((float(ious[j]), i, j))
    pairs.sort(key=lambda t: -t[0])
    used_o, used_n = set(), set()
    for _, i, j in pairs:
        if i not in used_o and j not in used_n:
            used_o.add(i)
            used_n.add(j)
    kept = len(used_o)
    return {"kept": kept, "added": int(len(nb) - kept), "removed": int(len(ob) - kept)}


def label_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
