"""Per-class counting error."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dccnet.eval.common import CLASS_NAMES, NUM_CLASSES, group_preds
from dccnet.types import Sample


def count_errors(samples: list[Sample], preds: list[dict], score_thr: float, data_root: Path) -> dict:
    """Signed per-image count error (#preds with score >= thr) - (#GT), per class.

    Returns ``{"per_image": {cls: [int]}, "MAE": {cls: float}, "rel": {cls: float | None}}`` where
    MAE is the mean |err| over images and rel = sum|err| / sum GT (None if sum GT = 0).
    """
    grouped = group_preds(samples, preds, data_root)
    per_image = {c: [] for c in CLASS_NAMES}
    gt_total = {c: 0 for c in CLASS_NAMES}
    for s, p in zip(samples, grouped):
        kept = p[p[:, 5] >= score_thr]
        for k, name in enumerate(CLASS_NAMES):
            n_gt = int((np.asarray(s.labels) == k).sum())
            n_pred = int((kept[:, 0] == k).sum())
            per_image[name].append(n_pred - n_gt)
            gt_total[name] += n_gt
    mae, rel = {}, {}
    for name in CLASS_NAMES:
        a = np.abs(np.asarray(per_image[name], dtype=np.float64))
        mae[name] = float(a.mean()) if len(a) else None
        rel[name] = float(a.sum() / gt_total[name]) if gt_total[name] > 0 else None
    return {"per_image": per_image, "MAE": mae, "rel": rel}
