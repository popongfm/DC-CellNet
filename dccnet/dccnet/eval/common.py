"""Shared helpers: class table, image/prediction grouping with validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dccnet.eval.export_json import image_key
from dccnet.types import Sample

CLASS_NAMES = ("RBC", "WBC", "Platelets")
NUM_CLASSES = len(CLASS_NAMES)


def sample_keys(samples: list[Sample], data_root: Path) -> list[str]:
    keys = [image_key(s, data_root) for s in samples]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})[:5]
        raise ValueError(f"Duplicate images in samples: {dup}")
    return keys


def group_preds(samples: list[Sample], preds: list[dict], data_root: Path) -> list[np.ndarray]:
    """Per-sample prediction arrays (M, 6) = [cat, x, y, w, h, score], in samples order."""
    keys = sample_keys(samples, data_root)
    index = {k: i for i, k in enumerate(keys)}
    unknown = []
    rows: list[list[list[float]]] = [[] for _ in samples]
    for p in preds:
        i = index.get(p["image"])
        if i is None:
            if p["image"] not in unknown:
                unknown.append(p["image"])
            continue
        cat = int(p["category_id"])
        if not 0 <= cat < NUM_CLASSES:
            raise ValueError(f"category_id {cat} outside 0..{NUM_CLASSES - 1} (image {p['image']})")
        x, y, w, h = (float(v) for v in p["bbox"])
        rows[i].append([cat, x, y, w, h, float(p["score"])])
    if unknown:
        raise ValueError(f"{len(unknown)} prediction image(s) not in samples; first up to 5: {unknown[:5]}")
    return [np.asarray(r, dtype=np.float64).reshape(-1, 6) for r in rows]


def gt_arrays(samples: list[Sample]) -> list[np.ndarray]:
    """Per-sample GT arrays (N, 5) = [cat, x, y, w, h] (xywh)."""
    out = []
    for s in samples:
        b = np.asarray(s.boxes, dtype=np.float64).reshape(-1, 4)
        l = np.asarray(s.labels).reshape(-1)
        out.append(np.column_stack([l, b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
                   if len(b) else np.zeros((0, 5)))
    return out
