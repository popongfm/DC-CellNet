"""Prediction-record I/O and image identity for the unified evaluator.

A prediction record is ``{"image": str, "category_id": int, "bbox": [x, y, w, h], "score": float,
"circle": [cx, cy, r] (optional)}`` in original image pixels. ``image`` is the image path relative to
``data_root`` with forward slashes, exactly what ``write_split`` stores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dccnet.types import Sample

REQUIRED_KEYS = ("image", "category_id", "bbox", "score")


def image_key(sample: Sample, data_root: Path) -> str:
    """Image identity: path relative to ``data_root``, POSIX slashes."""
    return Path(sample.image_path).relative_to(Path(data_root)).as_posix()


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o).__name__}")


def save_records(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), default=_default))


def load_records(path: Path) -> list[dict]:
    recs = json.loads(Path(path).read_text())
    if not isinstance(recs, list):
        raise ValueError(f"{path}: expected a JSON list of prediction records")
    for i, r in enumerate(recs):
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(f"{path}: record {i} is missing keys {missing}")
    return recs
