"""Shenggan BCCD_Dataset (VOC) loader. Root = the `BCCD` folder containing Annotations/ and JPEGImages/."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from dccnet.types import Sample

log = logging.getLogger(__name__)
_CLASSES = {"rbc": 0, "wbc": 1, "platelets": 2}


def _box(obj: ET.Element, w: int, h: int):
    name = (obj.findtext("name") or "").strip().lower()
    bb = obj.find("bndbox")
    if name not in _CLASSES or bb is None:
        return None
    try:
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(w)), min(y1, float(h))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1], _CLASSES[name]


def load_bccd_voc_with_stats(root: Path) -> tuple[list[Sample], dict[str, int]]:
    root = Path(root)
    samples: list[Sample] = []
    stats = {"images": 0, "skipped_boxes": 0}
    for xml in sorted((root / "Annotations").glob("*.xml")):
        tree = ET.parse(xml).getroot()
        fname = tree.findtext("filename") or f"{xml.stem}.jpg"
        img = root / "JPEGImages" / fname
        if not img.exists():
            img = root / "JPEGImages" / f"{xml.stem}.jpg"
        with Image.open(img) as im:
            w, h = im.size
        boxes, labels = [], []
        for obj in tree.findall("object"):
            r = _box(obj, w, h)
            if r is None:
                stats["skipped_boxes"] += 1
                log.warning("%s: skipped invalid box", xml.name)
                continue
            boxes.append(r[0])
            labels.append(r[1])
        samples.append(Sample(img, "BCCD_full", w, h,
                              np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                              np.asarray(labels, dtype=np.int64)))
        stats["images"] += 1
    return samples, stats


def load_bccd_voc(root: Path) -> list[Sample]:
    samples, stats = load_bccd_voc_with_stats(root)
    if stats["skipped_boxes"]:
        log.warning("BCCD: skipped %d invalid boxes", stats["skipped_boxes"])
    return samples
