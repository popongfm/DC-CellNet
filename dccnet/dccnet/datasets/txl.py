"""TXL-PBC loader.

Actual layout (inspected 2026-10-26):
    <root>/images/{train,val,test}/<md5>.png     official split = folder an image sits in
    <root>/labels/{train,val,test}/<md5>.txt     YOLO: "cls xc yc w h" normalized; may be empty
    <root>/classes.txt   lines "<idx> <name>": 0 White Blood Cell (WBC), 1 Red Blood Cell (RBC), 2 Platelet
                         (NOTE: TXL order is WBC,RBC,Platelet; ours is RBC=0,WBC=1,Platelets=2 -> mapped by name)
    <root>/data.yaml     names: ['WBC','RBC','Platelets'] (same order as classes.txt)
    <root>/metadata_file.xlsx   sheet1, columns: Original_Image_Name, Source_Dataset, New_Image_Name, Source_URL
                         New_Image_Name = image file name in images/; Source_Dataset in {PBC, Raabin-WBC, BCCD, BCDD}
    <root>/BCCD_selection.xlsx  (not used here)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dccnet.types import Sample

SPLITS = ("train", "val", "test")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _class_index(name: str) -> int:
    n = name.lower()
    if "wbc" in n or "white" in n:
        return 1
    if "rbc" in n or "red" in n:
        return 0
    if "platelet" in n:
        return 2
    raise ValueError(f"Unknown TXL class name: {name!r}")


def _read_class_map(root: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    lines = [ln.strip() for ln in (root / "classes.txt").read_text().splitlines() if ln.strip()]
    for pos, ln in enumerate(lines):
        m = re.match(r"^(\d+)\s+(.*)$", ln)
        idx, name = (int(m.group(1)), m.group(2)) if m else (pos, ln)
        mapping[idx] = _class_index(name)
    return mapping


def _norm_source(raw: str) -> str:
    r = str(raw).strip().lower()
    for key, val in (("raabin", "Raabin"), ("pbc", "PBC"), ("bccd", "BCCD"), ("bcdd", "BCDD")):
        if r.startswith(key):
            return val
    raise ValueError(f"Unrecognized TXL source name: {raw!r}")


def _read_labels(path: Path, cmap: dict[int, int], w: int, h: int):
    boxes, labels = [], []
    if path.exists():
        for ln in path.read_text().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
            labels.append(cmap[c])
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64))


def load_txl(root: Path) -> tuple[list[Sample], dict[str, str]]:
    root = Path(root)
    cmap = _read_class_map(root)
    meta = pd.read_excel(root / "metadata_file.xlsx")
    src_by_stem = {Path(str(n)).stem: _norm_source(s)
                   for n, s in zip(meta["New_Image_Name"], meta["Source_Dataset"])}
    samples: list[Sample] = []
    official: dict[str, str] = {}
    unmapped: list[str] = []
    for split in SPLITS:
        d = root / "images" / split
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if img.stem not in src_by_stem:
                unmapped.append(img.stem)
                continue
            with Image.open(img) as im:
                w, h = im.size
            boxes, labels = _read_labels(root / "labels" / split / f"{img.stem}.txt", cmap, w, h)
            samples.append(Sample(img, src_by_stem[img.stem], w, h, boxes, labels))
            official[img.stem] = split
    if unmapped:
        raise ValueError(f"{len(unmapped)} images have no source in metadata_file.xlsx; "
                         f"first stems: {unmapped[:10]}")
    return samples, official
