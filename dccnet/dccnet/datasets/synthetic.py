"""Synthetic blood-smear-like images with controlled disc overlap (M2 go/no-go experiment).

Boxes are the circles' bounding squares [cx-r, cy-r, cx+r, cy+r]; classes 0 RBC, 1 WBC, 2 Platelets.
Each image uses its own RNG seeded with ``seed * 100003 + image_index`` so it does not depend on
how many images are generated alongside it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dccnet.types import Sample

RBC, WBC, PLT = 0, 1, 2
ALPHA = 0.85
MAX_TRIES = 50

# RGB colours (converted to BGR only when writing the PNG).
BG_RGB = (235, 220, 225)
RBC_FILL = (232, 150, 150)
RBC_RIM = (205, 105, 115)
RBC_CENTER = (243, 190, 190)
WBC_CYTO = (200, 175, 220)
WBC_NUCLEUS = (95, 60, 140)
PLT_RGB = (110, 70, 150)


def _sample_center(rng: np.random.Generator, r: float, size: int, rbc_circles: list, clump_prob: float):
    """Return (cx, cy) with the whole disc inside the image, or None after MAX_TRIES failures."""
    lo, hi = r + 1.0, size - r - 1.0
    if hi <= lo:
        return None
    for _ in range(MAX_TRIES):
        if rbc_circles and rng.random() < clump_prob:
            ax, ay, _ = rbc_circles[rng.integers(len(rbc_circles))]
            ang = rng.uniform(0.0, 2.0 * np.pi)
            d = rng.uniform(0.6, 1.4) * r
            cx, cy = ax + d * np.cos(ang), ay + d * np.sin(ang)
        else:
            cx, cy = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if lo <= cx <= hi and lo <= cy <= hi:
            return float(cx), float(cy)
    return None


def _draw_cell(img: np.ndarray, rng: np.random.Generator, cls: int, cx: float, cy: float, r: float) -> None:
    """Alpha-blend one cell (in place) onto the float32 RGB canvas."""
    pad = int(np.ceil(r)) + 2
    x0, y0 = int(round(cx)) - pad, int(round(cy)) - pad
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(int(round(cx)) + pad + 1, img.shape[1]), min(int(round(cy)) + pad + 1, img.shape[0])
    lw, lh = x1 - x0, y1 - y0
    layer = np.zeros((lh, lw, 3), np.uint8)
    mask = np.zeros((lh, lw), np.uint8)
    c = (int(round(cx)) - x0, int(round(cy)) - y0)
    ri = max(int(round(r)), 1)
    cv2.circle(mask, c, ri, 1, -1, cv2.LINE_AA)
    if cls == RBC:
        cv2.circle(layer, c, ri, RBC_FILL, -1, cv2.LINE_AA)
        cv2.circle(layer, c, ri, RBC_RIM, 2, cv2.LINE_AA)
        cv2.circle(layer, c, max(int(round(0.4 * r)), 1), RBC_CENTER, -1, cv2.LINE_AA)
    elif cls == WBC:
        cv2.circle(layer, c, ri, WBC_CYTO, -1, cv2.LINE_AA)
        for _ in range(int(rng.integers(2, 4))):
            ox, oy = rng.uniform(-0.3 * r, 0.3 * r, 2)
            ax = max(int(round(rng.uniform(0.3, 0.5) * r)), 1)
            by = max(int(round(rng.uniform(0.25, 0.4) * r)), 1)
            cv2.ellipse(layer, (int(round(c[0] + ox)), int(round(c[1] + oy))), (ax, by),
                        float(rng.uniform(0, 180)), 0, 360, WBC_NUCLEUS, -1, cv2.LINE_AA)
    else:
        cv2.circle(layer, c, ri, PLT_RGB, -1, cv2.LINE_AA)
    a = (ALPHA * mask.astype(np.float32))[..., None]
    roi = img[y0:y1, x0:x1]
    roi[:] = roi * (1.0 - a) + layer.astype(np.float32) * a


def _make_one(rng: np.random.Generator, size: int, clump_prob: float):
    n_rbc = int(rng.integers(20, 61))
    n_wbc = int(rng.integers(0, 3))
    n_plt = int(rng.integers(0, 7))
    specs = ([(RBC, rng.uniform(18, 28)) for _ in range(n_rbc)]
             + [(WBC, rng.uniform(35, 45)) for _ in range(n_wbc)]
             + [(PLT, rng.uniform(5, 9)) for _ in range(n_plt)])
    cells: list[tuple[int, float, float, float]] = []
    rbcs: list[tuple[float, float, float]] = []
    for cls, r in specs:
        ctr = _sample_center(rng, float(r), size, rbcs, clump_prob)
        if ctr is None:
            continue
        cells.append((cls, ctr[0], ctr[1], float(r)))
        if cls == RBC:
            rbcs.append((ctr[0], ctr[1], float(r)))

    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(BG_RGB, np.float32)
    img += rng.normal(0.0, 2.0, img.shape).astype(np.float32)
    for k in rng.permutation(len(cells)):
        _draw_cell(img, rng, *cells[k])
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    labels = np.array([c[0] for c in cells], np.int64)
    boxes = np.array([[x - r, y - r, x + r, y + r] for _, x, y, r in cells], np.float32).reshape(-1, 4)
    return img, boxes, labels


def make_synthetic(out_dir: Path, n_images: int, seed: int, img_size: int = 640,
                   clump_prob: float = 0.5, start_index: int = 0) -> list[Sample]:
    """Generate ``n_images`` PNGs ``syn_{index:05d}.png`` in ``out_dir``; return their Samples."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for index in range(start_index, start_index + n_images):
        rng = np.random.default_rng(seed * 100003 + index)
        img, boxes, labels = _make_one(rng, img_size, clump_prob)
        path = out_dir / f"syn_{index:05d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to write {path}")
        samples.append(Sample(path, "SYN", img_size, img_size, boxes, labels))
    return samples
