"""Ground-truth target generation for DC-CellNet (CenterNet/CircleNet style).

Binding conventions (decoding, losses and the dataset rely on these):

* Map size ``H = W = img_size // stride`` (160 for 640 / 4). A circle center
  (cx, cy) in input px maps to feature coords ``u = cx / stride``,
  ``v = cy / stride``. Center cell: ``ix = clip(floor(u), 0, W-1)``,
  ``iy = clip(floor(v), 0, H-1)``; ``offset = (u - ix, v - iy)``;
  ``ind = iy * W + ix``.
* Heatmap: unnormalized Gaussian ``exp(-((x-ix)^2 + (y-iy)^2) / (2 sigma_h^2))``
  evaluated at integer cell coords, peak exactly 1.0 at (iy, ix),
  ``sigma_h = max(r / (3 * stride), 0.5)`` cells, truncated to radius
  ``ceil(3 * sigma_h)``; objects combine by element-wise max per class.
* Density: Gaussian centered at the sub-pixel location (u, v) evaluated at cell
  centers (x + 0.5, y + 0.5), ``sigma_d = max(r / (3 * stride), 0.5)`` cells
  (sigma = r/3 input px, floored at half a cell), truncated to radius
  ``ceil(3 * sigma_d) + 1`` around the center cell, clipped to the map, then
  normalized so each object's mass on the map is exactly 1; summed per class.
* ``log_r = ln(r)`` (r in input px; ``r <= 0`` raises ValueError). ``circle`` is
  (cx, cy, r) in input px. ``r_valid`` is passed through from the input.
* Float tensors are float32; ``ind``/``cls`` int64; ``mask``/``r_valid`` bool.
  ``K = max_objs`` slots are filled in input order; unused slots are zero/False.
  More than ``max_objs`` circles raises ValueError.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def build_targets(
    circles: np.ndarray,
    labels: np.ndarray,
    radius_valid: np.ndarray,
    img_size: int = 640,
    stride: int = 4,
    num_classes: int = 3,
    max_objs: int = 512,
) -> dict[str, torch.Tensor]:
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    radius_valid = np.asarray(radius_valid).reshape(-1).astype(bool)
    n = len(circles)
    if len(labels) != n or len(radius_valid) != n:
        raise ValueError("circles, labels and radius_valid must have the same length")
    if n > max_objs:
        raise ValueError(f"{n} circles exceed max_objs={max_objs}")
    if n and (circles[:, 2] <= 0).any():
        raise ValueError("circle radius must be > 0")
    if n and ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError(f"labels must be in [0, {num_classes})")

    H = W = img_size // stride
    heat = np.zeros((num_classes, H, W), dtype=np.float32)
    dens = np.zeros((num_classes, H, W), dtype=np.float32)
    ind = np.zeros(max_objs, dtype=np.int64)
    mask = np.zeros(max_objs, dtype=bool)
    log_r = np.zeros(max_objs, dtype=np.float32)
    offset = np.zeros((max_objs, 2), dtype=np.float32)
    r_valid = np.zeros(max_objs, dtype=bool)
    cls = np.zeros(max_objs, dtype=np.int64)
    circle = np.zeros((max_objs, 3), dtype=np.float32)

    for k in range(n):
        cx, cy, r = circles[k]
        c = int(labels[k])
        u, v = cx / stride, cy / stride
        ix = int(min(max(math.floor(u), 0), W - 1))
        iy = int(min(max(math.floor(v), 0), H - 1))
        sigma = max(r / (3.0 * stride), 0.5)

        # heatmap: integer cell coords, centered on the center cell
        rad = math.ceil(3.0 * sigma)
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None]
        xs = np.arange(x0, x1, dtype=np.float64)[None, :]
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma**2))
        patch = heat[c, y0:y1, x0:x1]
        np.maximum(patch, g.astype(np.float32), out=patch)

        # density: sub-pixel center, evaluated at cell centers, renormalized
        rad = math.ceil(3.0 * sigma) + 1
        y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
        x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        g = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2.0 * sigma**2))
        s = g.sum()
        if s > 0:
            dens[c, y0:y1, x0:x1] += (g / s).astype(np.float32)

        ind[k] = iy * W + ix
        mask[k] = True
        log_r[k] = math.log(r)
        offset[k] = (u - ix, v - iy)
        r_valid[k] = radius_valid[k]
        cls[k] = c
        circle[k] = (cx, cy, r)

    return {
        "heatmap": torch.from_numpy(heat),
        "density": torch.from_numpy(dens),
        "ind": torch.from_numpy(ind),
        "mask": torch.from_numpy(mask),
        "log_r": torch.from_numpy(log_r),
        "offset": torch.from_numpy(offset),
        "r_valid": torch.from_numpy(r_valid),
        "cls": torch.from_numpy(cls),
        "circle": torch.from_numpy(circle),
    }
