"""Ground-truth target generation for DC-CellNet (CenterNet/CircleNet style).

Binding conventions (decoding, losses and the dataset rely on these):

* Map size ``H = W = img_size // stride`` (160 for 640 / 4, 320 for 640 / 2). A circle center
  (cx, cy) in input px maps to feature coords ``u = cx / stride``,
  ``v = cy / stride``. Center cell: ``ix = clip(floor(u), 0, W-1)``,
  ``iy = clip(floor(v), 0, H-1)``; ``offset = (u - ix, v - iy)``;
  ``ind = iy * W + ix``.
* Heatmap: unnormalized Gaussian ``exp(-((x-ix)^2 + (y-iy)^2) / (2 sigma_h^2))``
  evaluated at integer cell coords, peak exactly 1.0 at (iy, ix),
  ``sigma_h = max(r / (3 * stride), 0.5)`` cells (``heatmap_sigma="radius"``, default),
  truncated to radius ``ceil(3 * sigma_h)``; objects combine by element-wise max per class.
  ``heatmap_sigma="adaptive"`` (M2d) shrinks sigma_h by the nearest SAME-class neighbour:
  ``d_nn`` = Euclidean distance (input px) from the cell's center to the nearest other center of
  the same class in the image (inf if none), ``sigma_h = max(min(r / (3 * stride),
  d_nn / (D * stride)), 0.5)`` cells with divisor ``D = heatmap_divisor`` (default 4).
  P3a rules (all distances in input px between centers present in THIS call, i.e. post-augmentation;
  "neighbour" = another center; all sigma floored at 0.5 cells):
  ``adaptive_agnostic``: as ``adaptive`` but d_nn over centers of ALL classes;
  ``knn3``: ``max(min(r / (3s), m3 / (D * s)), 0.5)``, m3 = mean distance to the <= 3 nearest
  same-class neighbours (fewer -> those available; none -> inf, i.e. r / (3s));
  ``mcnn``: ``max(0.3 * m3 / s, 0.5)``, no neighbour -> ``max(r / (3s), 0.5)`` (D ignored).
  ``radius`` ignores D as well. Only the heatmap changes; the density sigma below does not.
  Any other ``heatmap_sigma`` raises ValueError; ``heatmap_divisor`` must be finite and > 0.
* ``heatmap_shape="repel"`` (P3a; default ``"gaussian"``) replaces the Gaussian (the sigma rule is ignored):
  for each cell of class c, at integer cells (x, y) with Euclidean distance <= r / s (cells) from its
  center cell (ix, iy): d1 / d2 = distance (cells) from (x, y) to the nearest / second-nearest same-class
  center cell (d2 = inf if the class has one center), ``D' = d1 * (1 + d1 / d2)^2`` (0 if d1 = d2 = 0),
  value ``1 / (1 + REPEL_ALPHA * D')`` with ``REPEL_ALPHA = 0.8``; peak exactly 1.0 at each center cell,
  < 1 elsewhere; combined per class by max (zero outside every support disk).
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
from scipy.spatial import cKDTree

HEATMAP_SIGMAS = ("radius", "adaptive", "adaptive_agnostic", "knn3", "mcnn")
HEATMAP_SHAPES = ("gaussian", "repel")
REPEL_ALPHA = 0.8  # proximity-map alpha (Xie et al. / Liang et al.)
_KNN = 3
_MCNN_BETA = 0.3


def check_heatmap_sigma(heatmap_sigma: str) -> str:
    if heatmap_sigma not in HEATMAP_SIGMAS:
        raise ValueError(f"heatmap_sigma must be one of {HEATMAP_SIGMAS}, got {heatmap_sigma!r}")
    return heatmap_sigma


def check_heatmap_divisor(heatmap_divisor: float) -> float:
    try:
        d = float(heatmap_divisor)
    except (TypeError, ValueError):
        raise ValueError(f"heatmap_divisor must be a finite number > 0, got {heatmap_divisor!r}") from None
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"heatmap_divisor must be a finite number > 0, got {heatmap_divisor!r}")
    return d


def check_heatmap_shape(heatmap_shape: str) -> str:
    if heatmap_shape not in HEATMAP_SHAPES:
        raise ValueError(f"heatmap_shape must be one of {HEATMAP_SHAPES}, got {heatmap_shape!r}")
    return heatmap_shape


def _neighbour_dists(circles: np.ndarray, labels: np.ndarray | None, k: int) -> np.ndarray:
    """(n, k) sorted distances (input px) from each center to its k nearest other centers (same class if
    ``labels`` is given, any class if None); missing neighbours are inf."""
    n = len(circles)
    out = np.full((n, k), np.inf)
    groups = [np.arange(n)] if labels is None else [np.flatnonzero(labels == c) for c in np.unique(labels)]
    for idx in groups:
        if len(idx) < 2:
            continue
        p = circles[idx, :2]
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        m = min(k, len(idx) - 1)
        out[idx, :m] = np.sort(d, axis=1)[:, :m]
    return out


def heatmap_sigmas(circles: np.ndarray, labels: np.ndarray, stride: int, heatmap_sigma: str = "radius",
                   heatmap_divisor: float = 4.0) -> np.ndarray:
    """(n,) heatmap Gaussian sigma (cells) per circle under ``heatmap_sigma`` (see module docstring)."""
    check_heatmap_sigma(heatmap_sigma)
    D = check_heatmap_divisor(heatmap_divisor)
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    base = circles[:, 2] / (3.0 * stride)
    if heatmap_sigma == "radius":
        sig = base
    elif heatmap_sigma in ("adaptive", "adaptive_agnostic"):
        d_nn = _neighbour_dists(circles, labels if heatmap_sigma == "adaptive" else None, 1)[:, 0]
        sig = np.minimum(base, d_nn / (D * stride))
    else:  # knn3 / mcnn: mean distance to the <= 3 nearest same-class neighbours
        d = _neighbour_dists(circles, labels, _KNN)
        cnt = np.isfinite(d).sum(axis=1)
        tot = np.where(np.isfinite(d), d, 0.0).sum(axis=1)
        mean = np.divide(tot, cnt, out=np.full(len(circles), np.inf), where=cnt > 0)
        if heatmap_sigma == "knn3":
            sig = np.minimum(base, mean / (D * stride))
        else:
            sig = np.where(cnt > 0, _MCNN_BETA * mean / stride, base)  # mean = inf when cnt = 0
    return np.maximum(sig, 0.5)


def _repel_heatmap(heat: np.ndarray, centers: np.ndarray, labels: np.ndarray, support: np.ndarray) -> None:
    """Fill ``heat`` (C, H, W) in place with the repel map; ``centers`` (n, 2) integer (ix, iy) cells,
    ``support`` (n,) radius in cells."""
    _, H, W = heat.shape
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        sel = np.zeros((H, W), dtype=bool)
        for k in idx:
            ix, iy = int(centers[k, 0]), int(centers[k, 1])
            R = float(support[k])
            rad = int(math.floor(R))
            y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
            x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
            ys = np.arange(y0, y1, dtype=np.float64)[:, None]
            xs = np.arange(x0, x1, dtype=np.float64)[None, :]
            sel[y0:y1, x0:x1] |= (xs - ix) ** 2 + (ys - iy) ** 2 <= R * R
        yy, xx = np.nonzero(sel)
        q = np.stack([xx, yy], axis=1).astype(np.float64)
        tree = cKDTree(centers[idx].astype(np.float64))
        d, _ = tree.query(q, k=2)  # exact 2 nearest centers; d2 = inf when the class has one center
        d1, d2 = d[:, 0], d[:, 1]
        ratio = np.divide(d1, d2, out=np.zeros_like(d1), where=d2 > 0)  # d2 = 0 implies d1 = 0
        val = 1.0 / (1.0 + REPEL_ALPHA * (d1 * (1.0 + ratio) ** 2))
        heat[c, yy, xx] = np.maximum(heat[c, yy, xx], val.astype(np.float32))


def build_targets(
    circles: np.ndarray,
    labels: np.ndarray,
    radius_valid: np.ndarray,
    img_size: int = 640,
    stride: int = 4,
    num_classes: int = 3,
    max_objs: int = 512,
    heatmap_sigma: str = "radius",
    heatmap_divisor: float = 4.0,
    heatmap_shape: str = "gaussian",
) -> dict[str, torch.Tensor]:
    check_heatmap_sigma(heatmap_sigma)
    check_heatmap_divisor(heatmap_divisor)
    check_heatmap_shape(heatmap_shape)
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
    repel = heatmap_shape == "repel"
    sigma_hs = None if repel else heatmap_sigmas(circles, labels, stride, heatmap_sigma, heatmap_divisor)
    centers = np.zeros((n, 2), dtype=np.int64)

    for k in range(n):
        cx, cy, r = circles[k]
        c = int(labels[k])
        u, v = cx / stride, cy / stride
        ix = int(min(max(math.floor(u), 0), W - 1))
        iy = int(min(max(math.floor(v), 0), H - 1))
        sigma = max(r / (3.0 * stride), 0.5)
        centers[k] = (ix, iy)

        if not repel:
            # heatmap: integer cell coords, centered on the center cell
            sigma_h = float(sigma_hs[k])
            rad = math.ceil(3.0 * sigma_h)
            y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
            x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
            ys = np.arange(y0, y1, dtype=np.float64)[:, None]
            xs = np.arange(x0, x1, dtype=np.float64)[None, :]
            g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma_h**2))
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

    if repel and n:
        _repel_heatmap(heat, centers, labels, circles[:, 2] / stride)

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
