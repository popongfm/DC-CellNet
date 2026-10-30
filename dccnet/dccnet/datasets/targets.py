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
  Other ``heatmap_sigma`` rules (all in cells, ``s = stride``, ``D = heatmap_divisor`` (default 4),
  distances between centers in input px, all floored at 0.5):

  - ``adaptive`` (M2d): ``min(r / (3 s), d_nn / (D s))`` with ``d_nn`` the distance to the nearest
    other center of the SAME class (inf if none -> radius rule).
  - ``adaptive_agnostic`` (P3a): as ``adaptive`` but ``d_nn`` over centers of ALL classes.
  - ``knn3`` (P3a): ``min(r / (3 s), mean_knn / (D s))``, ``mean_knn`` = mean distance to the up-to-3
    nearest other same-class centers (fewer if the class has fewer; inf if none -> radius rule).
  - ``mcnn`` (P3a, MCNN/CSRNet geometry-adaptive kernel): ``0.3 * mean_knn / s`` (no radius cap,
    ``D`` unused); ``r / (3 s)`` when the cell has no same-class neighbour.

  Neighbour statistics use exactly the circles passed in (i.e. after augmentation drops).
  ``heatmap_shape="repel"`` (P3a, repel coding of Liang et al.) replaces the Gaussian (``heatmap_sigma``
  and ``heatmap_divisor`` are then ignored): for each class, with ``d1`` / ``d2`` the distances (cells)
  from a cell to the nearest / second-nearest same-class CENTER CELL (``d2 = inf`` if the class has one
  object), value ``1 / (1 + alpha * D')``, ``D' = d1 * (1 + d1 / d2)^2`` (``D' = 0`` where ``d1 = 0``),
  ``alpha = 0.8``, on the support ``{cells within r / s cells (Euclidean) of some object's center cell}``,
  0 elsewhere; peak exactly 1.0 at each center cell, < 1 elsewhere. Overlapping supports combine by max
  (which is the same value, since it depends only on the cell). Invalid ``heatmap_sigma`` /
  ``heatmap_shape`` or a non-positive / non-finite ``heatmap_divisor`` raises ValueError.
  Only the heatmap changes in every variant; density, offsets, radii and masks do not.
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
REPEL_ALPHA = 0.8
MCNN_BETA = 0.3
_KNN = 3


def check_heatmap_sigma(heatmap_sigma: str) -> str:
    if heatmap_sigma not in HEATMAP_SIGMAS:
        raise ValueError(f"heatmap_sigma must be one of {HEATMAP_SIGMAS}, got {heatmap_sigma!r}")
    return heatmap_sigma


def check_heatmap_shape(heatmap_shape: str) -> str:
    if heatmap_shape not in HEATMAP_SHAPES:
        raise ValueError(f"heatmap_shape must be one of {HEATMAP_SHAPES}, got {heatmap_shape!r}")
    return heatmap_shape


def check_heatmap_divisor(heatmap_divisor) -> float:
    if isinstance(heatmap_divisor, bool) or not isinstance(heatmap_divisor, (int, float, np.integer, np.floating)):
        raise ValueError(f"heatmap_divisor must be a positive finite number, got {heatmap_divisor!r}")
    d = float(heatmap_divisor)
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"heatmap_divisor must be a positive finite number, got {heatmap_divisor!r}")
    return d


def _pairwise(p: np.ndarray) -> np.ndarray:
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def _nearest_same_class_dist(circles: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(n,) distance (input px) from each center to the nearest other center of the same class; inf if none."""
    d_nn = np.full(len(circles), np.inf)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        d_nn[idx] = _pairwise(circles[idx, :2]).min(axis=1)
    return d_nn


def _nearest_any_class_dist(circles: np.ndarray) -> np.ndarray:
    """(n,) distance (input px) from each center to the nearest other center of any class; inf if none."""
    if len(circles) < 2:
        return np.full(len(circles), np.inf)
    return _pairwise(circles[:, :2]).min(axis=1)


def _mean_knn_same_class_dist(circles: np.ndarray, labels: np.ndarray, k: int = _KNN) -> np.ndarray:
    """(n,) mean distance (input px) to the up-to-k nearest other same-class centers; inf if none."""
    out = np.full(len(circles), np.inf)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        d = np.sort(_pairwise(circles[idx, :2]), axis=1)[:, : min(k, len(idx) - 1)]
        out[idx] = d.mean(axis=1)
    return out


def heatmap_sigmas(circles: np.ndarray, labels: np.ndarray, stride: int = 4, heatmap_sigma: str = "radius",
                   heatmap_divisor: float = 4.0) -> np.ndarray:
    """(n,) float64 Gaussian heatmap sigma (cells) per circle under ``heatmap_sigma`` (see module docstring)."""
    check_heatmap_sigma(heatmap_sigma)
    D = check_heatmap_divisor(heatmap_divisor)
    circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    n = len(circles)
    if heatmap_sigma == "adaptive":
        nb = _nearest_same_class_dist(circles, labels)
    elif heatmap_sigma == "adaptive_agnostic":
        nb = _nearest_any_class_dist(circles)
    elif heatmap_sigma in ("knn3", "mcnn"):
        nb = _mean_knn_same_class_dist(circles, labels)
    else:
        nb = None
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        r = circles[k, 2]
        # scalar expressions kept in the exact form of commit 42b0203 (bit-identical radius/adaptive targets)
        if nb is None:
            out[k] = max(r / (3.0 * stride), 0.5)
        elif heatmap_sigma == "mcnn":
            s = r / (3.0 * stride) if not np.isfinite(nb[k]) else MCNN_BETA * nb[k] / stride
            out[k] = max(s, 0.5)
        else:
            out[k] = max(min(r / (3.0 * stride), nb[k] / (D * stride)), 0.5)
    return out


def _repel_heatmap(heat: np.ndarray, circles: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                   stride: int) -> None:
    """Fill ``heat`` (C, H, W) in place with the repel code; ``centers`` (n, 2) integer center cells (ix, iy)."""
    _, H, W = heat.shape
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        support = np.zeros((H, W), dtype=bool)
        for k in idx:
            ix, iy = int(centers[k, 0]), int(centers[k, 1])
            rs = circles[k, 2] / stride
            rad = math.ceil(rs)
            y0, y1 = max(iy - rad, 0), min(iy + rad + 1, H)
            x0, x1 = max(ix - rad, 0), min(ix + rad + 1, W)
            ys = np.arange(y0, y1, dtype=np.float64)[:, None]
            xs = np.arange(x0, x1, dtype=np.float64)[None, :]
            support[y0:y1, x0:x1] |= np.sqrt((xs - ix) ** 2 + (ys - iy) ** 2) <= rs
            support[iy, ix] = True
        py, px = np.nonzero(support)
        pts = np.stack([px, py], axis=1).astype(np.float64)
        ctr = centers[idx].astype(np.float64)
        if len(idx) >= 2:
            d, _ = cKDTree(ctr).query(pts, k=2)
            d1, d2 = d[:, 0], d[:, 1]
        else:
            d1 = np.sqrt(((pts - ctr[0]) ** 2).sum(axis=1))
            d2 = np.full_like(d1, np.inf)
        ratio = np.divide(d1, d2, out=np.zeros_like(d1), where=d1 > 0)  # d1 = 0 (center cell) -> D' = 0
        val = 1.0 / (1.0 + REPEL_ALPHA * d1 * (1.0 + ratio) ** 2)
        heat[c, py, px] = val.astype(np.float32)


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
    sig_h = None if repel else heatmap_sigmas(circles, labels, stride, heatmap_sigma, heatmap_divisor)
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
            sigma_h = sig_h[k]
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
        _repel_heatmap(heat, circles, labels, centers, stride)

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
