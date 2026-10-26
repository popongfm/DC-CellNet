"""Circle geometry: box<->circle conversion, overlap levels, differentiable circle IoU."""
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-7


def boxes_to_circles(
    boxes: np.ndarray, img_w: int, img_h: int, edge_px: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """xyxy boxes -> (cx, cy, r) circles with r = sqrt(w*h)/2 (equal-area circle).

    Returns (circles float32 (N,3), radius_valid bool (N,)). radius_valid is False for
    boxes touching the image border (the box is truncated, so its radius is unreliable).
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = np.clip(x2 - x1, 0.0, None)
    h = np.clip(y2 - y1, 0.0, None)
    circles = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, np.sqrt(w * h) / 2], axis=1)
    near_edge = (x1 <= edge_px) | (y1 <= edge_px) | (x2 >= img_w - edge_px) | (y2 >= img_h - edge_px)
    return circles.astype(np.float32), ~near_edge


def circles_to_boxes(circles: np.ndarray) -> np.ndarray:
    """(cx, cy, r) -> xyxy [cx-r, cy-r, cx+r, cy+r], float32."""
    c = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    return np.stack([c[:, 0] - c[:, 2], c[:, 1] - c[:, 2], c[:, 0] + c[:, 2], c[:, 1] + c[:, 2]], axis=1).astype(
        np.float32
    )


def overlap_levels(circles: np.ndarray) -> np.ndarray:
    """Per-circle count of other circles it overlaps (d < r_i + r_j, strict), clipped to 3."""
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    n = len(c)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    ov = d < (c[:, None, 2] + c[None, :, 2])
    np.fill_diagonal(ov, False)
    return np.minimum(ov.sum(axis=1), 3).astype(np.int64)


def circle_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Closed-form, differentiable IoU of circles (...,3)=(cx,cy,r); broadcasts a with b.

    disjoint (d >= r1+r2) -> 0; contained (d <= |r1-r2|) -> (r_min/r_max)^2; else lens area.
    All branches are evaluated on clamped, finite inputs so torch.where yields finite grads.
    """
    a, b = torch.broadcast_tensors(a, b)
    r1 = a[..., 2].clamp_min(_EPS)
    r2 = b[..., 2].clamp_min(_EPS)
    dx = a[..., 0] - b[..., 0]
    dy = a[..., 1] - b[..., 1]
    d2 = dx * dx + dy * dy
    d_safe = torch.sqrt(d2 + _EPS)  # finite grad at d == 0

    disjoint = d2 >= (r1 + r2) ** 2
    contained = d2 <= (r1 - r2) ** 2

    one = 1.0 - _EPS
    cos1 = ((d_safe**2 + r1**2 - r2**2) / (2 * d_safe * r1)).clamp(-one, one)
    cos2 = ((d_safe**2 + r2**2 - r1**2) / (2 * d_safe * r2)).clamp(-one, one)
    tri = ((-d_safe + r1 + r2) * (d_safe + r1 - r2) * (d_safe - r1 + r2) * (d_safe + r1 + r2)).clamp_min(_EPS)
    lens = r1**2 * torch.acos(cos1) + r2**2 * torch.acos(cos2) - 0.5 * torch.sqrt(tri)

    small = torch.minimum(r1, r2)
    large = torch.maximum(r1, r2)
    inter = torch.where(contained, torch.pi * small**2, lens)
    inter = torch.where(disjoint, torch.zeros_like(inter), inter)
    union = torch.pi * (r1**2 + r2**2) - inter
    iou = inter / union.clamp_min(_EPS)
    # containment ratio computed explicitly for an exact (r_min/r_max)^2 value
    iou = torch.where(contained & ~disjoint, (small / large) ** 2, iou)
    return iou
