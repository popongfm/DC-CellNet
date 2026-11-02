"""NMS-free decoding of DC-CellNet outputs: heatmap peaks -> circles.

Conventions mirror ``dccnet.datasets.targets.build_targets``: the center of a peak at
cell (iy, ix) is ``((ix + offset_x) * stride, (iy + offset_y) * stride)`` in input px
(offset channel 0 = x, 1 = y), ``r = exp(log_r)`` and ``score`` is the heatmap value.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dccnet.datasets.geometry import circle_iou

_NMS_MODES = ("none", "circle", "box", "center")
_NMS_IOU = 0.5


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of the circles' bounding squares; a (...,3), b (...,3) broadcastable."""
    a, b = torch.broadcast_tensors(a, b)
    ax1, ay1, ax2, ay2 = a[..., 0] - a[..., 2], a[..., 1] - a[..., 2], a[..., 0] + a[..., 2], a[..., 1] + a[..., 2]
    bx1, by1, bx2, by2 = b[..., 0] - b[..., 2], b[..., 1] - b[..., 2], b[..., 0] + b[..., 2], b[..., 1] + b[..., 2]
    iw = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp_min(0)
    ih = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp_min(0)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union.clamp_min(1e-7)


def _greedy_nms(circles: torch.Tensor, scores: torch.Tensor, mode: str) -> torch.Tensor:
    """Greedy NMS over one class; returns kept indices (score-descending)."""
    order = torch.argsort(scores, descending=True, stable=True)
    c = circles[order]
    iou_fn = circle_iou if mode == "circle" else _box_iou
    iou = iou_fn(c[:, None, :], c[None, :, :])
    suppressed = torch.zeros(len(c), dtype=torch.bool)
    keep = []
    for i in range(len(c)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > _NMS_IOU
    return order[torch.tensor(keep, dtype=torch.long)]


def _center_suppress(circles: torch.Tensor, scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Greedy same-class duplicate-peak merge: drop a peak whose center lies closer than
    ``ratio * max(r_i, r_j)`` to a higher-scoring kept peak. Unlike IoU NMS it keeps genuinely
    overlapping cells (centers further apart than a fraction of the radius) and only removes the
    near-tie plateau peaks a 3x3 max-pool leaves on large cells. Returns kept indices."""
    order = torch.argsort(scores, descending=True, stable=True)
    c = circles[order]
    d = torch.cdist(c[:, :2], c[:, :2])
    rmax = torch.maximum(c[:, None, 2], c[None, :, 2])
    close = d < ratio * rmax
    suppressed = torch.zeros(len(c), dtype=torch.bool)
    keep = []
    for i in range(len(c)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= close[i]
    return order[torch.tensor(keep, dtype=torch.long)]


def _density_fill(
    hm: torch.Tensor,  # (C,H,W)
    peaks: torch.Tensor,  # (C,H,W) bool
    local_max: torch.Tensor,  # (C,H,W) bool
    density: torch.Tensor,  # (C,H,W)
    window: int,
    low_thr: float,
) -> torch.Tensor:
    """Return peaks augmented with candidates so each (class, tile) approaches its target count."""
    C, H, W = hm.shape
    out = peaks.clone()
    cand_all = local_max & (hm > low_thr) & ~peaks
    for c in range(C):
        for y0 in range(0, H, window):
            for x0 in range(0, W, window):
                sl = (c, slice(y0, y0 + window), slice(x0, x0 + window))
                target = int(torch.round(density[sl].sum()).item())
                missing = target - int(peaks[sl].sum().item())
                if missing <= 0:
                    continue
                cand = cand_all[sl]
                if not cand.any():
                    continue
                vals = hm[sl][cand]
                flat_idx = cand.flatten().nonzero().squeeze(1)
                order = torch.argsort(vals, descending=True, stable=True)[:missing]
                chosen = flat_idx[order]
                ys, xs = chosen // cand.shape[1], chosen % cand.shape[1]
                out[c, y0 + ys, x0 + xs] = True
    return out


@torch.no_grad()
def decode(
    heatmap: torch.Tensor,
    log_r: torch.Tensor,
    offset: torch.Tensor,
    density: torch.Tensor | None = None,
    score_thr: float = 0.3,
    stride: int = 4,
    density_fill: bool = False,
    window: int = 32,
    low_thr: float = 0.1,
    nms: str = "none",
    max_dets: int = 1000,
    center_ratio: float = 0.2,
) -> list[np.ndarray]:
    """Decode (B,C,H,W) heatmap + (B,1,H,W) log_r + (B,2,H,W) offset into per-image
    float32 arrays (M,5) of ``[cls, cx, cy, r, score]`` (input px), score-descending."""
    if nms not in _NMS_MODES:
        raise ValueError(f"nms must be one of {_NMS_MODES}, got {nms!r}")
    if nms == "center" and not center_ratio > 0:
        raise ValueError(f"center_ratio must be > 0, got {center_ratio!r}")
    if density_fill and density is None:
        raise ValueError("density_fill=True requires a density map")

    hm = heatmap.detach().float().cpu()
    lr = log_r.detach().float().cpu()
    off = offset.detach().float().cpu()
    dens = density.detach().float().cpu() if density is not None else None
    B, C, H, W = hm.shape

    local_max = hm == F.max_pool2d(hm, 3, stride=1, padding=1)
    peaks_all = local_max & (hm > score_thr)

    results: list[np.ndarray] = []
    for b in range(B):
        peaks = peaks_all[b]
        if density_fill:
            peaks = _density_fill(hm[b], peaks, local_max[b], dens[b], window, low_thr)
        cls_i, iy, ix = peaks.nonzero(as_tuple=True)
        if len(cls_i) == 0:
            results.append(np.zeros((0, 5), dtype=np.float32))
            continue
        scores = hm[b][cls_i, iy, ix]
        cx = (ix.float() + off[b, 0][iy, ix]) * stride
        cy = (iy.float() + off[b, 1][iy, ix]) * stride
        r = torch.exp(lr[b, 0][iy, ix])
        circles = torch.stack([cx, cy, r], dim=1)

        if nms != "none":
            kept = []
            for c in range(C):
                idx = (cls_i == c).nonzero().squeeze(1)
                if len(idx):
                    if nms == "center":
                        kept.append(idx[_center_suppress(circles[idx], scores[idx], float(center_ratio))])
                    else:
                        kept.append(idx[_greedy_nms(circles[idx], scores[idx], nms)])
            keep = torch.cat(kept)
            cls_i, scores, circles = cls_i[keep], scores[keep], circles[keep]

        order = torch.argsort(scores, descending=True, stable=True)[:max_dets]
        det = torch.cat([cls_i[order, None].float(), circles[order], scores[order, None]], dim=1)
        results.append(det.numpy().astype(np.float32))
    return results
