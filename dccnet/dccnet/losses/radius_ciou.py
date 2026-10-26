"""Radius (L1 on log r + circle-IoU) and offset losses at GT center cells."""
from __future__ import annotations

import torch

from dccnet.datasets.geometry import circle_iou


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B,Ch,H,W) gathered at flat indices ``ind`` (B,K) = iy*W+ix -> (B,K,Ch)."""
    B, Ch, H, W = feat.shape
    flat = feat.reshape(B, Ch, H * W)
    idx = ind.long().unsqueeze(1).expand(B, Ch, ind.shape[1])
    return flat.gather(2, idx).permute(0, 2, 1)


def radius_ciou_loss(
    pred_log_r: torch.Tensor,
    pred_offset: torch.Tensor,
    targets: dict[str, torch.Tensor],
    stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(l1, ciou)`` over objects with ``mask & r_valid`` (border-cut cells excluded).

    ``pred_log_r`` (B,1,H,W), ``pred_offset`` (B,2,H,W) channel 0 = x. Predictions are
    gathered at ``targets["ind"]`` (B,K).
    * ``l1`` = mean |pred_log_r - log_r|.
    * ``ciou`` = mean (1 - circle_iou(pred_circle, targets["circle"])) with
      pred_circle = ((ix + off_x)*stride, (iy + off_y)*stride, exp(pred_log_r)),
      ix = ind % W, iy = ind // W (input px; computed in float32 since circle_iou
      is not fp16-safe).
    With no valid objects both are ``0 * pred_log_r.sum()`` (zero, graph-attached).
    """
    pred_log_r = pred_log_r.float()
    pred_offset = pred_offset.float()
    W = pred_log_r.shape[-1]
    ind = targets["ind"].long()
    valid = targets["mask"].bool() & targets["r_valid"].bool()
    if not bool(valid.any()):
        zero = 0 * pred_log_r.sum()
        return zero, zero.clone()

    lr = _gather(pred_log_r, ind)[..., 0][valid]  # (N,)
    off = _gather(pred_offset, ind)[valid]  # (N,2)
    l1 = (lr - targets["log_r"].float()[valid]).abs().mean()

    ix = (ind % W)[valid].float()
    iy = (ind // W)[valid].float()
    pred_circle = torch.stack(
        [(ix + off[:, 0]) * stride, (iy + off[:, 1]) * stride, torch.exp(lr)], dim=-1
    )
    gt_circle = targets["circle"].float()[valid]
    ciou = (1.0 - circle_iou(pred_circle, gt_circle)).mean()
    return l1, ciou


def offset_loss(pred_offset: torch.Tensor, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean |pred - gt| sub-cell offset over all ``mask`` objects (border cells included)
    and both coordinates. No objects -> graph-attached zero."""
    pred_offset = pred_offset.float()
    mask = targets["mask"].bool()
    if not bool(mask.any()):
        return 0 * pred_offset.sum()
    off = _gather(pred_offset, targets["ind"].long())[mask]  # (N,2)
    return (off - targets["offset"].float()[mask]).abs().mean()
