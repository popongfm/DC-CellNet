"""Density-map loss: pixel MSE + per-image, per-class relative count L1."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def density_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """``F.mse_loss(pred, gt)`` (mean over all elements) + mean over (B,C) of
    ``|sum_hw pred - sum_hw gt| / max(sum_hw gt, 1)``. Inputs (B,C,H,W) cast to float32.

    The count term is normalized so dense RBC images (tens of cells) do not dominate the
    gradient and crush the sparse WBC / platelet channels (R17)."""
    pred = pred.float()
    gt = gt.float()
    mse = F.mse_loss(pred, gt)
    gt_cnt = gt.sum(dim=(-2, -1))
    count = ((pred.sum(dim=(-2, -1)) - gt_cnt).abs() / gt_cnt.clamp_min(1.0)).mean()
    return mse + count
