"""Density-map loss: pixel MSE + per-image, per-class count L1."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def density_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """``F.mse_loss(pred, gt)`` (mean over all elements) + mean over (B,C) of
    ``|sum_hw pred - sum_hw gt|``. Inputs (B,C,H,W) cast to float32."""
    pred = pred.float()
    gt = gt.float()
    mse = F.mse_loss(pred, gt)
    count = (pred.sum(dim=(-2, -1)) - gt.sum(dim=(-2, -1))).abs().mean()
    return mse + count
