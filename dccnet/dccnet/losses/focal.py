"""CenterNet penalty-reduced focal loss."""
from __future__ import annotations

import torch


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2, beta: float = 4) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (Law & Deng 2018; Zhou et al. 2019).

    ``pred`` (B,C,H,W) is the sigmoided heatmap already clamped to [1e-4, 1-1e-4];
    ``gt`` (B,C,H,W) is the Gaussian target with peaks exactly 1.0.

    Positives are cells where ``gt == 1`` (exact). Loss =
    ``(sum_pos -log(p)(1-p)^alpha + sum_neg -log(1-p) p^alpha (1-gt)^beta) / max(num_pos, 1)``.
    Inputs are cast to float32 (AMP safety); returns a 0-dim float32 tensor.
    """
    pred = pred.float()
    gt = gt.float()
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg
    num_pos = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos
