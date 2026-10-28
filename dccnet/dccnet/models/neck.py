"""FPN-lite neck fusing C2..C5 top-down to a single stride-4 map (optionally upsampled to stride 2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

OUT_STRIDES = (4, 2)


class FPNLite(nn.Module):
    """Top-down fusion of C2..C5 -> conv3x3+BN+ReLU at stride 4.

    ``out_stride=2`` (M2d) adds: bilinear x2 upsample of the fused stride-4 map, then
    conv3x3(out_channels)+BN+ReLU (module ``up``; absent at stride 4, so stride-4 state_dicts are unchanged).
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels: int = 128, out_stride: int = 4) -> None:
        super().__init__()
        if out_stride not in OUT_STRIDES:
            raise ValueError(f"out_stride must be one of {OUT_STRIDES}, got {out_stride!r}")
        self.out_stride = int(out_stride)
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if self.out_stride == 2:
            self.up = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        p = lat[-1]
        for l in reversed(lat[:-1]):
            p = l + F.interpolate(p, size=l.shape[-2:], mode="nearest")
        p = self.smooth(p)
        if self.out_stride == 2:
            p = self.up(F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False))
        return p
