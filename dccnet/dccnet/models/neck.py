"""FPN-lite neck fusing C2..C5 top-down to a single stride-4 map."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FPNLite(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        p = lat[-1]
        for l in reversed(lat[:-1]):
            p = l + F.interpolate(p, size=l.shape[-2:], mode="nearest")
        return self.smooth(p)
