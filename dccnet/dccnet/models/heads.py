"""Prediction heads: conv3x3 -> ReLU -> conv1x1."""
from __future__ import annotations

import math

from torch import nn


class Head(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias_init: float = 0.0, mid_ch: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(mid_ch, out_ch, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.constant_(self.out.bias, bias_init)

    def forward(self, x):
        return self.out(self.act(self.conv(x)))


HEATMAP_BIAS = -2.19
LOG_R_BIAS = math.log(16.0)
