"""DC-CellNet: ResNet-50 + FPN-lite (stride 4) + heatmap/log_r/offset/density heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import ResNet50Backbone
from .heads import DENSITY_BIAS, HEATMAP_BIAS, LOG_R_BIAS, Head
from .neck import FPNLite

NUM_CLASSES = 3
NECK_CH = 128


class DCCellNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        if mcfg.get("repr", "circle") != "circle":
            raise NotImplementedError(f"model.repr={mcfg['repr']!r} is planned for M3")
        if mcfg.get("backbone", "resnet50") != "resnet50":
            raise NotImplementedError(f"model.backbone={mcfg['backbone']!r} is planned for M3")
        self.backbone = ResNet50Backbone(pretrained=bool(mcfg.get("pretrained", True)))
        self.neck = FPNLite(ResNet50Backbone.out_channels, NECK_CH)
        heads = {
            "heatmap": Head(NECK_CH, NUM_CLASSES, HEATMAP_BIAS),
            "log_r": Head(NECK_CH, 1, LOG_R_BIAS),
            "offset": Head(NECK_CH, 2, 0.0),
        }
        if mcfg.get("heads", {}).get("density", True):
            heads["density"] = Head(NECK_CH, NUM_CLASSES, DENSITY_BIAS)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"input size must be divisible by 32, got {h}x{w}")
        f = self.neck(self.backbone(x))
        out = {
            "heatmap": torch.sigmoid(self.heads["heatmap"](f).float()).clamp(1e-4, 1 - 1e-4),
            "log_r": self.heads["log_r"](f).float(),
            "offset": self.heads["offset"](f).float(),
        }
        if "density" in self.heads:
            # softplus, not ReLU: a ReLU density head lets the count loss drive WBC/PLT channels dead (R17)
            out["density"] = F.softplus(self.heads["density"](f).float())
        return out
