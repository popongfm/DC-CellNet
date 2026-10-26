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
# ImageNet statistics of the pretrained ResNet-50 (torchvision), RGB order
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DCCellNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        if mcfg.get("repr", "circle") != "circle":
            raise NotImplementedError(f"model.repr={mcfg['repr']!r} is planned for M3")
        if mcfg.get("backbone", "resnet50") != "resnet50":
            raise NotImplementedError(f"model.backbone={mcfg['backbone']!r} is planned for M3")
        # non-persistent: constants, kept out of state_dict so checkpoints are unaffected
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
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
        """``x``: (B,3,H,W) RGB in [0,1] (as ``CellDataset`` yields); ImageNet-normalized here."""
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"input size must be divisible by 32, got {h}x{w}")
        x = (x - self.pixel_mean.to(x.dtype)) / self.pixel_std.to(x.dtype)
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
