"""DC-CellNet: ResNet-50 + FPN-lite (stride 4, or 2 with ``model.out_stride: 2``) + heatmap/log_r/offset/density heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import ResNet50Backbone
from .heads import DENSITY_BIAS, HEATMAP_BIAS, LOG_R_BIAS, DeepDensityHead, Head
from .neck import OUT_STRIDES, FPNLite

NUM_CLASSES = 3
NECK_CH = 128
DENSITY_BRANCHES = ("shared", "detached")
# ImageNet statistics of the pretrained ResNet-50 (torchvision), RGB order
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_out_stride(cfg: dict) -> int:
    """``model.out_stride`` (default 4; must be in ``OUT_STRIDES``), checked against ``data.stride`` when the
    config has one: targets / decode / losses use ``data.stride``, so a mismatch raises ValueError."""
    s = cfg.get("model", {}).get("out_stride", 4)
    if s not in OUT_STRIDES:
        raise ValueError(f"model.out_stride must be one of {OUT_STRIDES}, got {s!r}")
    data_stride = cfg.get("data", {}).get("stride")
    if data_stride is not None and int(data_stride) != int(s):
        raise ValueError(f"data.stride ({data_stride}) must equal model.out_stride ({s})")
    return int(s)


class DCCellNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]
        if mcfg.get("repr", "circle") != "circle":
            raise NotImplementedError(f"model.repr={mcfg['repr']!r} is planned for M3")
        if mcfg.get("backbone", "resnet50") != "resnet50":
            raise NotImplementedError(f"model.backbone={mcfg['backbone']!r} is planned for M3")
        self.out_stride = resolve_out_stride(cfg)
        # non-persistent: constants, kept out of state_dict so checkpoints are unaffected
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.backbone = ResNet50Backbone(pretrained=bool(mcfg.get("pretrained", True)))
        self.neck = FPNLite(ResNet50Backbone.out_channels, NECK_CH, out_stride=self.out_stride)
        self.density_detached = False
        heads = {
            "heatmap": Head(NECK_CH, NUM_CLASSES, HEATMAP_BIAS),
            "log_r": Head(NECK_CH, 1, LOG_R_BIAS),
            "offset": Head(NECK_CH, 2, 0.0),
        }
        if mcfg.get("heads", {}).get("density", True):
            branch = mcfg.get("density_branch", "shared")
            if branch not in DENSITY_BRANCHES:
                raise ValueError(f"model.density_branch must be one of {DENSITY_BRANCHES}, got {branch!r}")
            self.density_detached = branch == "detached"
            heads["density"] = (DeepDensityHead if self.density_detached else Head)(NECK_CH, NUM_CLASSES, DENSITY_BIAS)
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
            out["density"] = F.softplus(self.heads["density"](f.detach() if self.density_detached else f).float())
        return out
