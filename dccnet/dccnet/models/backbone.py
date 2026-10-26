"""ResNet-50 backbone returning C2..C5 feature maps."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """Strides 4/8/16/32, channels 256/512/1024/2048."""

    out_channels = (256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c2 = self.layer1(self.stem(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
