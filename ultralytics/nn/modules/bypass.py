# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("BypassCNN",)


class BypassCNN(nn.Module):
    """Small CNN branch for aligned depth images."""

    def __init__(self, ch_in: int = 3, channel: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(ch_in, channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(channel, channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 2),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 4),
            nn.SiLU(inplace=True),
        )
        self.out_channels = [channel * 2, channel * 4]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        return [p2, p3]
