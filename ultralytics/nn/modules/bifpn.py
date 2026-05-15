# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""BiFPN neck modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("BiFPN",)


class WeightedFeatureFusion(nn.Module):
    """Fast normalized weighted feature fusion used by BiFPN."""

    def __init__(self, n: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w)
        weight = w / (w.sum() + self.eps)
        return sum(weight[i] * x[i] for i in range(len(x)))


class SeparableConvBlock(nn.Module):
    """Depthwise separable convolution block used after BiFPN fusion."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BiFPN(nn.Module):
    """Bidirectional feature pyramid neck."""

    def __init__(self, c1: list[int], c2: list[int] | int | None = None, eps: float = 1e-4):
        super().__init__()
        if not isinstance(c1, list) or len(c1) < 2:
            raise ValueError("BiFPN expects a list of at least two input channel sizes")
        if c2 is None:
            c2 = c1
        elif isinstance(c2, int):
            c2 = [c2] * len(c1)
        if not isinstance(c2, list) or len(c2) != len(c1):
            raise ValueError("BiFPN output channels must be an int or a list matching input levels")

        self.c2 = c2
        self.lateral = nn.ModuleList(Conv(c_in, c_out, 1, 1) for c_in, c_out in zip(c1, c2))
        self.td_proj = nn.ModuleList(Conv(c2[i + 1], c2[i], 1, 1) for i in range(len(c2) - 1))
        self.td_fuse = nn.ModuleList(WeightedFeatureFusion(2, eps) for _ in range(len(c2) - 1))
        self.td_out = nn.ModuleList(SeparableConvBlock(c2[i]) for i in range(len(c2) - 1))
        self.bu_proj = nn.ModuleList(Conv(c2[i], c2[i + 1], 3, 2) for i in range(len(c2) - 1))
        self.bu_fuse = nn.ModuleList(WeightedFeatureFusion(3, eps) for _ in range(len(c2) - 1))
        self.bu_out = nn.ModuleList(SeparableConvBlock(c2[i + 1]) for i in range(len(c2) - 1))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        p = [proj(feat) for proj, feat in zip(self.lateral, x)]
        td = list(p)
        for i in range(len(p) - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=p[i].shape[2:], mode="nearest")
            td[i] = self.td_out[i](self.td_fuse[i]([p[i], self.td_proj[i](up)]))
        out = list(td)
        for i in range(1, len(p)):
            down = self.bu_proj[i - 1](out[i - 1])
            if down.shape[2:] != p[i].shape[2:]:
                down = F.interpolate(down, size=p[i].shape[2:], mode="nearest")
            out[i] = self.bu_out[i - 1](self.bu_fuse[i - 1]([p[i], td[i], down]))
        return out
