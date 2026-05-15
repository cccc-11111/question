# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""AFM fusion modules."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("AFM",)


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU block used by AFM."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SE(nn.Module):
    """Squeeze-and-excitation attention for RGB features."""

    def __init__(self, in_c: int, reduction: int = 16):
        super().__init__()
        hidden_channels = in_c // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_c, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_c, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = self.avg_pool(x)
        weight = self.se(pool)
        return x * weight


class SPA(nn.Module):
    """Spatial pyramid attention for depth features."""

    def __init__(self, in_c: int, bins: tuple[int, ...] | list[int] = (1, 4, 7), reduction: int = 4):
        super().__init__()
        self.bins = tuple(bins)
        inter_channels = max(in_c // reduction, 16)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_channels=in_c, out_channels=inter_channels, kernel_size=bin_size, bias=False),
                    nn.ReLU(inplace=True),
                )
                for bin_size in self.bins
            ]
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(in_c=inter_channels * len(self.bins), out_c=in_c, kernel_size=1),
            nn.Conv2d(in_c, in_c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for branch in self.branches:
            feats.append(branch(x))
        weight = self.fuse(torch.cat(feats, dim=1))
        return x * weight


class AFM(nn.Module):
    """Fuse aligned RGB and depth features at one pyramid level."""

    def __init__(
        self,
        rgb_channels: int,
        depth_channels: int | None = None,
        out_channels: int | None = None,
        reduction: int = 16,
        bins: tuple[int, ...] | list[int] = (1, 4, 7),
        use_fuse_conv: bool = True,
    ):
        super().__init__()
        if depth_channels is None:
            depth_channels = rgb_channels
        if out_channels is None:
            out_channels = rgb_channels
        self.out_channels = out_channels
        self.rgb_proj = (
            ConvBNReLU(in_c=rgb_channels, out_c=out_channels, kernel_size=1)
            if rgb_channels != out_channels
            else nn.Identity()
        )
        self.depth_proj = (
            ConvBNReLU(in_c=depth_channels, out_c=out_channels, kernel_size=1)
            if depth_channels != out_channels
            else nn.Identity()
        )
        self.rgb_attn = SE(in_c=out_channels, reduction=reduction)
        self.depth_attn = SPA(in_c=out_channels, bins=bins, reduction=reduction)
        self.fuse_conv = (
            ConvBNReLU(in_c=out_channels, out_c=out_channels, kernel_size=3, stride=1, padding=1)
            if use_fuse_conv
            else nn.Identity()
        )

    def forward(self, rgb_feat: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor], depth_feat=None):
        if depth_feat is None:
            if not isinstance(rgb_feat, (list, tuple)) or len(rgb_feat) != 2:
                raise TypeError("AFM expects either (rgb_feat, depth_feat) or a two-item [rgb_feat, depth_feat] input")
            rgb_feat, depth_feat = rgb_feat
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            raise ValueError(
                f"AFM requires rgb_feat and depth_feat to have the same spatial size, "
                f"but got rgb_feat={rgb_feat.shape}, depth_feat={depth_feat.shape}. "
                "Please align them in the encoder instead of using interpolate inside AFM."
            )
        rgb_feat = self.rgb_proj(rgb_feat)
        depth_feat = self.depth_proj(depth_feat)
        depth_enhanced = self.depth_attn(depth_feat)
        rgb_enhanced = self.rgb_attn(rgb_feat)
        fused = rgb_enhanced + depth_enhanced
        return fused if isinstance(self.fuse_conv, nn.Identity) else self.fuse_conv(fused)
