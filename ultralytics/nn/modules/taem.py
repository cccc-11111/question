# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""TAEM fusion modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("TAEM",)


class _CONV(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CBR(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbr(x)


class _CS(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 1):
        super().__init__()
        self.cs = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cs(x)


class _GFE(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.b_1 = _CONV(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.b_2 = _CONV(in_c, out_c, kernel_size=5, stride=2, padding=2)
        self.b_3 = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.cs = _CS(out_c * 3, out_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f1 = self.b_1(x)
        f2 = F.interpolate(self.b_2(x), size=(h, w), mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.b_3(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.cs(torch.cat([f1, f2, f3], dim=1))


class _CCS(nn.Module):
    def __init__(self, in_c: int, out_c: int = 1, kernel_size: int = 1):
        super().__init__()
        self.ccs = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, bias=False),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ccs(x)


class _FCCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, pool_type: str = "avg"):
        super().__init__()
        if pool_type not in {"avg", "max"}:
            raise ValueError("pool_type must be 'avg' or 'max'")
        self.pool_type = pool_type
        hidden_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    @staticmethod
    def _global_max_pool(x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(2, 3), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = self.avg_pool(x) if self.pool_type == "avg" else self._global_max_pool(x)
        fc = self.fc(pool)
        return x * (torch.sigmoid(fc) + F.softmax(fc, dim=1))


class _CA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, pool_type: str = "avg_max"):
        super().__init__()
        if pool_type not in {"avg", "max", "avg_max"}:
            raise ValueError("pool_type must be one of {'avg', 'max', 'avg_max'}")
        self.pool_type = pool_type
        hidden_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_max_pool = pool_type in {"max", "avg_max"}
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _global_max_pool(x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(2, 3), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.mlp(self.avg_pool(x))
        if self.use_max_pool:
            weight = weight + self.mlp(self._global_max_pool(x))
        return x * self.sigmoid(weight)


class TAEM(nn.Module):
    """Fuse aligned RGB and depth features at one pyramid level."""

    def __init__(self, channels: int, reduction: int = 16, fuse_mode: str = "sum", ca_pool_type: str = "avg_max"):
        super().__init__()
        if fuse_mode not in {"sum", "rgb", "depth"}:
            raise ValueError("fuse_mode must be one of {'sum', 'rgb', 'depth'}")
        self.fuse_mode = fuse_mode
        self.rgb_cbr = _CBR(channels, channels, kernel_size=3, stride=1, padding=1)
        self.depth_cbr = _CBR(channels, channels, kernel_size=3, stride=1, padding=1)
        self.rgb_cs = _CS(channels, channels, kernel_size=1)
        self.depth_cs = _CS(channels, channels, kernel_size=1)
        self.rgb_gfe = _GFE(channels, channels)
        self.depth_gfe = _GFE(channels, channels)
        self.rgb_ccs = _CCS(2, channels, kernel_size=1)
        self.fcca = _FCCA(channels, reduction)
        self.depth_conv = _CONV(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ca = _CA(channels, reduction, ca_pool_type)

    def forward(self, rgb: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor], depth=None):
        if depth is None:
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 2:
                raise TypeError("TAEM expects either (rgb, depth) or a two-item [rgb, depth] input")
            rgb, depth = rgb
        if rgb.shape != depth.shape:
            raise ValueError(f"TAEM input shapes must match, got {rgb.shape} and {depth.shape}")
        rgb_cbr = self.rgb_cbr(rgb)
        rgb_cs = self.rgb_cs(rgb_cbr)
        depth_cbr = self.depth_cbr(depth)
        depth_cs = self.depth_cs(depth_cbr)
        depth_guide_rgb = rgb_cs * depth_cs
        rgb_spatial_weight = torch.cat(
            [torch.max(depth_guide_rgb, dim=1, keepdim=True)[0], torch.mean(depth_guide_rgb, dim=1, keepdim=True)],
            dim=1,
        )
        rgb_out = self.fcca(self.rgb_ccs(rgb_spatial_weight) * rgb_cs)
        depth_gfe = self.depth_gfe(depth_cs)
        rgb_gfe = self.rgb_gfe(rgb_cs)
        depth_weight = torch.sigmoid(depth_gfe) * depth_gfe
        rgb_weight = torch.sigmoid(rgb_gfe) * rgb_gfe
        depth_out = self.ca(self.depth_conv(depth_weight + rgb_weight + depth))
        if self.fuse_mode == "rgb":
            return rgb_out
        if self.fuse_mode == "depth":
            return depth_out
        return rgb_out + depth_out
