# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ("BypassCNN",)


class BypassCNN(nn.Module):
    """Small CNN branch for aligned depth images."""

    def __init__(
        self,
        ch_in: int | str = 3,
        channel: int = 64,
        out_indices: list[int] | tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        self.out_indices = tuple(out_indices)
        self.max_out_index = max(self.out_indices)
        self.auto_channels = str(ch_in).lower() in {"auto", "both", "dynamic"}
        self.supported_input_channels = (1, 3) if self.auto_channels else (int(ch_in),)
        self.stem = (
            nn.ModuleDict({str(c): self._make_stem(c, channel) for c in self.supported_input_channels})
            if self.auto_channels
            else self._make_stem(int(ch_in), channel)
        )
        self.default_input_channels = 3 if 3 in self.supported_input_channels else self.supported_input_channels[0]
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
        self.stage3 = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 8),
            nn.SiLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(channel * 8, channel * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel * 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel * 16, channel * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel * 16),
            nn.SiLU(inplace=True),
        )
        all_channels = [channel * 2, channel * 4, channel * 8, channel * 16]
        self.out_channels = [all_channels[i] for i in self.out_indices]

    @staticmethod
    def _make_stem(ch_in: int, channel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(ch_in, channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
        )

    def dummy_input_channels(self) -> int:
        return self.default_input_channels

    def _select_stem(self, x: torch.Tensor) -> nn.Module:
        if not self.auto_channels:
            return self.stem
        key = str(x.shape[1])
        if key not in self.stem:
            raise ValueError(
                f"{self.__class__.__name__} with auto input channels supports {self.supported_input_channels}, "
                f"but got depth tensor with {x.shape[1]} channels."
            )
        return self.stem[key]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._select_stem(x)(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        if self.max_out_index < 2:
            features = [p2, p3]
            return [features[i] for i in self.out_indices]
        p4 = self.stage3(p3)
        if self.max_out_index < 3:
            features = [p2, p3, p4]
            return [features[i] for i in self.out_indices]
        p5 = self.stage4(p4)
        features = [p2, p3, p4, p5]
        return [features[i] for i in self.out_indices]
