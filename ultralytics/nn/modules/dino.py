# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("DINOBackbone",)


class _FeatureProjector(nn.Module):
    """Project one DINO feature map to the channel count expected by a YOLO neck."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DINOBackbone(nn.Module):
    """Use a local DINOv3 ViT checkpoint as a YOLO multi-scale backbone.

    The DINO ViT produces patch features at one stride. This wrapper exposes
    YOLO-friendly P3/P4/P5 feature maps by upsampling, keeping, and downsampling
    the projected DINO patch feature.

    Args:
        repo_dir (str): Local DINOv3 repository directory that contains ``hubconf.py``.
        weights (str): Local DINOv3 ``.pth`` checkpoint.
        model_name (str): DINOv3 hub backbone name, e.g. ``dinov3_vits16``.
        out_channels (list[int]): Output channels for P3, P4, and P5.
        train_backbone (bool): Whether DINO parameters are trainable.
        image_mean (list[float]): Normalization mean expected by DINO.
        image_std (list[float]): Normalization std expected by DINO.
    """

    _EMBED_DIMS = {
        "dinov3_vits16": 384,
        "dinov3_vits16plus": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vitl16plus": 1024,
    }

    def __init__(
        self,
        repo_dir: str,
        weights: str,
        model_name: str = "dinov3_vits16",
        out_channels: list[int] | tuple[int, int, int] = (256, 512, 1024),
        train_backbone: bool = False,
        image_mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        if model_name not in self._EMBED_DIMS:
            raise ValueError(f"Unsupported DINOv3 model_name={model_name!r}. Supported: {sorted(self._EMBED_DIMS)}")
        if len(out_channels) != 3:
            raise ValueError("out_channels must contain three channel counts for P3, P4, and P5")

        self.repo_dir = Path(repo_dir)
        self.weights = Path(weights)
        self.model_name = model_name
        self.patch_size = 16

        self.backbone = self._build_dinov3_backbone()
        embed_dim = self._EMBED_DIMS[model_name]
        self.projects = nn.ModuleList(_FeatureProjector(embed_dim, c) for c in out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self.set_backbone_trainable(train_backbone)

    def set_backbone_trainable(self, trainable: bool):
        """Set whether the wrapped DINOv3 backbone participates in training."""
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)
        if not trainable:
            self.backbone.eval()

    def _build_dinov3_backbone(self) -> nn.Module:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo_dir not found: {self.repo_dir}")
        if not self.weights.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights}")

        repo = str(self.repo_dir)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from dinov3.hub import backbones  # noqa: PLC0415

        build = getattr(backbones, self.model_name)
        model = build(pretrained=False)
        checkpoint = torch.load(self.weights, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Ultralytics inputs are already scaled to 0..1; DINO expects ImageNet normalization.
        x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        p4 = self.projects[1](feat)
        p3 = self.projects[0](F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False))
        p5 = self.projects[2](self.downsample(feat))
        return [p3, p4, p5]
