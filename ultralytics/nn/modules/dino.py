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
    _CONVNEXT_CHANNELS = {
        "dinov3_convnext_tiny": (96, 192, 384, 768),
        "dinov3_convnext_small": (96, 192, 384, 768),
        "dinov3_convnext_base": (128, 256, 512, 1024),
        "dinov3_convnext_large": (192, 384, 768, 1536),
    }

    def __init__(
        self,
        repo_dir: str,
        weights: str,
        model_name: str = "dinov3_vits16",
        out_channels: list[int] | tuple[int, ...] = (256, 512, 1024),
        train_backbone: bool = False,
        image_mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
        uses_original_input: bool = False,
        input_channels: int = 3,
    ):
        super().__init__()
        supported = {*self._EMBED_DIMS, *self._CONVNEXT_CHANNELS}
        if model_name not in supported:
            raise ValueError(f"Unsupported DINOv3 model_name={model_name!r}. Supported: {sorted(supported)}")
        if len(out_channels) not in {3, 4}:
            raise ValueError("out_channels must contain three channel counts for P3/P4/P5 or four for P2/P3/P4/P5")

        self.repo_dir = self._resolve_path(repo_dir)
        self.weights = self._resolve_path(weights)
        self.model_name = model_name
        self.is_convnext = model_name in self._CONVNEXT_CHANNELS
        self.patch_size = 16
        self.num_outputs = len(out_channels)
        self.uses_original_input = uses_original_input
        self.default_input_channels = input_channels

        self.backbone = self._build_dinov3_backbone()
        in_channels = self._feature_channels(model_name)
        self.projects = nn.ModuleList(_FeatureProjector(c1, c2) for c1, c2 in zip(in_channels, out_channels))
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self.set_backbone_trainable(train_backbone)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path

    def set_backbone_trainable(self, trainable: bool):
        """Set whether the wrapped DINOv3 backbone participates in training."""
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)
        if not trainable:
            self.backbone.eval()

    def dummy_input_channels(self) -> int:
        return self.default_input_channels

    def _build_dinov3_backbone(self) -> nn.Module:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo_dir not found: {self.repo_dir}")
        if not self.weights.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights}")

        if self.is_convnext:
            return self._build_hf_convnext_backbone()

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

    def _build_hf_convnext_backbone(self) -> nn.Module:
        try:
            from safetensors.torch import load_file  # noqa: PLC0415
            from transformers import AutoConfig, AutoModel  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("DINOv3 ConvNeXt loading requires transformers and safetensors to be installed.") from e

        config = AutoConfig.from_pretrained(self.repo_dir, local_files_only=True)
        model = AutoModel.from_config(config)
        state_dict = load_file(self.weights)
        if state_dict and not next(iter(state_dict)).startswith("model.") and hasattr(model, "model"):
            state_dict = {f"model.{k}" if k.startswith("stages.") else k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        return model

    def _feature_channels(self, model_name: str) -> tuple[int, ...]:
        if model_name in self._CONVNEXT_CHANNELS:
            channels = self._CONVNEXT_CHANNELS[model_name]
            return channels[-self.num_outputs :]
        return (self._EMBED_DIMS[model_name],) * self.num_outputs

    def _select_convnext_features(self, hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> list[torch.Tensor]:
        """Select real ConvNeXt stage features by their expected channel counts."""
        expected_channels = self._feature_channels(self.model_name)
        features = []
        start = 0
        spatial_states = [feat for feat in hidden_states if isinstance(feat, torch.Tensor) and feat.ndim == 4]
        for channels in expected_channels:
            match_idx, match = next(
                ((idx, feat) for idx, feat in enumerate(spatial_states[start:], start=start) if feat.shape[1] == channels),
                (None, None),
            )
            if match is None:
                shapes = [tuple(feat.shape) for feat in spatial_states]
                raise RuntimeError(
                    f"{self.__class__.__name__} expected real ConvNeXt stage channels {list(expected_channels)}, "
                    f"but hidden_states only exposed shapes {shapes}. Cannot build P2 by upsampling P3."
                )
            features.append(match)
            start = match_idx + 1
        return features

    def _forward_convnext_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward DINOv3 ConvNeXt stage modules directly and return real stage features."""
        stages = getattr(getattr(self.backbone, "model", None), "stages", None)
        if stages is None:
            outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
            return self._select_convnext_features(outputs.hidden_states or ())

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)
        return self._select_convnext_features(features)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Ultralytics inputs are already scaled to 0..1; DINO expects ImageNet normalization.
        x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        if self.is_convnext:
            features = self._forward_convnext_stages(x)
            return [project(feat) for project, feat in zip(self.projects, features)]

        _, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        if self.num_outputs == 4:
            p3_feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
            p2 = self.projects[0](F.interpolate(p3_feat, scale_factor=2, mode="bilinear", align_corners=False))
            p3 = self.projects[1](p3_feat)
            p4 = self.projects[2](feat)
            p5 = self.projects[3](self.downsample(feat))
            return [p2, p3, p4, p5]

        p4 = self.projects[1](feat)
        p3 = self.projects[0](F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False))
        p5 = self.projects[2](self.downsample(feat))
        return [p3, p4, p5]
