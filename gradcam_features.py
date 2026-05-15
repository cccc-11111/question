from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


IMG_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_LAYERS = "rgb,depth,fusion"
LOCAL_CONFIG_DIR = Path("runs") / ".ultralytics_config"


if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for RGB, depth, and fused feature layers.")
    parser.add_argument("--model", required=True, help="Path to trained best.pt.")
    parser.add_argument("--data", default="depth-gray.yaml", help="Dataset yaml used to match RGB/depth images.")
    parser.add_argument("--source", default=None, help="RGB image path. If omitted, the first val image is used.")
    parser.add_argument("--depth", default=None, help="Depth image path. If omitted, it is matched from --data.")
    parser.add_argument("--output", default="runs/gradcam", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--device", default="0", help="Device, e.g. 0, cuda:0, cpu.")
    parser.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help="Comma list: rgb,depth,fusion,all or explicit layer ids like 0,1,8.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity.")
    return parser.parse_args()


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_data_yaml(path: str | Path) -> tuple[dict[str, Any], Path]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = resolve_path(data.get("path", "."), data_path.parent).resolve()
    return data, root


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def default_rgb_image(data: dict[str, Any], root: Path) -> Path:
    val = data.get("val") or data.get("train")
    if not val:
        raise ValueError("Dataset yaml must contain val or train when --source is omitted.")
    images = collect_images(resolve_path(val, root))
    if not images:
        raise FileNotFoundError(f"No RGB images found under {resolve_path(val, root)}")
    return images[0]


def split_name_for_image(rgb_path: Path, data: dict[str, Any], root: Path) -> str:
    rgb_resolved = rgb_path.resolve()
    for split in ("train", "val", "test"):
        split_path = data.get(split)
        if not split_path:
            continue
        split_root = resolve_path(split_path, root).resolve()
        try:
            rgb_resolved.relative_to(split_root)
            return split
        except ValueError:
            continue
    return "val" if data.get("depth_valid") else "train"


def match_depth_image(rgb_path: Path, data: dict[str, Any], root: Path, explicit_depth: str | None = None) -> Path:
    if explicit_depth:
        return resolve_path(explicit_depth, root)

    split = split_name_for_image(rgb_path, data, root)
    depth_key = {"train": "depth_train", "val": "depth_valid", "test": "depth_test"}.get(split, "depth_valid")
    depth_root = data.get(depth_key) or data.get("depth_train")
    if not depth_root:
        raise ValueError(f"Dataset yaml has no {depth_key} or depth_train entry.")
    depth_root = resolve_path(depth_root, root)

    rgb_root = resolve_path(data.get(split, data.get("val", data.get("train"))), root)
    candidates: list[Path] = []
    try:
        rel = rgb_path.resolve().relative_to(rgb_root.resolve())
        candidates.append(depth_root / rel)
        candidates.extend((depth_root / rel).with_suffix(s) for s in IMG_SUFFIXES)
    except ValueError:
        pass
    candidates.extend([depth_root / rgb_path.name, depth_root / f"{rgb_path.stem}.png", depth_root / f"{rgb_path.stem}.jpg"])

    if depth_root.exists():
        by_name = {p.name.lower(): p for p in depth_root.rglob("*") if p.is_file()}
        by_stem = {p.stem.lower(): p for p in by_name.values()}
        candidates.extend([by_name.get(rgb_path.name.lower()), by_stem.get(rgb_path.stem.lower())])

    depth_path = next((p for p in candidates if p and p.exists()), None)
    if depth_path is None:
        raise FileNotFoundError(f"No matching depth image found for {rgb_path} under {depth_root}")
    return depth_path


def letterbox(image: np.ndarray, size: int, value: int = 114) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[..., None]
    canvas_shape = (size, size, resized.shape[2])
    canvas = np.full(canvas_shape, value, dtype=resized.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, (r, r), (left, top)


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def read_inputs(rgb_path: Path, depth_path: Path, size: int, depth_channels: int, device: torch.device):
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    rgb_lb, _, _ = letterbox(rgb_bgr, size)
    rgb_rgb = rgb_lb[..., ::-1]

    depth_flag = cv2.IMREAD_GRAYSCALE if depth_channels == 1 else cv2.IMREAD_COLOR
    depth = cv2.imread(str(depth_path), depth_flag)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    if depth.ndim == 2:
        depth = depth[..., None]
    depth_lb, _, _ = letterbox(depth, size)

    return rgb_bgr, image_to_tensor(rgb_rgb, device), image_to_tensor(depth_lb, device)


def expected_depth_channels(model: torch.nn.Module) -> tuple[int, ...] | None:
    bypass = next((m for m in model.modules() if m.__class__.__name__ == "BypassCNN"), None)
    if bypass is None:
        return None
    if hasattr(bypass, "supported_input_channels"):
        return tuple(int(c) for c in bypass.supported_input_channels)
    return (int(bypass.stem[0].in_channels),)


def adapt_depth_channels(depth: torch.Tensor, expected_channels: tuple[int, ...] | None) -> torch.Tensor:
    if expected_channels is None or depth.shape[1] in expected_channels:
        return depth
    raise ValueError(
        f"Depth channel mismatch: model supports {expected_channels}, input has {depth.shape[1]}. "
        "Use a model YAML/checkpoint whose BypassCNN input channels match depth_channels in the data YAML."
    )


def select_layer_ids(model: torch.nn.Module, spec: str) -> set[int]:
    requested = {item.strip().lower() for item in spec.split(",") if item.strip()}
    layers = list(model.model)
    if "all" in requested:
        return {m.i for m in layers}

    selected: set[int] = set()
    for item in requested:
        if item.isdigit():
            selected.add(int(item))
        elif item == "rgb":
            selected.update(m.i for m in layers if m.__class__.__name__ == "DINOBackbone")
        elif item == "depth":
            selected.update(m.i for m in layers if m.__class__.__name__ == "BypassCNN")
        elif item == "fusion":
            selected.update(m.i for m in layers if m.__class__.__name__ in {"TAEM", "BiFPN", "Segment"})
    return selected


def tensor_items(output: Any) -> list[tuple[str, torch.Tensor]]:
    if isinstance(output, torch.Tensor) and output.ndim == 4:
        return [("", output)]
    if isinstance(output, (list, tuple)):
        items = []
        for i, value in enumerate(output):
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                items.append((f"_out{i}", value))
        return items
    return []


def prediction_target(preds: Any, nc: int) -> torch.Tensor:
    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if pred.ndim != 3:
        raise RuntimeError(f"Unexpected prediction shape for Grad-CAM target: {tuple(pred.shape)}")
    class_scores = pred[:, 4 : 4 + nc, :]
    return class_scores.max()


def normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().float()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def save_cam(cam: np.ndarray, base_bgr: np.ndarray, path: Path, alpha: float) -> None:
    cam = cv2.resize(cam, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, heat, alpha, 0)
    cv2.imwrite(str(path), overlay)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CONFIG_DIR.resolve()))

    data, root = load_data_yaml(args.data)
    rgb_path = resolve_path(args.source, root) if args.source else default_rgb_image(data, root)
    depth_path = match_depth_image(rgb_path, data, root, args.depth)
    depth_channels = int(data.get("depth_channels", 3))

    from ultralytics import YOLO

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device.split(':')[-1]}")
    yolo = YOLO(args.model)
    model = yolo.model.to(device).eval()
    model_depth_channels = expected_depth_channels(model)
    layer_ids = select_layer_ids(model, args.layers)
    if not layer_ids:
        raise ValueError(f"No layers selected by --layers {args.layers!r}")

    rgb_bgr, rgb_tensor, depth_tensor = read_inputs(rgb_path, depth_path, args.imgsz, depth_channels, device)
    depth_tensor = adapt_depth_channels(depth_tensor, model_depth_channels)
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            for suffix, tensor in tensor_items(output):
                if tensor.requires_grad:
                    tensor.retain_grad()
                    captures[f"{layer.i:02d}_{layer.__class__.__name__}{suffix}"] = tensor

        return hook

    for layer in model.model:
        if layer.i in layer_ids:
            handles.append(layer.register_forward_hook(make_hook(layer)))

    model.zero_grad(set_to_none=True)
    preds = model((rgb_tensor, depth_tensor))
    target = prediction_target(preds, model.yaml.get("nc", 80))
    target.backward()

    saved = 0
    for name, activation in captures.items():
        grad = activation.grad
        if grad is None:
            continue
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1)[0])
        cam_np = normalize_cam(cam)
        save_cam(cam_np, rgb_bgr, output_dir / f"{rgb_path.stem}_{name}_gradcam.jpg", args.alpha)
        saved += 1

    for handle in handles:
        handle.remove()

    print(f"RGB: {rgb_path}")
    print(f"Depth: {depth_path}")
    if model_depth_channels is not None:
        print(f"Depth channels: data={depth_channels}, model={model_depth_channels}, used={depth_tensor.shape[1]}")
    print(f"Target score: {float(target.detach().cpu()):.6f}")
    print(f"Saved {saved} Grad-CAM heatmap(s) to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
