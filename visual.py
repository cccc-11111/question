from __future__ import annotations

import argparse
import os
import pathlib
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn

IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO image inference and optionally save per-layer Grad-CAM heatmaps."
    )
    parser.add_argument("--model", required=True, help="Path to the trained .pt model.")
    parser.add_argument("--source", required=True, help="Path to one image or a directory of images.")
    parser.add_argument("--output", default="runs/visual", help="Directory used to save predictions and heatmaps.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for normal prediction.")
    parser.add_argument("--device", default="0", help="Device, e.g. 0, cuda:0, cpu.")
    parser.add_argument("--recursive", action="store_true", help="Search source directory recursively.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in txt labels.")
    parser.add_argument("--no-predict", action="store_true", help="Skip annotated prediction image generation.")
    parser.add_argument("--no-cam", action="store_true", help="Skip Grad-CAM heatmap generation.")
    parser.add_argument(
        "--cam-layers",
        default="all",
        help=(
            "Grad-CAM layer indices: all, last, or a comma/range list such as 3,6,10-15. "
            "Only layers that produce 4D feature maps are saved."
        ),
    )
    parser.add_argument(
        "--cam-max-layers",
        type=int,
        default=0,
        help="Limit the number of CAM layers for quick debugging. 0 means no limit.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha in [0, 1].")
    return parser.parse_args()


def collect_images(source: str | Path, recursive: bool = False) -> list[Path]:
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMG_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")

    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images


def normalize_device(device: str) -> torch.device:
    if device.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def module_name(module: nn.Module) -> str:
    return module.__class__.__name__


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")


def parse_layer_selector(selector: str, layer_count: int) -> set[int] | str:
    selector = selector.strip().lower()
    if selector in {"all", "last"}:
        return selector

    selected: set[int] = set()
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return {i for i in selected if 0 <= i < layer_count}


def preprocess_image(image_path: Path, imgsz: int, stride: int, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    from ultralytics.data.augment import LetterBox

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    resized = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)(image=bgr)
    rgb = resized[..., ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0
    return bgr, tensor.to(device)


def find_score(output) -> torch.Tensor:
    tensors: list[torch.Tensor] = []

    def visit(value) -> None:
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(output)
    if not tensors:
        raise RuntimeError("Model output does not contain a floating-point tensor for Grad-CAM.")

    for tensor in tensors:
        if tensor.ndim == 3 and min(tensor.shape[1], tensor.shape[2]) > 4:
            if tensor.shape[1] <= tensor.shape[2]:
                return tensor[:, 4:, :].amax()
            return tensor[..., 4:].amax()
    return max((tensor.max() for tensor in tensors if tensor.numel()), key=lambda x: float(x.detach()))


def eligible_layers(model: nn.Module, selector: str) -> list[tuple[int, nn.Module]]:
    layers = list(getattr(model, "model", []))
    selected = parse_layer_selector(selector, len(layers))
    candidates: list[tuple[int, nn.Module]] = []

    for idx, layer in enumerate(layers):
        if selected != "all" and selected != "last" and idx not in selected:
            continue
        if any(name in module_name(layer).lower() for name in ("detect", "segment", "pose", "classify", "obb")):
            continue
        candidates.append((idx, layer))

    if selected == "last" and candidates:
        return [candidates[-1]]
    return candidates


def make_gradcam(model: nn.Module, image_tensor: torch.Tensor, layer: nn.Module) -> np.ndarray | None:
    activation = None
    gradient = None

    def forward_hook(_module, _inputs, output):
        nonlocal activation
        activation = output if isinstance(output, torch.Tensor) and output.ndim == 4 else None

    def backward_hook(_module, _grad_inputs, grad_outputs):
        nonlocal gradient
        grad = grad_outputs[0] if grad_outputs else None
        gradient = grad if isinstance(grad, torch.Tensor) and grad.ndim == 4 else None

    handles = [layer.register_forward_hook(forward_hook), layer.register_full_backward_hook(backward_hook)]
    try:
        model.zero_grad(set_to_none=True)
        output = model(image_tensor)
        if activation is None:
            return None
        score = find_score(output)
        score.backward(retain_graph=False)
        if gradient is None:
            return None

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam.detach().float().cpu().numpy()
        cam -= cam.min()
        max_value = cam.max()
        return cam / max_value if max_value > 1e-8 else None
    finally:
        for handle in handles:
            handle.remove()


def save_cam(cam: np.ndarray, original_bgr: np.ndarray, output_path: Path, alpha: float) -> None:
    cam = cv2.resize(cam, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap, alpha, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def unique_output_path(output_dir: Path, image_path: str | Path) -> Path:
    path = Path(image_path)
    output_path = output_dir / path.name
    if not output_path.exists():
        return output_path

    parent_parts = [safe_name(part) for part in path.parent.parts if part not in (path.anchor, "", ".")]
    prefix = "_".join(part for part in parent_parts[-3:] if part)
    stem = f"{prefix}_{path.stem}" if prefix else path.stem
    return output_dir / f"{stem}{path.suffix}"


def save_final_predictions(results, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        has_masks = result.masks is not None
        plotted = result.plot(
            boxes=not has_masks,
            masks=True,
            labels=not has_masks,
            conf=not has_masks,
        )
        output_path = unique_output_path(output_dir, result.path)
        cv2.imwrite(str(output_path), plotted)


def run_prediction(yolo, source: str, args: argparse.Namespace) -> None:
    results = yolo.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.output,
        name="predict",
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )
    save_final_predictions(results, Path(args.output) / "final")


def run_gradcam(yolo, images: Iterable[Path], args: argparse.Namespace) -> None:
    device = normalize_device(args.device)
    model = yolo.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    layers = eligible_layers(model, args.cam_layers)
    if args.cam_max_layers > 0:
        layers = layers[: args.cam_max_layers]

    cam_root = Path(args.output) / "gradcam"
    for image_path in images:
        original, tensor = preprocess_image(image_path, args.imgsz, stride, device)
        tensor.requires_grad_(True)
        image_dir = cam_root / image_path.stem
        for idx, layer in layers:
            cam = make_gradcam(model, tensor, layer)
            if cam is None:
                continue
            name = f"layer{idx:03d}_{safe_name(module_name(layer))}.jpg"
            save_cam(cam, original, image_dir / name, args.alpha)


def main() -> None:
    args = parse_args()
    config_parent = Path(args.output) / ".ultralytics_config"
    config_parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_parent.resolve())

    from ultralytics import YOLO

    images = collect_images(args.source, args.recursive)
    yolo = YOLO(args.model)

    if not args.no_predict:
        run_prediction(yolo, args.source, args)
    if not args.no_cam:
        run_gradcam(yolo, images, args)

    print(f"Done. Processed {len(images)} image(s). Results saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
