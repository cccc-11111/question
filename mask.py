from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path

import numpy as np


IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = "runs/dinov3/yolo11-dinov3/weights/best.pt"


if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export binary masks, RGB cutouts, and prediction images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the trained segmentation .pt model.")
    parser.add_argument("--source", required=True, help="Path to one image or a directory of images.")
    parser.add_argument("--output", default="runs/masks", help="Directory used to save output images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Device, e.g. 0, cuda:0, cpu.")
    parser.add_argument("--recursive", action="store_true", help="Search source directory recursively.")
    parser.add_argument(
        "--empty",
        choices=("save", "skip"),
        default="save",
        help="What to do when no mask is detected. save writes a black mask; skip writes nothing.",
    )
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
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMG_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images


def unique_output_path(output_dir: Path, image_path: str | Path, suffix: str) -> Path:
    path = Path(image_path)
    output_path = output_dir / f"{path.stem}_{suffix}.jpg"
    if not output_path.exists():
        return output_path

    parent_parts = [part for part in path.parent.parts if part not in (path.anchor, "", ".")]
    prefix = "_".join(parent_parts[-3:])
    stem = f"{prefix}_{path.stem}" if prefix else path.stem
    return output_dir / f"{stem}_{suffix}.jpg"


def to_numpy_masks(result) -> np.ndarray | None:
    import cv2
    import torch

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    masks = result.masks.data
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    masks = masks.astype(np.float32)
    height, width = result.orig_shape
    if masks.shape[-2:] != (height, width):
        resized = [cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) for mask in masks]
        masks = np.stack(resized, axis=0)
    return masks > 0.5


def build_binary_mask(result, save_empty: bool) -> np.ndarray | None:
    height, width = result.orig_shape
    masks = to_numpy_masks(result)
    if masks is None:
        return np.zeros((height, width), dtype=np.uint8) if save_empty else None

    return masks.any(axis=0).astype(np.uint8) * 255


def build_rgb_cutout(result, save_empty: bool) -> np.ndarray | None:
    height, width = result.orig_shape
    masks = to_numpy_masks(result)
    if masks is None:
        return np.zeros((height, width, 3), dtype=np.uint8) if save_empty else None

    foreground = masks.any(axis=0)
    rgb_image = result.orig_img[..., ::-1]
    rgb_cutout = np.zeros_like(rgb_image, dtype=np.uint8)
    rgb_cutout[foreground] = rgb_image[foreground]
    return rgb_cutout


def main() -> None:
    args = parse_args()
    import cv2

    output_dir = Path(args.output)
    binary_dir = output_dir / "binary"
    rgb_dir = output_dir / "rgb"
    predict_dir = output_dir / "predict"
    binary_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)

    config_parent = output_dir / ".ultralytics_config"
    config_parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_parent.resolve())

    from ultralytics import YOLO

    images = collect_images(args.source, args.recursive)
    model = YOLO(args.model)

    saved = 0
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    for result in results:
        predict_path = unique_output_path(predict_dir, result.path, "predict")
        plotted = result.plot(boxes=True, masks=True, labels=True, conf=True)
        cv2.imwrite(str(predict_path), plotted, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved += 1

        binary_mask = build_binary_mask(result, save_empty=args.empty == "save")
        rgb_cutout = build_rgb_cutout(result, save_empty=args.empty == "save")
        if binary_mask is None or rgb_cutout is None:
            continue
        binary_path = unique_output_path(binary_dir, result.path, "binary")
        rgb_path = unique_output_path(rgb_dir, result.path, "rgb")
        cv2.imwrite(str(binary_path), binary_mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_cutout, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved += 2

    print(f"Done. Processed {len(images)} image(s), saved {saved} image(s) to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
