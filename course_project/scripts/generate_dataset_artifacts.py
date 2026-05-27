#!/usr/bin/env python3
"""Generate Label Studio tasks, starter annotations, YOLO files and report assets."""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
SOURCES_CSV = ROOT / "data" / "sources.csv"
TASKS_JSON = ROOT / "label_studio" / "tasks.json"
EXPORT_JSON = ROOT / "annotations" / "label_studio_export.json"
YOLO_DIR = ROOT / "annotations" / "yolo"
ASSET_DIR = ROOT / "docs" / "assets"
DATA_NAMES = ROOT / "data" / "obj.names"
STATS_JSON = ROOT / "data" / "stats.json"

LABEL = "animal_track"
FROM_NAME = "label"
TO_NAME = "image"
MAX_PREVIEW = 6


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        return self.w * self.h


def read_sources() -> list[dict[str, str]]:
    with SOURCES_CSV.open(newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def local_file_url(filename: str) -> str:
    return f"/data/local-files/?d=images/{filename}"


def normalize_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    x_center = (box.x + box.w / 2) / width
    y_center = (box.y + box.h / 2) / height
    box_width = box.w / width
    box_height = box.h / height
    return x_center, y_center, box_width, box_height


def iou(a: Box, b: Box) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h
    inter_w = max(0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def binary_cleanup(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.filter(ImageFilter.MaxFilter(5))
    image = image.filter(ImageFilter.MinFilter(3))
    image = image.filter(ImageFilter.MinFilter(3))
    image = image.filter(ImageFilter.MaxFilter(3))
    return np.asarray(image) > 0


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)

    for start_y, start_x in zip(ys, xs, strict=False):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        area = 0

        while queue:
            y, x = queue.popleft()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

        components.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return components


def detect_boxes(image_path: Path) -> tuple[list[Box], tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    full_width, full_height = image.size

    work = ImageOps.grayscale(image)
    work.thumbnail((760, 760), Image.Resampling.LANCZOS)
    work_width, work_height = work.size
    arr = np.asarray(work).astype(np.float32)

    blur = np.asarray(work.filter(ImageFilter.GaussianBlur(radius=10))).astype(np.float32)
    local_delta = np.abs(arr - blur)
    dark_delta = np.maximum(blur - arr, 0)

    threshold = max(float(np.percentile(local_delta, 94.5)), float(local_delta.mean() + 1.3 * local_delta.std()), 7.0)
    dark_threshold = max(float(np.percentile(dark_delta, 91.0)), 5.0)
    mask = (local_delta >= threshold) | (dark_delta >= dark_threshold)

    # Ignore the outer frame where captions, borders, and compression artifacts often live.
    border_x = max(2, int(work_width * 0.01))
    border_y = max(2, int(work_height * 0.01))
    mask[:border_y, :] = False
    mask[-border_y:, :] = False
    mask[:, :border_x] = False
    mask[:, -border_x:] = False

    mask = binary_cleanup(mask)
    components = connected_components(mask)

    scale_x = full_width / work_width
    scale_y = full_height / work_height
    image_area = work_width * work_height
    candidates: list[Box] = []

    for min_x, min_y, max_x, max_y, area in components:
        box_w = max_x - min_x
        box_h = max_y - min_y
        if area < max(35, image_area * 0.00008):
            continue
        if area > image_area * 0.09:
            continue
        if box_w < 6 or box_h < 6:
            continue
        aspect = box_w / box_h
        if aspect < 0.12 or aspect > 8.5:
            continue
        density = area / (box_w * box_h)
        if density < 0.08:
            continue

        pad_x = max(2, int(box_w * 0.08))
        pad_y = max(2, int(box_h * 0.08))
        x = max(0, math.floor((min_x - pad_x) * scale_x))
        y = max(0, math.floor((min_y - pad_y) * scale_y))
        x2 = min(full_width, math.ceil((max_x + pad_x) * scale_x))
        y2 = min(full_height, math.ceil((max_y + pad_y) * scale_y))
        confidence = min(0.99, 0.45 + density + min(area / (image_area * 0.025), 0.3))
        candidates.append(Box(x=x, y=y, w=max(1, x2 - x), h=max(1, y2 - y), confidence=confidence))

    candidates.sort(key=lambda box: (box.confidence, box.area), reverse=True)
    selected: list[Box] = []
    for candidate in candidates:
        if candidate.w * candidate.h > full_width * full_height * 0.18:
            continue
        if all(iou(candidate, existing) < 0.35 for existing in selected):
            selected.append(candidate)
        if len(selected) >= 14:
            break

    selected.sort(key=lambda box: (box.y, box.x))
    return selected, (full_width, full_height)


def load_yolo_boxes(label_path: Path, image_size: tuple[int, int]) -> list[Box]:
    width, height = image_size
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, box_width, box_height = parts
        xc = float(x_center) * width
        yc = float(y_center) * height
        bw = float(box_width) * width
        bh = float(box_height) * height
        x = max(0, int(round(xc - bw / 2)))
        y = max(0, int(round(yc - bh / 2)))
        x2 = min(width, int(round(xc + bw / 2)))
        y2 = min(height, int(round(yc + bh / 2)))
        boxes.append(Box(x=x, y=y, w=max(1, x2 - x), h=max(1, y2 - y), confidence=1.0))
    return boxes


def label_studio_result(box: Box, width: int, height: int, idx: int) -> dict[str, object]:
    return {
        "id": f"bbox_{idx:04d}",
        "type": "rectanglelabels",
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": box.x / width * 100,
            "y": box.y / height * 100,
            "width": box.w / width * 100,
            "height": box.h / height * 100,
            "rotation": 0,
            "rectanglelabels": [LABEL],
        },
        "score": round(box.confidence, 4),
    }


def draw_preview(image_path: Path, boxes: list[Box], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 450))
    for box in boxes:
        xy = [box.x, box.y, box.x + box.w, box.y + box.h]
        draw.rectangle(xy, outline=(220, 40, 35), width=line_width)
    image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
    image.save(output_path, "PNG", optimize=True)


def generate_charts(objects_per_image: list[int], bbox_areas: list[float]) -> None:
    plt.figure(figsize=(8, 4.5))
    bins = range(0, max(objects_per_image + [0]) + 2)
    plt.hist(objects_per_image, bins=bins, color="#4677c8", edgecolor="white")
    plt.title("Распределение числа bbox на изображение")
    plt.xlabel("bbox на изображение")
    plt.ylabel("количество изображений")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "chart_objects_per_image.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.hist(bbox_areas, bins=24, color="#4f9d69", edgecolor="white")
    plt.title("Распределение относительной площади bbox")
    plt.xlabel("доля площади изображения")
    plt.ylabel("количество bbox")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "chart_bbox_area.png", dpi=160)
    plt.close()


def main() -> None:
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    use_existing_yolo = any(YOLO_DIR.glob("track_*.txt"))
    if not use_existing_yolo:
        for old_file in YOLO_DIR.glob("*.txt"):
            old_file.unlink()
    for old_file in ASSET_DIR.glob("sample_annotation_*.png"):
        old_file.unlink()

    sources = read_sources()
    tasks: list[dict[str, object]] = []
    export: list[dict[str, object]] = []
    objects_per_image: list[int] = []
    bbox_areas: list[float] = []
    preview_items: list[tuple[Path, list[Box]]] = []

    for task_id, row in enumerate(sources, start=1):
        filename = row["filename"]
        image_path = IMAGE_DIR / filename
        with Image.open(image_path) as image:
            width, height = image.size
        if use_existing_yolo:
            boxes = load_yolo_boxes(YOLO_DIR / f"{Path(filename).stem}.txt", (width, height))
        else:
            boxes, (width, height) = detect_boxes(image_path)
        objects_per_image.append(len(boxes))
        bbox_areas.extend([box.area / (width * height) for box in boxes])

        tasks.append({"id": task_id, "data": {"image": local_file_url(filename)}})

        yolo_path = YOLO_DIR / f"{Path(filename).stem}.txt"
        if not use_existing_yolo:
            with yolo_path.open("w", encoding="utf-8") as yolo_file:
                for box in boxes:
                    x_center, y_center, box_width, box_height = normalize_box(box, width, height)
                    yolo_file.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        results = [label_studio_result(box, width, height, idx) for idx, box in enumerate(boxes, start=1)]
        export.append(
            {
                "id": task_id,
                "data": {"image": local_file_url(filename)},
                "annotations": [
                    {
                        "id": task_id,
                        "completed_by": 1,
                        "result": results,
                        "was_cancelled": False,
                        "ground_truth": False,
                    }
                ],
                "predictions": [],
            }
        )

        if boxes and len(preview_items) < MAX_PREVIEW:
            preview_items.append((image_path, boxes))

    TASKS_JSON.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    EXPORT_JSON.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    DATA_NAMES.write_text(f"{LABEL}\n", encoding="utf-8")

    for idx, (image_path, boxes) in enumerate(preview_items, start=1):
        draw_preview(image_path, boxes, ASSET_DIR / f"sample_annotation_{idx:02d}.png")

    generate_charts(objects_per_image, bbox_areas)

    stats = {
        "images": len(sources),
        "bbox_count": int(sum(objects_per_image)),
        "empty_images": int(sum(1 for count in objects_per_image if count == 0)),
        "mean_bbox_per_image": round(float(np.mean(objects_per_image)), 3) if objects_per_image else 0.0,
        "median_bbox_per_image": round(float(np.median(objects_per_image)), 3) if objects_per_image else 0.0,
        "mean_bbox_area": round(float(np.mean(bbox_areas)), 6) if bbox_areas else 0.0,
        "median_bbox_area": round(float(np.median(bbox_areas)), 6) if bbox_areas else 0.0,
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
