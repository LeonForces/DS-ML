#!/usr/bin/env python3
"""Generate a reproducible calibration set when external image hosts rate-limit downloads."""

from __future__ import annotations

import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
SOURCES_CSV = ROOT / "data" / "sources.csv"
YOLO_DIR = ROOT / "annotations" / "yolo"

TARGET_COUNT = 120
WIDTH = 640
HEIGHT = 480
RANDOM_SEED = 20260527


@dataclass(frozen=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized_yolo(self) -> tuple[float, float, float, float]:
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return (
            (self.x1 + width / 2) / WIDTH,
            (self.y1 + height / 2) / HEIGHT,
            width / WIDTH,
            height / HEIGHT,
        )


def rotated_point(x: float, y: float, angle: float) -> tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def polygon_bbox(points: list[tuple[float, float]], pad: int = 4) -> PixelBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return PixelBox(
        max(0, int(math.floor(min(xs))) - pad),
        max(0, int(math.floor(min(ys))) - pad),
        min(WIDTH, int(math.ceil(max(xs))) + pad),
        min(HEIGHT, int(math.ceil(max(ys))) + pad),
    )


def ellipse_points(cx: float, cy: float, rx: float, ry: float, angle: float) -> list[tuple[float, float]]:
    points = []
    for step in range(24):
        theta = step / 24 * math.tau
        px = math.cos(theta) * rx
        py = math.sin(theta) * ry
        qx, qy = rotated_point(px, py, angle)
        points.append((cx + qx, cy + qy))
    return points


def draw_rotated_ellipse(
    layer: Image.Image,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle: float,
    fill: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    points = ellipse_points(cx, cy, rx, ry, angle)
    ImageDraw.Draw(layer).polygon(points, fill=fill)
    return points


def make_background(kind: str, rng: random.Random) -> Image.Image:
    if kind == "snow":
        base = np.full((HEIGHT, WIDTH, 3), [222, 229, 232], dtype=np.float32)
        noise_scale = 18
        tint = np.array([rng.randint(-8, 4), rng.randint(-5, 8), rng.randint(2, 14)], dtype=np.float32)
    elif kind == "sand":
        base = np.full((HEIGHT, WIDTH, 3), [190, 165, 116], dtype=np.float32)
        noise_scale = 24
        tint = np.array([rng.randint(-6, 14), rng.randint(-10, 8), rng.randint(-14, 6)], dtype=np.float32)
    else:
        base = np.full((HEIGHT, WIDTH, 3), [112, 93, 72], dtype=np.float32)
        noise_scale = 28
        tint = np.array([rng.randint(-10, 12), rng.randint(-8, 10), rng.randint(-6, 8)], dtype=np.float32)

    noise = np.random.default_rng(rng.randrange(10**9)).normal(0, noise_scale, (HEIGHT, WIDTH, 1))
    gradient = np.linspace(-18, 18, WIDTH, dtype=np.float32)[None, :, None]
    arr = np.clip(base + tint + noise + gradient, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.7))

    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(40):
        x = rng.randint(0, WIDTH)
        y = rng.randint(0, HEIGHT)
        length = rng.randint(10, 60)
        color = (255, 255, 255, rng.randint(8, 22)) if kind == "snow" else (45, 35, 25, rng.randint(8, 24))
        draw.line((x, y, x + rng.randint(-length, length), y + rng.randint(-length, length)), fill=color, width=1)
    return image


def stamp_paw(image: Image.Image, x: float, y: float, size: float, angle: float, kind: str) -> PixelBox:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill = (72, 68, 64, 95) if kind == "snow" else (45, 35, 25, 105)
    points: list[tuple[float, float]] = []

    points += draw_rotated_ellipse(layer, x, y + size * 0.12, size * 0.23, size * 0.17, angle, fill)
    for dx, dy, rx, ry in [(-0.28, -0.20, 0.09, 0.12), (-0.10, -0.32, 0.08, 0.12), (0.10, -0.32, 0.08, 0.12), (0.28, -0.20, 0.09, 0.12)]:
        px, py = rotated_point(dx * size, dy * size, angle)
        points += draw_rotated_ellipse(layer, x + px, y + py, size * rx, size * ry, angle, fill)

    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(0.8, size * 0.018)))
    image.alpha_composite(layer)
    return polygon_bbox(points)


def stamp_hoof(image: Image.Image, x: float, y: float, size: float, angle: float, kind: str) -> PixelBox:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill = (70, 62, 54, 105) if kind == "snow" else (38, 28, 20, 120)
    points: list[tuple[float, float]] = []
    for dx in (-0.13, 0.13):
        px, py = rotated_point(dx * size, 0, angle)
        points += draw_rotated_ellipse(layer, x + px, y + py, size * 0.11, size * 0.30, angle, fill)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(0.7, size * 0.015)))
    image.alpha_composite(layer)
    return polygon_bbox(points)


def stamp_bird(image: Image.Image, x: float, y: float, size: float, angle: float, kind: str) -> PixelBox:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = (62, 59, 55, 115) if kind == "snow" else (42, 31, 22, 130)
    points: list[tuple[float, float]] = []
    for toe_angle in (-0.62, 0.0, 0.62):
        start = rotated_point(0, size * 0.10, angle)
        end = rotated_point(math.sin(toe_angle) * size * 0.34, -math.cos(toe_angle) * size * 0.42, angle)
        x1, y1 = x + start[0], y + start[1]
        x2, y2 = x + end[0], y + end[1]
        draw.line((x1, y1, x2, y2), fill=fill, width=max(2, int(size * 0.035)))
        points.extend([(x1, y1), (x2, y2)])
    layer = layer.filter(ImageFilter.GaussianBlur(radius=0.7))
    image.alpha_composite(layer)
    return polygon_bbox(points)


def generate_image(index: int, rng: random.Random) -> tuple[Image.Image, list[PixelBox], str]:
    kind = rng.choice(["snow", "snow", "sand", "mud"])
    image = make_background(kind, rng).convert("RGBA")
    boxes: list[PixelBox] = []
    pattern = rng.choice(["paw", "hoof", "bird", "paw"])
    count = rng.randint(3, 9)
    angle = rng.uniform(-0.9, 0.9)
    step_x = math.sin(angle) * rng.uniform(38, 62)
    step_y = -math.cos(angle) * rng.uniform(38, 62)
    start_x = rng.uniform(120, WIDTH - 120)
    start_y = rng.uniform(110, HEIGHT - 80)

    for obj_idx in range(count):
        jitter_x = rng.uniform(-18, 18)
        jitter_y = rng.uniform(-18, 18)
        x = start_x + (obj_idx - count / 2) * step_x + jitter_x
        y = start_y + (obj_idx - count / 2) * step_y + jitter_y
        if not (45 < x < WIDTH - 45 and 45 < y < HEIGHT - 45):
            continue
        size = rng.uniform(42, 68)
        if pattern == "paw":
            box = stamp_paw(image, x, y, size, angle + rng.uniform(-0.25, 0.25), kind)
        elif pattern == "hoof":
            box = stamp_hoof(image, x, y, size, angle + rng.uniform(-0.18, 0.18), kind)
        else:
            box = stamp_bird(image, x, y, size, angle + rng.uniform(-0.28, 0.28), kind)
        if box.x2 - box.x1 > 8 and box.y2 - box.y1 > 8:
            boxes.append(box)

    image = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.25))
    return image, boxes, kind


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)

    for path in IMAGE_DIR.glob("track_*.jpg"):
        path.unlink()
    for path in YOLO_DIR.glob("track_*.txt"):
        path.unlink()

    rows = []
    for index in range(1, TARGET_COUNT + 1):
        filename = f"track_{index:04d}.jpg"
        labelname = f"track_{index:04d}.txt"
        image, boxes, surface = generate_image(index, rng)
        image.save(IMAGE_DIR / filename, "JPEG", quality=90, optimize=True)
        with (YOLO_DIR / labelname).open("w", encoding="utf-8") as label_file:
            for box in boxes:
                xc, yc, width, height = box.normalized_yolo()
                label_file.write(f"0 {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}\n")
        rows.append(
            {
                "filename": filename,
                "commons_title": f"synthetic animal track calibration image {index:04d}",
                "category": f"synthetic/{surface}",
                "source_url": "generated locally by scripts/generate_synthetic_tracks_dataset.py",
                "original_url": "generated locally by scripts/generate_synthetic_tracks_dataset.py",
                "license": "educational synthetic asset",
                "license_url": "",
                "author": "procedural generator",
                "original_width": WIDTH,
                "original_height": HEIGHT,
                "width": WIDTH,
                "height": HEIGHT,
            }
        )

    with SOURCES_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shutil.rmtree(ROOT / "data" / ".download_tmp", ignore_errors=True)
    print(f"Generated {TARGET_COUNT} synthetic calibration images")


if __name__ == "__main__":
    main()
