#!/usr/bin/env python3
"""Validate the course project dataset artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
SOURCES_CSV = ROOT / "data" / "sources.csv"
TASKS_JSON = ROOT / "label_studio" / "tasks.json"
EXPORT_JSON = ROOT / "annotations" / "label_studio_export.json"
YOLO_DIR = ROOT / "annotations" / "yolo"
TARGET_COUNT = 120


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def main() -> None:
    images = sorted(IMAGE_DIR.glob("track_*.jpg"))
    if len(images) != TARGET_COUNT:
        fail(f"expected {TARGET_COUNT} images, found {len(images)}")

    with SOURCES_CSV.open(newline="", encoding="utf-8") as csvfile:
        sources = list(csv.DictReader(csvfile))
    if len(sources) != TARGET_COUNT:
        fail(f"expected {TARGET_COUNT} source rows, found {len(sources)}")

    image_names = {path.name for path in images}
    source_names = {row["filename"] for row in sources}
    if image_names != source_names:
        fail("image filenames and sources.csv filenames differ")

    for image_path in images:
        with Image.open(image_path) as image:
            image.verify()

    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    if len(tasks) != TARGET_COUNT:
        fail(f"expected {TARGET_COUNT} Label Studio tasks, found {len(tasks)}")

    export = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    if len(export) != TARGET_COUNT:
        fail(f"expected {TARGET_COUNT} Label Studio export tasks, found {len(export)}")

    total_boxes = 0
    empty_files = 0
    for image_path in images:
        label_path = YOLO_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            fail(f"missing YOLO label file for {image_path.name}")
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_files += 1
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                fail(f"{label_path.name}: expected 5 YOLO fields, got {len(parts)}")
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
            if class_id != 0:
                fail(f"{label_path.name}: expected class id 0, got {class_id}")
            if any(value < 0 or value > 1 for value in values):
                fail(f"{label_path.name}: normalized coordinates outside [0, 1]")
            total_boxes += 1

    preview_count = len(list((ROOT / "docs" / "assets").glob("sample_annotation_*.png")))
    if preview_count < 4:
        fail(f"expected at least 4 annotated previews, found {preview_count}")

    print("Validation passed")
    print(f"images: {len(images)}")
    print(f"bbox: {total_boxes}")
    print(f"empty YOLO files: {empty_files}")
    print(f"preview images: {preview_count}")


if __name__ == "__main__":
    main()
