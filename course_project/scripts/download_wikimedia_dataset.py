#!/usr/bin/env python3
"""Download a small animal-tracks calibration dataset from Wikimedia Commons."""

from __future__ import annotations

import csv
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
TMP_DIR = ROOT / "data" / ".download_tmp"
SOURCES_CSV = ROOT / "data" / "sources.csv"

API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_URL = "https://commons.wikimedia.org/wiki/"
FILEPATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/"
USER_AGENT = "CollectionDatasetCourseWork/1.0 (educational dataset preparation)"

TARGET_COUNT = 120
MAX_DIMENSION = 512
REQUEST_PAUSE_SECONDS = 0.8
IMAGEINFO_BATCH_SIZE = 10
DOWNLOAD_WORKERS = 2
MAX_DOWNLOAD_BYTES = 5_000_000

CATEGORIES = [
    "Category:Animal tracks on snow",
    "Category:Animal tracks on sand",
    "Category:Animal tracks on mud",
    "Category:Animal tracks on dust",
    "Category:Hare tracks",
    "Category:Unidentified animal tracks",
    "Category:Bird tracks on sand",
    "Category:Animal tracks",
]

SEARCH_QUERIES = [
    "animal tracks footprints snow",
    "wild animal tracks footprints",
    "bird tracks sand",
    "deer tracks snow",
    "hare tracks snow",
    "boar tracks snow",
]

EXCLUDE_TITLE = re.compile(
    r"("
    r"\.svg|\.pdf|\.djvu|diagram|drawing|illustration|icon|map|symbol|"
    r"human|happisburgh|hominin|person|people|barefoot|shoeprint|"
    r"dog|cat|pig|paw|tegula|aquincum|archaeolog|fouilles|terracotta|"
    r"vehicle|tire|tyre|wheel|tractor|ski|snowshoe|"
    r"fossil|dinosaur|museum|pottery|ware|book|serial"
    r")",
    re.IGNORECASE,
)
INCLUDE_TITLE = re.compile(
    r"(track|tracks|footprint|footprints|spoor|sporen|spuren|spur|trace|traces|"
    r"empreinte|empreintes|petjad|pugmark|tierspuren|huella|pates|spoor\))",
    re.IGNORECASE,
)


def api_get(params: dict[str, Any]) -> dict[str, Any]:
    payload = {"format": "json", "formatversion": 2, **params}
    for attempt in range(5):
        response = requests.get(
            API_URL,
            params=payload,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if response.status_code != 429:
            response.raise_for_status()
            time.sleep(REQUEST_PAUSE_SECONDS)
            return response.json()
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = int(retry_after)
        else:
            wait_seconds = 5 * (attempt + 1)
        time.sleep(wait_seconds)
    response.raise_for_status()
    return response.json()


def collect_category_titles(category: str) -> list[tuple[str, str]]:
    category_path = category.replace(" ", "_")
    url = f"{COMMONS_URL}{quote(category_path, safe=':/')}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE_SECONDS)

    titles: list[tuple[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a href="/wiki/(File:[^"#?<>]+)"[^>]*>\s*<img[^>]+>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(response.text):
        raw_html = match.group(0)
        raw_title = unquote(match.group(1)).replace("_", " ")
        if raw_title not in seen:
            thumb_url = ""
            srcset_match = re.search(r'srcset="([^"]+)"', raw_html)
            if srcset_match:
                srcset_items = [item.strip().split()[0] for item in srcset_match.group(1).split(",")]
                thumb_url = srcset_items[-1] if srcset_items else ""
            if not thumb_url:
                src_match = re.search(r'src="([^"]+)"', raw_html)
                thumb_url = src_match.group(1) if src_match else ""
            if thumb_url.startswith("//"):
                thumb_url = f"https:{thumb_url}"
            if not thumb_url:
                continue
            seen.add(raw_title)
            titles.append((raw_title, thumb_url))
    return titles


def collect_search_titles(query: str) -> list[str]:
    return []


def metadata_value(meta: dict[str, Any], key: str) -> str:
    value = meta.get(key, {}).get("value", "")
    return re.sub(r"<[^>]+>", "", value).strip()


def commons_file_url(title: str) -> str:
    return COMMONS_URL + quote(title.replace(" ", "_"), safe=":/")


def special_filepath_url(title: str) -> str:
    filename = title.removeprefix("File:").replace(" ", "_")
    return f"{FILEPATH_URL}{quote(filename, safe='')}?width={MAX_DIMENSION}"


def build_candidates() -> list[dict[str, Any]]:
    title_to_source: dict[str, tuple[str, str]] = {}

    for category in CATEGORIES:
        for title, thumb_url in collect_category_titles(category):
            if INCLUDE_TITLE.search(title) and not EXCLUDE_TITLE.search(title):
                title_to_source.setdefault(title, (category, thumb_url))

    if len(title_to_source) < TARGET_COUNT:
        for query in SEARCH_QUERIES:
            for title in collect_search_titles(query):
                if INCLUDE_TITLE.search(title) and not EXCLUDE_TITLE.search(title):
                    title_to_source.setdefault(title, (f"search:{query}", special_filepath_url(title)))

    candidates: list[dict[str, Any]] = []
    for title, (category, thumb_url) in title_to_source.items():
        candidates.append(
            {
                "title": title,
                "category": category,
                "download_url": thumb_url,
                "original_url": special_filepath_url(title).removesuffix(f"?width={MAX_DIMENSION}"),
                "source_url": commons_file_url(title),
                "meta": {},
            }
        )

    candidates.sort(key=lambda item: (item["category"], item["title"].lower()))
    return candidates


def download_and_save(candidate: dict[str, Any], output_path: Path) -> tuple[int, int] | None:
    try:
        response = requests.get(
            candidate["download_url"],
            headers={"User-Agent": USER_AGENT},
            stream=True,
            timeout=(5, 5),
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"not an image response: {content_type}")
        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > MAX_DOWNLOAD_BYTES:
                raise ValueError("download exceeds size limit")
            chunks.append(chunk)
        image = Image.open(BytesIO(b"".join(chunks)))
        image = ImageOps.exif_transpose(image)
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
        if image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, "white")
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        image.save(output_path, "JPEG", quality=88, optimize=True)
        return image.size
    except Exception as exc:  # noqa: BLE001 - continue dataset collection on bad files.
        print(f"skip {candidate['title']}: {exc}")
        return None


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)

    for old_file in IMAGE_DIR.glob("track_*.jpg"):
        old_file.unlink()
    for old_file in TMP_DIR.glob("*.jpg"):
        old_file.unlink()

    candidates = build_candidates()
    print(f"Found {len(candidates)} candidate image pages", flush=True)
    rows: list[dict[str, Any]] = []

    executor = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    future_to_item = {
        executor.submit(download_and_save, candidate, TMP_DIR / f"candidate_{idx:04d}.jpg"): (
            candidate,
            TMP_DIR / f"candidate_{idx:04d}.jpg",
        )
        for idx, candidate in enumerate(candidates, start=1)
    }

    try:
        for future in as_completed(future_to_item):
            candidate, tmp_path = future_to_item[future]
            if len(rows) >= TARGET_COUNT:
                future.cancel()
                continue
            try:
                saved_size = future.result()
            except Exception as exc:  # noqa: BLE001 - keep collecting.
                print(f"skip {candidate['title']}: {exc}", flush=True)
                continue
            if saved_size is None:
                continue
            if saved_size[0] < 180 or saved_size[1] < 80:
                tmp_path.unlink(missing_ok=True)
                continue

            filename = f"track_{len(rows) + 1:04d}.jpg"
            saved_path = IMAGE_DIR / filename
            shutil.move(str(tmp_path), saved_path)

            meta = candidate["meta"]
            print(f"{filename}: {candidate['title']}", flush=True)
            rows.append(
                {
                    "filename": filename,
                    "commons_title": candidate["title"],
                    "category": candidate["category"],
                    "source_url": candidate["source_url"],
                    "original_url": candidate["original_url"],
                    "license": metadata_value(meta, "LicenseShortName") or "see Wikimedia Commons source page",
                    "license_url": metadata_value(meta, "LicenseUrl"),
                    "author": metadata_value(meta, "Artist")
                    or metadata_value(meta, "Credit")
                    or "see Wikimedia Commons source page",
                    "original_width": saved_size[0],
                    "original_height": saved_size[1],
                    "width": saved_size[0],
                    "height": saved_size[1],
                }
            )
            if len(rows) >= TARGET_COUNT:
                for pending in future_to_item:
                    pending.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if len(rows) < TARGET_COUNT:
        raise SystemExit(f"Only downloaded {len(rows)} images, expected {TARGET_COUNT}")

    with SOURCES_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filename",
                "commons_title",
                "category",
                "source_url",
                "original_url",
                "license",
                "license_url",
                "author",
                "original_width",
                "original_height",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Downloaded {len(rows)} images into {IMAGE_DIR}")
    print(f"Wrote metadata to {SOURCES_CSV}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
