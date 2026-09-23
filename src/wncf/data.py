"""Manifest and crops.

Photos: data/raw/<session>/<type>_<role>_<specimen>_<view>.jpg, e.g. s1/usba_socket_02_v1.jpg.
object_id = <type>_<role>_<specimen> identifies the physical object across sessions and views.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from wncf import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
MANIFEST = REPO_ROOT / "data" / "manifest.csv"
COMPAT = REPO_ROOT / "data" / "compat.csv"
MANIFEST_COLS = ["object_id", "type", "role", "specimen", "session", "view", "path", "crop_x", "crop_y", "crop_w", "crop_h"]
COMPAT_COLS = ["plug_id", "socket_id", "compatible", "note"]
ROLES = {"plug", "socket"}
VIEWS = {"v1", "v2", "v3"}


@dataclass(frozen=True)
class PhotoName:
    object_id: str
    type: str
    role: str
    specimen: str
    view: str


def parse_photo_name(path: Path) -> PhotoName:
    # rsplit so connector types may contain underscores (dc_jack_socket_01_v2)
    parts = path.stem.rsplit("_", 3)
    if len(parts) != 4 or parts[1] not in ROLES or not parts[2].isdigit() or parts[3] not in VIEWS:
        raise ValueError(f"bad photo name {path.name}: want <type>_<plug|socket>_<NN>_<v1|v2|v3>.jpg")
    type_, role, specimen, view = parts
    return PhotoName(f"{type_}_{role}_{specimen}", type_, role, specimen, view)


def open_upright(path: Path) -> Image.Image:
    """Open with EXIF rotation applied, matching cv2.imread, so crop boxes line up."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["session"], r["object_id"], r["view"]))
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def load_crop(row: dict) -> Image.Image:
    img = open_upright(REPO_ROOT / row["path"])
    x, y, w, h = (int(row[k]) for k in ("crop_x", "crop_y", "crop_w", "crop_h"))
    return img.crop((x, y, x + w, y + h))
