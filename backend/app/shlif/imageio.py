"""Image loading and dataset discovery.

Handles the two regimes in this dataset:
  * ordinary "по сортам" photos (a few MP) — load in full;
  * gigapixel panoramas (up to ~574 MP) — decode at a reduced scale via the JPEG
    ``draft`` mode so we never allocate billions of pixels at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

# our images are legitimately huge; disable the decompression-bomb guard
Image.MAX_IMAGE_PIXELS = None

# Full dataset (~1180 photos): ч1 + the much larger ч2 sub-folders.
CLASS_DIRS = {
    "ordinary": [
        "Фото руд по сортам. ч1/Рядовые руды",
        "Фото руд по сортам. ч2/рядовые",
    ],
    "hard": [
        "Фото руд по сортам. ч1/Труднообогатимые руды",
        "Фото руд по сортам. ч2/тонкие",
    ],
    "talcose": [
        "Фото руд по сортам. ч1/Оталькованные руды",
        "Фото руд по сортам. ч2/оталькованные",
    ],
}
MIN_BYTES = 60_000  # skip thumbnails / junk (real photos are >>700 KB)
TALC_ANNOT_DIR = "Фото руд по сортам. ч1/Оталькованные руды/Области оталькования"
PANORAMA_DIR = "Панорамы"

_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) from the header without decoding pixels."""
    with Image.open(path) as im:
        return im.size


def load_rgb(path: str | Path, max_pixels: int | None = None) -> np.ndarray:
    """Load an image as an HxWx3 uint8 RGB array.

    If ``max_pixels`` is given and the image is larger, it is decoded at the
    smallest power-of-two reduction that fits the budget (memory-safe for
    gigapixel panoramas). Returns the array at whatever scale was decoded.
    """
    im = Image.open(path)
    w, h = im.size
    if max_pixels is not None and w * h > max_pixels:
        factor = decode_factor(w, h, max_pixels)
        im.draft("RGB", (w // factor, h // factor))
    return np.asarray(im.convert("RGB"))


def decode_factor(w: int, h: int, max_pixels: int) -> int:
    """Smallest power-of-two divisor so that (w/f)*(h/f) <= max_pixels."""
    factor = 1
    while (w // factor) * (h // factor) > max_pixels:
        factor *= 2
    return factor


def list_class_images(root: str | Path) -> list[tuple[Path, str]]:
    """Return [(path, class_label)] for every labelled "по сортам" photo.

    The talc annotation sub-folder is skipped (those are label overlays, not
    independent samples).
    """
    root = Path(root)
    out: list[tuple[Path, str]] = []
    annot = (root / TALC_ANNOT_DIR).resolve()
    for label, dirs in CLASS_DIRS.items():
        for d in dirs:
            base = root / d
            if not base.exists():
                continue
            for p in sorted(base.iterdir()):
                if p.suffix.lower() not in _IMG_EXT:
                    continue  # skips the stray 1.bmp
                if p.resolve().parent == annot:
                    continue
                if p.stat().st_size < MIN_BYTES:
                    continue  # skip thumbnails
                out.append((p, label))
    return out


def list_panoramas(root: str | Path) -> list[Path]:
    """Unique panorama paths, sorted numerically, de-duplicated by content hash."""
    import hashlib

    base = Path(root) / PANORAMA_DIR
    if not base.exists():
        return []

    def key(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    paths = sorted((p for p in base.iterdir() if p.suffix.lower() in _IMG_EXT), key=key)
    seen: dict[str, Path] = {}
    for p in paths:
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        seen.setdefault(digest, p)  # keep first (lowest-numbered) of any duplicates
    return sorted(seen.values(), key=key)


def annotated_talc_pairs(root: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yield (raw_image, annotated_image) for every talc image that has a
    blue-contour annotation twin (same filename in the annotation sub-folder)."""
    root = Path(root)
    raw_dir = root / "Фото руд по сортам. ч1/Оталькованные руды"
    annot_dir = root / TALC_ANNOT_DIR
    if not annot_dir.exists():
        return
    for a in sorted(annot_dir.iterdir()):
        if a.suffix.lower() not in _IMG_EXT:
            continue
        raw = raw_dir / a.name
        if raw.exists():
            yield raw, a
