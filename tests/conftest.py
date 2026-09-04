"""Shared test fixtures and image generation helpers for photo_curator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_solid_image(
    width: int = 100,
    height: int = 100,
    color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Create a solid-color BGR numpy array (OpenCV format)."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def make_noisy_image(width: int = 100, height: int = 100) -> np.ndarray:
    """Create a high-frequency noise image (very sharp)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def save_test_image(
    path: Path,
    color: tuple[int, int, int] = (128, 128, 128),
    fmt: str = "JPEG",
    exif_orientation: int | None = None,
) -> None:
    """Save a small solid-color image to disk.

    Optionally writes an EXIF orientation tag for rotation testing.
    """
    img = Image.new("RGB", (50, 50), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif_orientation is not None:
        exif = img.getexif()
        exif[0x0112] = exif_orientation  # 0x0112 is Orientation tag
        img.save(path, format=fmt, exif=exif)
    else:
        img.save(path, format=fmt)
