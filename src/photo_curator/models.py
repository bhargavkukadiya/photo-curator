"""Data models and constants for photo_curator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Optional dependencies detection
try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError as e:
    logger.info("DeepFace not found; emotion scoring disabled. (%s)", e)
    DEEPFACE_AVAILABLE = False

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}
if HEIC_SUPPORTED:
    SUPPORTED_EXTENSIONS.add(".heic")
    SUPPORTED_EXTENSIONS.add(".heif")

DEFAULT_WEIGHT_TECHNICAL = 0.4
DEFAULT_WEIGHT_AESTHETIC = 0.4
DEFAULT_WEIGHT_EMOTION = 0.2

SHARPNESS_MIN = 0.0
SHARPNESS_MAX = 1000.0
MAX_ANALYSIS_DIMENSION = 1024

MANIFEST_FILENAME = ".curator_manifest.json"
LEGACY_MANIFEST_FILENAME = ".curator_manifest.txt"


@dataclass
class ScoredImage:
    """An image path together with its component scores and optional embedding."""

    path: Path
    total: float
    technical: float
    aesthetic: float
    emotion: float
    embedding: torch.Tensor | None = None
