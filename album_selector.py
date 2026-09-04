#!/usr/bin/env python3
"""Backward-compatibility proxy for photo_curator.

Enables legacy scripts, tests, and CLI calls pointing to album_selector.py
to run seamlessly against the modular src/photo_curator package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path when running from the repository root
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Re-exported for mock compatibility in existing unit tests
import sentence_transformers.util as util

DeepFace = None

from photo_curator.cli import logger, main
from photo_curator.models import (
    DEEPFACE_AVAILABLE,
    DEFAULT_WEIGHT_AESTHETIC,
    DEFAULT_WEIGHT_EMOTION,
    DEFAULT_WEIGHT_TECHNICAL,
    HEIC_SUPPORTED,
    LEGACY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    MAX_ANALYSIS_DIMENSION,
    SHARPNESS_MAX,
    SHARPNESS_MIN,
    SUPPORTED_EXTENSIONS,
    ScoredImage,
)
from photo_curator.scoring import (
    aesthetic_score,
    aesthetic_score_batch,
    calculate_total_score,
    emotion_score,
    encode_reference_text,
    is_image_file,
    load_clip_model,
    load_image,
    technical_score,
)
from photo_curator.selection import filter_near_duplicates
from photo_curator.storage import copy_top_images, export_scores_csv

__all__ = [
    "ScoredImage",
    "is_image_file",
    "technical_score",
    "aesthetic_score",
    "aesthetic_score_batch",
    "emotion_score",
    "calculate_total_score",
    "load_clip_model",
    "encode_reference_text",
    "load_image",
    "filter_near_duplicates",
    "export_scores_csv",
    "copy_top_images",
    "main",
    "util",
    "logger",
    "SUPPORTED_EXTENSIONS",
    "DEFAULT_WEIGHT_TECHNICAL",
    "DEFAULT_WEIGHT_AESTHETIC",
    "DEFAULT_WEIGHT_EMOTION",
    "SHARPNESS_MIN",
    "SHARPNESS_MAX",
    "MAX_ANALYSIS_DIMENSION",
    "MANIFEST_FILENAME",
    "LEGACY_MANIFEST_FILENAME",
    "DEEPFACE_AVAILABLE",
    "HEIC_SUPPORTED",
]

if __name__ == "__main__":
    main()
