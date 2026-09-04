"""Photo Curator: AI-powered photo selector and album curator."""

from __future__ import annotations

from photo_curator.cli import main
from photo_curator.models import ScoredImage
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

__version__ = "1.1.0"

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
]
