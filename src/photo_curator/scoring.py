"""Computer vision, deep learning, and quality scoring functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer, util

from photo_curator.models import (
    DEEPFACE_AVAILABLE,
    DEFAULT_WEIGHT_AESTHETIC,
    DEFAULT_WEIGHT_EMOTION,
    DEFAULT_WEIGHT_TECHNICAL,
    MAX_ANALYSIS_DIMENSION,
    SHARPNESS_MAX,
    SHARPNESS_MIN,
    SUPPORTED_EXTENSIONS,
)

DeepFace = None
if DEEPFACE_AVAILABLE:
    try:
        from deepface import DeepFace
    except ImportError:
        pass

logger = logging.getLogger(__name__)


def is_image_file(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def technical_score(image: np.ndarray) -> float:
    """Sharpness + exposure score, normalized to [0, 1].

    Sharpness is the variance of the Laplacian, standardized by resizing
    the image to a canonical resolution (max edge MAX_ANALYSIS_DIMENSION)
    to prevent sensor resolution bias, clamped to [SHARPNESS_MIN, SHARPNESS_MAX]
    and scaled to [0, 1].

    Exposure is normalized to [0, 1] using distance from midpoint (0.5),
    where 0.5 is 1.0 (ideal exposure) and 0.0 or 1.0 is 0.0 (pure black or blown-out white).
    The two sub-scores are averaged.
    """
    h, w = image.shape[:2]
    if max(h, w) > MAX_ANALYSIS_DIMENSION:
        scale = MAX_ANALYSIS_DIMENSION / max(h, w)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    if resized.ndim == 2:
        gray = resized
    elif resized.ndim == 3:
        channels = resized.shape[2]
        if channels == 1:
            gray = resized.squeeze(axis=2)
        elif channels == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image array shape: {image.shape}")

    # Sharpness: clamp then scale to [0, 1]
    raw_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = float(
        np.clip(raw_sharpness, SHARPNESS_MIN, SHARPNESS_MAX) / SHARPNESS_MAX
    )

    # Exposure: 1.0 when mean brightness is exactly 0.5, 0.0 at extremes (0.0 or 1.0)
    mean_brightness = float(np.mean(gray)) / 255.0
    exposure = float(np.clip(1.0 - 2.0 * abs(mean_brightness - 0.5), 0.0, 1.0))

    return (sharpness + exposure) / 2.0


def aesthetic_score_batch(
    pil_images: Sequence[Image.Image],
    clip_model: SentenceTransformer,
    ref_embedding: torch.Tensor,
    batch_size: int = 32,
) -> tuple[list[float], list[torch.Tensor]]:
    """CLIP cosine similarity between a batch of images and reference text embedding.

    Raw CLIP cosine similarity for photos typically falls in ~[0.1, 0.4].
    We rescale that range to [0, 1] so the score is comparable with the
    other signals.

    Returns ``(scores, embeddings)``. Raises RuntimeError on inference failure.
    """
    if not pil_images:
        return [], []
    try:
        img_embs = clip_model.encode(
            list(pil_images),
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        sims = util.cos_sim(img_embs, ref_embedding).squeeze(dim=-1)
        if sims.ndim == 0:
            sims = sims.unsqueeze(0)

        scores: list[float] = []
        embeddings: list[torch.Tensor] = []
        for i in range(len(pil_images)):
            sim = sims[i].item()
            score = float(np.clip((sim - 0.1) / 0.3, 0.0, 1.0))
            scores.append(score)
            embeddings.append(img_embs[i])
        return scores, embeddings
    except Exception as e:
        logger.error("Batch aesthetic scoring failed: %s", e)
        raise RuntimeError(f"CLIP aesthetic scoring failed: {e}") from e


def aesthetic_score(
    pil_img: Image.Image,
    clip_model: SentenceTransformer,
    ref_embedding: torch.Tensor,
) -> float:
    """Single-image aesthetic scoring helper."""
    scores, _ = aesthetic_score_batch(
        [pil_img], clip_model, ref_embedding, batch_size=1
    )
    return scores[0] if scores else 0.0


def emotion_score(
    image_input: Path | np.ndarray,
    *,
    deepface_enabled: bool,
) -> float:
    """Score based on detected happiness using DeepFace.

    Supports either a Path to an image file or an in-memory OpenCV BGR array.
    Aggregates all detected faces in group photos by taking the mean happiness.
    Filters out fallback results where no face was detected (face_confidence <= 0).
    Returns 0.0 if DeepFace is unavailable, disabled, or no faces are detected.
    """
    import sys

    album_mod = sys.modules.get("album_selector")
    is_available = getattr(album_mod, "DEEPFACE_AVAILABLE", None)
    if is_available is None:
        is_available = DEEPFACE_AVAILABLE

    deepface_cls = getattr(album_mod, "DeepFace", None) or globals().get("DeepFace")

    if not deepface_enabled or not is_available or deepface_cls is None:
        return 0.0

    try:
        target = (
            str(image_input) if isinstance(image_input, Path) else image_input
        )
        result = deepface_cls.analyze(
            img_path=target,
            actions=["emotion"],
            enforce_detection=False,
        )

        if not isinstance(result, list):
            result = [result]

        face_scores: list[float] = []
        for face in result:
            if isinstance(face, dict) and "emotion" in face:
                confidence = face.get(
                    "face_confidence", face.get("confidence", 1.0)
                )
                if confidence is None or confidence <= 0:
                    continue
                happy = face["emotion"].get("happy", 0.0)
                face_scores.append(happy / 100.0)

        if not face_scores:
            return 0.0

        return float(np.mean(face_scores))

    except Exception as e:
        name = (
            image_input.name
            if isinstance(image_input, Path)
            else "in-memory image"
        )
        logger.warning("DeepFace analysis failed for %s: %s", name, e)
        return 0.0


def calculate_total_score(
    technical: float,
    aesthetic: float,
    emotion: float,
    *,
    weight_technical: float = DEFAULT_WEIGHT_TECHNICAL,
    weight_aesthetic: float = DEFAULT_WEIGHT_AESTHETIC,
    weight_emotion: float = DEFAULT_WEIGHT_EMOTION,
    emotion_active: bool = True,
) -> float:
    """Calculate normalized total score across active signals.

    When emotion scoring is disabled or unavailable, weights dynamically
    rebalance across technical and aesthetic signals so total scores span [0, 1].
    """
    if not emotion_active:
        total_w = weight_technical + weight_aesthetic
        if total_w <= 0:
            return 0.0
        return (
            weight_technical * technical + weight_aesthetic * aesthetic
        ) / total_w

    total_w = weight_technical + weight_aesthetic + weight_emotion
    if total_w <= 0:
        return 0.0
    return (
        weight_technical * technical
        + weight_aesthetic * aesthetic
        + weight_emotion * emotion
    ) / total_w


def load_clip_model(device: str = "cpu") -> SentenceTransformer:
    """Load the CLIP SentenceTransformer model."""
    return SentenceTransformer("clip-ViT-B-32", device=device)


def encode_reference_text(
    clip_model: SentenceTransformer,
    text: str = "a beautiful photograph",
) -> torch.Tensor:
    """Encode reference text once so it can be reused for every image."""
    return clip_model.encode([text], convert_to_tensor=True)


def load_image(
    path: Path,
    max_dimension: int = MAX_ANALYSIS_DIMENSION,
) -> tuple[Image.Image | None, np.ndarray | None]:
    """Read an image once and return both PIL (RGB) and OpenCV (BGR) arrays.

    Applies EXIF orientation transpose so phone/camera photos are correctly rotated.
    If the image exceeds *max_dimension* along its longest edge, it is resized
    using high-quality downsampling to protect working memory from multi-gigabyte spikes.
    Returns ``(pil_img, cv_img)`` or ``(None, None)`` on failure.
    """
    try:
        with Image.open(path) as raw_img:
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
            if max_dimension is not None and max(pil_img.size) > max_dimension:
                pil_img.thumbnail(
                    (max_dimension, max_dimension),
                    Image.Resampling.LANCZOS,
                )
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except (OSError, SyntaxError, ValueError) as e:
        logger.warning("Could not read image %s: %s", path.name, e)
        return None, None
