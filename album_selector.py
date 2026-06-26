#!/usr/bin/env python3
"""AI-based photo selector for albums.

Scores photos using technical quality (sharpness + exposure), aesthetic quality
(CLIP cosine similarity), and optional emotion detection (DeepFace happiness),
then selects the top-N images and copies them to an output folder.
"""

from __future__ import annotations

import csv
import logging
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

# DeepFace for emotion scoring
try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError as e:
    logger.info("DeepFace not found; emotion scoring disabled. (%s)", e)
    DEEPFACE_AVAILABLE = False

# pillow-heif for HEIC support
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
if HEIC_SUPPORTED:
    SUPPORTED_EXTENSIONS.add(".heic")

# Default scoring weights
WEIGHT_TECHNICAL = 0.4
WEIGHT_AESTHETIC = 0.4
WEIGHT_EMOTION = 0.2

# Sharpness normalization — Laplacian variance is clamped to this range
# before mapping to [0, 1]. Typical values: blurry <50, sharp >500.
SHARPNESS_MIN = 0.0
SHARPNESS_MAX = 1000.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ScoredImage:
    """An image path together with its component scores."""

    path: Path
    total: float
    technical: float
    aesthetic: float
    emotion: float


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def is_image_file(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def technical_score(image: np.ndarray) -> float:
    """Sharpness + exposure score, normalized to [0, 1].

    Sharpness is the variance of the Laplacian, clamped to
    [SHARPNESS_MIN, SHARPNESS_MAX] and scaled to [0, 1].
    Exposure is how close mean brightness is to the midpoint (0.5).
    The two sub-scores are averaged.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness: clamp then scale to [0, 1]
    raw_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = float(
        np.clip(raw_sharpness, SHARPNESS_MIN, SHARPNESS_MAX) / SHARPNESS_MAX
    )

    # Exposure: 1.0 when mean brightness is exactly 0.5, 0.5 at extremes
    mean_brightness = float(np.mean(gray)) / 255.0
    exposure = 1.0 - abs(mean_brightness - 0.5)

    return (sharpness + exposure) / 2.0


def aesthetic_score(
    pil_img: Image.Image,
    clip_model: SentenceTransformer,
    ref_embedding: torch.Tensor,
) -> float:
    """CLIP cosine similarity between *pil_img* and a cached reference text embedding.

    Raw CLIP cosine similarity for photos typically falls in ~[0.1, 0.4].
    We rescale that range to [0, 1] so the score is comparable with the
    other signals.
    """
    try:
        img_emb = clip_model.encode([pil_img], convert_to_tensor=True)
        sim = util.cos_sim(img_emb, ref_embedding).item()
        # Rescale from ~[0.1, 0.4] → [0, 1]
        return float(np.clip((sim - 0.1) / 0.3, 0.0, 1.0))
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Aesthetic scoring failed: %s", e)
        return 0.0


def emotion_score(image_path: Path, *, deepface_enabled: bool) -> float:
    """Score based on detected happiness using DeepFace.

    Returns 0.0 if DeepFace is unavailable or disabled.
    """
    if not deepface_enabled or not DEEPFACE_AVAILABLE:
        return 0.0

    try:
        result = DeepFace.analyze(
            img_path=str(image_path),
            actions=["emotion"],
            enforce_detection=False,
        )

        if isinstance(result, list) and result:
            result = result[0]

        return result.get("emotion", {}).get("happy", 0) / 100.0

    except (OSError, ValueError, AttributeError) as e:
        logger.warning("DeepFace analysis failed for %s: %s", image_path.name, e)
        return 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_clip_model(device: str = "cpu") -> SentenceTransformer:
    """Load the CLIP SentenceTransformer model."""
    return SentenceTransformer("clip-ViT-B-32", device=device)


def encode_reference_text(
    clip_model: SentenceTransformer,
    text: str = "a beautiful photograph",
) -> torch.Tensor:
    """Encode reference text once so it can be reused for every image."""
    return clip_model.encode([text], convert_to_tensor=True)


# ---------------------------------------------------------------------------
# Image I/O helper
# ---------------------------------------------------------------------------


def load_image(path: Path) -> tuple[Image.Image | None, np.ndarray | None]:
    """Read an image once and return both PIL (RGB) and OpenCV (BGR) arrays.

    Returns ``(pil_img, cv_img)`` or ``(None, None)`` on failure.
    """
    try:
        pil_img = Image.open(path).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except (OSError, SyntaxError) as e:
        logger.warning("Could not read image %s: %s", path.name, e)
        return None, None


# ---------------------------------------------------------------------------
# Export & Copy helpers
# ---------------------------------------------------------------------------


def export_scores_csv(scored_images: list[ScoredImage], csv_path: Path) -> None:
    """Write scored images to a CSV file with formatted values."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Path", "TotalScore", "Technical", "Aesthetic", "Emotion"])
        for s in scored_images:
            writer.writerow(
                [
                    s.path,
                    f"{s.total:.4f}",
                    f"{s.technical:.4f}",
                    f"{s.aesthetic:.4f}",
                    f"{s.emotion:.4f}",
                ]
            )
    logger.info("Preview CSV saved to %s", csv_path)


def copy_top_images(images: list[ScoredImage], output_path: Path) -> None:
    """Copy the selected images to *output_path* with zero-padded index prefixes."""
    width = len(str(len(images))) if images else 1
    for idx, scored in enumerate(tqdm(images, desc="Copying"), start=1):
        dest = output_path / f"{idx:0{width}d}_{scored.path.name}"
        shutil.copy2(scored.path, dest)
    logger.info("Copied %d images to %s", len(images), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, score images, and copy the best ones."""
    parser = argparse.ArgumentParser(
        description="AI-based photo selector for albums.",
    )
    parser.add_argument("--input", required=True, help="Input folder with photos")
    parser.add_argument(
        "--output", required=True, help="Output folder for selected photos"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=200,
        help="Number of photos to select (default: 200)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, or mps (default: cpu)",
    )
    parser.add_argument(
        "--preview_csv",
        default=None,
        help="Optional: save all scores to a CSV before copying",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Only score and preview; do not copy photos",
    )
    parser.add_argument(
        "--no_deepface",
        action="store_true",
        help="Disable DeepFace emotion scoring",
    )
    parser.add_argument(
        "--ref_text",
        default="a beautiful photograph",
        help="Reference text for CLIP aesthetic scoring (default: 'a beautiful photograph')",
    )
    args = parser.parse_args()

    # --- Input validation ---------------------------------------------------
    input_path = Path(args.input)
    if not input_path.is_dir():
        parser.error(f"Input path does not exist or is not a directory: {input_path}")

    if args.target <= 0:
        parser.error(f"--target must be a positive integer, got {args.target}")

    # Validate torch device early so we fail fast
    try:
        torch.zeros(1, device=args.device)
    except RuntimeError as e:
        parser.error(f"Invalid --device '{args.device}': {e}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Collect images -----------------------------------------------------
    images = sorted(p for p in input_path.rglob("*") if is_image_file(p))
    if not images:
        logger.error("No supported images found in %s", input_path)
        return

    logger.info("Found %d images.", len(images))

    # --- Load models --------------------------------------------------------
    clip_model = load_clip_model(device=args.device)
    ref_embedding = encode_reference_text(clip_model, text=args.ref_text)
    use_deepface = not args.no_deepface

    # --- Score images -------------------------------------------------------
    scored_images: list[ScoredImage] = []

    for img_path in tqdm(images, desc="Scoring"):
        pil_img, cv_img = load_image(img_path)
        if pil_img is None or cv_img is None:
            continue

        try:
            tech = technical_score(cv_img)
            aes = aesthetic_score(pil_img, clip_model, ref_embedding)
            emo = emotion_score(img_path, deepface_enabled=use_deepface)

            total = (
                WEIGHT_TECHNICAL * tech
                + WEIGHT_AESTHETIC * aes
                + WEIGHT_EMOTION * emo
            )
            scored_images.append(
                ScoredImage(
                    path=img_path,
                    total=total,
                    technical=tech,
                    aesthetic=aes,
                    emotion=emo,
                )
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Error scoring %s: %s", img_path.name, e)

    # --- Sort by total score (descending) -----------------------------------
    scored_images.sort(key=lambda s: s.total, reverse=True)

    # --- Optional CSV export ------------------------------------------------
    if args.preview_csv:
        export_scores_csv(scored_images, Path(args.preview_csv))

    # --- Copy top images ----------------------------------------------------
    if not args.dryrun:
        copy_top_images(scored_images[: args.target], output_path)
    else:
        logger.info("Dry run complete. No files were copied.")


if __name__ == "__main__":
    main()
