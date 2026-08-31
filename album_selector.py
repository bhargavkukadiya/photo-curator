#!/usr/bin/env python3
"""AI-based photo selector for albums.

Scores photos using technical quality (sharpness + exposure), aesthetic quality
(CLIP cosine similarity), and optional emotion detection (DeepFace happiness),
then selects the top-N images and copies them to an output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
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

# pillow-heif for HEIC/HEIF support
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Default scoring weights
DEFAULT_WEIGHT_TECHNICAL = 0.4
DEFAULT_WEIGHT_AESTHETIC = 0.4
DEFAULT_WEIGHT_EMOTION = 0.2

# Sharpness normalization — Laplacian variance is clamped to this range
# before mapping to [0, 1]. Typical values: blurry <50, sharp >500.
SHARPNESS_MIN = 0.0
SHARPNESS_MAX = 1000.0
MAX_ANALYSIS_DIMENSION = 1024  # Standard canonical max edge for Laplacian variance

MANIFEST_FILENAME = ".curator_manifest.json"
LEGACY_MANIFEST_FILENAME = ".curator_manifest.txt"



# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ScoredImage:
    """An image path together with its component scores and optional embedding."""

    path: Path
    total: float
    technical: float
    aesthetic: float
    emotion: float
    embedding: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


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

    gray = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if len(resized.shape) == 3
        else resized
    )

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
    if not deepface_enabled or not DEEPFACE_AVAILABLE:
        return 0.0

    try:
        target = (
            str(image_input) if isinstance(image_input, Path) else image_input
        )
        result = DeepFace.analyze(
            img_path=target,
            actions=["emotion"],
            enforce_detection=False,
        )

        if not isinstance(result, list):
            result = [result]

        face_scores: list[float] = []
        for face in result:
            if isinstance(face, dict) and "emotion" in face:
                # DeepFace returns face_confidence=0 when enforce_detection=False
                # and no actual face was detected.
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


def filter_near_duplicates(
    scored_images: list[ScoredImage],
    threshold: float | None = None,
    target_count: int | None = None,
) -> list[ScoredImage]:
    """Filter out near-duplicate burst shots using CLIP cosine similarity.

    Assumes *scored_images* is pre-sorted descending by score.
    Iteratively selects the highest scoring candidate that is not too similar
    (cosine similarity >= threshold) to any already selected candidate.
    """
    if threshold is None or threshold > 1.0 or not scored_images:
        return scored_images[:target_count] if target_count else scored_images

    selected: list[ScoredImage] = []
    for candidate in scored_images:
        if candidate.embedding is None:
            selected.append(candidate)
            if target_count and len(selected) >= target_count:
                break
            continue

        is_duplicate = False
        for chosen in selected:
            if chosen.embedding is not None:
                sim = util.cos_sim(
                    candidate.embedding, chosen.embedding
                ).item()
                if sim >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            selected.append(candidate)
            if target_count and len(selected) >= target_count:
                break

    return selected


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

    Applies EXIF orientation transpose so phone/camera photos are correctly rotated.
    Returns ``(pil_img, cv_img)`` or ``(None, None)`` on failure.
    """
    try:
        with Image.open(path) as raw_img:
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return pil_img, cv_img
    except (OSError, SyntaxError, ValueError) as e:
        logger.warning("Could not read image %s: %s", path.name, e)
        return None, None


# ---------------------------------------------------------------------------
# Export & Copy helpers
# ---------------------------------------------------------------------------


def export_scores_csv(scored_images: list[ScoredImage], csv_path: Path) -> None:
    """Write scored images to a CSV file with formatted values."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Path", "TotalScore", "Technical", "Aesthetic", "Emotion"]
        )
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


def copy_top_images(
    images: list[ScoredImage],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Copy the selected images to *output_path* with zero-padded index prefixes.

    Uses transactional staging and two-phase commit with full rollback capability:
    - All new images and manifest are staged in an isolated temporary directory.
    - Previous manifest and files to be replaced are safely backed up before committing.
    - If any commit operation fails, all changes are rolled back to the original state.
    - JSON manifest avoids corruption from newlines or special characters in filenames.

    Returns the number of successfully copied images.
    Raises RuntimeError if any file copy, manifest validation, cleanup, or write fails.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_output = output_path.resolve()
    manifest_file = output_path / MANIFEST_FILENAME
    legacy_manifest_file = output_path / LEGACY_MANIFEST_FILENAME

    # Reject symlinked manifest
    if manifest_file.is_symlink():
        raise RuntimeError(
            f"Manifest file '{manifest_file}' is a symlink. Refusing to proceed."
        )
    if legacy_manifest_file.is_symlink():
        raise RuntimeError(
            f"Manifest file '{legacy_manifest_file}' is a symlink. Refusing to proceed."
        )

    # P2: Enforce overwrite guard inside copy_top_images
    if not overwrite:
        has_existing_files = output_path.exists() and any(
            True for _ in output_path.iterdir()
        )
        if has_existing_files:
            raise RuntimeError(
                f"Output directory '{output_path}' is not empty. "
                "Specify overwrite=True to allow writing to it."
            )

    validated_previous_entries: set[str] = set()
    raw_entries: list[str] = []


    if manifest_file.exists():
        if not manifest_file.is_file():
            raise RuntimeError(
                f"Manifest path '{manifest_file}' exists and is not a regular file."
            )
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and "files" in data
                and isinstance(data["files"], list)
            ):
                raw_entries = data["files"]
            elif isinstance(data, list):
                raw_entries = data
            else:
                raise ValueError("Invalid manifest structure")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to read/parse manifest '{manifest_file}': {e}"
            ) from e
    elif legacy_manifest_file.exists():
        if not legacy_manifest_file.is_file():
            raise RuntimeError(
                f"Manifest path '{legacy_manifest_file}' exists and is not a regular file."
            )
        try:
            raw_entries = legacy_manifest_file.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as e:
            raise RuntimeError(
                f"Failed to read existing legacy manifest '{legacy_manifest_file}': {e}"
            ) from e

    for item in raw_entries:
        if not isinstance(item, str):
            raise RuntimeError(
                f"Manifest security error: invalid non-string entry in manifest: {item!r}"
            )
        fname = item.strip()
        if not fname:
            continue
        # P0: Path traversal protection
        if Path(fname).name != fname or fname in (".", ".."):
            raise RuntimeError(
                f"Manifest security error: invalid or path-traversal entry '{fname}' in manifest."
            )
        target = output_path / fname
        try:
            resolved_target = target.resolve()
            if resolved_target.parent != resolved_output:
                raise RuntimeError(
                    f"Manifest security error: entry '{fname}' resolves outside output directory."
                )
        except (OSError, RuntimeError) as e:
            raise RuntimeError(
                f"Manifest security error validating '{fname}': {e}"
            ) from e

        # P1: Reject directories and non-regular tracked entries before starting transaction
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(
                    f"Manifest security error: tracked entry '{fname}' in '{manifest_file}' is not a regular file (directory or special file)."
                )

        validated_previous_entries.add(fname)

    if not images:
        return 0

    width = len(str(len(images)))
    new_manifest_entries: list[str] = []

    # Pre-validate all target destinations before copying anything
    for idx, scored in enumerate(images, start=1):
        filename = f"{idx:0{width}d}_{scored.path.name}"
        dest = output_path / filename

        # P1: Reject untracked destination collisions
        if dest.exists() or dest.is_symlink():
            if filename not in validated_previous_entries:
                raise RuntimeError(
                    f"Destination file '{dest}' already exists and is not tracked by the previous manifest. "
                    "Refusing to overwrite untracked file."
                )
        new_manifest_entries.append(filename)

    # P2: Allocate staging and backup directories under cleanup protection
    staging_dir: Path | None = None
    backup_dir: Path | None = None
    preserve_backup = False

    try:
        staging_dir = Path(
            tempfile.mkdtemp(dir=output_path, prefix=".curator_stage_")
        )
        backup_dir = Path(
            tempfile.mkdtemp(dir=output_path, prefix=".curator_backup_")
        )

        # Step 1: Copy all images into the staging directory
        for scored, filename in zip(
            tqdm(images, desc="Copying"), new_manifest_entries
        ):
            staged_dest = staging_dir / filename
            try:
                shutil.copy2(scored.path, staged_dest)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to copy '{scored.path}' to staging: {e}"
                ) from e

        # Step 2: Create staged JSON manifest
        staged_manifest = staging_dir / MANIFEST_FILENAME
        manifest_payload = {
            "version": 1,
            "files": new_manifest_entries,
        }
        manifest_bytes = json.dumps(manifest_payload, indent=2).encode("utf-8")
        try:
            fd = os.open(
                staged_manifest,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with open(fd, "wb", closefd=True) as mf:
                mf.write(manifest_bytes)
        except OSError as e:
            raise RuntimeError(
                f"Failed to write staging manifest '{staged_manifest}': {e}"
            ) from e

        # Step 3: Transactional Two-Phase Commit with Full Rollback
        moved_to_backup: list[tuple[Path, Path]] = []
        committed_new_files: list[Path] = []
        manifest_backed_up = False
        legacy_manifest_backed_up = False

        try:
            # 3A: Backup existing manifest files
            if manifest_file.exists():
                os.replace(manifest_file, backup_dir / MANIFEST_FILENAME)
                manifest_backed_up = True
            if legacy_manifest_file.exists():
                os.replace(
                    legacy_manifest_file, backup_dir / LEGACY_MANIFEST_FILENAME
                )
                legacy_manifest_backed_up = True

            # Backup previous tracked files (both obsolete ones and ones being replaced)
            for fname in sorted(validated_previous_entries):
                dest_file = output_path / fname
                if dest_file.exists() or dest_file.is_symlink():
                    bkp_file = backup_dir / fname
                    os.replace(dest_file, bkp_file)
                    moved_to_backup.append((dest_file, bkp_file))

            # 3B: Move all staged files into output_path using atomic no-clobber
            for filename in new_manifest_entries:
                staged_src = staging_dir / filename
                final_dest = output_path / filename
                linked_ok = False
                try:
                    os.link(staged_src, final_dest)
                    committed_new_files.append(final_dest)
                    linked_ok = True
                except FileExistsError as e:
                    raise RuntimeError(
                        f"Destination file '{final_dest}' already exists (untracked file created during run). "
                        "Refusing to overwrite untracked file."
                    ) from e
                except OSError:
                    # Hard links not supported on this filesystem (e.g. FAT32, exFAT, network shares).
                    # Perform race-safe atomic exclusive file creation using O_CREAT | O_EXCL.
                    try:
                        fd = os.open(
                            final_dest,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                    except FileExistsError as e:
                        raise RuntimeError(
                            f"Destination file '{final_dest}' already exists (untracked file created during run). "
                            "Refusing to overwrite untracked file."
                        ) from e
                    except OSError as e:
                        raise RuntimeError(
                            f"Exclusive file creation failed for '{final_dest}': {e}"
                        ) from e

                    # Destination now exists on disk; register immediately for rollback tracking
                    committed_new_files.append(final_dest)

                    try:
                        with open(fd, "wb", closefd=True) as out_f, open(
                            staged_src, "rb"
                        ) as in_f:
                            shutil.copyfileobj(in_f, out_f)
                        # Preserve metadata (mtime, atime, mode, flags) from staged photo
                        shutil.copystat(
                            staged_src, final_dest, follow_symlinks=False
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to stream copy data/metadata to '{final_dest}': {e}"
                        ) from e

                if linked_ok:
                    try:
                        staged_src.unlink()
                    except OSError as e:
                        logger.warning(
                            "Failed to remove staging link '%s': %s", staged_src, e
                        )





            # 3C: Atomically move new manifest into output_path
            if manifest_file.is_symlink():
                raise RuntimeError(
                    f"Manifest '{manifest_file}' was replaced with a symlink during run."
                )
            os.replace(staged_manifest, manifest_file)

        except Exception as e:
            # Rollback on ANY commit failure
            logger.error("Commit failed, rolling back changes: %s", e)
            rollback_errors: list[str] = []

            for cf in committed_new_files:
                try:
                    if cf.exists() or cf.is_symlink():
                        cf.unlink()
                except OSError as err:
                    rollback_errors.append(
                        f"Failed to unlink uncommitted file '{cf}': {err}"
                    )

            for orig_dest, bkp_file in reversed(moved_to_backup):
                try:
                    if bkp_file.exists() or bkp_file.is_symlink():
                        os.replace(bkp_file, orig_dest)
                except OSError as err:
                    rollback_errors.append(
                        f"Failed to restore '{orig_dest}' from backup: {err}"
                    )

            if manifest_backed_up and (backup_dir / MANIFEST_FILENAME).exists():
                try:
                    os.replace(backup_dir / MANIFEST_FILENAME, manifest_file)
                except OSError as err:
                    rollback_errors.append(
                        f"Failed to restore manifest '{manifest_file}': {err}"
                    )
            if (
                legacy_manifest_backed_up
                and (backup_dir / LEGACY_MANIFEST_FILENAME).exists()
            ):
                try:
                    os.replace(
                        backup_dir / LEGACY_MANIFEST_FILENAME,
                        legacy_manifest_file,
                    )
                except OSError as err:
                    rollback_errors.append(
                        f"Failed to restore legacy manifest: {err}"
                    )

            if rollback_errors:
                preserve_backup = True
                logger.critical(
                    "Rollback encountered errors. Backup files preserved at: %s",
                    backup_dir,
                )
                raise RuntimeError(
                    f"Commit failed ({e}) and rollback encountered errors. "
                    f"Recovery files preserved in '{backup_dir}': {'; '.join(rollback_errors)}"
                ) from e

            raise RuntimeError(
                f"Commit phase failed and was rolled back: {e}"
            ) from e

    finally:
        # Cleanup temporary staging and backup directories
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if (
            backup_dir is not None
            and backup_dir.exists()
            and not preserve_backup
        ):
            shutil.rmtree(backup_dir, ignore_errors=True)

    logger.info("Copied %d images to %s", len(images), output_path)
    return len(images)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, score images, and copy the best ones."""
    parser = argparse.ArgumentParser(
        description="AI-based photo selector for albums.",
    )
    parser.add_argument(
        "--input", required=True, help="Input folder with photos"
    )
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP inference (default: 32)",
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
    parser.add_argument(
        "--weight_technical",
        type=float,
        default=DEFAULT_WEIGHT_TECHNICAL,
        help=f"Weight for technical quality score (default: {DEFAULT_WEIGHT_TECHNICAL})",
    )
    parser.add_argument(
        "--weight_aesthetic",
        type=float,
        default=DEFAULT_WEIGHT_AESTHETIC,
        help=f"Weight for aesthetic quality score (default: {DEFAULT_WEIGHT_AESTHETIC})",
    )
    parser.add_argument(
        "--weight_emotion",
        type=float,
        default=DEFAULT_WEIGHT_EMOTION,
        help=f"Weight for emotion score (default: {DEFAULT_WEIGHT_EMOTION})",
    )
    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=None,
        help="Optional cosine similarity threshold [0.0-1.0] to suppress near-duplicates (e.g. 0.90)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting non-empty output directory or preview CSV",
    )
    args = parser.parse_args()

    # --- Input validation ---------------------------------------------------
    input_path = Path(args.input)
    if not input_path.is_dir():
        parser.error(
            f"Input path does not exist or is not a directory: {input_path}"
        )

    if args.target <= 0:
        parser.error(f"--target must be a positive integer, got {args.target}")

    if args.batch_size <= 0:
        parser.error(
            f"--batch_size must be a positive integer, got {args.batch_size}"
        )

    # Validate weights are finite non-negative floats
    for name, val in [
        ("--weight_technical", args.weight_technical),
        ("--weight_aesthetic", args.weight_aesthetic),
        ("--weight_emotion", args.weight_emotion),
    ]:
        if not math.isfinite(val) or val < 0:
            parser.error(
                f"{name} must be a finite non-negative number, got {val}"
            )

    use_deepface = (not args.no_deepface) and DEEPFACE_AVAILABLE

    active_weight_sum = args.weight_technical + args.weight_aesthetic
    if use_deepface:
        active_weight_sum += args.weight_emotion

    if active_weight_sum <= 0:
        if not use_deepface and args.weight_emotion > 0:
            parser.error(
                "Sum of active scoring weights must be greater than 0. "
                "Emotion scoring is disabled/unavailable, but technical and aesthetic weights are 0."
            )
        parser.error("Sum of active scoring weights must be greater than 0.")

    if args.dedup_threshold is not None:
        if not math.isfinite(args.dedup_threshold) or not (
            0.0 <= args.dedup_threshold <= 1.0
        ):
            parser.error(
                f"--dedup_threshold must be a finite number between 0.0 and 1.0, got {args.dedup_threshold}"
            )

    # Validate torch device early (catches RuntimeError, AssertionError, ValueError)
    try:
        device = torch.device(args.device)
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError, ValueError) as e:
        parser.error(f"Invalid or unavailable --device '{args.device}': {e}")

    output_path = Path(args.output).resolve()
    resolved_input = input_path.resolve()

    if output_path.exists() and not output_path.is_dir():
        parser.error(
            f"--output path '{output_path}' exists and is not a directory."
        )

    if output_path == resolved_input or resolved_input.is_relative_to(output_path):
        parser.error(
            f"--output path '{output_path}' cannot be the input directory or an ancestor of the input directory."
        )

    manifest_path = output_path / MANIFEST_FILENAME
    legacy_manifest_path = output_path / LEGACY_MANIFEST_FILENAME
    if manifest_path.is_symlink() or legacy_manifest_path.is_symlink():
        parser.error(
            f"Manifest file in '{output_path}' is a symlink. Refusing to operate."
        )

    if not args.dryrun:
        has_existing_files = output_path.exists() and any(
            True for _ in output_path.iterdir()
        )
        if has_existing_files and not args.overwrite:
            parser.error(
                f"Output directory '{output_path}' is not empty. "
                "Specify --overwrite to allow writing to it."
            )
        output_path.mkdir(parents=True, exist_ok=True)


    # Validate preview CSV path
    if args.preview_csv:
        csv_path = Path(args.preview_csv).resolve()
        if csv_path.exists() and csv_path.is_dir():
            parser.error(
                f"--preview_csv path '{csv_path}' is a directory, not a file."
            )
        if csv_path == resolved_input:
            parser.error(
                f"--preview_csv path '{csv_path}' cannot be the input directory."
            )
        if csv_path in (manifest_path, legacy_manifest_path) or csv_path.name in (
            MANIFEST_FILENAME,
            LEGACY_MANIFEST_FILENAME,
        ):
            parser.error(
                f"--preview_csv path '{csv_path}' conflicts with the internal curator manifest."
            )
        if is_image_file(csv_path):
            parser.error(
                f"--preview_csv path '{csv_path}' conflicts with an image file extension."
            )

        if csv_path.exists() and not args.overwrite:
            parser.error(
                f"--preview_csv file '{csv_path}' already exists. Specify --overwrite to replace it."
            )

    # --- Collect images (excluding output directory if inside input) ---------
    images: list[Path] = []
    for p in sorted(input_path.rglob("*")):
        if not is_image_file(p):
            continue
        # Avoid scanning files that are inside the output directory (even in dryrun)
        if output_path.is_relative_to(resolved_input):
            try:
                if p.resolve().is_relative_to(output_path):
                    continue
            except ValueError:
                pass
        images.append(p)

    if not images:
        logger.error("No supported images found in %s", input_path)
        sys.exit(1)

    # Prevent preview CSV collision with collected source images
    if args.preview_csv:
        resolved_csv = Path(args.preview_csv).resolve()
        for img_p in images:
            if img_p.resolve() == resolved_csv:
                parser.error(
                    f"--preview_csv path '{resolved_csv}' conflicts with an input photo ({img_p.name})."
                )

    logger.info("Found %d images.", len(images))

    # --- Load models --------------------------------------------------------
    clip_model = load_clip_model(device=args.device)
    ref_embedding = encode_reference_text(clip_model, text=args.ref_text)

    # --- Score images in batches --------------------------------------------
    scored_images: list[ScoredImage] = []
    need_embeddings = args.dedup_threshold is not None

    for i in tqdm(
        range(0, len(images), args.batch_size), desc="Scoring Batches"
    ):
        batch_paths = images[i : i + args.batch_size]
        batch_pil: list[Image.Image] = []
        batch_cv: list[np.ndarray] = []
        valid_paths: list[Path] = []

        for p in batch_paths:
            pil_img, cv_img = load_image(p)
            if pil_img is not None and cv_img is not None:
                batch_pil.append(pil_img)
                batch_cv.append(cv_img)
                valid_paths.append(p)

        if not batch_pil:
            continue

        # Batch aesthetic scoring via CLIP
        try:
            batch_aes_scores, batch_embs = aesthetic_score_batch(
                batch_pil,
                clip_model,
                ref_embedding,
                batch_size=args.batch_size,
            )
        except RuntimeError as e:
            logger.error("Aesthetic scoring aborted: %s", e)
            sys.exit(1)

        # Technical and emotion scoring per valid image in the batch
        for p, pil_img, cv_img, aes, emb in zip(
            valid_paths,
            batch_pil,
            batch_cv,
            batch_aes_scores,
            batch_embs,
        ):
            try:
                tech = technical_score(cv_img)
                emo = emotion_score(cv_img, deepface_enabled=use_deepface)

                total = calculate_total_score(
                    technical=tech,
                    aesthetic=aes,
                    emotion=emo,
                    weight_technical=args.weight_technical,
                    weight_aesthetic=args.weight_aesthetic,
                    weight_emotion=args.weight_emotion,
                    emotion_active=use_deepface,
                )

                # Only retain embedding if deduplication is active, moved to CPU
                saved_emb = emb.detach().cpu() if need_embeddings else None

                scored_images.append(
                    ScoredImage(
                        path=p,
                        total=total,
                        technical=tech,
                        aesthetic=aes,
                        emotion=emo,
                        embedding=saved_emb,
                    )
                )
            except Exception as e:
                logger.warning("Error scoring %s: %s", p.name, e)

    # --- Check for successfully scored images -------------------------------
    if not scored_images:
        logger.error("No images were successfully read and scored.")
        sys.exit(1)

    # --- Sort by total score (descending) -----------------------------------
    scored_images.sort(key=lambda s: s.total, reverse=True)

    # --- Optional CSV export (before dedup, exports full ranking) -----------
    if args.preview_csv:
        export_scores_csv(scored_images, Path(args.preview_csv))

    # --- Deduplication / Selection ------------------------------------------
    selected_images = filter_near_duplicates(
        scored_images,
        threshold=args.dedup_threshold,
        target_count=args.target,
    )

    # --- Copy top images ----------------------------------------------------
    if not args.dryrun:
        try:
            copy_top_images(
                selected_images, output_path, overwrite=args.overwrite
            )
        except RuntimeError as e:
            logger.error("Failed to complete copying: %s", e)
            sys.exit(1)
    else:
        logger.info(
            "Dry run complete. Selected %d images. No files were copied.",
            len(selected_images),
        )


if __name__ == "__main__":
    main()
