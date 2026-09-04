"""Command line interface and orchestration for photo_curator."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from photo_curator.models import (
    DEEPFACE_AVAILABLE,
    DEFAULT_WEIGHT_AESTHETIC,
    DEFAULT_WEIGHT_EMOTION,
    DEFAULT_WEIGHT_TECHNICAL,
    LEGACY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    ScoredImage,
)
from photo_curator.scoring import (
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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

    # Validate torch device early
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
        if not p.is_file() or not is_image_file(p):
            continue
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

    # --- Deduplication / Selection ------------------------------------------
    selected_images, suppressed_paths = filter_near_duplicates(
        scored_images,
        threshold=args.dedup_threshold,
        target_count=args.target,
        return_suppressed=True,
    )

    # --- Optional CSV export (includes full ranking and selection status) ---
    if args.preview_csv:
        selected_set = {s.path for s in selected_images}
        statuses = {}
        for s in scored_images:
            if s.path in selected_set:
                statuses[s.path] = "Selected"
            elif s.path in suppressed_paths:
                statuses[s.path] = "Duplicate_Suppressed"
            else:
                statuses[s.path] = "Rank_Cutoff"
        export_scores_csv(
            scored_images, Path(args.preview_csv), statuses=statuses
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
