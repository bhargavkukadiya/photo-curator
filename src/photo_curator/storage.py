"""Filesystem operations, transactional staging, and manifest management."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from photo_curator.models import (
    LEGACY_MANIFEST_FILENAME,
    MANIFEST_FILENAME,
    ScoredImage,
)

logger = logging.getLogger(__name__)


def export_scores_csv(
    scored_images: list[ScoredImage],
    csv_path: Path,
    statuses: dict[Path, str] | None = None,
) -> None:
    """Write scored images to a CSV file with formatted values and optional status."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Path", "TotalScore", "Technical", "Aesthetic", "Emotion"]
        if statuses is not None:
            header.append("Status")
        writer.writerow(header)
        for s in scored_images:
            row = [
                s.path,
                f"{s.total:.4f}",
                f"{s.technical:.4f}",
                f"{s.aesthetic:.4f}",
                f"{s.emotion:.4f}",
            ]
            if statuses is not None:
                row.append(statuses.get(s.path, "Unselected"))
            writer.writerow(row)
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

    # Enforce overwrite guard inside copy_top_images
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
        # Path traversal protection
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

        # Reject directories and non-regular tracked entries before starting transaction
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

        # Reject untracked destination collisions
        if dest.exists() or dest.is_symlink():
            if filename not in validated_previous_entries:
                raise RuntimeError(
                    f"Destination file '{dest}' already exists and is not tracked by the previous manifest. "
                    "Refusing to overwrite untracked file."
                )
        new_manifest_entries.append(filename)

    # Allocate staging and backup directories under cleanup protection
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
