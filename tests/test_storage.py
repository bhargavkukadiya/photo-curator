"""Unit tests for filesystem operations, transactional staging, and manifest security."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from photo_curator.models import ScoredImage
from photo_curator.storage import copy_top_images, export_scores_csv
from tests.conftest import save_test_image


class TestExportScoresCSV:
    """Test CSV output via export_scores_csv()."""

    def test_header_and_values(self, tmp_path):
        """CSV should have the correct header and formatted values."""
        csv_path = tmp_path / "scores.csv"
        scored = [
            ScoredImage(Path("a.jpg"), total=0.1234, technical=0.5678, aesthetic=0.9012, emotion=0.3456),
            ScoredImage(Path("b.jpg"), total=0.9999, technical=0.1111, aesthetic=0.2222, emotion=0.3333),
        ]

        export_scores_csv(scored, csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Path", "TotalScore", "Technical", "Aesthetic", "Emotion"]

            row1 = next(reader)
            assert row1[0] == "a.jpg"
            assert row1[1] == "0.1234"
            assert row1[2] == "0.5678"

            row2 = next(reader)
            assert row2[0] == "b.jpg"
            assert row2[1] == "0.9999"

    def test_creates_nested_parent_directory(self, tmp_path):
        """Should automatically create non-existent parent folders for the CSV."""
        nested_csv = tmp_path / "deep" / "nested" / "report.csv"
        export_scores_csv([], nested_csv)
        assert nested_csv.exists()

    def test_export_scores_csv_with_statuses(self, tmp_path):
        """export_scores_csv should include Status column when statuses dict is supplied."""
        csv_path = tmp_path / "preview.csv"
        scored = [
            ScoredImage(Path("a.jpg"), 0.9, 0.5, 0.5, 0.0),
            ScoredImage(Path("b.jpg"), 0.8, 0.5, 0.5, 0.0),
            ScoredImage(Path("c.jpg"), 0.7, 0.5, 0.5, 0.0),
        ]
        statuses = {
            Path("a.jpg"): "Selected",
            Path("b.jpg"): "Duplicate_Suppressed",
            Path("c.jpg"): "Rank_Cutoff",
        }
        export_scores_csv(scored, csv_path, statuses=statuses)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Path", "TotalScore", "Technical", "Aesthetic", "Emotion", "Status"]
            row1 = next(reader)
            assert row1[0] == "a.jpg" and row1[5] == "Selected"
            row2 = next(reader)
            assert row2[0] == "b.jpg" and row2[5] == "Duplicate_Suppressed"
            row3 = next(reader)
            assert row3[0] == "c.jpg" and row3[5] == "Rank_Cutoff"


class TestCopyTopImages:
    """Test the file copy logic via copy_top_images()."""

    def test_dynamic_filename_width(self, tmp_path):
        """Filename index width should adapt to the number of images."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"

        scored = []
        for i in range(12):
            img_path = src_dir / f"img_{i}.jpg"
            save_test_image(img_path)
            scored.append(
                ScoredImage(img_path, total=1.0 - i * 0.05, technical=0.5, aesthetic=0.5, emotion=0.0)
            )

        copied = copy_top_images(scored, out_dir)
        assert copied == 12

        output_files = sorted(p.name for p in out_dir.iterdir() if not p.name.startswith("."))
        assert output_files[0] == "01_img_0.jpg"
        assert output_files[9] == "10_img_9.jpg"
        assert output_files[11] == "12_img_11.jpg"

    def test_creates_nested_output_directory(self, tmp_path):
        """Should create nested output directory if it does not exist."""
        src_dir = tmp_path / "src"
        img_path = src_dir / "photo.jpg"
        save_test_image(img_path)

        out_dir = tmp_path / "deep" / "nested" / "album"
        scored = [ScoredImage(img_path, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0)]

        copied = copy_top_images(scored, out_dir)
        assert copied == 1
        assert (out_dir / "1_photo.jpg").exists()

    def test_empty_list(self, tmp_path):
        """Empty images list copies 0 and does not crash."""
        out_dir = tmp_path / "out"
        copied = copy_top_images([], out_dir)
        assert copied == 0

    def test_copy_failure_raises_runtime_error(self, tmp_path):
        """Shutil copy failure should raise RuntimeError and not leave corrupted manifest."""
        src_dir = tmp_path / "src"
        img_path = src_dir / "photo.jpg"
        save_test_image(img_path)

        out_dir = tmp_path / "out"
        scored = [ScoredImage(img_path, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0)]

        with mock.patch("shutil.copy2", side_effect=OSError("Disk full")):
            with pytest.raises(RuntimeError, match="Failed to copy"):
                copy_top_images(scored, out_dir)

        manifest = out_dir / ".curator_manifest.json"
        assert not manifest.exists()

    def test_overwrite_manifest_cleans_tracked_files(self, tmp_path):
        """Overwriting should safely remove previous manifest files."""
        src_dir = tmp_path / "src"
        img_a = src_dir / "a.jpg"
        img_b = src_dir / "b.jpg"
        save_test_image(img_a)
        save_test_image(img_b)

        out_dir = tmp_path / "out"

        scored_run1 = [ScoredImage(img_a, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0)]
        copy_top_images(scored_run1, out_dir)
        assert (out_dir / "1_a.jpg").exists()

        scored_run2 = [ScoredImage(img_b, total=0.95, technical=0.5, aesthetic=0.5, emotion=0.0)]
        copy_top_images(scored_run2, out_dir, overwrite=True)

        assert not (out_dir / "1_a.jpg").exists()
        assert (out_dir / "1_b.jpg").exists()

    def test_manifest_path_traversal_rejected(self, tmp_path):
        """Manifest containing path traversal (e.g. ../victim.txt) must raise RuntimeError."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        manifest_file = out_dir / ".curator_manifest.json"
        manifest_file.write_text('{"version": 1, "files": ["../victim.txt"]}')

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="Manifest security error: invalid or path-traversal entry"):
            copy_top_images(scored, out_dir, overwrite=True)

    def test_manifest_symlink_rejected(self, tmp_path):
        """Manifest pointing to a symlink must be rejected."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        target_file = tmp_path / "fake_manifest.json"
        target_file.write_text('{"version": 1, "files": []}')
        (out_dir / ".curator_manifest.json").symlink_to(target_file)

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is a symlink"):
            copy_top_images(scored, out_dir, overwrite=True)

    def test_mid_run_copy_failure_leaves_output_clean(self, tmp_path):
        """Failure mid-run in staging leaves output directory completely clean."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img1 = src_dir / "1.jpg"
        img2 = src_dir / "2.jpg"
        save_test_image(img1)
        save_test_image(img2)

        scored = [
            ScoredImage(img1, 0.9, 0.5, 0.5, 0.0),
            ScoredImage(img2, 0.8, 0.5, 0.5, 0.0),
        ]

        orig_copy2 = shutil.copy2
        call_count = [0]

        def fail_on_second(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("Disk write error")
            return orig_copy2(src, dst)

        with mock.patch("shutil.copy2", side_effect=fail_on_second):
            with pytest.raises(RuntimeError, match="Failed to copy"):
                copy_top_images(scored, out_dir)

        assert list(out_dir.iterdir()) == []

    def test_commit_failure_rolls_back_completely(self, tmp_path):
        """Failure during commit phase should roll back new files and restore previous files."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        save_test_image(old_img)

        # Run 1: initial commit
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)
        assert (out_dir / "1_old.jpg").exists()

        # Run 2: commit failure simulated during os.replace on manifest
        new_img = src_dir / "new.jpg"
        save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        orig_replace = os.replace

        def failing_replace(src, dst):
            if str(dst).endswith(".curator_manifest.json") and ".curator_stage_" in str(src):
                raise OSError("Simulated disk error replacing manifest")
            return orig_replace(src, dst)

        with mock.patch("os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError, match="Commit phase failed and was rolled back"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        assert (out_dir / "1_old.jpg").exists()
        assert not (out_dir / "1_new.jpg").exists()
        manifest_data = json.loads((out_dir / ".curator_manifest.json").read_text(encoding="utf-8"))
        assert manifest_data["files"] == ["1_old.jpg"]

    def test_newline_in_filename_handled_safely_by_json_manifest(self, tmp_path):
        """Newlines in filenames should be preserved safely without corrupting JSON manifest."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "photo\nwith\nnewline.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        copy_top_images(scored, out_dir)

        manifest = json.loads((out_dir / ".curator_manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["files"]) == 1
        assert manifest["files"][0] == "1_photo\nwith\nnewline.jpg"

    def test_unrelated_tmp_manifest_not_deleted(self, tmp_path):
        """Arbitrary files like .curator_manifest.tmp should not be deleted by overwrite."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        unrelated = out_dir / ".curator_manifest.tmp"
        unrelated.write_text("unrelated user file")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        copy_top_images(scored, out_dir, overwrite=True)

        assert unrelated.exists()
        assert unrelated.read_text() == "unrelated user file"

    def test_overwrite_false_rejects_non_empty_output_dir(self, tmp_path):
        """copy_top_images should refuse to write to a non-empty directory when overwrite=False."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing.txt").write_text("data")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is not empty. Specify overwrite=True"):
            copy_top_images(scored, out_dir, overwrite=False)

    def test_preview_csv_manifest_conflict_rejected(self, tmp_path):
        """Preview CSV pointing to manifest file name should trigger an error in CLI."""
        from photo_curator.cli import main

        test_args = [
            "photo-curator",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--preview_csv", str(tmp_path / ".curator_manifest.json"),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_commit_concurrent_untracked_collision_rejected(self, tmp_path):
        """Untracked file created concurrently during run should trigger rollback, not overwrite."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        save_test_image(old_img)
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        orig_copy2 = shutil.copy2

        def inject_concurrent_file(src, dst):
            res = orig_copy2(src, dst)
            (out_dir / "1_new.jpg").write_text("concurrent intruder")
            return res

        with mock.patch("shutil.copy2", side_effect=inject_concurrent_file):
            with pytest.raises(RuntimeError, match="Refusing to overwrite untracked file"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        assert (out_dir / "1_new.jpg").read_text() == "concurrent intruder"

    def test_untracked_collision_rejected(self, tmp_path):
        """Pre-existing untracked file collision before staging must be rejected."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "1_a.jpg").write_text("untracked original")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="Refusing to overwrite untracked file"):
            copy_top_images(scored, out_dir, overwrite=True)

        assert (out_dir / "1_a.jpg").read_text() == "untracked original"

    def test_manifest_directory_entry_rejected(self, tmp_path):
        """Manifest entry that resolves to a directory instead of a regular file is rejected."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        subfolder = out_dir / "subfolder"
        subfolder.mkdir()
        (subfolder / "nested_file.txt").write_text("content")

        manifest = out_dir / ".curator_manifest.json"
        manifest.write_text(json.dumps({"version": 1, "files": ["subfolder"]}))

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is not a regular file"):
            copy_top_images(scored, out_dir, overwrite=True)

        assert (subfolder / "nested_file.txt").exists()

    def test_failed_rollback_preserves_backup_dir(self, tmp_path):
        """If restoring files during rollback fails, the backup directory is preserved on disk."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        save_test_image(old_img)

        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        original_replace = os.replace

        def failing_replace(src, dst):
            if str(dst).endswith(".curator_manifest.json") and ".curator_stage_" in str(src):
                raise OSError("Commit disk error")
            if ".curator_backup_" in str(src) and str(dst).endswith("1_old.jpg"):
                raise OSError("Rollback restore error")
            return original_replace(src, dst)

        with mock.patch("os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError, match="Recovery files preserved in"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        backup_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_backup_")]
        assert len(backup_dirs) == 1
        assert (backup_dirs[0] / "1_old.jpg").exists()

    def test_backup_allocation_failure_cleans_staging(self, tmp_path):
        """If backup dir allocation fails, the staging dir is cleaned up without leaking."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        orig_mkdtemp = tempfile.mkdtemp
        call_count = [0]

        def fail_second_mkdtemp(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("Disk full creating backup dir")
            return orig_mkdtemp(**kwargs)

        with mock.patch("tempfile.mkdtemp", side_effect=fail_second_mkdtemp):
            with pytest.raises(OSError, match="Disk full"):
                copy_top_images(scored, out_dir)

        staged_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_stage_")]
        assert staged_dirs == []

    def test_staging_unlink_failure_tracked_in_committed(self, tmp_path):
        """Failure to unlink staged_src does not crash commit and final_dest is tracked."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        orig_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if ".curator_stage_" in str(self):
                raise OSError("Cannot unlink staged link")
            return orig_unlink(self, *args, **kwargs)

        with mock.patch.object(Path, "unlink", failing_unlink):
            copied = copy_top_images(scored, out_dir)
            assert copied == 1
            assert (out_dir / "1_a.jpg").exists()
            assert (out_dir / ".curator_manifest.json").exists()

    def test_exclusive_copy_fallback_when_hard_links_unsupported(self, tmp_path):
        """When os.link is unsupported (e.g. FAT32/exFAT), O_CREAT|O_EXCL stream copy succeeds."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with mock.patch("os.link", side_effect=OSError("Operation not supported on filesystem")):
            copied = copy_top_images(scored, out_dir)
            assert copied == 1
            assert (out_dir / "1_a.jpg").exists()
            assert (out_dir / ".curator_manifest.json").exists()

    def test_exclusive_copy_fallback_rejects_concurrent_collision(self, tmp_path):
        """When os.link fails and destination exists, O_CREAT|O_EXCL rejects collision without overwrite."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        save_test_image(old_img)
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        orig_copy2 = shutil.copy2

        def inject_concurrent(src, dst):
            res = orig_copy2(src, dst)
            (out_dir / "1_new.jpg").write_text("concurrent content")
            return res

        with mock.patch("shutil.copy2", side_effect=inject_concurrent):
            with mock.patch("os.link", side_effect=OSError("Operation not supported")):
                with pytest.raises(RuntimeError, match="Refusing to overwrite untracked file"):
                    copy_top_images(scored_new, out_dir, overwrite=True)

        assert (out_dir / "1_new.jpg").read_text() == "concurrent content"

    def test_exclusive_copy_fallback_preserves_metadata(self, tmp_path):
        """Exclusive stream copy fallback must preserve mtime from source photo via copystat."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)

        target_mtime = 946684800.0  # 2000-01-01 00:00:00 UTC
        os.utime(img, (target_mtime, target_mtime))

        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with mock.patch("os.link", side_effect=OSError("Operation not supported")):
            copy_top_images(scored, out_dir)

        dest_file = out_dir / "1_a.jpg"
        assert dest_file.exists()
        assert abs(dest_file.stat().st_mtime - target_mtime) < 1.0

    def test_exclusive_copy_fallback_partial_failure_routes_to_rollback(self, tmp_path):
        """Failed stream copy must register destination and route through rollback_errors if unlink fails."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        orig_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if str(self).endswith("1_a.jpg"):
                raise OSError("Cannot delete partial destination")
            return orig_unlink(self, *args, **kwargs)

        with mock.patch("os.link", side_effect=OSError("Operation not supported")):
            with mock.patch("shutil.copyfileobj", side_effect=OSError("Disk write error mid-stream")):
                with mock.patch.object(Path, "unlink", failing_unlink):
                    with pytest.raises(RuntimeError, match="Recovery files preserved in"):
                        copy_top_images(scored, out_dir)

        backup_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_backup_")]
        assert len(backup_dirs) == 1
