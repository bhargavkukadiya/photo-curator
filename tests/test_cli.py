"""Unit tests for the CLI entry point, argument parsing, and end-to-end orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from photo_curator.cli import main
from tests.conftest import save_test_image


class TestCLIValidation:
    """Tests for CLI argument validation and error handling."""

    def test_batch_size_passed_to_encode(self):
        """Batch size should be validated as a positive integer."""
        test_args = ["photo-curator", "--input", "./sample_photos", "--output", "./out", "--batch_size", "0"]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_nan_weight_validation(self):
        """NaN weights should trigger an argparse validation error."""
        test_args = ["photo-curator", "--input", "./sample_photos", "--output", "./out", "--weight_technical", "nan"]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_device_assertion_error_handled(self):
        """Device initialization raising AssertionError should be caught gracefully."""
        test_args = ["photo-curator", "--input", "./sample_photos", "--output", "./out", "--device", "invalid_cuda"]
        with mock.patch("sys.argv", test_args):
            with mock.patch("torch.zeros", side_effect=AssertionError("Torch not compiled with CUDA")):
                with pytest.raises(SystemExit):
                    main()

    def test_preview_csv_collision_with_input_image(self, tmp_path):
        """Preview CSV path conflicting with an input image should trigger validation error."""
        img = tmp_path / "photo.jpg"
        save_test_image(img)

        test_args = [
            "photo-curator",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--preview_csv", str(img),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_preview_csv_is_dir_raises_error(self, tmp_path):
        """Preview CSV pointing to an existing directory should trigger validation error."""
        csv_dir = tmp_path / "csv_folder"
        csv_dir.mkdir()

        test_args = [
            "photo-curator",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--preview_csv", str(csv_dir),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_output_is_file_raises_error(self, tmp_path):
        """Output path pointing to an existing file should trigger validation error."""
        out_file = tmp_path / "out_file.txt"
        out_file.write_text("hello")

        test_args = [
            "photo-curator",
            "--input", str(tmp_path),
            "--output", str(out_file),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_output_is_input_ancestor_raises_error(self, tmp_path):
        """Output path being equal to or an ancestor of input should trigger validation error."""
        nested_input = tmp_path / "nested" / "photos"
        nested_input.mkdir(parents=True)

        test_args = [
            "photo-curator",
            "--input", str(nested_input),
            "--output", str(tmp_path),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_non_empty_output_requires_overwrite(self, tmp_path):
        """Non-empty output directory without --overwrite should trigger validation error."""
        input_dir = tmp_path / "input"
        save_test_image(input_dir / "photo.jpg")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing.jpg").write_text("dummy")

        test_args = [
            "photo-curator",
            "--input", str(input_dir),
            "--output", str(out_dir),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_directory_with_image_extension_ignored_in_discovery(self, tmp_path):
        """A directory named folder.jpg inside the input path should not be treated as an image."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        save_test_image(in_dir / "real.jpg")
        fake_dir = in_dir / "subfolder.jpg"
        fake_dir.mkdir()

        out_dir = tmp_path / "out"
        preview_file = tmp_path / "preview.csv"

        test_args = [
            "photo-curator",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--dryrun",
            "--preview_csv", str(preview_file),
            "--no_deepface",
        ]

        mock_clip = mock.MagicMock()
        mock_clip.encode.return_value = torch.tensor([[0.5]])

        with mock.patch("sys.argv", test_args):
            with mock.patch("photo_curator.cli.load_clip_model", return_value=mock_clip):
                with mock.patch("photo_curator.cli.encode_reference_text", return_value=torch.tensor([[0.5]])):
                    with mock.patch("photo_curator.cli.aesthetic_score_batch", return_value=([0.5], [torch.tensor([0.5])])):
                        main()

        assert preview_file.exists()
        content = preview_file.read_text(encoding="utf-8")
        assert "real.jpg" in content
        assert "subfolder.jpg" not in content

    def test_main_end_to_end_mocked_copy_run(self, tmp_path):
        """End-to-end execution of main() copying top images and writing manifest."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        save_test_image(in_dir / "photo1.jpg", color=(255, 0, 0))
        save_test_image(in_dir / "photo2.jpg", color=(0, 255, 0))

        out_dir = tmp_path / "out"

        test_args = [
            "photo-curator",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--target", "1",
            "--no_deepface",
        ]

        mock_clip = mock.MagicMock()
        mock_clip.encode.return_value = torch.tensor([[0.5]])

        with mock.patch("sys.argv", test_args):
            with mock.patch("photo_curator.cli.load_clip_model", return_value=mock_clip):
                with mock.patch("photo_curator.cli.encode_reference_text", return_value=torch.tensor([[0.5]])):
                    with mock.patch("photo_curator.cli.aesthetic_score_batch", return_value=([0.8, 0.4], [torch.tensor([0.8]), torch.tensor([0.4])])):
                        main()

        manifest_file = out_dir / ".curator_manifest.json"
        assert manifest_file.exists()
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert len(manifest_data["files"]) == 1
        assert (out_dir / manifest_data["files"][0]).exists()
