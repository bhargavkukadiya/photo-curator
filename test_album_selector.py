"""Unit tests for album_selector.py.

All heavy ML models (CLIP, DeepFace) are mocked so tests run fast,
in-memory, and without GPU / model downloads.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from album_selector import (
    ScoredImage,
    aesthetic_score,
    aesthetic_score_batch,
    calculate_total_score,
    copy_top_images,
    emotion_score,
    export_scores_csv,
    filter_near_duplicates,
    is_image_file,
    load_image,
    technical_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solid_image(width: int = 100, height: int = 100, color=(128, 128, 128)):
    """Create a solid-color BGR numpy array (OpenCV format)."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_noisy_image(width: int = 100, height: int = 100):
    """Create a high-frequency noise image (very sharp)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _save_test_image(
    path: Path,
    color=(128, 128, 128),
    fmt: str = "JPEG",
    exif_orientation: int | None = None,
):
    """Save a small solid-color image to disk.

    Optionally writes an EXIF orientation tag for rotation testing.
    """
    img = Image.new("RGB", (50, 50), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exif_orientation is not None:
        exif = img.getexif()
        exif[0x0112] = exif_orientation  # 0x0112 is Orientation tag
        img.save(path, format=fmt, exif=exif)
    else:
        img.save(path, format=fmt)


# ---------------------------------------------------------------------------
# is_image_file
# ---------------------------------------------------------------------------


class TestIsImageFile:
    """Tests for the is_image_file() extension checker."""

    def test_supported_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]:
            assert is_image_file(Path(f"photo{ext}")), f"{ext} should be supported"

    def test_case_insensitive(self):
        assert is_image_file(Path("photo.JPG"))
        assert is_image_file(Path("photo.Jpeg"))
        assert is_image_file(Path("photo.PNG"))
        assert is_image_file(Path("photo.WEBP"))

    def test_unsupported_extensions(self):
        for ext in [".gif", ".svg", ".pdf", ".txt", ".mp4", ".json"]:
            assert not is_image_file(Path(f"file{ext}")), f"{ext} should not be supported"

    def test_no_extension(self):
        assert not is_image_file(Path("photo"))

    def test_extension_substring_no_match(self):
        """A filename like '.jpg_backup' should not match '.jpg'."""
        assert not is_image_file(Path(".jpg_backup"))


# ---------------------------------------------------------------------------
# technical_score
# ---------------------------------------------------------------------------


class TestTechnicalScore:
    """Tests for sharpness + exposure scoring."""

    def test_output_in_unit_range(self):
        """Score must always be in [0, 1]."""
        for img in [
            _make_solid_image(color=(0, 0, 0)),       # black
            _make_solid_image(color=(255, 255, 255)),  # white
            _make_solid_image(color=(128, 128, 128)),  # mid-gray
            _make_noisy_image(),                       # noisy / sharp
        ]:
            score = technical_score(img)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_noisy_sharper_than_solid(self):
        """A noisy image should score higher on sharpness than a solid one."""
        solid_score = technical_score(_make_solid_image())
        noisy_score = technical_score(_make_noisy_image())
        assert noisy_score > solid_score

    def test_exposure_extremes_and_midpoint(self):
        """Pure black and white should get exposure 0.0; mid-gray gets ~1.0."""
        # Pure black solid: sharpness = 0.0, exposure = 0.0 -> total technical = 0.0
        black_score = technical_score(_make_solid_image(color=(0, 0, 0)))
        assert black_score == pytest.approx(0.0, abs=0.01)

        # Pure white solid: sharpness = 0.0, exposure = 0.0 -> total technical = 0.0
        white_score = technical_score(_make_solid_image(color=(255, 255, 255)))
        assert white_score == pytest.approx(0.0, abs=0.01)

        # Mid-gray solid: sharpness = 0.0, exposure = 1.0 -> total technical = 0.5
        gray_score = technical_score(_make_solid_image(color=(128, 128, 128)))
        assert gray_score == pytest.approx(0.5, abs=0.02)

    def test_extreme_sharpness_clamped(self):
        """Very high Laplacian variance should still produce score <= 1.0."""
        checker = np.zeros((100, 100, 3), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        score = technical_score(checker)
        assert score <= 1.0

    def test_single_pixel_image(self):
        """A 1x1 image should not crash and should return a valid score."""
        tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
        score = technical_score(tiny)
        assert 0.0 <= score <= 1.0

    def test_large_image_rescaling(self):
        """Images larger than MAX_ANALYSIS_DIMENSION should be scaled and scored."""
        large = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        score = technical_score(large)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# aesthetic_score & aesthetic_score_batch
# ---------------------------------------------------------------------------


class TestAestheticScore:
    """Tests for CLIP-based aesthetic scoring (model is mocked)."""

    @staticmethod
    def _score_with_sim(similarity: float) -> float:
        """Run aesthetic_score with a mocked CLIP model returning *similarity*."""
        model = mock.MagicMock()
        model.encode.return_value = torch.tensor([[1.0]])
        pil_img = Image.new("RGB", (50, 50))
        ref_emb = torch.tensor([[1.0]])

        with mock.patch(
            "album_selector.util.cos_sim",
            return_value=torch.tensor([[similarity]]),
        ):
            return aesthetic_score(pil_img, model, ref_emb)

    def test_mid_range_similarity(self):
        """Similarity of 0.25 (midpoint of [0.1, 0.4]) -> ~0.5 after rescaling."""
        score = self._score_with_sim(0.25)
        assert score == pytest.approx(0.5, abs=0.02)

    def test_high_similarity(self):
        """Similarity of 0.4 (top of range) -> 1.0."""
        score = self._score_with_sim(0.4)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_low_similarity(self):
        """Similarity of 0.1 (bottom of range) -> 0.0."""
        score = self._score_with_sim(0.1)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_clamps_above_range(self):
        """Similarity above 0.4 should clamp to 1.0."""
        score = self._score_with_sim(0.6)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_clamps_below_range(self):
        """Similarity below 0.1 should clamp to 0.0."""
        score = self._score_with_sim(0.02)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_batch_scoring(self):
        """Batch scoring should return scores and embeddings for all images."""
        model = mock.MagicMock()
        # Mock 3 image embeddings
        model.encode.return_value = torch.tensor([[1.0], [2.0], [3.0]])
        ref_emb = torch.tensor([[1.0]])
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]

        with mock.patch(
            "album_selector.util.cos_sim",
            return_value=torch.tensor([[0.1], [0.25], [0.4]]),
        ):
            scores, embs = aesthetic_score_batch(images, model, ref_emb)
            assert len(scores) == 3
            assert len(embs) == 3
            assert scores[0] == pytest.approx(0.0, abs=0.01)
            assert scores[1] == pytest.approx(0.5, abs=0.02)
            assert scores[2] == pytest.approx(1.0, abs=0.01)

    def test_empty_batch(self):
        """Empty batch should return empty lists."""
        model = mock.MagicMock()
        ref_emb = torch.tensor([[1.0]])
        scores, embs = aesthetic_score_batch([], model, ref_emb)
        assert scores == []
        assert embs == []

    def test_raises_on_batch_error(self):
        """Should raise RuntimeError when batch encoding fails."""
        model = mock.MagicMock()
        model.encode.side_effect = RuntimeError("CUDA OOM")
        pil_img = Image.new("RGB", (50, 50))
        ref_emb = torch.tensor([[1.0]])

        with pytest.raises(RuntimeError, match="CLIP aesthetic scoring failed"):
            aesthetic_score_batch([pil_img], model, ref_emb)



# ---------------------------------------------------------------------------
# emotion_score
# ---------------------------------------------------------------------------


class TestEmotionScore:
    """Tests for DeepFace emotion scoring (DeepFace is mocked)."""

    def test_disabled_returns_zero(self):
        """When deepface_enabled=False, should return 0.0 immediately."""
        score = emotion_score(Path("any.jpg"), deepface_enabled=False)
        assert score == 0.0

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", False)
    def test_unavailable_returns_zero(self):
        """When DEEPFACE_AVAILABLE is False, should return 0.0."""
        score = emotion_score(Path("any.jpg"), deepface_enabled=True)
        assert score == 0.0

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_happy_face_dict_response(self, mock_deepface):
        """DeepFace returns a dict -> extract happiness correctly."""
        mock_deepface.analyze.return_value = {"emotion": {"happy": 80.0}}
        score = emotion_score(Path("happy.jpg"), deepface_enabled=True)
        assert score == pytest.approx(0.8, abs=0.01)

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_happy_face_list_response(self, mock_deepface):
        """DeepFace returns a list -> unwrap and extract happiness."""
        mock_deepface.analyze.return_value = [{"emotion": {"happy": 50.0}}]
        score = emotion_score(Path("smile.jpg"), deepface_enabled=True)
        assert score == pytest.approx(0.5, abs=0.01)

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_multi_face_average(self, mock_deepface):
        """DeepFace returns multiple faces -> average happiness across faces."""
        mock_deepface.analyze.return_value = [
            {"emotion": {"happy": 100.0}},
            {"emotion": {"happy": 50.0}},
        ]
        score = emotion_score(Path("group.jpg"), deepface_enabled=True)
        assert score == pytest.approx(0.75, abs=0.01)

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_in_memory_numpy_array(self, mock_deepface):
        """DeepFace accepts numpy array directly without disk reload."""
        mock_deepface.analyze.return_value = [{"emotion": {"happy": 90.0}}]
        cv_img = _make_solid_image(50, 50)
        score = emotion_score(cv_img, deepface_enabled=True)
        assert score == pytest.approx(0.9, abs=0.01)

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_empty_list_response(self, mock_deepface):
        """DeepFace returns an empty list -> should not crash."""
        mock_deepface.analyze.return_value = []
        score = emotion_score(Path("empty.jpg"), deepface_enabled=True)
        assert score == 0.0

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_no_face_detected_zero_confidence(self, mock_deepface):
        """DeepFace full-frame fallback with face_confidence=0 should return 0.0."""
        mock_deepface.analyze.return_value = [
            {"emotion": {"happy": 95.0}, "face_confidence": 0.0}
        ]
        score = emotion_score(Path("sunset.jpg"), deepface_enabled=True)
        assert score == 0.0

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_no_face_detected(self, mock_deepface):

        """DeepFace returns empty emotion -> default to 0."""
        mock_deepface.analyze.return_value = {"emotion": {}}
        score = emotion_score(Path("landscape.jpg"), deepface_enabled=True)
        assert score == 0.0

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_error_returns_zero(self, mock_deepface):
        """DeepFace raises -> return 0.0 gracefully."""
        mock_deepface.analyze.side_effect = ValueError("No face")
        score = emotion_score(Path("err.jpg"), deepface_enabled=True)
        assert score == 0.0


# ---------------------------------------------------------------------------
# calculate_total_score
# ---------------------------------------------------------------------------


class TestCalculateTotalScore:
    """Tests for dynamic weight normalization and scoring calculation."""

    def test_all_signals_active(self):
        """Default weights: 0.4 tech, 0.4 aes, 0.2 emo."""
        score = calculate_total_score(
            technical=1.0,
            aesthetic=0.5,
            emotion=0.0,
            emotion_active=True,
        )
        # (0.4 * 1.0 + 0.4 * 0.5 + 0.2 * 0.0) / 1.0 = 0.60
        assert score == pytest.approx(0.60, abs=0.01)

    def test_emotion_inactive_dynamic_rebalance(self):
        """When emotion is inactive, technical & aesthetic rebalance to sum to 1.0."""
        score = calculate_total_score(
            technical=1.0,
            aesthetic=0.5,
            emotion=0.0,
            emotion_active=False,
        )
        # (0.4 * 1.0 + 0.4 * 0.5) / 0.8 = 0.60 / 0.8 = 0.75
        assert score == pytest.approx(0.75, abs=0.01)

    def test_custom_weights(self):
        """Custom weights should be normalized properly."""
        score = calculate_total_score(
            technical=1.0,
            aesthetic=0.0,
            emotion=0.5,
            weight_technical=0.7,
            weight_aesthetic=0.1,
            weight_emotion=0.2,
            emotion_active=True,
        )
        # (0.7 * 1.0 + 0.1 * 0.0 + 0.2 * 0.5) / 1.0 = 0.80
        assert score == pytest.approx(0.80, abs=0.01)

    def test_zero_weights_safe(self):
        """Zero weights should return 0.0 without division by zero error."""
        score = calculate_total_score(
            technical=1.0,
            aesthetic=1.0,
            emotion=1.0,
            weight_technical=0.0,
            weight_aesthetic=0.0,
            weight_emotion=0.0,
            emotion_active=False,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# filter_near_duplicates
# ---------------------------------------------------------------------------


class TestFilterNearDuplicates:
    """Tests for burst shot deduplication based on CLIP embedding similarity."""

    def test_dedup_removes_similar_images(self):
        """Images with similarity >= threshold should be filtered out."""
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([0.99, 0.01])  # Near duplicate of emb1
        emb3 = torch.tensor([0.0, 1.0])    # Completely different image

        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb1),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb2),
            ScoredImage(Path("c.jpg"), total=0.7, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb3),
        ]

        with mock.patch("album_selector.util.cos_sim") as mock_sim:
            # emb2 compared to emb1 is 0.98 (duplicate)
            # emb3 compared to emb1 is 0.10 (not duplicate)
            def sim_side_effect(t1, t2):
                if torch.allclose(t1, emb2) and torch.allclose(t2, emb1):
                    return torch.tensor([[0.98]])
                return torch.tensor([[0.10]])

            mock_sim.side_effect = sim_side_effect

            selected = filter_near_duplicates(images, threshold=0.90)
            assert len(selected) == 2
            assert selected[0].path == Path("a.jpg")
            assert selected[1].path == Path("c.jpg")

    def test_dedup_disabled_when_none(self):
        """When threshold is None, all scored images are returned."""
        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0),
        ]
        selected = filter_near_duplicates(images, threshold=None)
        assert len(selected) == 2

    def test_target_count_limit(self):
        """Deduplication stops when target_count is reached."""
        images = [
            ScoredImage(Path(f"{i}.jpg"), total=1.0 - i * 0.1, technical=0.5, aesthetic=0.5, emotion=0.0)
            for i in range(10)
        ]
        selected = filter_near_duplicates(images, threshold=None, target_count=3)
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Tests for the single-read image loader."""

    def test_valid_jpeg(self, tmp_path):
        """Should return (PIL.Image, np.ndarray) for a valid JPEG."""
        img_path = tmp_path / "test.jpg"
        _save_test_image(img_path, color=(255, 0, 0))

        pil_img, cv_img = load_image(img_path)

        assert pil_img is not None
        assert cv_img is not None
        assert isinstance(pil_img, Image.Image)
        assert isinstance(cv_img, np.ndarray)
        assert pil_img.mode == "RGB"
        assert cv_img.shape[2] == 3

    def test_valid_png(self, tmp_path):
        """Should also work with PNG files."""
        img_path = tmp_path / "test.png"
        _save_test_image(img_path, color=(0, 128, 255), fmt="PNG")

        pil_img, cv_img = load_image(img_path)

        assert pil_img is not None
        assert cv_img is not None

    def test_exif_orientation_handling(self, tmp_path):
        """Images with EXIF orientation metadata should load without error."""
        img_path = tmp_path / "rotated.jpg"
        _save_test_image(img_path, color=(255, 128, 0), exif_orientation=6)

        pil_img, cv_img = load_image(img_path)
        assert pil_img is not None
        assert cv_img is not None

    def test_corrupt_file(self, tmp_path):
        """Should return (None, None) for a corrupt file."""
        bad_file = tmp_path / "corrupt.jpg"
        bad_file.write_text("not an image")

        pil_img, cv_img = load_image(bad_file)

        assert pil_img is None
        assert cv_img is None

    def test_missing_file(self, tmp_path):
        """Should return (None, None) for a non-existent file."""
        pil_img, cv_img = load_image(tmp_path / "missing.jpg")

        assert pil_img is None
        assert cv_img is None

    def test_rgb_bgr_conversion(self, tmp_path):
        """PIL (RGB) and OpenCV (BGR) should have swapped channels."""
        img_path = tmp_path / "red.png"
        _save_test_image(img_path, color=(255, 0, 0), fmt="PNG")

        pil_img, cv_img = load_image(img_path)

        r, g, b = pil_img.getpixel((25, 25))
        assert (r, g, b) == (255, 0, 0)

        bgr_pixel = cv_img[25, 25]
        assert bgr_pixel[2] == 255  # R channel in BGR
        assert bgr_pixel[0] == 0    # B channel in BGR


# ---------------------------------------------------------------------------
# ScoredImage dataclass
# ---------------------------------------------------------------------------


class TestScoredImage:
    """Tests for the ScoredImage data model."""

    def test_creation(self):
        si = ScoredImage(
            path=Path("photo.jpg"),
            total=0.75,
            technical=0.8,
            aesthetic=0.7,
            emotion=0.6,
        )
        assert si.path == Path("photo.jpg")
        assert si.total == 0.75
        assert si.embedding is None

    def test_sorting(self):
        """ScoredImages should be sortable by total score."""
        images = [
            ScoredImage(Path("c.jpg"), total=0.3, technical=0, aesthetic=0, emotion=0),
            ScoredImage(Path("a.jpg"), total=0.9, technical=0, aesthetic=0, emotion=0),
            ScoredImage(Path("b.jpg"), total=0.6, technical=0, aesthetic=0, emotion=0),
        ]
        images.sort(key=lambda s: s.total, reverse=True)

        assert images[0].path == Path("a.jpg")
        assert images[1].path == Path("b.jpg")
        assert images[2].path == Path("c.jpg")


# ---------------------------------------------------------------------------
# export_scores_csv
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# copy_top_images
# ---------------------------------------------------------------------------


class TestCopyTopImages:
    """Test the file copy logic via copy_top_images()."""

    def test_dynamic_filename_width(self, tmp_path):
        """Filename index width should adapt to the number of images."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"

        scored = []
        for i in range(12):
            img_path = src_dir / f"img_{i}.jpg"
            _save_test_image(img_path)
            scored.append(
                ScoredImage(
                    path=img_path,
                    total=1.0 - i * 0.01,
                    technical=0.5,
                    aesthetic=0.5,
                    emotion=0.0,
                )
            )

        copy_top_images(scored, out_dir)

        copied = sorted(
            p
            for p in out_dir.iterdir()
            if p.name not in (".curator_manifest.json", ".curator_manifest.txt")
        )
        assert len(copied) == 12
        assert copied[0].name == "01_img_0.jpg"
        assert copied[-1].name == "12_img_11.jpg"

    def test_creates_nested_output_directory(self, tmp_path):
        """Should automatically create destination directory if it does not exist."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "deep" / "output"
        img_path = src_dir / "solo.jpg"
        _save_test_image(img_path)
        scored = [ScoredImage(path=img_path, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0)]

        copied = copy_top_images(scored, out_dir)
        assert copied == 1
        assert out_dir.exists()
        assert (out_dir / "1_solo.jpg").exists()

    def test_empty_list(self, tmp_path):
        """Copying an empty list should not crash."""
        out_dir = tmp_path / "out"
        copied = copy_top_images([], out_dir)
        assert copied == 0
        assert list(out_dir.iterdir()) == []

    def test_copy_failure_raises_runtime_error(self, tmp_path):
        """If copying any file fails, copy_top_images should raise RuntimeError."""
        out_dir = tmp_path / "out"
        scored = [
            ScoredImage(path=Path("/non/existent/img1.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0),
        ]
        with pytest.raises(RuntimeError, match="Failed to copy"):
            copy_top_images(scored, out_dir)

    def test_overwrite_manifest_cleans_tracked_files(self, tmp_path):
        """When overwrite=True, only manifest-tracked files are removed; untracked files are preserved."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        untracked_file = out_dir / "untracked_report.csv"
        untracked_file.write_text("keep this")

        src_dir = tmp_path / "src"
        img1 = src_dir / "img1.jpg"
        img2 = src_dir / "img2.jpg"
        _save_test_image(img1)
        _save_test_image(img2)

        # Run 1: Copy 2 images into non-empty dir (with overwrite=True)
        scored1 = [
            ScoredImage(path=img1, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0),
            ScoredImage(path=img2, total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0),
        ]
        copy_top_images(scored1, out_dir, overwrite=True)
        assert (out_dir / "1_img1.jpg").exists()
        assert (out_dir / "2_img2.jpg").exists()
        assert (out_dir / ".curator_manifest.json").exists()


        # Run 2: Copy only 1 image with overwrite=True
        scored2 = [
            ScoredImage(path=img1, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0),
        ]
        copy_top_images(scored2, out_dir, overwrite=True)

        assert (out_dir / "1_img1.jpg").exists()
        assert not (out_dir / "2_img2.jpg").exists()  # Stale tracked file removed
        assert untracked_file.exists()  # Untracked file untouched!



# ---------------------------------------------------------------------------
# Additional tests for findings (batch_size, dedup 1.0, extreme aspect ratio)
# ---------------------------------------------------------------------------


class TestFindingFixes:
    """Targeted tests for specific review finding resolutions."""

    def test_batch_size_passed_to_encode(self):
        """SentenceTransformer.encode should receive the specified batch_size."""
        model = mock.MagicMock()
        model.encode.return_value = torch.tensor([[1.0]])
        ref_emb = torch.tensor([[1.0]])
        images = [Image.new("RGB", (10, 10))]

        with mock.patch("album_selector.util.cos_sim", return_value=torch.tensor([[0.2]])):
            aesthetic_score_batch(images, model, ref_emb, batch_size=64)

        model.encode.assert_called_once()
        _, kwargs = model.encode.call_args
        assert kwargs.get("batch_size") == 64

    def test_dedup_threshold_one_removes_exact_duplicates(self):
        """Threshold 1.0 should remove candidates with exact similarity (1.0)."""
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([1.0, 0.0])  # Exact duplicate

        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb1),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb2),
        ]

        with mock.patch("album_selector.util.cos_sim", return_value=torch.tensor([[1.0]])):
            selected = filter_near_duplicates(images, threshold=1.0)
            assert len(selected) == 1
            assert selected[0].path == Path("a.jpg")

    def test_extreme_aspect_ratio_narrow(self):
        """Narrow image (1x5000) should scale to at least 1px without OpenCV error."""
        narrow = np.full((5000, 1, 3), 128, dtype=np.uint8)
        score = technical_score(narrow)
        assert 0.0 <= score <= 1.0

    def test_extreme_aspect_ratio_wide(self):
        """Wide image (5000x1) should scale to at least 1px without OpenCV error."""
        wide = np.full((1, 5000, 3), 128, dtype=np.uint8)
        score = technical_score(wide)
        assert 0.0 <= score <= 1.0

    def test_nan_weight_validation(self):
        """NaN weights should trigger an argparse validation error."""
        from album_selector import main

        test_args = ["album_selector.py", "--input", "./sample_photos", "--output", "./out", "--weight_technical", "nan"]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_device_assertion_error_handled(self):
        """Device initialization raising AssertionError should be caught gracefully."""
        from album_selector import main

        test_args = ["album_selector.py", "--input", "./sample_photos", "--output", "./out", "--device", "invalid_cuda"]
        with mock.patch("sys.argv", test_args):
            with mock.patch("torch.zeros", side_effect=AssertionError("Torch not compiled with CUDA")):
                with pytest.raises(SystemExit):
                    main()

    def test_preview_csv_collision_with_input_image(self, tmp_path):
        """Preview CSV path conflicting with an input image should trigger validation error."""
        from album_selector import main

        img = tmp_path / "photo.jpg"
        _save_test_image(img)

        test_args = [
            "album_selector.py",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--preview_csv", str(img),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_preview_csv_is_dir_raises_error(self, tmp_path):
        """Preview CSV pointing to an existing directory should trigger validation error."""
        from album_selector import main

        csv_dir = tmp_path / "csv_folder"
        csv_dir.mkdir()

        test_args = [
            "album_selector.py",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--preview_csv", str(csv_dir),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_output_is_file_raises_error(self, tmp_path):
        """Output path pointing to an existing file should trigger validation error."""
        from album_selector import main

        out_file = tmp_path / "out_file.txt"
        out_file.write_text("hello")

        test_args = [
            "album_selector.py",
            "--input", str(tmp_path),
            "--output", str(out_file),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_output_is_input_ancestor_raises_error(self, tmp_path):
        """Output path being equal to or an ancestor of input should trigger validation error."""
        from album_selector import main

        nested_input = tmp_path / "nested" / "photos"
        nested_input.mkdir(parents=True)

        test_args = [
            "album_selector.py",
            "--input", str(nested_input),
            "--output", str(tmp_path),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_non_empty_output_requires_overwrite(self, tmp_path):
        """Non-empty output directory without --overwrite should trigger validation error."""
        from album_selector import main

        input_dir = tmp_path / "input"
        _save_test_image(input_dir / "photo.jpg")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing.jpg").write_text("dummy")

        test_args = [
            "album_selector.py",
            "--input", str(input_dir),
            "--output", str(out_dir),
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_manifest_path_traversal_rejected(self, tmp_path):
        """Manifest containing path traversal (e.g. ../victim.txt) must raise RuntimeError."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        manifest_file = out_dir / ".curator_manifest.json"
        manifest_file.write_text('{"version": 1, "files": ["../victim.txt"]}')

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="path-traversal entry"):
            copy_top_images(scored, out_dir, overwrite=True)

    def test_manifest_symlink_rejected(self, tmp_path):
        """Symlinked manifest must be rejected with RuntimeError."""
        target_file = tmp_path / "target.txt"
        target_file.write_text("important")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        manifest_file = out_dir / ".curator_manifest.json"
        manifest_file.symlink_to(target_file)

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is a symlink"):
            copy_top_images(scored, out_dir, overwrite=True)

        assert target_file.read_text() == "important"  # Target not overwritten

    def test_mid_run_copy_failure_leaves_output_clean(self, tmp_path):
        """A failure on a subsequent copy must rollback and leave output unchanged."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        _save_test_image(old_img)

        # Run 1: previous selection
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)
        assert (out_dir / "1_old.jpg").exists()

        # Run 2: image 1 exists, but image 2 fails
        new_img1 = src_dir / "new1.jpg"
        _save_test_image(new_img1)
        new_img2 = src_dir / "missing.jpg"  # Does not exist

        scored_new = [
            ScoredImage(new_img1, 0.95, 0.5, 0.5, 0.0),
            ScoredImage(new_img2, 0.85, 0.5, 0.5, 0.0),
        ]
        with pytest.raises(RuntimeError, match="Failed to copy"):
            copy_top_images(scored_new, out_dir, overwrite=True)

        # Ensure no new orphaned files were committed into out_dir
        assert not (out_dir / "1_new1.jpg").exists()
        # Ensure previous files are still intact
        assert (out_dir / "1_old.jpg").exists()

    def test_commit_failure_rolls_back_completely(self, tmp_path):
        """If a failure occurs during the commit phase, all changes are rolled back."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        _save_test_image(old_img)

        # Initial commit
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)
        assert (out_dir / "1_old.jpg").exists()

        new_img = src_dir / "new.jpg"
        _save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        # Mock os.replace to fail when moving the final manifest
        original_replace = os.replace

        def failing_replace(src, dst):
            if str(dst).endswith(".curator_manifest.json") and ".curator_stage_" in str(src):
                raise OSError("Simulated disk error during manifest commit")
            return original_replace(src, dst)

        with mock.patch("os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError, match="Commit phase failed"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        # Rollback check: old file restored, new file removed, manifest intact
        assert (out_dir / "1_old.jpg").exists()
        assert not (out_dir / "1_new.jpg").exists()
        assert (out_dir / ".curator_manifest.json").exists()

    def test_newline_in_filename_handled_safely_by_json_manifest(self, tmp_path):
        """Filenames with newlines or special characters are safely escaped in JSON manifest."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        newline_img = src_dir / "photo\nwith\nnewline.jpg"
        _save_test_image(newline_img)

        scored = [ScoredImage(newline_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored, out_dir)

        manifest_file = out_dir / ".curator_manifest.json"
        assert manifest_file.exists()
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert len(data["files"]) == 1
        assert data["files"][0] == "1_photo\nwith\nnewline.jpg"

        # Overwriting on next run must parse it correctly without collision error
        copy_top_images(scored, out_dir, overwrite=True)
        assert (out_dir / "1_photo\nwith\nnewline.jpg").exists()

    def test_unrelated_tmp_manifest_not_deleted(self, tmp_path):
        """An unrelated file named .curator_manifest.tmp must not be deleted."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        unrelated_tmp = out_dir / ".curator_manifest.tmp"
        unrelated_tmp.write_text("keep this user data")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        copy_top_images(scored, out_dir, overwrite=True)

        assert unrelated_tmp.exists()
        assert unrelated_tmp.read_text() == "keep this user data"

    def test_overwrite_false_rejects_non_empty_output_dir(self, tmp_path):
        """copy_top_images with overwrite=False must raise RuntimeError on non-empty dir."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        existing_file = out_dir / "existing.txt"
        existing_file.write_text("content")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is not empty. Specify overwrite=True"):
            copy_top_images(scored, out_dir, overwrite=False)

    def test_preview_csv_manifest_conflict_rejected(self, tmp_path):
        """Preview CSV path conflicting with manifest filename must trigger validation error."""
        from album_selector import main

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = src_dir / "photo.jpg"
        _save_test_image(img)

        out_dir = tmp_path / "out"
        manifest_conflict = out_dir / ".curator_manifest.json"

        test_args = [
            "album_selector.py",
            "--input", str(src_dir),
            "--output", str(out_dir),
            "--preview_csv", str(manifest_conflict),
            "--overwrite",
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_commit_concurrent_untracked_collision_rejected(self, tmp_path):
        """A file created at destination during staging must trigger atomic no-clobber rollback."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        _save_test_image(old_img)

        # Initial run
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        _save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        # Simulate a concurrent file being placed at destination during copy
        orig_copy2 = shutil.copy2

        def inject_concurrent_file(src, dst):
            result = orig_copy2(src, dst)
            # Create a concurrent untracked file at the intended destination right after staging copy
            concurrent = out_dir / "1_new.jpg"
            concurrent.write_text("concurrently created untracked file")
            return result

        with mock.patch("shutil.copy2", side_effect=inject_concurrent_file):
            with pytest.raises(RuntimeError, match="Refusing to overwrite untracked file"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        # Ensure the concurrent file was NOT clobbered
        assert (out_dir / "1_new.jpg").read_text() == "concurrently created untracked file"

    def test_untracked_collision_rejected(self, tmp_path):
        """Colliding with an untracked existing file must raise RuntimeError."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        collision_file = out_dir / "1_a.jpg"
        collision_file.write_text("untracked original")

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="Refusing to overwrite untracked file"):
            copy_top_images(scored, out_dir, overwrite=True)

        assert collision_file.read_text() == "untracked original"

    def test_manifest_directory_entry_rejected(self, tmp_path):
        """Manifest listing an existing subdirectory must be rejected before the transaction."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        subfolder = out_dir / "subfolder"
        subfolder.mkdir()
        (subfolder / "nested_file.txt").write_text("should not be deleted")

        manifest_file = out_dir / ".curator_manifest.json"
        manifest_file.write_text('{"version": 1, "files": ["subfolder"]}')

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        with pytest.raises(RuntimeError, match="is not a regular file"):
            copy_top_images(scored, out_dir, overwrite=True)

        assert (subfolder / "nested_file.txt").exists()  # Subfolder content not deleted

    def test_failed_rollback_preserves_backup_dir(self, tmp_path):
        """If restoring files during rollback fails, the backup directory is preserved on disk."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        old_img = src_dir / "old.jpg"
        _save_test_image(old_img)

        # Initial commit
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        _save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        # Mock os.replace to fail during commit, and also fail during rollback
        original_replace = os.replace

        def failing_replace(src, dst):
            # Fail during commit
            if str(dst).endswith(".curator_manifest.json") and ".curator_stage_" in str(src):
                raise OSError("Commit disk error")
            # Fail during rollback when restoring old file
            if ".curator_backup_" in str(src) and str(dst).endswith("1_old.jpg"):
                raise OSError("Rollback restore error")
            return original_replace(src, dst)

        with mock.patch("os.replace", side_effect=failing_replace):
            with pytest.raises(RuntimeError, match="Recovery files preserved in"):
                copy_top_images(scored_new, out_dir, overwrite=True)

        # Verify a backup directory was preserved on disk
        backup_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_backup_")]
        assert len(backup_dirs) == 1
        assert (backup_dirs[0] / "1_old.jpg").exists()  # Old file preserved in backup

    def test_backup_allocation_failure_cleans_staging(self, tmp_path):
        """If backup dir allocation fails, the staging dir is cleaned up without leaking."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
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

        # Verify staging directory was cleaned up and not leaked
        staged_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_stage_")]
        assert staged_dirs == []

    def test_staging_unlink_failure_tracked_in_committed(self, tmp_path):
        """Failure to unlink staged_src does not crash commit and final_dest is tracked."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        src_dir = tmp_path / "src"
        img = src_dir / "a.jpg"
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        # Mock Path.unlink to fail on staged files
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
        _save_test_image(img)
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
        _save_test_image(old_img)
        scored_old = [ScoredImage(old_img, 0.9, 0.5, 0.5, 0.0)]
        copy_top_images(scored_old, out_dir)

        new_img = src_dir / "new.jpg"
        _save_test_image(new_img)
        scored_new = [ScoredImage(new_img, 0.95, 0.5, 0.5, 0.0)]

        # Simulate concurrent file created during staging AND os.link unsupported
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
        _save_test_image(img)

        # Set specific historical timestamp (Year 2000)
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
        _save_test_image(img)
        scored = [ScoredImage(img, 0.9, 0.5, 0.5, 0.0)]

        # Mock copyfileobj to fail mid-stream, and mock Path.unlink to fail during rollback
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

        # Backup dir preserved due to failed rollback
        backup_dirs = [p for p in out_dir.iterdir() if p.name.startswith(".curator_backup_")]
        assert len(backup_dirs) == 1
