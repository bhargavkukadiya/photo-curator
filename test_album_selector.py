"""Unit tests for album_selector.py.

All heavy ML models (CLIP, DeepFace) are mocked so tests run fast
and without GPU / model downloads.
"""

import csv
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from album_selector import (
    ScoredImage,
    aesthetic_score,
    copy_top_images,
    emotion_score,
    export_scores_csv,
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


def _save_test_image(path: Path, color=(128, 128, 128), fmt: str = "JPEG"):
    """Save a small solid-color image to disk.

    Use fmt="PNG" for lossless round-trip tests.
    """
    img = Image.new("RGB", (50, 50), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt)


# ---------------------------------------------------------------------------
# is_image_file
# ---------------------------------------------------------------------------


class TestIsImageFile:
    """Tests for the is_image_file() extension checker."""

    def test_supported_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            assert is_image_file(Path(f"photo{ext}")), f"{ext} should be supported"

    def test_case_insensitive(self):
        assert is_image_file(Path("photo.JPG"))
        assert is_image_file(Path("photo.Jpeg"))
        assert is_image_file(Path("photo.PNG"))

    def test_unsupported_extensions(self):
        for ext in [".gif", ".bmp", ".tiff", ".svg", ".pdf", ".txt"]:
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

    def test_mid_gray_best_exposure(self):
        """Mid-gray (brightness ~0.5) should have the best exposure sub-score."""
        gray_score = technical_score(_make_solid_image(color=(128, 128, 128)))
        black_score = technical_score(_make_solid_image(color=(0, 0, 0)))
        white_score = technical_score(_make_solid_image(color=(255, 255, 255)))
        # All are solid so sharpness is ~0; differences come from exposure
        assert gray_score >= black_score
        assert gray_score >= white_score

    def test_extreme_sharpness_clamped(self):
        """Very high Laplacian variance should still produce score <= 1.0."""
        # Create a checkerboard pattern for extreme edges
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


# ---------------------------------------------------------------------------
# aesthetic_score
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

    def test_returns_zero_on_error(self):
        """Should return 0.0 when the model raises an error."""
        model = mock.MagicMock()
        model.encode.side_effect = RuntimeError("CUDA OOM")
        pil_img = Image.new("RGB", (50, 50))
        ref_emb = torch.tensor([[1.0]])

        score = aesthetic_score(pil_img, model, ref_emb)
        assert score == 0.0


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
        """DeepFace returns a list -> unwrap first element."""
        mock_deepface.analyze.return_value = [{"emotion": {"happy": 50.0}}]
        score = emotion_score(Path("smile.jpg"), deepface_enabled=True)
        assert score == pytest.approx(0.5, abs=0.01)

    @mock.patch("album_selector.DEEPFACE_AVAILABLE", True)
    @mock.patch("album_selector.DeepFace", create=True)
    def test_empty_list_response(self, mock_deepface):
        """DeepFace returns an empty list -> should not crash."""
        mock_deepface.analyze.return_value = []
        score = emotion_score(Path("empty.jpg"), deepface_enabled=True)
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
        """PIL (RGB) and OpenCV (BGR) should have swapped channels.

        Uses PNG for lossless round-trip so pixel values are exact.
        """
        img_path = tmp_path / "red.png"
        _save_test_image(img_path, color=(255, 0, 0), fmt="PNG")

        pil_img, cv_img = load_image(img_path)

        # PIL pixel should be exactly red
        r, g, b = pil_img.getpixel((25, 25))
        assert (r, g, b) == (255, 0, 0)

        # OpenCV BGR: channel 2 = R, channel 0 = B
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
# export_scores_csv (calls production code)
# ---------------------------------------------------------------------------


class TestExportScoresCSV:
    """Test CSV output via the extracted export_scores_csv() function."""

    def test_header_and_values(self, tmp_path):
        """CSV should have the correct header and formatted values."""
        csv_path = tmp_path / "scores.csv"
        scored = [
            ScoredImage(Path("a.jpg"), total=0.1234, technical=0.5678, aesthetic=0.9012, emotion=0.3456),
            ScoredImage(Path("b.jpg"), total=0.9999, technical=0.1111, aesthetic=0.2222, emotion=0.3333),
        ]

        export_scores_csv(scored, csv_path)

        with open(csv_path) as f:
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

    def test_empty_list(self, tmp_path):
        """Exporting an empty list should produce a CSV with only a header."""
        csv_path = tmp_path / "empty.csv"
        export_scores_csv([], csv_path)

        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# copy_top_images (calls production code)
# ---------------------------------------------------------------------------


class TestCopyTopImages:
    """Test the file copy logic via the extracted copy_top_images() function."""

    def test_dynamic_filename_width(self, tmp_path):
        """Filename index width should adapt to the number of images."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Create 12 test images -> width should be 2 (not 3)
        scored = []
        for i in range(12):
            img_path = src_dir / f"img_{i}.jpg"
            _save_test_image(img_path)
            scored.append(
                ScoredImage(path=img_path, total=1.0 - i * 0.01,
                            technical=0.5, aesthetic=0.5, emotion=0.0)
            )

        copy_top_images(scored, out_dir)

        copied = sorted(out_dir.iterdir())
        assert len(copied) == 12
        assert copied[0].name == "01_img_0.jpg"
        assert copied[-1].name == "12_img_11.jpg"

    def test_single_image(self, tmp_path):
        """Copying a single image should use width 1 (no zero-padding)."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        img_path = src_dir / "solo.jpg"
        _save_test_image(img_path)
        scored = [ScoredImage(path=img_path, total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0)]

        copy_top_images(scored, out_dir)

        copied = list(out_dir.iterdir())
        assert len(copied) == 1
        assert copied[0].name == "1_solo.jpg"

    def test_empty_list(self, tmp_path):
        """Copying an empty list should not crash."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        copy_top_images([], out_dir)

        assert list(out_dir.iterdir()) == []

    def test_copied_files_are_valid(self, tmp_path):
        """Copied files should be valid, openable images."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        img_path = src_dir / "photo.png"
        _save_test_image(img_path, color=(0, 255, 0), fmt="PNG")
        scored = [ScoredImage(path=img_path, total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0)]

        copy_top_images(scored, out_dir)

        copied = list(out_dir.iterdir())
        img = Image.open(copied[0])
        assert img.size == (50, 50)
