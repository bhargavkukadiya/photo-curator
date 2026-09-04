"""Unit tests for computer vision and quality scoring modules."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from photo_curator.scoring import (
    aesthetic_score,
    aesthetic_score_batch,
    calculate_total_score,
    emotion_score,
    is_image_file,
    load_image,
    technical_score,
)
from tests.conftest import make_noisy_image, make_solid_image, save_test_image


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


class TestTechnicalScore:
    """Tests for sharpness + exposure scoring."""

    def test_output_in_unit_range(self):
        """Score must always be in [0, 1]."""
        for img in [
            make_solid_image(color=(0, 0, 0)),       # black
            make_solid_image(color=(255, 255, 255)),  # white
            make_solid_image(color=(128, 128, 128)),  # mid-gray
            make_noisy_image(),                       # noisy / sharp
        ]:
            score = technical_score(img)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_noisy_sharper_than_solid(self):
        """A noisy image should score higher on sharpness than a solid one."""
        solid_score = technical_score(make_solid_image())
        noisy_score = technical_score(make_noisy_image())
        assert noisy_score > solid_score

    def test_exposure_extremes_and_midpoint(self):
        """Pure black and white should get exposure 0.0; mid-gray gets ~1.0."""
        black_score = technical_score(make_solid_image(color=(0, 0, 0)))
        assert black_score == pytest.approx(0.0, abs=0.01)

        white_score = technical_score(make_solid_image(color=(255, 255, 255)))
        assert white_score == pytest.approx(0.0, abs=0.01)

        gray_score = technical_score(make_solid_image(color=(128, 128, 128)))
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

    def test_technical_score_single_channel_3d(self):
        """Single-channel 3D array (H, W, 1) should be squeezed and scored without cvtColor error."""
        img_3d = np.full((100, 100, 1), 128, dtype=np.uint8)
        score = technical_score(img_3d)
        assert 0.0 <= score <= 1.0

    def test_technical_score_4channel_bgra(self):
        """4-channel BGRA array (H, W, 4) should convert using COLOR_BGRA2GRAY and score successfully."""
        bgra = np.full((100, 100, 4), 128, dtype=np.uint8)
        score = technical_score(bgra)
        assert 0.0 <= score <= 1.0


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
            "photo_curator.scoring.util.cos_sim",
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
        """Similarity > 0.4 should clamp to 1.0."""
        score = self._score_with_sim(0.6)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_clamps_below_range(self):
        """Similarity < 0.1 should clamp to 0.0."""
        score = self._score_with_sim(0.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_batch_scoring(self):
        """Batch aesthetic scoring handles multiple images and returns embeddings."""
        model = mock.MagicMock()
        model.encode.return_value = torch.tensor([[1.0], [2.0], [3.0]])
        ref_emb = torch.tensor([[1.0]])
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]

        with mock.patch(
            "photo_curator.scoring.util.cos_sim",
            return_value=torch.tensor([[0.25], [0.4], [0.1]]),
        ):
            scores, embs = aesthetic_score_batch(images, model, ref_emb, batch_size=2)
            assert len(scores) == 3
            assert len(embs) == 3
            assert scores[0] == pytest.approx(0.5, abs=0.02)
            assert scores[1] == pytest.approx(1.0, abs=0.01)
            assert scores[2] == pytest.approx(0.0, abs=0.01)

    def test_empty_batch(self):
        """Empty input returns empty lists."""
        scores, embs = aesthetic_score_batch([], mock.MagicMock(), torch.tensor([[1.0]]))
        assert scores == []
        assert embs == []

    def test_raises_on_batch_error(self):
        """Propagate RuntimeError if batch scoring fails."""
        model = mock.MagicMock()
        model.encode.side_effect = RuntimeError("CUDA OOM")
        with pytest.raises(RuntimeError, match="CLIP aesthetic scoring failed"):
            aesthetic_score_batch(
                [Image.new("RGB", (10, 10))], model, torch.tensor([[1.0]])
            )


class TestEmotionScore:
    """Tests for DeepFace emotion scoring (DeepFace is mocked)."""

    def test_disabled_returns_zero(self):
        """When deepface_enabled=False, return 0.0 immediately."""
        score = emotion_score(Path("photo.jpg"), deepface_enabled=False)
        assert score == 0.0

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", False)
    def test_unavailable_returns_zero(self):
        """When DeepFace is not installed, return 0.0."""
        score = emotion_score(Path("photo.jpg"), deepface_enabled=True)
        assert score == 0.0

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_happy_face_dict_response(self):
        """DeepFace returns a dict -> extract happiness correctly."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = {"emotion": {"happy": 80.0}}
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("happy.jpg"), deepface_enabled=True)
            assert score == pytest.approx(0.8, abs=0.01)

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_happy_face_list_response(self):
        """DeepFace returns a list -> unwrap and extract happiness."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = [{"emotion": {"happy": 50.0}}]
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("smile.jpg"), deepface_enabled=True)
            assert score == pytest.approx(0.5, abs=0.01)

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_multi_face_average(self):
        """DeepFace returns multiple faces -> average happiness across faces."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = [
            {"emotion": {"happy": 100.0}},
            {"emotion": {"happy": 50.0}},
        ]
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("group.jpg"), deepface_enabled=True)
            assert score == pytest.approx(0.75, abs=0.01)

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_in_memory_numpy_array(self):
        """DeepFace accepts numpy array directly without disk reload."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = [{"emotion": {"happy": 90.0}}]
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            cv_img = make_solid_image(50, 50)
            score = emotion_score(cv_img, deepface_enabled=True)
            assert score == pytest.approx(0.9, abs=0.01)

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_empty_list_response(self):
        """DeepFace returns an empty list -> score is 0.0."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = []
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("photo.jpg"), deepface_enabled=True)
            assert score == 0.0

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_no_face_detected_zero_confidence(self):
        """DeepFace returns face_confidence=0 -> treat as no face detected."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = [
            {"emotion": {"happy": 100.0}, "face_confidence": 0}
        ]
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("landscape.jpg"), deepface_enabled=True)
            assert score == 0.0

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_no_face_detected(self):
        """DeepFace returns a dict without emotion key -> score is 0.0."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.return_value = [{"other": 123}]
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("photo.jpg"), deepface_enabled=True)
            assert score == 0.0

    @mock.patch("photo_curator.scoring.DEEPFACE_AVAILABLE", True)
    def test_error_returns_zero(self):
        """DeepFace exception should return 0.0, not crash."""
        mock_deepface = mock.MagicMock()
        mock_deepface.analyze.side_effect = ValueError("Corrupt image")
        with mock.patch("photo_curator.scoring.DeepFace", mock_deepface, create=True):
            score = emotion_score(Path("corrupt.jpg"), deepface_enabled=True)
            assert score == 0.0


class TestCalculateTotalScore:
    """Tests for the composite weighted total score calculation."""

    def test_all_signals_active(self):
        """Default weights: 0.4*tech + 0.4*aes + 0.2*emo."""
        score = calculate_total_score(1.0, 1.0, 1.0, emotion_active=True)
        assert score == pytest.approx(1.0)

        score = calculate_total_score(0.5, 0.5, 0.5, emotion_active=True)
        assert score == pytest.approx(0.5)

    def test_emotion_inactive_dynamic_rebalance(self):
        """When emotion is inactive, technical and aesthetic rebalance to sum to 1.0."""
        score = calculate_total_score(
            technical=0.8,
            aesthetic=0.6,
            emotion=0.0,
            weight_technical=0.4,
            weight_aesthetic=0.4,
            weight_emotion=0.2,
            emotion_active=False,
        )
        # (0.4*0.8 + 0.4*0.6) / (0.4 + 0.4) = (0.32 + 0.24) / 0.8 = 0.70
        assert score == pytest.approx(0.70, abs=0.001)

    def test_custom_weights(self):
        """Custom weights should be normalized by their active sum."""
        score = calculate_total_score(
            technical=1.0,
            aesthetic=0.0,
            emotion=0.0,
            weight_technical=2.0,
            weight_aesthetic=1.0,
            weight_emotion=1.0,
            emotion_active=True,
        )
        # (2.0*1.0 + 0 + 0) / 4.0 = 0.5
        assert score == pytest.approx(0.5)

    def test_zero_weights_safe(self):
        """All-zero weights should safely return 0.0 without division by zero."""
        score = calculate_total_score(
            1.0, 1.0, 1.0,
            weight_technical=0.0,
            weight_aesthetic=0.0,
            weight_emotion=0.0,
            emotion_active=True,
        )
        assert score == 0.0


class TestLoadImage:
    """Tests for the single-read image loader."""

    def test_valid_jpeg(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        save_test_image(img_path)
        pil_img, cv_img = load_image(img_path)
        assert pil_img is not None
        assert cv_img is not None
        assert pil_img.size == (50, 50)
        assert cv_img.shape == (50, 50, 3)

    def test_valid_png(self, tmp_path):
        img_path = tmp_path / "test.png"
        save_test_image(img_path, fmt="PNG")
        pil_img, cv_img = load_image(img_path)
        assert pil_img is not None
        assert cv_img is not None

    def test_exif_orientation_handling(self, tmp_path):
        """Images with EXIF rotation tags should load without errors."""
        img_path = tmp_path / "rotated.jpg"
        save_test_image(img_path, exif_orientation=6)  # 6 = 90 deg CW
        pil_img, cv_img = load_image(img_path)
        assert pil_img is not None
        assert cv_img is not None

    def test_corrupt_file(self, tmp_path):
        """Corrupt file returns (None, None) gracefully."""
        bad_path = tmp_path / "corrupt.jpg"
        bad_path.write_bytes(b"not an image file data at all")
        pil_img, cv_img = load_image(bad_path)
        assert pil_img is None
        assert cv_img is None

    def test_missing_file(self, tmp_path):
        """Missing file returns (None, None) gracefully."""
        missing = tmp_path / "nonexistent.jpg"
        pil_img, cv_img = load_image(missing)
        assert pil_img is None
        assert cv_img is None

    def test_rgb_bgr_conversion(self, tmp_path):
        """Verify RGB to BGR channel ordering."""
        img_path = tmp_path / "red.png"
        save_test_image(img_path, color=(255, 0, 0), fmt="PNG")
        pil_img, cv_img = load_image(img_path)
        assert pil_img.getpixel((0, 0)) == (255, 0, 0)
        assert tuple(cv_img[0, 0]) == (0, 0, 255)

    def test_load_image_downsamples_large_image_saving_memory(self, tmp_path):
        """Large image exceeding max_dimension should be resized during load_image."""
        large_path = tmp_path / "huge.jpg"
        img = Image.new("RGB", (3000, 2000), (128, 128, 128))
        img.save(large_path, format="JPEG")

        pil_img, cv_img = load_image(large_path, max_dimension=1024)
        assert pil_img is not None
        assert cv_img is not None
        assert max(pil_img.size) <= 1024
        assert max(cv_img.shape[:2]) <= 1024
