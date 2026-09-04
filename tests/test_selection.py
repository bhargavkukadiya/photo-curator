"""Unit tests for selection, ranking, and deduplication modules."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from photo_curator.models import ScoredImage
from photo_curator.selection import filter_near_duplicates


class TestScoredImage:
    """Tests for the ScoredImage dataclass."""

    def test_creation(self):
        img = ScoredImage(Path("a.jpg"), total=0.8, technical=0.7, aesthetic=0.9, emotion=0.0)
        assert img.path == Path("a.jpg")
        assert img.total == 0.8
        assert img.embedding is None

    def test_sorting(self):
        """ScoredImage objects should sort correctly by total descending."""
        img1 = ScoredImage(Path("1.jpg"), total=0.5, technical=0.5, aesthetic=0.5, emotion=0.0)
        img2 = ScoredImage(Path("2.jpg"), total=0.9, technical=0.9, aesthetic=0.9, emotion=0.0)
        img3 = ScoredImage(Path("3.jpg"), total=0.1, technical=0.1, aesthetic=0.1, emotion=0.0)

        sorted_images = sorted([img1, img2, img3], key=lambda s: s.total, reverse=True)
        assert [s.path.name for s in sorted_images] == ["2.jpg", "1.jpg", "3.jpg"]


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

        with mock.patch("photo_curator.selection.util.cos_sim") as mock_sim:
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

    def test_dedup_threshold_one_removes_exact_duplicates(self):
        """Threshold 1.0 should remove candidates with exact similarity (1.0)."""
        emb1 = torch.tensor([1.0, 0.0])
        emb2 = torch.tensor([1.0, 0.0])

        images = [
            ScoredImage(Path("a.jpg"), total=0.9, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb1),
            ScoredImage(Path("b.jpg"), total=0.8, technical=0.5, aesthetic=0.5, emotion=0.0, embedding=emb2),
        ]

        with mock.patch("photo_curator.selection.util.cos_sim", return_value=torch.tensor([[1.0]])):
            selected = filter_near_duplicates(images, threshold=1.0)
            assert len(selected) == 1
            assert selected[0].path == Path("a.jpg")

    def test_vectorized_dedup_multiple_candidates_correctness(self):
        """Vectorized deduplication correctly tracks multiple chosen candidates."""
        c1_a = torch.tensor([1.0, 0.0, 0.0])
        c1_b = torch.tensor([0.98, 0.02, 0.0])  # Near dup of c1_a
        c2_a = torch.tensor([0.0, 1.0, 0.0])
        c2_b = torch.tensor([0.0, 0.99, 0.01])  # Near dup of c2_a
        c3_a = torch.tensor([0.0, 0.0, 1.0])

        candidates = [
            ScoredImage(Path("1.jpg"), 0.95, 0.5, 0.5, 0.0, c1_a),
            ScoredImage(Path("2.jpg"), 0.90, 0.5, 0.5, 0.0, c1_b),  # Suppressed by 1.jpg
            ScoredImage(Path("3.jpg"), 0.85, 0.5, 0.5, 0.0, c2_a),  # Selected
            ScoredImage(Path("4.jpg"), 0.80, 0.5, 0.5, 0.0, c2_b),  # Suppressed by 3.jpg
            ScoredImage(Path("5.jpg"), 0.75, 0.5, 0.5, 0.0, c3_a),  # Selected
        ]

        selected, suppressed = filter_near_duplicates(
            candidates, threshold=0.90, return_suppressed=True
        )
        assert [s.path for s in selected] == [Path("1.jpg"), Path("3.jpg"), Path("5.jpg")]
        assert suppressed == {Path("2.jpg"), Path("4.jpg")}
