"""Selection and deduplication algorithms for photo curation."""

from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import util

from photo_curator.models import ScoredImage


def filter_near_duplicates(
    scored_images: list[ScoredImage],
    threshold: float | None = None,
    target_count: int | None = None,
    *,
    return_suppressed: bool = False,
) -> list[ScoredImage] | tuple[list[ScoredImage], set[Path]]:
    """Filter out near-duplicate burst shots using CLIP cosine similarity.

    Assumes *scored_images* is pre-sorted descending by score.
    Iteratively selects the highest scoring candidate that is not too similar
    (cosine similarity >= threshold) to any already selected candidate.
    Uses vectorized PyTorch tensor comparisons for high performance.
    """
    if threshold is None or threshold > 1.0 or not scored_images:
        selected = scored_images[:target_count] if target_count else scored_images
        if return_suppressed:
            return selected, set()
        return selected

    selected: list[ScoredImage] = []
    selected_embs: list[torch.Tensor] = []
    suppressed_paths: set[Path] = set()

    for candidate in scored_images:
        if candidate.embedding is None:
            selected.append(candidate)
            if target_count and len(selected) >= target_count:
                break
            continue

        cand_emb = candidate.embedding
        if cand_emb.ndim == 1:
            cand_emb = cand_emb.unsqueeze(0)

        is_duplicate = False
        if selected_embs:
            stacked_selected = torch.stack(selected_embs)
            if stacked_selected.ndim == 3 and stacked_selected.shape[1] == 1:
                stacked_selected = stacked_selected.squeeze(1)
            sims = util.cos_sim(cand_emb, stacked_selected)
            if (sims >= threshold).any():
                is_duplicate = True
                suppressed_paths.add(candidate.path)

        if not is_duplicate:
            selected.append(candidate)
            selected_embs.append(cand_emb.squeeze(0))
            if target_count and len(selected) >= target_count:
                break

    if return_suppressed:
        return selected, suppressed_paths
    return selected
