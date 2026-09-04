# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-04

### Added
- **Vectorized Near-Duplicate Filtering**: Accelerated burst-shot deduplication with batched PyTorch tensor cosine similarity (`torch.matmul`), delivering up to **37×–100× faster filtering** on large photo libraries.
- **Selection Status in CSV Preview**: Added `Status` column to `--preview_csv` (`Selected`, `Duplicate_Suppressed`, `Rank_Cutoff`) for clear inspection during dry runs.
- **Multichannel & 3D Array Safety**: Support for 2D grayscale, 3D single-channel `(H, W, 1)`, 3-channel BGR, and 4-channel BGRA arrays in `technical_score`.
- **Packaging Standards**: Added `pyproject.toml` with console script entrypoint `photo-curator`, optional extras (`[emotion]`, `[heic]`, `[dev]`), and Pytest configuration.
- **Community Health & Documentation**: Added `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue templates, and pull request template.
- **7 New Unit Tests**: Added unit tests for channel safety, memory downscaling, vectorized dedup correctness, preview CSV status annotations, and end-to-end `main()` execution (total: 87 tests).

### Changed
- **Retained Batch Memory Optimization**: `load_image` now downsamples high-resolution photos (max edge > 1024) post-decode using Lanczos interpolation, reducing retained dual-buffer memory per 48MP photo by **>98%** (~288 MB to ~4.72 MB combined across PIL RGB and OpenCV BGR, or ~151 MB for a batch of 32) and preventing cumulative memory exhaustion across multi-image processing batches.
- **Directory Traversal**: File discovery now enforces `p.is_file()` during recursive scans, preventing subfolders with image extensions (e.g. `Trip.2024.jpg/`) from being processed as images.

---

## [1.0.0] - 2026-09-01

### Added
- **Multi-Signal AI Scoring Engine**:
  - Technical Quality scoring via canonical Laplacian variance (sharpness) and exposure normalization.
  - Aesthetic Quality scoring using batched CLIP (ViT-B/32) neural embeddings against reference text prompts.
  - Optional facial emotion analysis via DeepFace (happiness score).
- **Near-Duplicate Suppression**: Cosine similarity threshold filtering (`--dedup_threshold`) to eliminate redundant burst shots.
- **Safe & Transactional File Operations**:
  - Two-phase commit with staging directory isolation (`.curator_stage_*`) and automatic rollback on failure.
  - Non-destructive JSON manifest tracking (`.curator_manifest.json`) with path-traversal and symlink protection.
  - Filesystem-agnostic fallback using `O_CREAT | O_EXCL` and metadata preservation (`copystat`) for FAT32, exFAT, and network shares.
- **Hardware Acceleration**: Automatic device routing for Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and CPU (`cpu`).
- **Dry-Run & CSV Previews**: Preview rankings and export scores to CSV before executing file operations.
- **In-Memory Mock Test Suite**: 80+ fast unit tests with zero model download overhead.
