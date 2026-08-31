# 📸 Photo Curator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-80%2B%20passed-brightgreen.svg)](test_album_selector.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

> **AI-powered photo selector that automatically curates and ranks the best photos from large collections for your albums.**

---

## ✨ Features

- 🎯 **Multi-Signal AI Scoring**:
  - **Technical Quality (40%)**: Standardized Laplacian variance for sharpness + balanced exposure.
  - **Aesthetic Quality (40%)**: Batched CLIP (ViT-B/32) neural aesthetics scoring against configurable reference prompts.
  - **Emotion Detection (20%, optional)**: DeepFace facial expression and happiness detection.
- ⚡ **Near-Duplicate Suppression**: Automatically skips redundant burst shots based on embedding cosine similarity (`--dedup_threshold`).
- 🛡️ **Safe & Transactional File Operations**:
  - Staging and two-phase atomic commit with automatic rollback on errors.
  - Non-destructive manifest tracking (`.curator_manifest.json`) and untracked-file protection.
  - Supports all major filesystems (APFS, ext4, NTFS, FAT32, exFAT, SMB network shares) with metadata preservation (`copystat`).
- 🚀 **Hardware Accelerated**: Native support for Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), and multi-core `cpu`.
- 📊 **Dry-Run & CSV Previews**: Preview, inspect, and export rankings to CSV before moving or copying files.
- 🧪 **Fast In-Memory Test Suite**: 80+ unit tests with zero model-download overhead.

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/bhargavkukadiya/photo-curator.git
cd photo-curator

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Optional: Install DeepFace for facial emotion scoring
pip install deepface

# 4. Optional: Install HEIC/HEIF image format support
pip install pillow-heif
```

> [!NOTE]
> On the first run, the CLIP model (`clip-ViT-B-32`, ~350 MB) downloads automatically and caches locally.

---

## 🚀 Quick Start

### Basic Usage

Select the top 50 photos from a folder and copy them with index prefixes:

```bash
python album_selector.py --input ./my_photos --output ./selected --target 50
```

### Dry Run with CSV Preview & Burst Deduplication

Score all photos, filter out burst duplicates with similarity $\ge 0.90$, and export the results to CSV without copying:

```bash
python album_selector.py \
  --input ./sample_photos \
  --output ./album \
  --target 5 \
  --dedup_threshold 0.90 \
  --preview_csv scores.csv \
  --dryrun
```

Sample output rankings on included sample photos (with `--no_deepface`):

| Photo | Total Score | Technical | Aesthetic | Emotion |
|---|---|---|---|---|
| `city_night.jpg` | **0.6832** | 0.9774 | 0.3889 | 0.0000 |
| `beach_sunset.jpg` | **0.6787** | 0.9836 | 0.3737 | 0.0000 |
| `ocean.jpg` | **0.6478** | 0.9604 | 0.3351 | 0.0000 |
| `mountain.jpg` | **0.6374** | 0.9360 | 0.3387 | 0.0000 |
| `flower.jpg` | **0.6303** | 0.8938 | 0.3668 | 0.0000 |

---

## ⚙️ Advanced Options

### Custom Scoring Weights

Tailor curation for landscapes (e.g. 50% technical, 50% aesthetic, no emotion):

```bash
python album_selector.py \
  --input ./nature_trip \
  --output ./album \
  --no_deepface \
  --weight_technical 0.5 \
  --weight_aesthetic 0.5
```

> [!TIP]
> When `--no_deepface` is passed or emotion scoring is unavailable, weights automatically rebalance dynamically so total scores always use the full `[0.0, 1.0]` range.

### Hardware Acceleration

```bash
# Apple Silicon (M1/M2/M3/M4)
python album_selector.py --input ./photos --output ./album --device mps --batch_size 32

# NVIDIA CUDA
python album_selector.py --input ./photos --output ./album --device cuda --batch_size 64
```

---

## 🛠️ CLI Reference

| Flag | Description | Default |
|---|---|---|
| `--input` | Input folder with photos (scanned recursively) | *required* |
| `--output` | Output folder for selected photos | *required* |
| `--target` | Number of photos to select | `200` |
| `--batch_size` | Batch size for CLIP inference | `32` |
| `--device` | Torch device: `cpu`, `cuda`, or `mps` | `cpu` |
| `--preview_csv` | Save all computed scores to a CSV before copying | `None` |
| `--ref_text` | Reference text prompt for aesthetic scoring | `"a beautiful photograph"` |
| `--no_deepface` | Disable DeepFace facial emotion scoring | `false` |
| `--weight_technical` | Weight for technical quality (sharpness + exposure) | `0.4` |
| `--weight_aesthetic` | Weight for aesthetic quality | `0.4` |
| `--weight_emotion` | Weight for emotion score | `0.2` |
| `--dedup_threshold` | Cosine similarity threshold `[0.0-1.0]` to suppress near-duplicates (e.g. `0.90`) | *disabled* |
| `--overwrite` | Allow overwriting non-empty output directory or preview CSV | `false` |
| `--dryrun` | Score and rank only; do not copy files | `false` |

---

## 🖼️ Supported Formats

| Format | Support |
|---|---|
| JPEG (`.jpg`, `.jpeg`) | ✅ Built-in (auto EXIF orientation correction) |
| PNG (`.png`) | ✅ Built-in |
| WebP (`.webp`) | ✅ Built-in |
| BMP (`.bmp`) | ✅ Built-in |
| TIFF (`.tiff`, `.tif`) | ✅ Built-in |
| HEIC / HEIF (`.heic`, `.heif`) | ⚙️ Requires `pip install pillow-heif` |

---

## 🧪 Testing

Run the test suite:

```bash
pip install pytest
python -m pytest test_album_selector.py -v
```

All 80+ unit tests run in-memory with mocked ML models — fast execution with no GPU or model downloads required.

---

## 🤝 Contributing

Contributions are welcome!
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run test suite (`pytest test_album_selector.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [sentence-transformers](https://www.sbert.net/) by UKP Lab
- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil (optional, GPL-3.0 licensed)
