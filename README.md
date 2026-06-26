# 📸 Album Selector

AI-powered photo selector that automatically picks the best photos from a large collection for your album.

It scores every image using three signals and selects the top-N:

| Signal | Model | Weight | What it measures |
|---|---|---|---|
| **Technical quality** | OpenCV (Laplacian + brightness) | 40% | Sharpness and proper exposure |
| **Aesthetic quality** | CLIP (ViT-B/32) | 40% | How "beautiful" the photo looks |
| **Emotion** *(optional)* | DeepFace | 20% | Happiness / smiles detected |

## Installation

```bash
# Clone the repo
git clone https://github.com/bhargavkukadiya/Album-selector.git
cd Album-selector

# Install dependencies
pip install -r requirements.txt

# Optional: Install DeepFace for emotion scoring
pip install deepface

# Optional: Install HEIC support
pip install pillow-heif
```

> [!NOTE]
> On first run, the CLIP model (`clip-ViT-B-32`, ~350 MB) will be downloaded automatically and cached locally.

## Usage

### Basic usage

```bash
python album_selector.py --input ./my_photos --output ./selected --target 50
```

### All options

| Flag | Description | Default |
|---|---|---|
| `--input` | Input folder with photos (scanned recursively) | *required* |
| `--output` | Output folder for selected photos | *required* |
| `--target` | Number of photos to select | `200` |
| `--device` | Torch device: `cpu`, `cuda`, or `mps` | `cpu` |
| `--preview_csv` | Save all scores to a CSV before copying | — |
| `--ref_text` | Reference text for CLIP aesthetic scoring | `"a beautiful photograph"` |
| `--no_deepface` | Disable DeepFace emotion scoring | `false` |
| `--dryrun` | Score only, don't copy files | `false` |

### Dry run with CSV preview

Score all images and export results without copying anything:

```bash
python album_selector.py \
  --input ./vacation_photos \
  --output ./album \
  --target 100 \
  --preview_csv scores.csv \
  --dryrun
```

Then inspect `scores.csv` to review the rankings before committing. Here's real output from the included sample photos:

| Photo | Total Score | Technical | Aesthetic | Emotion |
|---|---|---|---|---|
| city_night.jpg | **0.5510** | 0.9887 | 0.3889 | 0.0000 |
| beach_sunset.jpg | **0.5462** | 0.9918 | 0.3737 | 0.0000 |
| ocean.jpg | **0.5261** | 0.9802 | 0.3351 | 0.0000 |
| mountain.jpg | **0.5227** | 0.9680 | 0.3387 | 0.0000 |
| blurry_abstract.jpg | **0.3508** | 0.4670 | 0.4102 | 0.0000 |

> [!TIP]
> Run with `--no_deepface` if you don't need emotion scoring — it's significantly faster.

### GPU acceleration

```bash
# NVIDIA GPU
python album_selector.py --input ./photos --output ./album --device cuda

# Apple Silicon
python album_selector.py --input ./photos --output ./album --device mps
```

## Supported Formats

| Format | Support |
|---|---|
| JPEG (`.jpg`, `.jpeg`) | ✅ Built-in |
| PNG (`.png`) | ✅ Built-in |
| WebP (`.webp`) | ✅ Built-in |
| HEIC (`.heic`) | ⚙️ Requires `pip install pillow-heif` |

## How Scoring Works

1. **Technical score** — Combines Laplacian variance (sharpness) and mean brightness deviation (exposure), both normalized to `[0, 1]`.
2. **Aesthetic score** — CLIP cosine similarity between the image and the reference text `"a beautiful photograph"` (configurable via `--ref_text`), rescaled from CLIP's typical `[0.1, 0.4]` range to `[0, 1]`.
3. **Emotion score** — DeepFace happiness detection, already in `[0, 1]`. Returns `0.0` if DeepFace is not installed or `--no_deepface` is passed.

Final score = `0.4 × technical + 0.4 × aesthetic + 0.2 × emotion`

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt) for dependencies

## Testing

```bash
pip install pytest
python -m pytest test_album_selector.py -v
```

All 36 tests run with mocked ML models — no GPU or model downloads needed.

## License

[MIT](LICENSE)

## Acknowledgements

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [sentence-transformers](https://www.sbert.net/) by UKP Lab
- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil (optional, GPL-3.0 licensed)
