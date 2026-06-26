# 📸 Album Selector

AI-powered photo selector that automatically picks the best photos from a large collection for your album.

It scores every image using three signals and selects the top-N:

| Signal | Model | Weight | What it measures |
|---|---|---|---|
| **Technical quality** | OpenCV Laplacian + brightness | 40% | Sharpness and proper exposure |
| **Aesthetic quality** | CLIP (ViT-B/32) | 40% | How "beautiful" the photo looks |
| **Emotion** *(optional)* | DeepFace | 20% | Happiness / smiles detected |

## Installation

```bash
# Clone the repo
git clone https://github.com/bhargavkukadiya/album-selector.git
cd album-selector

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

```bash
python album_selector.py \
  --input ./my_photos \          # Input folder (scanned recursively)
  --output ./selected \          # Output folder for selected photos
  --target 200 \                 # Number of photos to select (default: 200)
  --device cuda \                # Torch device: cpu, cuda, or mps (default: cpu)
  --preview_csv scores.csv \     # Save all scores to CSV before copying
  --ref_text "a stunning photo"\ # Custom CLIP reference text
  --no_deepface \                # Disable emotion scoring
  --dryrun                       # Score only, don't copy files
```

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

Then inspect `scores.csv` to review the rankings before committing:

```
Path,TotalScore,Technical,Aesthetic,Emotion
/photos/sunset.jpg,0.7823,0.6500,0.9200,0.8100
/photos/group.jpg,0.7150,0.5800,0.8000,0.8500
...
```

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
| HEIC (`.heic`) | ✅ Requires `pip install pillow-heif` |

## How Scoring Works

1. **Technical score** — Combines Laplacian variance (sharpness) and mean brightness deviation (exposure), both normalized to `[0, 1]`.
2. **Aesthetic score** — CLIP cosine similarity between the image and the reference text `"a beautiful photograph"` (configurable via `--ref_text`), rescaled from CLIP's typical `[0.1, 0.4]` range to `[0, 1]`.
3. **Emotion score** — DeepFace happiness detection, already in `[0, 1]`. Disabled by default if DeepFace is not installed.

Final score = `0.4 × technical + 0.4 × aesthetic + 0.2 × emotion`

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for dependencies

## License

[MIT](LICENSE)

## Acknowledgements

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [sentence-transformers](https://www.sbert.net/) by UKP Lab
- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil (optional, GPL-3.0 licensed)
