# Intelligent Sky Objects Detector

## Introduction

This project is part of the TeideSat Cubesat Tracker. Its goal is to train a Convolutional Neural Network (CNN) to detect bright, point-like or streak-like objects in images of the night sky — stars, satellites, orbital debris, or airplanes — and to estimate whether a given satellite (in particular, TeideSat I) would be detectable by the resulting model.

The project has grown from a single-dataset pipeline into a strategy-based framework: normalization, masking (ground-truth generation), augmentation, loss functions, model architectures, and postprocessing are all swappable components, so different datasets and training regimes can be combined without touching the training loop itself.

Two dataset sources are currently supported:

- **Astrometry.net** — wide-field astrometric solves with verified star catalog annotations.
- **Frigate** — a public dataset of real LEO (Low Earth Orbit) surveillance imagery with actual moving-object streaks, generated via frame-differencing.

A dedicated analysis tool cross-matches detections against the GAIA DR3 catalog to empirically estimate the model's detection limit in apparent magnitude, and compares it against TeideSat I's theoretical brightness.

---

## Results

The repository includes evaluation and analysis tools for measuring model performance on the held-out test split and for estimating the apparent-magnitude detection limit.

The evaluation pipeline reports object-level precision, recall, and F1, and generates visual grids showing inputs, predicted masks, and detected positions. The satellite-visibility analysis additionally produces a GAIA DR3-calibrated completeness curve, interpolated 50%/80% detection limits, precision/purity statistics, positional errors, and per-object records.

> No fixed benchmark values are hard-coded in this README because results depend on the dataset, model checkpoint, preprocessing strategies, and training configuration used for each experiment. Generated results are saved under the configured performance directory.

---

## Requirements

The project is designed to run through Docker and Docker Compose.

- Docker
- Docker Compose
- NVIDIA Container Toolkit for GPU execution
- NVIDIA GPU with a compatible NVIDIA driver for GPU training/inference

The repository mounts `./data` and `./src` into the containers. Dataset files and generated models/results are therefore kept outside the container filesystem.

---

## Quick Start

Build the Docker images once:

```bash
docker compose build
```

The main entry points can then be run through the corresponding Docker Compose services. For example, to train an Astrometry.net model:

```bash
docker compose run --rm train
```

To preprocess Frigate data:

```bash
docker compose run --rm preprocess_frigate
```

and to train using the resulting pairs:

```bash
docker compose run --rm train_frigate
```

See the sections below for the commands and arguments used by each pipeline.

---

## Datasets

### 1. Astrometry.net dataset

Downloaded via `src/dataset_downloader/dataset_downloader.py`, which logs in to the [Astrometry.net](https://nova.astrometry.net/) API and, for each solved job, retrieves:

| File | Description |
|---|---|
| `<id>-image.fits` | The sky image. |
| `<id>-axy.fits` | The raw list of detected sources (X, Y, flux) found by Astrometry.net's own source extractor. This is noisy and not tied to a real catalog. |
| `<id>-annotations.json` | Objects from the solve that were cross-matched against real astronomical catalogs (HD, Tycho-2, 2MASS, USNO-B), each with a pixel position (`pixelx`, `pixely`). Only star-type annotations are kept. This is the reliable ground truth. |
| `rdls.fits` (when present) | RA/Dec of the reference stars used to solve the field. No magnitudes. |

The downloader is re-runnable: it skips entries that are already fully downloaded and only fetches missing pieces for a given job range.

### 2. Frigate dataset

[Frigate](https://doi.org/10.6084/m9.figshare.29545667) (Roll et al., *Scientific Data*, 2025) is a public dataset of real LEO surveillance frames captured by a commercial ground-based optical network (ExoAnalytic Solutions). Frames are 9600×6422 px, 0.5s exposure, with real satellite/debris streaks moving against the star field — unlike Astrometry.net, which only has static point sources.

Frigate frames are converted into training pairs by `src/model_training/data_preprocessing/frigate_auxiliar.py`:

1. Frames are loaded and normalized from their 16-bit unsigned encoding.
2. For each frame, a difference image is computed between the average of the preceding and following frames in a sliding window, isolating anything that moves (streaks) from the static star field.
3. The difference is thresholded (mean + `threshold_sigma` × std) and cleaned with morphological opening/dilation.
4. Because the raw frames are ~37× larger per axis than the training resolution, the mask is deliberately fattened *before* downscaling (to survive the resize) and resized with `INTER_AREA` (area-averaging) rather than nearest-neighbour, so that thin streaks are not lost between samples.
5. A final dilation gives the mask object a real, learnable area (radius configurable via `--mask-radius`), consistent with the radii used by the Astrometry.net masking strategies.
6. Image/mask pairs are saved as `<id>_image.npy` / `<id>_mask.npy`.

The script prints diagnostics (empty-mask count, average positive pixels per mask) so mask degeneracy can be caught before a wasted training run.

`FrigatePairLoader` then loads these `.npy` pairs directly — no normalization or masking strategy is applied at load time, since both were already baked in during preprocessing.

---

## Data layout

The repository expects datasets and generated artifacts under `data/`. A typical layout is:

```
data/
├── dataset/
│   ├── <astrometry-job-id>/
│   │   ├── <id>-image.fits
│   │   ├── <id>-axy.fits
│   │   ├── <id>-annotations.json
│   │   ├── rdls.fits
│   │   └── ...
│   ├── ...
│
├── frigate/
│   └── raw/
│       └── <Frigate frames>
│
├── frigate_pairs/
│   ├── <id>_image.npy
│   ├── <id>_mask.npy
│   └── ...
│
├── trained-models/
└── performance/
```

The exact contents depend on which dataset and pipeline are being used. The datasets themselves are not included in the repository; see the dataset-specific sections above for their sources and preparation steps.

---

## Data Preprocessing — Astrometry.net path

For the Astrometry.net pipeline, `DatasetLoader` reads each entry's FITS image and object table, converts to monochrome, applies a **normalization strategy**, resizes to the target shape, and then delegates ground-truth mask generation to a **masking strategy**. Results are cached per-entry (keyed by masking strategy name) so repeated loads are fast, and the cache is automatically invalidated if the annotations file is newer than the cache.

### Normalization strategies (`data_preprocessing/normalization/`)

- **`LogPercentileNormalization`** — subtracts the sky background (median), applies `log1p` compression, then stretches between the 1st/99th (or configurable) percentile. This is the default; it's relative to each image, which matters when reasoning about how bright an external object (like a satellite) would appear relative to training data.
- **`SimpleMaxNormalization`** — divides by the global maximum pixel value.
- **`IdentityNormalization`** (in `train.py`) — no-op, used for Frigate pairs since normalization already happened during `.npy` generation.

### Masking strategies (`data_preprocessing/masking/`)

- **`AnnotationMasking`** *(current default for Astrometry.net)* — draws a fixed-radius circle at each catalog-verified star position from `-annotations.json`. Ignores the noisy `.axy` file entirely. If no annotations exist for an entry, returns an empty mask.
- **`VerifiedLocalSNRMasking`** — accepts an `.axy` object only if its position is a statistically significant local peak (z-score over a local window) in the actual image; can also skip this check entirely.
- **`CircularDynamicMasking`** — filters `.axy` objects by a flux percentile threshold, merges nearby detections (KD-tree + union-find) to avoid duplicate marks on the same source, and scales each circle's radius by log-flux.
- **`bbox_masking`** — bounding-box variant (see source for details).

---

## Augmentation

`data_preprocessing/augmentation/` provides composable, tensor-level augmentations applied per-batch during training only:

- **`FlipRotateAugmentation`** — random horizontal/vertical flips plus 90°-multiple rotations.
- **`BrightnessContrastAugmentation`**, **`ElasticAugmentation`** — additional strategies available, combinable via **`ComposeAugmentation`**.

`limit_empty_images` (in `auxiliary_functions.py`) keeps all images containing at least one verified object and only a configurable fraction of completely empty images, preventing the empty-image majority from dominating the loss average and biasing the model toward predicting nothing.

---

## Model

The core model is a **U-Net**, available in two depths:

- **`UNet3Levels`** — 3 encoder/decoder levels, less capacity, weaker bottleneck.
- **`UNet4Levels`** — 4 levels, more capacity, stronger bottleneck.

Both use `GroupNorm` + `ReLU` conv blocks and skip connections. The output convolution's bias is initialized analytically from a `prior_prob` (expected foreground fraction), which stabilizes early training under heavy class imbalance — a single random initialization can otherwise push the model toward the "predict background everywhere" trivial solution when foreground pixels are extremely rare.

---

## Loss functions

All losses implement a common `LossStrategy` interface, which also declares whether the model needs a final activation and how many output channels it expects — allowing losses to be swapped (and combined) without touching the model or training loop:

- **`BCELoss`**, **`WeightedBCELoss`** (with `pos_weight` for imbalance)
- **`DiceLoss`**
- **`FocalLoss`** (`alpha`, `gamma`)
- **`FocalTverskyLoss`** (independent false-positive/false-negative weighting via `alpha`/`beta`, plus a focal `gamma`)
- **`LovaszHingeLoss`** — directly optimizes IoU; expects raw logits (`needs_activation = False`)
- **`CrossEntropyLoss`** — two-channel softmax variant
- **`CombinedLoss`** — weighted sum of any two compatible strategies (e.g. `Dice + Focal`, the current default)

---

## Training

`src/model_training/train.py` auto-detects the dataset type (presence of `.npy` files → Frigate; otherwise Astrometry.net) and swaps the loader/normalization/masking accordingly. It performs a 70/15/15 train/val/test split (`random_state=42`), reports theoretical maximum IoU and positive-pixel density diagnostics, trains with early-stopping-style checkpointing (best validation IoU), and logs full experiment configuration + per-epoch history/config as JSON under `<output>/logs/`.

```bash
python -m src.model_training.train \
    --dataset /app/data/dataset \
    --output /app/data/trained-models \
    --performance /app/data/performance \
    --epochs 150
```

For Frigate:

```bash
python -m src.model_training.data_preprocessing.frigate_auxiliar \
    --input /app/frigate/raw --output /app/data/frigate_pairs --max 1200 --size 256

python -m src.model_training.train \
    --dataset /app/data/frigate_pairs \
    --output /app/data/trained-models \
    --performance /app/data/performance \
    --epochs 100 --batch-size 4
```

All of the above are also wired up as `docker-compose` services (`train`, `preprocess_frigate`, `train_frigate`).

---

## Input and Output

The model input is a monochrome image with a size equal to the one used for training. The output is a per-pixel probability map (or logits, depending on the loss strategy), thresholded into a binary mask. To obtain object positions from the mask:

1. Apply morphological closing (`MorphologicalClosing`) or plain contour detection (`SimpleContour`) to group pixels belonging to the same object.
2. Compute each contour's centroid (or bounding-box center) to get its (x, y) position.

`Detector` wraps a trained model together with its normalization and postprocessing strategies, exposing `predict_mask()` (from a preprocessed image) and `predict_from_raw()` (from a raw FITS array). It auto-adapts to single-channel Sigmoid or multi-channel Softmax model heads based on metadata saved alongside the checkpoint.

---

## Evaluation

`run_evaluation.py` loads the test split, runs inference across it, and produces:

- An inference grid (input / predicted mask / detected positions).
- A dataset sample grid (input / ground truth objects / ground truth mask).
- Object-level precision/recall/F1 (`Evaluator.compute_object_detection_metrics`), matching predicted vs. ground-truth positions within a pixel tolerance.

```bash
python -m src.model_applications.run_evaluation \
    --model /app/data/trained-models/model-xxx.pt \
    --dataset /app/data/dataset/ \
    --output /app/data/performance/eval-xxx/
```

---

## Single-image inference

`run_inference.py` runs the full pipeline (normalize → resize → predict → postprocess) on one FITS file and saves a three-panel figure (input, predicted mask, detected positions).

```bash
python -m src.model_applications.run_inference \
    --model /app/data/trained-models/model-xxx.pt \
    --image /app/data/dataset/1544485/1544485-image.fits \
    --output /app/data/performance/infer_result.png
```

---

## Satellite visibility analysis

`run_satellite_visibility.py` answers the project's core applied question: **would the model detect TeideSat I?**

It builds an empirical detection-completeness curve calibrated against real photometry, rather than relying on the model's own (uncalibrated) internal flux units:

1. For each test entry, verified catalog annotations (`-annotations.json`) are read — these are the same objects used as training ground truth via `AnnotationMasking`, so the analysis only evaluates objects the model was actually trained to find.
2. Their original-FITS pixel positions are projected to sky coordinates via WCS (from `rdls.fits`/`-wcs.fits`/`-image.fits` header, with automatic fallback between sources), and cross-matched against **GAIA DR3** (queried live via `astroquery`, using RP-band magnitude as a Cousins-R proxy, with G-band fallback) to obtain a real, calibrated apparent magnitude for each object.
3. GAIA query results are cached to disk per-entry (`<output>/gaia_cache/`), so re-running the analysis after a code fix or on new checkpoints is near-instant instead of re-querying hundreds of fields.
4. The model's predictions are compared to the annotation positions (scaled to the training resolution) to determine per-object detection status.
5. Objects are binned by GAIA magnitude to build a completeness curve, from which the 50%/80% detection-limit magnitudes are interpolated.
6. TeideSat I's theoretical apparent magnitude — computed from its LED datasheet via the physical conversion pipeline documented in the TeideSat I "Magnitud aparente" report (Marrero García, 2022): lumens → radiant intensity at the LED's actual emission wavelength → spectral flux density → Vega magnitude in the Cousins R band — is compared against the empirical limit, producing a verdict with margin in magnitudes.

Precision/purity metrics (true vs. false positive predictions, average positional error in pixels) and the raw per-object records are also exported to CSV for further analysis.

```bash
python -m src.model_applications.run_satellite_visibility \
    --model /app/data/trained-models/model-xxx.pt \
    --dataset /app/data/dataset/ \
    --output /app/data/performance/ \
    --led-lumens 8000 --num-leds 4

# or, if the apparent magnitude is already known:
    --sat-mag 4.2
```

### Known limitations of this analysis

- The comparison is between a **moving satellite** (whose apparent brightness depends on range, phase angle, and appears as a streak rather than a static point in longer exposures) and **static reference stars**. The magnitude-to-detectability mapping is a reasonable first-order proxy, not a full radiometric simulation of a LEO pass.
- `LogPercentileNormalization` rescales each image relative to its own percentile range; an object dramatically brighter than anything else in a given frame can dominate and distort that frame's normalization in ways not represented in the training set. Empirical validation via synthetic source injection (rendering a calibrated-flux point source into a real FITS frame and running it through the full pipeline) is the recommended next step to validate this analysis further.
- Coverage depends on how many test entries actually have usable WCS/annotations; entries missing both are skipped and do not contribute to the completeness curve.

---

## Project structure

```
src/
├── dataset_downloader/
│   └── dataset_downloader.py        # Astrometry.net API downloader (re-runnable)
├── model_training/
│   ├── train.py                     # Entry point; auto-detects Astrometry.net vs Frigate
│   ├── trainer.py                   # Training loop, checkpointing, experiment logging
│   ├── auxiliary_functions.py       # Tensor conversion, empty-image limiting, config printing
│   ├── data_preprocessing/
│   │   ├── loader.py                # Astrometry.net loader (with per-entry cache)
│   │   ├── loader_frigate.py        # Frigate .npy pair loader
│   │   ├── frigate_auxiliar.py      # FITS → image/mask .npy pair generator for Frigate
│   │   ├── entry.py                 # DatasetEntry data structure
│   │   ├── normalization/           # LogPercentileNormalization, SimpleMaxNormalization
│   │   ├── masking/                 # AnnotationMasking, VerifiedLocalSNRMasking,
│   │   │                            # CircularDynamicMasking, bbox_masking
│   │   └── augmentation/            # FlipRotate, BrightnessContrast, Elastic, Compose
│   └── modelling_specs/
│       ├── models/                  # UNet3Levels, UNet4Levels
│       └── losses/                  # BCE, WeightedBCE, Dice, Focal, FocalTversky,
│                                     # LovaszHinge, CrossEntropy, CombinedLoss
├── model_analysis/
│   ├── detector.py                  # Inference wrapper (activation-agnostic)
│   ├── evaluator.py                 # Plots + object-level precision/recall/F1
│   └── postprocessing/              # MorphologicalClosing, SimpleContour
└── model_applications/
    ├── run_inference.py             # Single-image inference
    ├── run_evaluation.py            # Test-split evaluation
    └── run_satellite_visibility.py  # GAIA-calibrated TeideSat I detectability analysis
```

---

## Docker

All entry points are wired up as `docker-compose` services (`download`, `train`, `preprocess_frigate`, `train_frigate`, `eval`, `eval_frigate`, `infer`, `infer_frigate`, `visibility`), each mounting `./data` and `./src` and requesting GPU access via the NVIDIA runtime. Build once (`docker compose build`) and run the relevant service (`docker compose run --rm <service>`).

---

## References

- **Astrometry.net** — online astronomical image solving service and source of the Astrometry.net dataset used by this project.
- **Roll et al. (2025)** — Frigate, *Scientific Data*. Public dataset of real LEO surveillance imagery used for moving-object detection.
- **Gaia DR3** — Gaia Data Release 3, used to obtain calibrated stellar photometry for the detection-completeness analysis.
- **Marrero García (2022)** — TeideSat I — Magnitud aparente. Physical conversion pipeline used to estimate TeideSat I's theoretical apparent magnitude from its LED characteristics.