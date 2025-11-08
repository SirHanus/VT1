# VT1 Project

PyQt6 GUI and offline video pipeline for YOLO Pose + optional SAM2 segmentation, with team clustering tools.

## Repository layout

- `src/vt1/gui/` – GUI (entry: `vt1.gui.main`)
- `src/vt1/pipeline/` – pipeline runtime scripts (`sam_offline.py`, `sam_general.py`, etc.)
- `offline_pipeline/team_clustering/` – current location for team clustering artifacts (umap.pkl, kmeans.pkl)
- `videos_all/`, `videos_all_processed/` – datasets (optional)
- `requirements.txt`, `pyproject.toml`

## Quick start

1) Create/select your Python 3.10+ environment and install dependencies.

```cmd
pip install -r requirements.txt
```

2) Launch the GUI.

```cmd
python -m vt1.gui.main
```

Alternatively, running the script directly also works:

```cmd
python src\vt1\gui\main.py
```

3) In the GUI, open the Pipeline tab and set:

- Video source: a video file (default example: `data_hockey.mp4`).
- Pose model: Ultralytics YOLO pose model path/name (default: `models\yolo11x-pose.pt`).
- Device: `cuda` (GPU) or `cpu`.
  Then click Run Pipeline. Output videos are written under `src/vt1/pipeline/outputs/`.

## Pipeline tab

- Video source: Path to an input video file.
- Pose model: Ultralytics YOLO pose checkpoint or model id.
- SAM2 id: Hugging Face model id for SAM2 (optional if disabled).
- Device: `cuda` or `cpu`.
- Image size, Confidence: YOLO inference parameters.
- Max frames: >0 to limit processing for a quick test.
- Flags:
    - Show live window: preview during processing.
    - Disable SAM: faster, pose-only overlay.
    - Use FP16: half precision on CUDA.
- SAM options: frequency (SAM every), Top-K boxes, periodic Reinit.
- Metrics JSON: if set, writes per-run metrics to a JSON file.
- Team models dir: folder with `umap.pkl` and `kmeans.pkl` for team coloring (default: `models\team_clustering`).
- SigLIP id: HF id for SigLIP (vision) used by team clustering.

## Team Clustering tab

Four sub-tabs to build and evaluate team clustering:

1) Build Set

- Sample crops from videos and compute embeddings.
- Inputs: videos dir + glob, detector (MMDetection RF-DETR-S or YOLO fallback), FPS, crop settings, SigLIP id, device.
- Outputs: per-video folders with embeddings; optional crop images.

2) Cluster

- Fit UMAP + KMeans on collected embeddings.
- Options: K, UMAP dim/neighbors/metric/min_dist, reuse existing UMAP, limit rows, save models.
- Persisting writes `umap.pkl` and `kmeans.pkl` for inference in the Pipeline tab.

3) Audit

- Produce image mosaics to visually inspect sampled crops per video.

4) Evaluate

- Quick detection + team coloring on images or sampled video frames.
- Useful to sanity-check trained clustering models.
- In the Evaluate tab, set:
    - Team models: default `models\team_clustering`
    - YOLO model: default `models\yolo11n.pt`

## Configuration

Defaults are centralized in `config_defaults.toml` with optional overrides:

1. `config_defaults.toml` (committed) – base defaults.
2. `config_local.toml` (optional, not committed) – developer overrides.
3. Environment variables `VT1_*` – highest precedence (e.g. `VT1_MODELS_DIR`, `VT1_PIPELINE_OUTPUT_DIR`).

All resolved via `vt1.config.settings()` and exposed as:

- models_dir
- pose_model
- yolo_model
- team_models_dir
- pipeline_output_dir (root-level `outputs/`)
- team_output_dir (`outputs/team_clustering`)

Example `config_local.toml`:

```
models_dir = "D:/data/vt1_models"
pipeline_output_dir = "D:/data/vt1_outputs"
log_level = "DEBUG"
```

Environment override (Windows CMD):

```
set VT1_PIPELINE_OUTPUT_DIR=D:\fast_nvme\vt1_outputs
set VT1_MODELS_DIR=D:\models\vt1
```

## Where are outputs saved?

- Pipeline videos: `outputs/` (root-level)
- Team clustering artifacts: `outputs/team_clustering/` (default; can override via config)

## Troubleshooting

- No CUDA? Set Device to `cpu`.
- Missing models? Verify file paths, or allow first run to download from Hugging Face (SAM2/SigLIP).
- Large videos: set Max frames to test quickly.
- Windows paths: prefer local drives (e.g., `D:\`) instead of network paths.

## Run pipeline directly (CLI)

You can also run the pipeline script from a terminal:

```cmd
python -m vt1.pipeline.sam_offline --source data_hockey.mp4
```

## Build Windows executables (.exe)

You can bundle the GUI and pipeline into standalone executables using PyInstaller.

### Option A: Use the provided batch script

```cmd
build_exe.bat
```

This creates `dist\vt1-gui.exe` and `dist\vt1-pipeline.exe`.

### Option B: Manual commands

Install build extras:

```cmd
pip install .[build]
```

Pipeline CLI:

```cmd
pyinstaller --onefile --name vt1-pipeline --paths src src\vt1\pipeline\sam_offline.py
```

GUI:

```cmd
pyinstaller --onefile --name vt1-gui --paths src -w src\vt1\gui\main.py
```

After building, run:

```cmd
dist\vt1-gui.exe
```

If the GUI detects `vt1-pipeline.exe` in the same folder it will invoke it directly for pipeline runs; otherwise it
falls back to the embedded Python module.

### Notes

- First launches may download SAM2 / SigLIP weights (internet required).
- Keep model `.pt` files accessible; bundle them separately if needed.
- For reproducible builds, pin exact versions in `pyproject.toml`.

---

# Model files

All model files are centralized under `models/`:

- `models/yolo11x-pose.pt` (YOLO pose)
- `models/yolo11n.pt` (YOLO detect fallback / eval)
- `outputs/team_clustering/umap.pkl`, `outputs/team_clustering/kmeans.pkl` (team clustering models generated after
  running clustering)

### Generate team clustering models

If `umap.pkl` and `kmeans.pkl` are missing:

1. Build training set (embeddings):
   ```cmd
   python -m vt1.team_clustering.build_training_set --videos-dir videos_all/CAR_vs_NYR --glob *.mp4 --fps 1 --device cuda --out-dir outputs/team_clustering --save-crops
   ```
2. Cluster and save models:
   ```cmd
   python -m vt1.team_clustering.cluster_umap_kmeans --in-root outputs/team_clustering --out-dir outputs/team_clustering --k 2 --umap-dim 16 --save-models
   ```

The files `outputs/team_clustering/umap.pkl` and `kmeans.pkl` will then be available for Evaluate and Pipeline team
coloring.

---

## Change log

- 0.1.0 (08-11-2025): Initial release with GUI, offline pipeline, and team clustering tools.