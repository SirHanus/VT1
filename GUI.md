# VT1 GUI Documentation

This document provides detailed usage instructions for the PyQt6 GUI. For a high-level overview and CLI examples see `README.md`.

## First Run: Quick Start Workflow

On first launch (when team models don't exist), VT1 shows a **startup dialog** with the option to run the complete training workflow automatically:

1. **Build training set** - Extracts player crops and embeddings from videos
2. **Cluster** - Creates team classification models (UMAP + KMeans)
3. **Run pipeline demo** - Processes first video with team coloring enabled

**Requirements:**
- Training videos must exist in the configured directory (default: `videos_all/CAR_vs_NYR`)
- Videos should match the glob pattern (default: `*.mp4`)

**What happens:**
- The GUI runs `build_training_set` → `cluster_umap_kmeans` → `sam_offline` with all config defaults
- Models are saved to `models/team_clustering/` (configurable)
- Demo video (300 frames) saved to `outputs/` showing team coloring in action
- Progress is shown with live logs
- Takes 5-30 minutes depending on dataset size and hardware

**Options:**
- **Run Training Workflow** - Automatic setup (recommended for first-time users)
- **Skip** - Configure manually using Team Clustering tabs
- **Exit** - Close the application

After completion, you can immediately use the Pipeline tab with team coloring enabled.

---

## Tabs Overview
1. Pipeline
2. Team Clustering (Build Set, Cluster, Audit, Evaluate sub-tabs)
3. Help (renders this document and README)

---
## 1. Pipeline Tab
Run the offline video processing pipeline.

Controls:
- Video source: Path to input video file (required).
- Pose model: YOLO pose model path / id (e.g. `models/yolo11x-pose.pt`).
- SAM2 id: HuggingFace SAM2 model id (e.g. `facebook/sam2-hiera-large`). Leave as default or disable SAM.
- Device: `cuda` or `cpu`.
- Image size: YOLO inference size (trade speed vs accuracy). Smaller is faster.
- Confidence: YOLO detection confidence threshold.
- Max frames: Limit processing for quick tests (0 = all).

Flags:
- Show live window: Display frame-by-frame overlay.
- Disable SAM: Skip SAM2 segmentation (pose only; faster).
- Use FP16: Half precision (CUDA only) to improve speed / reduce VRAM.

SAM options:
- SAM every: Run SAM segmentation every N frames.
- SAM top-K: Limit SAM processing to top K detection boxes.
- SAM reinit: Reinitialize SAM periodically (0 = never) if drift is observed.
- Empty CUDA cache every N: Calls `torch.cuda.empty_cache()` to reduce fragmentation.

Team clustering:
- Team models dir: Directory containing `umap.pkl` + `kmeans.pkl` (coloring players by cluster/team).
- SigLIP id: Embedding backbone for clustering inference.
- Central crop ratio: Fraction of bbox retained around center when embedding (lower zooms in).
- Disable team coloring: Ignore team models even if present.

Metrics JSON: Path to write a summary metrics JSON (optional).

Run behavior:
- On start, the GUI spawns a subprocess (or uses packaged exe) and streams logs.
- Progress bar switches from indeterminate to frame counts if available.

Output:
- Artifacts under `outputs/` (see config for overrides).

---
## 2. Team Clustering Tab
Provides a 4-stage workflow to build and evaluate team clustering models.

### 2.1 Build Set
Extract player crops and embeddings.

Key inputs:
- Videos dir + Glob: Source video selection.
- Detector: RF-DETR-S (MMDetection) or YOLO fallback (auto if RF-DETR config/weights absent).
- FPS: Frame sampling rate.
- Central ratio / Min crop size: Crop refinement.
- SigLIP model: Embedding backbone.
- Batch size: Embedding batch (speed vs memory).
- Save crops: Persist JPEG crops for auditing.

Outputs per video: folder containing `embeddings.npy`, `index.csv`, optional `crops/`.

### 2.2 Cluster
Fit UMAP + KMeans.

Parameters:
- K: Number of clusters (teams).
- UMAP dim / neighbors / metric / min_dist: Dimensionality reduction behavior.
- Reuse UMAP: Provide an existing `umap.pkl` for consistent mapping.
- Limit rows: Cap total samples (memory or fast test).
- Plot: Save `umap_scatter.png` (if dim >=2).
- Save models: Write `umap.pkl` + `kmeans.pkl` to models directory.

Artifacts: `combined_embeddings.npy`, `umap_<dim>.npy`, `labels_kK.npy`, `labeled_index.csv`, `summary.json`, optional plot.

### 2.3 Audit
Visual check of sampled crops.
- Per-video limit: number of crops per video in mosaic.
- Save grid: Write mosaics for quick manual inspection.

### 2.4 Evaluate
Apply clustering models to images or sampled video frames.

Inputs:
- Images dir + glob OR video with frame-step + max-frames.
- Team models dir: Must contain `umap.pkl` and `kmeans.pkl`.
- SigLIP id and YOLO model.
- Inference: Image size, confidence, max boxes, central crop ratio.
- Output dir: Root evaluation folder (timestamped subfolder).
- Show: live preview windows.
- Save grid: mosaic of first N annotated frames.
- Limit images: cap annotated outputs.

Outputs: annotated images, `summary.json`, optional `mosaic.jpg`.

---
## 3. Help Tab
Renders `GUI.md` (this file) if present; falls back to `README.md`. Ensures documentation is available in packaged builds.

---
## Configuration Reference (Summary)
See `config_defaults.toml` for full inline comments. Common adjustments:
- Storage paths: `models_dir`, `pipeline_output_dir`, `team_output_dir`.
- Detector/clustering thresholds: `yolo_conf`, `det_score_thr_default`, `cluster_k`, `umap_*`.
- Sampling granularity: `build_fps`, `eval_frame_step`.
- Team cropping: `central_ratio_default`.

Override strategy:
```cmd
set VT1_CLUSTER_K=3
set VT1_UMAP_DIM=8
```

---
## Generating Team Models Recap
```cmd
python -m vt1.team_clustering.build_training_set --videos-dir videos_all/CAR_vs_NYR --glob *.mp4 --fps 1 --device cuda --out-dir outputs/team_clustering
python -m vt1.team_clustering.cluster_umap_kmeans --in-root outputs/team_clustering --out-dir outputs/team_clustering --k 2 --umap-dim 16 --save-models
```
Resulting files moved/used from `models/team_clustering/` (if configured) or default output folder.

---
## Tips
- Start with a small FPS (1) for dataset build; increase if coverage is insufficient.
- Use `--limit` during clustering to experiment with hyperparameters quickly.
- Regenerate `umap.pkl` only when embedding distribution changes (e.g. different SigLIP model).
- Keep model files on fast storage to reduce load latency.

---
## Changelog (brief)
- 0.1.0: Initial release (GUI, offline pipeline, clustering workflow).

For a concise overview, read `README.md`.

