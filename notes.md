# Hockey Video Analysis Pipeline

## Overview

This project implements a complete pipeline for analyzing hockey game footage with:
- **Player detection and pose estimation** using YOLO11
- **Player segmentation** using SAM2
- **Unsupervised team clustering** using SigLIP embeddings + UMAP + KMeans
- **Batch video processing** with metrics tracking

## Main Pipeline: offline_pipeline/

The core processing pipeline lives in `offline_pipeline/` and processes videos with pose detection, segmentation, and optional team coloring.

### sam_offline.py - Main Processing Script

Processes hockey videos with YOLO pose detection, optional SAM2 segmentation, and team clustering.

**Key Features:**
- YOLO11 pose detection with configurable confidence thresholds
- SAM2 segmentation for player masks
- Team clustering inference (colors players by team)
- FP16 support for faster CUDA processing
- Memory management (periodic SAM re-init, cache clearing)
- Progress tracking with tqdm
- JSON metrics output

**Basic Usage:**

```cmd
python offline_pipeline\sam_offline.py --source data_hockey.mp4
```

**With Team Coloring:**

```cmd
python offline_pipeline\sam_offline.py ^
  --source data_hockey.mp4 ^
  --team-models offline_pipeline\team_clustering\clustering
```

**Performance Options:**

```cmd
python offline_pipeline\sam_offline.py --source data_hockey.mp4 --half --sam-every 5 --sam-topk 3 --sam-reinit 60 --empty-cache-interval 25
```

**Key Arguments:**
- `--source` - Input video path
- `--out` - Output video path (auto-named if omitted)
- `--max-frames` - Limit frames processed
- `--no-sam` - Skip SAM segmentation (pose-only, faster)
- `--half` - Use FP16 for CUDA (faster, less memory)
- `--sam-every N` - Run SAM every N frames
- `--sam-topk K` - Limit SAM to K highest-confidence detections
- `--sam-reinit N` - Reinitialize SAM every N frames (prevents memory creep)
- `--empty-cache-interval N` - Clear CUDA cache every N frames
- `--team-models DIR` - Path to clustering models (umap.pkl, kmeans.pkl)

## Team Clustering: offline_pipeline/team_clustering/

Unsupervised learning pipeline to automatically separate players into teams based on uniform appearance.

### Workflow

**1. Build Training Set** (`build_training_set.py`)

Samples frames from game videos, detects players, crops them, and generates SigLIP embeddings.

```cmd
python offline_pipeline\team_clustering\build_training_set.py ^
  --videos-dir videos_all\CAR_vs_NYR ^
  --glob "*.mp4" ^
  --yolo-model yolo11n.pt ^
  --central-ratio 0.6 ^
  --save-crops
```

**Key Arguments:**
- `--videos-dir` - Directory containing training videos (one game)
- `--glob` - File pattern (default: `*.mp4`)
- `--yolo-model` - YOLO model path for player detection
- `--central-ratio` - Crop ratio to focus on uniform (default: 0.6)
- `--save-crops` - Save cropped player images for inspection
- `--frame-step` - Sample every Nth frame (default: 30, ~1fps)
- `--max-frames` - Limit total frames processed per video

**Outputs:** 
- `offline_pipeline/team_clustering/clustering/<video_stem>/embeddings.npy`
- `offline_pipeline/team_clustering/clustering/<video_stem>/index.csv`
- `offline_pipeline/team_clustering/clustering/<video_stem>/crops/` (if --save-crops)

**2. Train Clustering Model** (`cluster_umap_kmeans.py`)

Combines embeddings from all videos, applies UMAP dimensionality reduction, and trains KMeans (K=2 for two teams).

```cmd
python offline_pipeline\team_clustering\cluster_umap_kmeans.py ^
  --in-root offline_pipeline\team_clustering\clustering ^
  --out-dir offline_pipeline\team_clustering\clustering ^
  --k 2 ^
  --umap-dim 16 ^
  --umap-neighbors 15 ^
  --plot ^
  --save-models
```

**Key Arguments:**
- `--in-root` - Directory with per-video embeddings
- `--out-dir` - Where to save models and results
- `--k` - Number of clusters (2 for two teams)
- `--umap-dim` - UMAP target dimensions (default: 16)
- `--umap-neighbors` - UMAP n_neighbors parameter
- `--save-models` - Save umap.pkl and kmeans.pkl for inference
- `--plot` - Generate UMAP visualization plots

**Outputs:**
- `offline_pipeline/team_clustering/clustering/umap.pkl`
- `offline_pipeline/team_clustering/clustering/kmeans.pkl`
- `offline_pipeline/team_clustering/clustering/combined_embeddings.npy`
- `offline_pipeline/team_clustering/clustering/labeled_index.csv`
- `offline_pipeline/team_clustering/clustering/summary.json`
- `offline_pipeline/team_clustering/clustering/umap_scatter.png` (if --plot)

**3. Evaluate Clustering** (`eval_clustering.py`)

Test the clustering model on sample images to verify team separation quality.

```cmd
python offline_pipeline\team_clustering\eval_clustering.py ^
  --video videos_all\CAR_vs_NYR\CAR_vs_NYR_001.mp4 ^
  --models-dir offline_pipeline\team_clustering\clustering ^
  --max-frames 100 ^
  --out-dir offline_pipeline\team_clustering\eval_output
```

**Key Arguments:**
- `--video` - Test video path
- `--models-dir` - Directory with umap.pkl and kmeans.pkl
- `--max-frames` - Number of frames to evaluate
- `--out-dir` - Where to save annotated images

**Outputs:**
- Annotated images showing detected players with team color overlays
- Console output showing cluster distribution statistics

### Important Notes on Team Clustering

⚠️ **Per-Match Training Required:** Each match should have its own clustering model trained on that specific game's footage. Different games have different team uniforms, lighting conditions, and camera angles. Train on 3-5 video clips from the same match for best results.

⚠️ **Model Location for Inference:** The `sam_offline.py` script expects to find `umap.pkl` and `kmeans.pkl` in the directory specified by `--team-models`. Make sure these files exist there before running with team coloring enabled.

## Development Timeline

### 3/10
- Base demos with Camera input for:
    - MediaPipe Pose
    - OpenVINO MoveNet
    - TRT Pose
    - PyTorch Keypoint Baseline
    - MMPose

### 16/10
- Minimal YOLO Pose + SAM2 POC on data_hockey.mp4 (offline and online)
- Pass yolo poses into SAM2 for mask generation
- Performance and memory reset options added (run larger videos reliably, try different model sizes)
- Batch run script
- Logging and progress bars (saving in json)

### 30/10
- Add unsupervised clustering (into teams) - separate
- Maybe add number identification? connect with skeleton
- ~~Try to run it on bigger gpu and parallelize?~~

### 14/11
- Starting writing the paper (some chapter split etc.)
- Combine the clustering with the offline-pipeline (maybe still do it online based on the perf.)

### 31/10 (Current)
- **Team clustering integrated** - Full workflow from data generation, training to inference
- **Per-match model training** - Build custom clustering models per game (needed so far)
- **Evaluation tools** - Visual validation of team separation quality
- *FUTURE:*
  - Finetune yolo model? Make dataset and improve it. (Yolo performance currently affect other parts too)
  - Still paralelization?
  - Boundry for the playing field (cut the background)
  - Dataset finetuning - Take whole perspective and see peroformance (topic for discussion)
  - Make it more user friendly (gui etc.)

## Batch Processing (PowerShell)

- Script: `process_all.ps1`
- What it does:
    - Recursively scans `videos_all` for all `*.mp4` files (including subfolders).
    - Runs `offline_pipeline/sam_offline.py` on each video with configurable flags.
    - Writes processed videos to `videos_all_processed`, preserving the original subfolder structure.
    - Saves per-video metrics as JSON next to each output `.mp4` and an aggregate `metrics_aggregate.json` in
      `videos_all_processed`.
- Outputs:
    - Videos: `videos_all_processed\<subfolders>\<filename>.mp4`
    - Per-video metrics: `videos_all_processed\<subfolders>\<filename>.json`
    - Aggregate metrics: `videos_all_processed\metrics_aggregate.json`

**Example batch run with team coloring:**
```powershell
.\process_all.ps1 -TeamModels "offline_pipeline\team_clustering\clustering"
```

## Complete Example Workflow

### 1. Train Team Clustering for a Match

```cmd
REM Step 1: Build training set from 3-5 clips of the same match
python offline_pipeline\team_clustering\build_training_set.py ^
  --videos-dir videos_all\CAR_vs_NYR ^
  --glob "*.mp4" ^
  --yolo-model yolo11n.pt ^
  --central-ratio 0.6 ^
  --save-crops ^
  --frame-step 30 ^
  --max-frames 500

REM Step 2: Train UMAP + KMeans and save models
python offline_pipeline\team_clustering\cluster_umap_kmeans.py ^
  --in-root offline_pipeline\team_clustering\clustering ^
  --out-dir offline_pipeline\team_clustering\clustering ^
  --k 2 ^
  --save-models ^
  --plot

REM Step 3: Evaluate clustering quality (optional)
python offline_pipeline\team_clustering\eval_clustering.py ^
  --video videos_all\CAR_vs_NYR\CAR_vs_NYR_001.mp4 ^
  --models-dir offline_pipeline\team_clustering\clustering ^
  --max-frames 100 ^
  --out-dir offline_pipeline\team_clustering\eval_output
```

### 2. Process Videos with Team Coloring

```cmd
REM Single video
python offline_pipeline\sam_offline.py ^
  --source videos_all\CAR_vs_NYR\CAR_vs_NYR_001.mp4 ^
  --team-models offline_pipeline\team_clustering\clustering ^
  --half ^
  --sam-every 2 ^
  --max-frames 800

REM Batch process all videos in a folder
.\process_all.ps1 -TeamModels "offline_pipeline\team_clustering\clustering"
```

## Tips & Troubleshooting

### Improving Team Clustering Quality

If team separation is poor:

1. **Check training data quality**
   - Use `--save-crops` when building training set
   - Inspect crops in `clustering/<video_stem>/crops/`
   - Ensure both teams are well-represented
   - Verify crops show mostly uniform, minimal background

2. **Adjust cropping parameters**
   - Increase `--central-ratio` (e.g., 0.7-0.8) to focus more on uniform center
   - Lower values (0.5-0.6) include more context but also more noise

3. **Use more training data**
   - Process 3-5 clips from the same match
   - Aim for 300-500 player detections minimum per team
   - Ensure variety: different camera angles, ice positions

4. **Tune UMAP/KMeans parameters**
   - Try different `--umap-neighbors` (10-20)
   - Adjust `--umap-dim` (8-32)
   - Lower dimensions = simpler separation but may lose nuance

5. **Verify model files**
   - Check that `umap.pkl` and `kmeans.pkl` exist in models directory
   - Check `summary.json` for cluster statistics

### Memory Management

For long videos or limited VRAM:
- Use `--half` for FP16 precision
- Increase `--sam-every` (process fewer frames)
- Reduce `--sam-topk` (segment fewer players per frame)
- Enable periodic resets: `--sam-reinit 60 --empty-cache-interval 25`

### Performance Tuning

- **Fastest (pose-only):** `--no-sam`
- **Balanced:** `--half --sam-every 5 --sam-topk 3`
- **High quality:** `--sam-every 1 --sam-topk 10` (slow)

## Project Structure

```
D:\WORK\VT1\
├── offline_pipeline/              # Main processing pipeline
│   ├── sam_offline.py            # Main video processor (pose + SAM + teams)
│   ├── sam_general.py            # SAM2 wrapper utilities
│   ├── list_sources.py           # Helper to list video devices
│   ├── yolo11n_realtime_metrics.py  # Real-time YOLO testing
│   ├── outputs/                  # Processed video outputs
│   └── team_clustering/          # Team clustering sub-pipeline
│       ├── build_training_set.py # Step 1: Extract embeddings
│       ├── cluster_umap_kmeans.py # Step 2: Train models
│       ├── eval_clustering.py    # Step 3: Evaluate quality
│       ├── audit_training_set.py # Inspect training data
│       ├── README.md             # Clustering documentation
│       └── clustering/           # Training data and models
│           ├── <video_stem>/     # Per-video embeddings/crops
│           ├── umap.pkl          # Trained UMAP model
│           ├── kmeans.pkl        # Trained KMeans model
│           └── summary.json      # Training statistics
├── videos_all/                   # Raw input videos (organized by match)
├── videos_all_processed/         # Batch processed outputs
├── process_all.ps1               # Batch processing script
├── requirements.txt              # Python dependencies
└── notes.md                      # This file
```


