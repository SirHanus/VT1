# VT1

Minimal toolkit for offline hockey (or sports) video processing:
- Player pose estimation (YOLO pose)
- Optional segmentation (SAM2) overlay
- Unsupervised team clustering (SigLIP embeddings → UMAP + KMeans)
- PyQt6 GUI and CLI scripts

## Quick Start

```cmd
pip install -r requirements.txt
python -m vt1.gui.main
```
Or run the pipeline directly:
```cmd
python -m vt1.pipeline.sam_offline --source data_hockey.mp4 --device cuda
```

## Configuration (Single Source of Truth)
All defaults live in `config_defaults.toml` and can be overridden by:
1. `config_local.toml` 
2. Environment variables `VT1_*` (highest precedence)

Example override (Windows CMD):
```cmd
set VT1_PIPELINE_OUTPUT_DIR=D:\fast_nvme\vt1_outputs
set VT1_MODELS_DIR=D:\models\vt1
```
Access resolved settings anywhere:
```python
from vt1.config import settings
cfg = settings()
print(cfg.team_output_dir)
```
Key paths (after resolution):
- `models_dir` – root for model files
- `pose_model` / `yolo_model` – default model paths
- `team_models_dir` – where `umap.pkl` / `kmeans.pkl` live
- `pipeline_output_dir` – root outputs
- `team_output_dir` – clustering artifacts & training set

See `GUI.md` for a full explanation of each config value.

## Pipeline CLI Essentials
```cmd
python -m vt1.pipeline.sam_offline --source your_video.mp4 ^
  --pose-model models\yolo11x-pose.pt ^
  --device cuda ^
  --conf 0.25 --imgsz 640 ^
  --team-models models\team_clustering ^
  --siglip google/siglip-base-patch16-224
```
Useful flags:
- `--max-frames N` limit frames for smoke test
- `--no-sam` disable SAM2 for speed
- `--half` FP16 on CUDA
- `--metrics-json run_metrics.json` write summary

## Team Clustering Workflow (CLI)
1. Build training embeddings:
   ```cmd
   python -m vt1.team_clustering.build_training_set ^
     --videos-dir videos_all/CAR_vs_NYR --glob *.mp4 ^
     --fps 1 --device cuda --save-crops ^
     --out-dir outputs/team_clustering
   ```
2. Cluster & save models:
   ```cmd
   python -m vt1.team_clustering.cluster_umap_kmeans ^
     --in-root outputs/team_clustering --out-dir outputs/team_clustering ^
     --k 2 --umap-dim 16 --save-models
   ```
3. Evaluate models quickly:
   ```cmd
   python -m vt1.team_clustering.eval_clustering ^
     --video data_hockey.mp4 --team-models models/team_clustering ^
     --yolo-model models/yolo11n.pt --siglip google/siglip-base-patch16-224
   ```

## Build Windows Executables
Using PyInstaller (already defined in `pyproject.toml` extras):
```cmd
pip install .[build]
pyinstaller --onefile --name vt1-pipeline --paths src src\vt1\pipeline\sam_offline.py
pyinstaller --onefile --name vt1-gui --paths src -w src\vt1\gui\main.py
```
Or run the helper script:
```cmd
scripts\build_exe.bat
```
Result: `dist\vt1-pipeline.exe`, `dist\vt1-gui.exe`.

## Models & Assets
Place model weights in `models/` (defaults already referenced):
- `models/yolo11x-pose.pt`
- `models/yolo11n.pt`
Generated after clustering:
- `models/team_clustering/umap.pkl`
- `models/team_clustering/kmeans.pkl`

## GUI Documentation
Detailed GUI usage (tabs, options, tips) moved to: **`GUI.md`**.
The in-app Help tab renders `GUI.md` (and links back here).

## License / Attribution
Internal tooling; review dependencies (PyQt6, Ultralytics, Transformers, SAM2 model licensing) before redistribution.

---
Changelog: see bottom of `GUI.md` for detailed feature notes.
