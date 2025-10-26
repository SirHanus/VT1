# Unsupervised Team Clustering (SigLIP + UMAP + KMeans)

This folder contains two scripts:

1) build_training_set.py
   - Samples 1 frame per second from selected game videos
   - Runs player detection using RF-DETR-S (MMDetection) or YOLO fallback
   - Central-crops each detection to minimize background noise
   - Generates SigLIP embeddings for each crop
   - Saves per-video index.csv and embeddings.npy (and optional crop images)

2) cluster_umap_kmeans.py
   - Loads the saved embeddings from all processed videos
   - Applies UMAP dimensionality reduction
   - Clusters with KMeans (default K=2)
   - Saves labels and summary artifacts

Folder layout (new default):
- team_clustering/
  - build_training_set.py
  - cluster_umap_kmeans.py
  - clustering/                <- both per-video embeddings and final models live here
    - <video_stem>/
      - embeddings.npy
      - index.csv
      - crops/ (if --save-crops)
    - combined_embeddings.npy, combined_index.csv
    - umap_16.npy, labels_k2.npy, labeled_index.csv, summary.json
    - umap.pkl, kmeans.pkl     <- saved with --save-models

Quick start (Windows cmd):

1) Build training set

   RF-DETR-S (preferred):
   ```cmd
   python offline_pipeline\team_clustering\build_training_set.py ^
     --videos-dir videos_all\CAR_vs_NYR ^
     --glob "*.mp4" ^
     --det-config path\to\rfdetr_s_config.py ^
     --det-weights path\to\rfdetr_s.pth ^
     --central-ratio 0.6 ^
     --save-crops
   ```

   Or YOLO fallback:
   ```cmd
   python offline_pipeline\team_clustering\build_training_set.py ^
     --videos-dir videos_all\CAR_vs_NYR ^
     --glob "*.mp4" ^
     --yolo-fallback --yolo-model yolo11n.pt ^
     --central-ratio 0.6 ^
     --save-crops
   ```

   Outputs per video under `offline_pipeline/team_clustering/clustering/<video_stem>/`.

2) Cluster with UMAP + KMeans and save models

   ```cmd
   python offline_pipeline\team_clustering\cluster_umap_kmeans.py ^
     --in-root offline_pipeline\team_clustering\clustering ^
     --out-dir offline_pipeline\team_clustering\clustering ^
     --k 2 --umap-dim 16 --umap-neighbors 15 --plot --save-models
   ```

   This writes `umap.pkl` and `kmeans.pkl` into the same `clustering` folder.

3) Run sam_offline with team coloring

   ```cmd
   python offline_pipeline\sam_offline.py ^
     --source data_hockey.mp4 ^
     --team-models offline_pipeline\team_clustering\clustering
   ```

Notes
- SigLIP model default is `google/siglip-base-patch16-224`. First run may download weights (cached afterwards).
- For RF-DETR-S, ensure MMDetection is installed and versions match mmcv.
- For YOLO fallback, install `ultralytics` and pass a person-detection model.
- Use `--device cpu` if CUDA isn't available.
- For reproducibility, use `--seed` in both scripts.
