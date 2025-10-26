# What I tried

Webcam pose-detection tools I tested:
- MediaPipe (mediapipe_webcam.py) — best real-time performance and stability.
- OpenVINO MoveNet (openvino_movenet_webcam.py) — fast, good accuracy.
- TRT Pose / TensorRT (trtpose_webcam.py) — fast after optimization.
- PyTorch keypoint baseline (torchkeypoiny_webcam.py) — okay but heavier; slower startup.
- MMPose (mmpose_webcam.py) — slowest on my setup; worst real-time performance. (Lagging)

Quick takeaway:
- Best: MediaPipe
- Worst: MMPose

Offline video processing:
- Ultralytics YOLO Pose on hockey video (with_ultralytics.py) — processed data_hockey.mp4 offline using yolo11n-pose.pt; output saved to outputs/data_hockey_annotated.mp4.
- YOLO Pose + SAM2 POC (ultralytics_video/sam_offline.py) — processes data_hockey.mp4; output ultralytics_video/outputs/data_hockey_sam_pose.mp4. Use --no-sam for pose-only (faster).

Efficiency/memory updates (sam_offline.py):
- Faster + lighter:
  - FP16 on CUDA (--half), cuDNN autotune, adjustable --imgsz/--conf.
  - Throttle SAM: --sam-every N, limit boxes: --sam-topk K.
- Avoid slowdown/memory creep for longer videos:
  - Periodic SAM re-init (--sam-reinit N) + CUDA cache/GC (--empty-cache-interval) to reset memory; allows running larger/longer clips without stalling.
  - Reduced capture buffering and in-place drawing to cut copies.
- QoL:
  - tqdm progress bar with proper totals (respects --max-frames).
  - Auto-named outputs based on params when --out is omitted.

Example longer-run (balanced):
- `python .\sam_offline.py --half --sam-every 5 --sam-topk 3 --sam-reinit 60 --empty-cache-interval 25`


## Real-time processing tests

- Ultralytics YOLOv11 Pose (realtime) — `ultralytics_video/yolo11n_realtime_metrics.py`
  - Run from the `ultralytics_video` folder:
    - `python .\yolo11n_realtime_metrics.py --source ..\data_hockey.mp4 --model ..\hockeypose_1\yolo11n-pose.pt`

- MediaPipe Tasks Pose Landmarker (multi-person, realtime) — `mediapipe_video/mp4_realtime_metrics.py`
  - Run from the `mediapipe_video` folder:
    - `python .\mp4_realtime_metrics.py --video ..\data_hockey.mp4 --model .\pose_landmarker_lite.task --num-poses 10`

- Offline POC (SAM + YOLO Pose) — `ultralytics_video/sam_offline.py`
  - Run from the `ultralytics_video` folder:
    - Pose-only quick check (30 frames): `python .\sam_offline.py --no-sam --max-frames 30`
    - With SAM masks (CPU example, 3 frames): `python .\sam_offline.py --device cpu --max-frames 3`


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
 - batch run script
 - logging and progress bars (saving in json)

### 30/10
 - Add unsupervised clustering (into teams)
 - Maybe add number identification? connect with skeleton
 - Try to run it on bigger gpu and parallelize?

## Batch processing (PowerShell)

- Script: `process_all.ps1`
- What it does:
  - Recursively scans `videos_all` for all `*.mp4` files (including subfolders).
  - Runs `ultralytics_video/sam_offline.py` on each video with flags: `--max-frames 800 --sam-every 1 --sam-reinit 70`.
  - Writes processed videos to `videos_all_processed`, preserving the original subfolder structure.
  - Saves per-video metrics as JSON next to each output `.mp4` and an aggregate `metrics_aggregate.json` in `videos_all_processed`.
- Outputs:
  - Videos: `videos_all_processed\<subfolders>\<filename>.mp4`
  - Per-video metrics: `videos_all_processed\<subfolders>\<filename>.json`
  - Aggregate metrics: `videos_all_processed\metrics_aggregate.json`
