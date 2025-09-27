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


## Real-time processing tests

- Ultralytics YOLOv11 Pose (realtime) — `ultralytics_video/yolo11n_realtime_metrics.py`
  - Run from the `ultralytics_video` folder:
    - `python .\yolo11n_realtime_metrics.py --source ..\data_hockey.mp4 --model ..\hockeypose_1\yolo11n-pose.pt`

- MediaPipe Tasks Pose Landmarker (multi-person, realtime) — `mediapipe_video/mp4_realtime_metrics.py`
  - Run from the `mediapipe_video` folder:
    - `python .\mp4_realtime_metrics.py --video ..\data_hockey.mp4 --model .\pose_landmarker_lite.task --num-poses 10`
