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
