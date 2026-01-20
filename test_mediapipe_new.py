"""Test MediaPipe 0.10+ new Tasks API for pose estimation."""

import urllib.request
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download pose landmarker model if not present
model_path = Path("models/pose_landmarker_heavy.task")
model_path.parent.mkdir(exist_ok=True)

if not model_path.exists():
    print(f"Downloading pose landmarker model to {model_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")
else:
    print(f"Model already exists at {model_path}")

# Test the model
print("\nTesting MediaPipe PoseLandmarker...")
try:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=10,  # Detect multiple people
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("✅ PoseLandmarker created successfully!")
    print(f"   Model path: {model_path}")
    print(f"   Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    landmarker.close()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
