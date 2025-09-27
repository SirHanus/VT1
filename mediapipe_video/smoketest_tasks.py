import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VIDEO = r"D:\WORK\VT1\mediapipe_video\data_hockey.mp4"
MODEL = r"D:\WORK\VT1\mediapipe_video\pose_landmarker_full.task"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("ERROR: cannot open video", VIDEO)
    sys.exit(2)

ok, frame = cap.read()
cap.release()
if not ok:
    print("ERROR: cannot read first frame")
    sys.exit(3)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

base_options = mp_python.BaseOptions(model_asset_path=MODEL)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=6,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    min_pose_presence_confidence=0.5,
    output_segmentation_masks=False,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)
res = landmarker.detect_for_video(mp_image, 0)
print("OK: poses detected:", len(res.pose_landmarks) if res and res.pose_landmarks else 0)

