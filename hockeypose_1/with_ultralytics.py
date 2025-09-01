from ultralytics import YOLO
import cv2, os

SRC = "data_hockey.mp4"            # your input video
OUT_DIR = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "data_hockey_annotated.mp4")

# Read FPS from the source (fallback to 30 if unreadable)
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
if not fps or fps <= 0 or fps > 240:
    fps = 30

model = YOLO("yolo11n-pose.pt")
writer = None

for res in model.predict(source=SRC, conf=0.25, stream=True):
    frame = res.plot()  # annotated BGR frame (numpy array)
    if writer is None:
        h, w = frame.shape[:2]
        # 'mp4v' works everywhere; try 'avc1' if you specifically want H.264 and have codecs
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for MP4.")
    writer.write(frame)

if writer:
    writer.release()

print(f"Wrote {OUT}")
