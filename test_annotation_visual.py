"""
Quick test to visualize annotations on an image to see if coordinates are correct.
"""

import json
from pathlib import Path

import cv2

# Load annotation
ann_path = Path("D:/WORK/VT1/labelstudio_data/annotations/001_frame_000000.json")
img_path = Path("D:/WORK/VT1/labelstudio_data/images/001_frame_000000.jpg")

with open(ann_path, "r", encoding="utf-8") as f:
    ann_data = json.load(f)

# Load image
img = cv2.imread(str(img_path))
h, w = img.shape[:2]

print(f"Image size: {w}x{h}")

# Draw annotations
for result in ann_data["annotations"][0]["result"]:
    if result["type"] == "rectanglelabels":
        # Draw bounding box
        x_pct = result["value"]["x"]
        y_pct = result["value"]["y"]
        w_pct = result["value"]["width"]
        h_pct = result["value"]["height"]

        # Convert to pixels
        x1 = int((x_pct / 100) * w)
        y1 = int((y_pct / 100) * h)
        x2 = int(((x_pct + w_pct) / 100) * w)
        y2 = int(((y_pct + h_pct) / 100) * h)

        print(f"BBox: ({x1},{y1}) to ({x2},{y2}) - size: {x2-x1}x{y2-y1}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            result["id"],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    elif result["type"] == "keypointlabels":
        # Draw keypoint
        x_pct = result["value"]["x"]
        y_pct = result["value"]["y"]

        x_px = int((x_pct / 100) * w)
        y_px = int((y_pct / 100) * h)

        label = result["value"]["keypointlabels"][0]
        print(f"  {label}: ({x_px},{y_px})")
        cv2.circle(img, (x_px, y_px), 3, (0, 0, 255), -1)

# Save output
out_path = Path("D:/WORK/VT1/test_annotation_visual.jpg")
cv2.imwrite(str(out_path), img)
print(f"\nSaved visualization to: {out_path}")
print("Open this image to see if annotations align with players")
