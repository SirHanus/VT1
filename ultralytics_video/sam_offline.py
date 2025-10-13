# Minimal POC: YOLO pose + optional SAM2 segmentation on data_hockey.mp4
import argparse
from pathlib import Path
import sys
import time
from typing import List, Tuple

import cv2
import torch
from ultralytics import YOLO

# Reuse SAM2 wrapper if available
try:
    from sam_general import SAM2VideoWrapper  # same folder
except Exception:
    SAM2VideoWrapper = None  # will gracefully disable SAM if import fails

# COCO keypoint skeleton pairs (17-keypoint format)
# (from COCO: https://cocodataset.org/#keypoints-2018)
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 3), (0, 2), (2, 4),      # head/eyes-ears (varies by training, still fine for viz)
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 11), (6, 12)  # torso connects
]

COLORS = [
    (255, 99, 71), (50, 205, 50), (255, 215, 0), (65, 105, 225), (255, 105, 180),
    (255, 140, 0), (0, 206, 209), (160, 32, 240), (0, 191, 255), (124, 252, 0)
]

def color_for_id(idx: int) -> Tuple[int, int, int]:
    return COLORS[idx % len(COLORS)]

def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser("POC: YOLO pose + SAM2 segmentation on data_hockey.mp4")
    ap.add_argument("--source", type=str, default=str(root / "data_hockey.mp4"), help="Video source path")
    ap.add_argument("--pose-model", type=str, default=str(root / "hockeypose_1" / "yolo11n-pose.pt"),
                    help="Ultralytics YOLO pose model path/name")
    ap.add_argument("--sam2", type=str, default="facebook/sam2-hiera-large", help="HF SAM2 model id")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "data_hockey_sam_pose.mp4"),
                    help="Output mp4 path")
    ap.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--no-sam", action="store_true", help="Disable SAM and only draw pose (faster)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Open source video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {args.source}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 30.0

    # Load YOLO pose
    try:
        pose_model = YOLO(args.pose_model)
    except Exception as e:
        print(f"[ERROR] Could not load pose model '{args.pose_model}': {e}")
        return 1

    # Init SAM2 (optional)
    sam2 = None
    if not args.no_sam:
        if SAM2VideoWrapper is None:
            print("[WARN] SAM2 wrapper not available (transformers missing?). Proceeding without SAM.")
        else:
            try:
                sam2 = SAM2VideoWrapper(model_id=args.sam2, device=device, dtype=torch.bfloat16)
            except Exception as e:
                print(f"[WARN] SAM2 init failed: {e}\nProceeding without SAM.")
                sam2 = None

    # Prepare writer
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe first frame to get size
    ok, first = cap.read()
    if not ok:
        print("[ERROR] Could not read first frame")
        return 1
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (W, H))
    if not writer.isOpened():
        print(f"[ERROR] Could not open writer at: {out_path}")
        return 1

    # Reset capture to frame 0 for full processing, write first frame later
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win_name = "SAM2 + YOLO Pose (q to quit)"
    frame_idx = 0
    t0_global = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO pose inference
        with torch.inference_mode():
            res = pose_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=None if device == "cuda" else device,
                verbose=False,
            )[0]

        # Gather detections and keypoints
        boxes = []  # list of [x1,y1,x2,y2]
        kpts = None  # (N, K, 3)
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            boxes = xyxy.tolist()
        if hasattr(res, "keypoints") and res.keypoints is not None and len(res.keypoints) > 0:
            kpts = res.keypoints.data.detach().cpu().numpy()  # (N, K, 3)

        # Prepare visualization layers
        vis = frame.copy()

        # SAM2 segmentation per frame using YOLO boxes
        if sam2 is not None and boxes:
            try:
                obj_ids = list(range(len(boxes)))
                sam2.add_box_prompts(vis, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes)
                masks_by_id = sam2.segment_frame(vis, frame_idx)
                overlay = vis.copy()
                for oid, mask in masks_by_id.items():
                    if mask is None or mask.sum() == 0:
                        continue
                    color = color_for_id(int(oid))
                    overlay[mask > 0] = color
                    # draw outline
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, color, thickness=2)
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
            except Exception as e:
                # Non-fatal: continue without SAM for this frame
                print(f"[WARN] SAM2 frame error at {frame_idx}: {e}")

        # Draw poses
        if kpts is not None and len(kpts) > 0:
            for i in range(kpts.shape[0]):
                pts = kpts[i]
                # draw skeleton
                for a, b in COCO_SKELETON:
                    if a < pts.shape[0] and b < pts.shape[0]:
                        xa, ya, sa = pts[a]
                        xb, yb, sb = pts[b]
                        if sa > 0.05 and sb > 0.05:
                            cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
                # draw joints
                for j in range(pts.shape[0]):
                    x, y, s = pts[j]
                    if s > 0.05:
                        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # HUD
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)
        if args.show:
            cv2.imshow(win_name, vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    dt = time.perf_counter() - t0_global
    print(f"Saved: {out_path} | frames={frame_idx} | time={dt:.1f}s | fps~{frame_idx / max(1e-6, dt):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
