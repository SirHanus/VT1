# python
# Minimal POC: YOLO pose + optional SAM2 segmentation on data_hockey.mp4 (optimized for speed/memory)
import argparse
from pathlib import Path
import sys
import time
import gc

from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Reuse SAM2 wrapper if available
try:
    from sam_general import SAM2VideoWrapper  # same folder
except Exception:
    SAM2VideoWrapper = None  # will gracefully disable SAM if import fails

# COCO keypoint skeleton pairs (17-keypoint format)
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
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
    ap = argparse.ArgumentParser("POC: YOLO pose + SAM2 segmentation on data_hockey.mp4 (optimized)")
    ap.add_argument("--source", type=str, default=str(root / "data_hockey.mp4"), help="Video source path")
    ap.add_argument("--pose-model", type=str, default=str(root / "hockeypose_1" / "yolo11n-pose.pt"),
                    help="Ultralytics YOLO pose model path/name")
    ap.add_argument("--sam2", type=str, default="facebook/sam2-hiera-large", help="HF SAM2 model id")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--out", type=str, default="", help="Output mp4 path (empty=auto name from params)")
    ap.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--no-sam", action="store_true", help="Disable SAM and only draw pose (faster)")

    # Performance/memory controls
    ap.add_argument("--half", action="store_true", help="Use FP16 for YOLO on CUDA")
    ap.add_argument("--sam-every", type=int, default=1, help="Run SAM every N frames (1=every frame)")
    ap.add_argument("--sam-topk", type=int, default=5, help="Limit SAM to top-K boxes per frame")
    ap.add_argument("--sam-reinit", type=int, default=0, help="Re-init SAM2 every N frames (0=never)")
    ap.add_argument("--empty-cache-interval", type=int, default=25, help="Call torch.cuda.empty_cache() every N frames on CUDA (0=never)")
    return ap.parse_args()


def _build_out_path(args) -> Path:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem or "video"
    stem = stem.replace(" ", "_")
    conf_pct = int(round(args.conf * 100))
    sam_enabled = not args.no_sam
    tokens = [stem, "pose"]
    if sam_enabled:
        tokens.append("sam")
    tokens.append(args.device)
    if args.device == "cuda" and args.half:
        tokens.append("fp16")
    tokens += [f"img{args.imgsz}", f"c{conf_pct}"]
    if sam_enabled:
        tokens += [f"se{max(1, int(args.sam_every))}", f"sk{max(0, int(args.sam_topk))}"]
        if args.sam_reinit and int(args.sam_reinit) > 0:
            tokens.append(f"sr{int(args.sam_reinit)}")
    if args.max_frames and int(args.max_frames) > 0:
        tokens.append(f"n{int(args.max_frames)}")
    filename = "_".join(tokens) + ".mp4"
    return out_dir / filename


def init_sam2(model_id: str, device: str, prefer_half: bool) -> Optional[object]:
    if SAM2VideoWrapper is None:
        print("[WARN] SAM2 wrapper not available. Proceeding without SAM.")
        return None
    try:
        dtype = torch.float16 if (device == "cuda" and prefer_half) else torch.float32
        return SAM2VideoWrapper(model_id=model_id, device=device, dtype=dtype)
    except Exception as e:
        print(f"[WARN] SAM2 init failed: {e}\nProceeding without SAM.")
        return None


def main():
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # cuDNN autotune for convs
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Open source video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {args.source}")
        return 1

    # Reduce capture buffering if supported (helps latency/memory on some backends)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:
        pass

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 30.0

    # Load YOLO pose
    try:
        pose_model = YOLO(args.pose_model)
        # Move to device and optionally use FP16
        if device == "cuda":
            pose_model.to("cuda")
            if args.half:
                # Half precision gives big speedups and memory savings on GPU
                pose_model.model.half()
    except Exception as e:
        print(f"[ERROR] Could not load pose model '{args.pose_model}': {e}")
        return 1

    # Init SAM2 (optional)
    sam2 = None
    if not args.no_sam:
        sam2 = init_sam2(args.sam2, device=device, prefer_half=(args.half and device == "cuda"))

    # Prepare writer
    out_path = Path(args.out) if args.out else _build_out_path(args)
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

    # Reset capture to frame 0 for full processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Progress bar setup respecting --max-frames
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else None
    effective_total = None
    if args.max_frames and args.max_frames > 0:
        effective_total = min(total_frames, args.max_frames) if total_frames is not None else args.max_frames
    else:
        effective_total = total_frames
    pbar = tqdm(total=effective_total, desc="Processing", unit="frame")

    win_name = "SAM2 + YOLO Pose (q to quit)"
    frame_idx = 0
    t0_global = time.perf_counter()

    # Helper to maybe reinit SAM2 to prevent state growth
    def maybe_reinit_sam2():
        nonlocal sam2
        if args.no_sam or args.sam_reinit <= 0 or sam2 is None:
            return
        if frame_idx > 0 and (frame_idx % args.sam_reinit == 0):
            # Drop old instance and GC to free memory
            try:
                del sam2
            except Exception:
                pass
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            sam2_local = init_sam2(args.sam2, device=device, prefer_half=(args.half and device == "cuda"))
            sam2 = sam2_local

    # Main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO pose inference
        with torch.inference_mode():
            # Use explicit device for clarity
            res = pose_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )[0]

        # Gather detections and keypoints
        boxes = []  # list of [x1,y1,x2,y2]
        kpts = None  # (N, K, 3)
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            # Limit SAM workload to top-K by confidence
            if hasattr(res.boxes, "conf") and res.boxes.conf is not None:
                conf = res.boxes.conf.detach().cpu().numpy()
                order = np.argsort(-conf)
                topk = min(len(order), max(0, int(args.sam_topk)))
                boxes = xyxy[order[:topk]].tolist()
            else:
                # Fallback: top-K by area
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                order = np.argsort(-areas)
                topk = min(len(order), max(0, int(args.sam_topk)))
                boxes = xyxy[order[:topk]].tolist()

        if hasattr(res, "keypoints") and res.keypoints is not None and len(res.keypoints) > 0:
            kpts = res.keypoints.data.detach().cpu().numpy()  # (N, K, 3)

        # Prepare visualization
        vis = frame  # draw in-place to avoid extra copies

        # SAM2 segmentation using YOLO boxes (throttled)
        if sam2 is not None and boxes and (frame_idx % max(1, args.sam_every) == 0):
            try:
                obj_ids = list(range(len(boxes)))
                sam2.add_box_prompts(vis, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes)
                masks_by_id = sam2.segment_frame(vis, frame_idx)
                # Overlay masks with minimal allocations
                for oid, mask in masks_by_id.items():
                    if mask is None:
                        continue
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.sum() == 0:
                        continue
                    color = color_for_id(int(oid))
                    # draw filled contours directly (avoids full-frame copy)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, color, thickness=cv2.FILLED)
                    cv2.drawContours(vis, contours, -1, color, thickness=2)
            except Exception as e:
                print(f"[WARN] SAM2 frame error at {frame_idx}: {e}")

        # Draw poses
        if kpts is not None and len(kpts) > 0:
            for i in range(kpts.shape[0]):
                pts = kpts[i]
                # skeleton
                for a, b in COCO_SKELETON:
                    if a < pts.shape[0] and b < pts.shape[0]:
                        xa, ya, sa = pts[a]
                        xb, yb, sb = pts[b]
                        if sa > 0.05 and sb > 0.05:
                            cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
                # joints
                for j in range(pts.shape[0]):
                    x, y, s = pts[j]
                    if s > 0.05:
                        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # HUD
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Write/display
        writer.write(vis)
        if args.show:
            cv2.imshow("SAM2 + YOLO Pose (q to quit)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

        # Cleanup to prevent growth
        del res
        if device == "cuda" and args.empty_cache_interval > 0 and (frame_idx % args.empty_cache_interval == 0):
            torch.cuda.empty_cache()
        if (frame_idx % 50) == 0:
            gc.collect()

        frame_idx += 1
        pbar.update(1)

        if args.max_frames and frame_idx >= args.max_frames:
            break

        # Maybe reinit SAM to drop any accumulated state
        maybe_reinit_sam2()

    pbar.close()
    writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    dt = time.perf_counter() - t0_global
    print(f"Saved: {out_path} | frames={frame_idx} | time={dt:.1f}s | fps~{frame_idx / max(1e-6, dt):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
