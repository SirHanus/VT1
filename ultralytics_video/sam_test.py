# yolo_sam2_demo.py
import argparse
import os
import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- SAM2 via Hugging Face (video) ---
# Requires: pip install transformers accelerate --upgrade
try:
    from transformers import Sam2VideoModel, Sam2VideoProcessor
except Exception as e:
    Sam2VideoModel = None
    Sam2VideoProcessor = None

# --------------------------
# Utility helpers
# --------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter + 1e-6
    return inter / denom

def nice_fps(dt_hist: deque) -> float:
    if not dt_hist:
        return 0.0
    avg_dt = sum(dt_hist) / len(dt_hist)
    return 1.0 / max(1e-6, avg_dt)

def color_for_id(idx: int) -> Tuple[int, int, int]:
    # Deterministic pleasant colors (BGR)
    palette = [
        (255, 99, 71),   # tomato
        (50, 205, 50),   # lime green
        (255, 215, 0),   # gold
        (65, 105, 225),  # royal blue
        (255, 105, 180), # hot pink
        (255, 140, 0),   # dark orange
        (0, 206, 209),   # dark turquoise
        (160, 32, 240),  # purple
        (0, 191, 255),   # deep sky blue
        (124, 252, 0),   # lawn green
    ]
    return palette[idx % len(palette)]

def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# --------------------------
# Simple tracker (IoU-based)
# --------------------------
class IoUTracker:
    def __init__(self, iou_thres=0.3, max_miss=30):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}  # id -> {bbox, miss, score, class}
        self.iou_thres = iou_thres
        self.max_miss = max_miss

    def update(self, detections: List[Tuple[List[float], int, float]]) -> Dict[int, dict]:
        """
        detections: list of (bbox[x1,y1,x2,y2], cls, conf)
        returns: dict id -> track record (with bbox, class, score)
        """
        assigned = set()
        det_idxs = list(range(len(detections)))
        # Try to match existing tracks
        for tid, tr in self.tracks.items():
            best_iou, best_j = 0.0, -1
            for j in det_idxs:
                if j in assigned:
                    continue
                bb, c, sc = detections[j]
                iou = iou_xyxy(tr["bbox"], bb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thres and best_j >= 0:
                bb, c, sc = detections[best_j]
                tr.update({"bbox": bb, "miss": 0, "score": sc, "class": c})
                assigned.add(best_j)
            else:
                tr["miss"] += 1

        # New tracks for unmatched detections
        for j in det_idxs:
            if j in assigned:
                continue
            bb, c, sc = detections[j]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": bb, "miss": 0, "score": sc, "class": c}

        # Remove stale tracks
        to_del = [tid for tid, tr in self.tracks.items() if tr["miss"] > self.max_miss]
        for tid in to_del:
            del self.tracks[tid]

        return self.tracks


# noinspection PyPackageRequirements
# python
class SAM2VideoWrapper:
    def __init__(self, model_id: str = "facebook/sam2-hiera-large",
                 device="cuda", dtype=None):
        if Sam2VideoModel is None or Sam2VideoProcessor is None:
            raise RuntimeError(
                "SAM2 (transformers) not available. Install with: pip install transformers accelerate --upgrade"
            )
        # Resolve device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
        # Prefer float32 on CPU to avoid potential unsupported half/bfloat16 ops
        if self.device == "cpu":
            self.model = self.model.to(dtype=torch.float32)
        self.model_dtype = next(self.model.parameters()).dtype

        # Align session dtype with model dtype; prefer float32 on CPU
        session_dtype = torch.float32 if self.device == "cpu" else self.model_dtype
        self.session = self.processor.init_video_session(
            inference_device=device,
            dtype=session_dtype,
        )
        self.frames_with_prompts = set()
        self.known_ids = set()
        # Cache last prompts per frame to allow session recovery on mismatch
        self._last_prompts: Dict[int, Tuple[List[int], List[List[float]], Tuple[int, int]]] = {}

    def add_box_prompt(self, frame_bgr: np.ndarray, frame_idx: int, obj_id: int, box_xyxy: List[float]):
        proc = self.processor(images=frame_bgr, return_tensors="pt")
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_boxes=[[[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])]]],
            original_size=proc.original_sizes[0],
            clear_old_inputs=True
        )
        self.frames_with_prompts.add(frame_idx)
        self.known_ids.add(obj_id)
        self._last_prompts[frame_idx] = ([obj_id], [box_xyxy], proc.original_sizes[0])

    def add_box_prompts(self, frame_bgr: np.ndarray, frame_idx: int, obj_ids: List[int], boxes_xyxy: List[List[float]]):
        """Batch-add box prompts for multiple objects on the same frame.
        This ensures all obj_ids are registered as having new inputs together.
        """
        if not obj_ids:
            return
        proc = self.processor(images=frame_bgr, return_tensors="pt")
        # input_boxes shape: [images=1][num_boxes][4]
        input_boxes = [[[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes_xyxy]]
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session,
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
            original_size=proc.original_sizes[0],
            clear_old_inputs=True,
        )
        self.frames_with_prompts.add(frame_idx)
        self.known_ids.update(obj_ids)
        self._last_prompts[frame_idx] = (obj_ids, boxes_xyxy, proc.original_sizes[0])

    def _reset_session(self):
        # Recreate a fresh inference session with aligned dtype and device
        session_dtype = torch.float32 if self.device == "cpu" else self.model_dtype
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            dtype=session_dtype,
        )

    def segment_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> Dict[int, np.ndarray]:
        with torch.inference_mode():
            inputs = self.processor(images=frame_bgr, return_tensors="pt")
            pixel = inputs["pixel_values"][0].to(device=self.device, dtype=self.model_dtype)
            try:
                out = self.model(
                    inference_session=self.session,
                    frame=pixel,
                    frame_idx=frame_idx,
                )
            except ValueError as e:
                # Recover from potential session/input desync by resetting the session and re-adding prompts
                if "maskmem_features in conditioning outputs cannot be empty" in str(e):
                    self._reset_session()
                    if frame_idx in self._last_prompts:
                        obj_ids, boxes, orig_size = self._last_prompts[frame_idx]
                        self.processor.add_inputs_to_inference_session(
                            inference_session=self.session,
                            frame_idx=frame_idx,
                            obj_ids=obj_ids,
                            input_boxes=[[[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]],
                            original_size=orig_size,
                            clear_old_inputs=True,
                        )
                    # retry once
                    out = self.model(
                        inference_session=self.session,
                        frame=pixel,
                        frame_idx=frame_idx,
                    )
                else:
                    raise
            masks = self.processor.post_process_masks(
                [out.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True
            )[0]
        obj_ids = list(self.session.obj_ids)
        result = {}
        for i, oid in enumerate(obj_ids):
            if i < masks.shape[0]:
                result[int(oid)] = masks[i].detach().cpu().numpy().astype(np.uint8)
        return result

# --------------------------
# Main pipeline
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser("YOLOv11n + SAM2 video segmentation demo")
    ap.add_argument("--source", type=str, default="0", help="webcam index or path to video file")
    ap.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Ultralytics YOLO model path/name")
    ap.add_argument("--sam2", type=str, default="facebook/sam2-hiera-large", help="SAM2 model id")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--width", type=int, default=0, help="Optional resize width (keep aspect)")
    ap.add_argument("--class-filter", type=str, default="", help="Comma-separated class ids to keep (empty=all)")
    ap.add_argument("--blur-bg", action="store_true", help="Blur background outside masks")
    return ap.parse_args()

def parse_source(src: str):
    if os.path.exists(src):
        return src
    try:
        return int(src)
    except ValueError:
        return src

def main():
    args = parse_args()
    # Video source
    cap_src = parse_source(args.source)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open source: {args.source}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 0 or src_fps > 240:
        src_fps = 30.0

    # Load YOLO
    try:
        yolo = YOLO(args.yolo_model)
    except Exception as e:
        print(f"[ERROR] Could not load YOLO model '{args.yolo_model}': {e}")
        return 1

    # Load SAM2
    try:
        sam2 = SAM2VideoWrapper(model_id=args.sam2, device=args.device, dtype=torch.bfloat16)
    except Exception as e:
        print(f"[ERROR] SAM2 init failed: {e}")
        print("Hint: pip install transformers accelerate --upgrade")
        return 1

    # Optional class filter
    class_filter = None
    if args.class_filter.strip():
        class_filter = set(int(x) for x in args.class_filter.split(",") if x.strip().isdigit())

    tracker = IoUTracker(iou_thres=0.35, max_miss=30)

    # Metrics
    dt_hist = deque(maxlen=60)
    frame_idx = 0
    window = "YOLOv11n + SAM2 (q: quit)"

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        # Optional resize for speed/consistency
        if args.width and frame.shape[1] != args.width:
            new_w = args.width
            new_h = int(frame.shape[0] * (new_w / frame.shape[1]))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]

        # YOLO detect
        with torch.inference_mode():
            res = yolo.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device if args.device != "cuda" else None,
                verbose=False
            )[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            # Ultralytics boxes.xyxy, boxes.cls, boxes.conf
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            conf = res.boxes.conf.detach().cpu().numpy()
            for i in range(len(xyxy)):
                if class_filter is not None and cls[i] not in class_filter:
                    continue
                x1, y1, x2, y2 = xyxy[i].tolist()
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                detections.append(([x1, y1, x2, y2], int(cls[i]), float(conf[i])))

        # Update tracker & get stable IDs
        tracks = tracker.update(detections)

        # Introduce NEW/UPDATED objects to SAM2 for this frame
        # We only need to add prompts when a track is new or when we want to re-anchor it.
        # Submit all boxes in a single call so all IDs are treated as having new inputs on this frame
        if tracks:
            obj_ids = list(tracks.keys())
            boxes = [tracks[tid]["bbox"] for tid in obj_ids]
            sam2.add_box_prompts(frame, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes)

        # If there are no tracks and no known objects yet, skip SAM2 to avoid errors
        if not tracks and len(sam2.known_ids) == 0:
            # HUD metrics and display only
            dt = time.perf_counter() - t0
            dt_hist.append(dt)
            fps = nice_fps(dt_hist)
            vis = frame.copy()
            draw_text(vis, f"FPS: {fps:.1f}", 10, 24)
            draw_text(vis, f"Frames: {frame_idx}", 10, 46)
            draw_text(vis, f"Tracks: 0 (SAM2 IDs: 0)", 10, 68)
            draw_text(vis, f"Res: {W}x{H}", 10, 90)
            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_idx += 1
            continue

        # SAM2 segmentation for current frame
        masks_by_id = sam2.segment_frame(frame, frame_idx)

        # Visualization
        vis = frame.copy()
        overlay = vis.copy()
        combined_foreground_mask = np.zeros((H, W), dtype=np.uint8)

        for tid, mask in masks_by_id.items():
            if mask.sum() == 0:
                continue
            color = color_for_id(tid)
            # fill
            overlay[mask > 0] = color
            combined_foreground_mask = np.maximum(combined_foreground_mask, mask)

            # outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, thickness=2)

            # label
            # put label near top-left of bbox (from tracker if present)
            tr = tracks.get(tid)
            if tr:
                x1, y1, x2, y2 = tr["bbox"]
                draw_text(vis, f"ID {tid}", int(x1), max(20, int(y1) - 6), scale=0.6, color=(255, 255, 255))

        # blend colored masks
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # optional background blur (outside all masks)
        if args.blur_bg and combined_foreground_mask.any():
            blurred = cv2.GaussianBlur(frame, (25, 25), 0)
            bg_mask = (combined_foreground_mask == 0)
            vis[bg_mask] = blurred[bg_mask]

        # HUD metrics
        dt = time.perf_counter() - t0
        dt_hist.append(dt)
        fps = nice_fps(dt_hist)
        draw_text(vis, f"FPS: {fps:.1f}", 10, 24)
        draw_text(vis, f"Frames: {frame_idx}", 10, 46)
        draw_text(vis, f"Tracks: {len(tracks)} (SAM2 IDs: {len(masks_by_id)})", 10, 68)
        draw_text(vis, f"Res: {W}x{H}", 10, 90)

        cv2.imshow(window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
