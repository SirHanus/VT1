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
import json
from datetime import datetime

# Reuse SAM2 wrapper if available
from sam_general import SAM2VideoWrapper  # same folder

# --- Team clustering inference additions ---
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    from transformers import AutoImageProcessor
    from transformers.models.siglip import SiglipVisionModel  # type: ignore
    _HAS_SIGLIP = True
except Exception:
    _HAS_SIGLIP = False


def central_crop_from_bbox(img: np.ndarray, bbox: Tuple[float, float, float, float], ratio: float = 0.6) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    r = float(max(1e-3, min(1.0, ratio)))
    nw = bw * r
    nh = bh * r
    nx1 = int(round(cx - nw * 0.5))
    ny1 = int(round(cy - nh * 0.5))
    nx2 = int(round(cx + nw * 0.5))
    ny2 = int(round(cy + nh * 0.5))
    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(0, min(w, nx2))
    ny2 = max(0, min(h, ny2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    crop = img[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    return crop


class TeamClusteringInfer:
    def __init__(self, models_dir: Path, siglip_id: str, device: str = "cuda"):
        if not _HAS_JOBLIB:
            raise RuntimeError("joblib not available to load models")
        umap_p = Path(models_dir) / "umap.pkl"
        kmeans_p = Path(models_dir) / "kmeans.pkl"
        if not umap_p.exists() or not kmeans_p.exists():
            raise FileNotFoundError(f"Expected umap.pkl and kmeans.pkl in {models_dir}")
        self.reducer = joblib.load(umap_p)
        self.kmeans = joblib.load(kmeans_p)
        if not _HAS_SIGLIP:
            raise RuntimeError("transformers SigLIP not available")
        self.processor = AutoImageProcessor.from_pretrained(siglip_id)
        self.model = SiglipVisionModel.from_pretrained(siglip_id)
        self.device = device
        if device == "cuda":
            self.model.to("cuda")
        self.model.eval()

    @torch.inference_mode()
    def predict_labels(self, frame: np.ndarray, boxes_xyxy: List[List[float]], central_ratio: float) -> List[int]:
        crops = []
        idxs = []
        for i, bb in enumerate(boxes_xyxy):
            crop = central_crop_from_bbox(frame, (bb[0], bb[1], bb[2], bb[3]), central_ratio)
            if crop is None:
                continue
            if crop.ndim == 2:
                crop = np.stack([crop, crop, crop], axis=-1)
            if crop.shape[2] == 4:
                crop = crop[:, :, :3]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            idxs.append(i)
        if not crops:
            return [-1] * len(boxes_xyxy)
        inputs = self.processor(images=[cv2.cvtColor(c, cv2.COLOR_RGB2BGR)[..., ::-1] for c in crops], return_tensors="pt")
        # Note: crops are already RGB above, converting back/forth to be robust to processor expectations
        px = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixel_values=px)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            x = out.last_hidden_state
            emb = x.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1).cpu().numpy()
        Z = self.reducer.transform(emb)
        labels = self.kmeans.predict(Z).astype(int).tolist()
        # Map back to full list
        full = [-1] * len(boxes_xyxy)
        for j, i in enumerate(idxs):
            full[i] = labels[j]
        return full


# COCO keypoint skeleton pairs (17-keypoint format)
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]

# Two team colors default; fallback colors if k>2
TEAM_COLORS = [
    (255, 0, 0),   # Team 0 - Blue-ish (BGR)
    (0, 165, 255), # Team 1 - Orange-ish (BGR)
    (50, 205, 50), (255, 105, 180), (255, 215, 0)
]


def color_for_team(label: int) -> Tuple[int, int, int]:
    if label is None or label < 0:
        return (200, 200, 200)
    return TEAM_COLORS[label % len(TEAM_COLORS)]


def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser("POC: YOLO pose + SAM2 segmentation on data_hockey.mp4 (optimized)")
    ap.add_argument("--source", type=str, default=str(root / "data_hockey.mp4"), help="Video source path")
    ap.add_argument("--pose-model", type=str, default=str(root / "hockeypose_1" / "yolo11x-pose.pt"),
                    help="Ultralytics YOLO pose model path/name")
    ap.add_argument("--sam2", type=str, default="facebook/sam2-hiera-large", help="HF SAM2 model id")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    ap.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    ap.add_argument("--show", action="store_true", help="Show a live window")
    ap.add_argument("--no-sam", action="store_true", help="Disable SAM and only draw pose (faster)")

    ap.add_argument("--half", action="store_true", help="Use FP16 for YOLO on CUDA")
    ap.add_argument("--sam-every", type=int, default=1, help="Run SAM every N frames (1=every frame)")
    ap.add_argument("--sam-topk", type=int, default=10, help="Limit SAM to top-K boxes per frame")
    ap.add_argument("--sam-reinit", type=int, default=60, help="Re-init SAM2 every N frames (0=never)")
    ap.add_argument("--empty-cache-interval", type=int, default=25, help="Call torch.cuda.empty_cache() every N frames on CUDA (0=never)")

    ap.add_argument("--metrics-json", type=str, default="", help="If set, write per-run metrics JSON to this path")

    ap.add_argument("--team-models", type=str, default=str(root / "offline_pipeline" / "team_clustering" / "clustering"),
                    help="Directory containing umap.pkl and kmeans.pkl")
    ap.add_argument("--siglip", type=str, default="google/siglip-base-patch16-224", help="SigLIP model id (vision)")
    ap.add_argument("--central-ratio", type=float, default=0.6, help="Central crop ratio for team inference")
    ap.add_argument("--disable-team", action="store_true", help="Disable team coloring even if models are present")
    return ap.parse_args()


def _build_out_path(args) -> Path:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem or "video"
    stem = stem.replace(" ", "_")
    conf_pct = int(round(args.conf * 100))
    sam_enabled = not args.no_sam
    # Add date as first token
    date_str = datetime.now().strftime("%Y%m%d")
    tokens = [date_str, stem, "pose"]
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
    if not args.disable_team:
        tokens.append("teams")
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

    # Team clustering inference init
    team_infer: Optional[TeamClusteringInfer] = None
    if not args.disable_team:
        try:
            team_infer = TeamClusteringInfer(models_dir=Path(args.team_models), siglip_id=args.siglip, device=device)
        except Exception as e:
            print(f"[WARN] Team clustering inference disabled: {e}")
            team_infer = None

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
    # Always auto-generate output path now (previously respected --out)
    out_path = _build_out_path(args)
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
    sam_frames = 0  # count frames where SAM actually ran

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


        #

        # YOLO pose inference
        with torch.inference_mode():

            res = pose_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )[0]

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
                conf = conf[order[:topk]]
            else:
                # Fallback: top-K by area
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                order = np.argsort(-areas)
                topk = min(len(order), max(0, int(args.sam_topk)))
                boxes = xyxy[order[:topk]].tolist()
                conf = None
        else:
            conf = None

        if hasattr(res, "keypoints") and res.keypoints is not None and len(res.keypoints) > 0:
            kpts = res.keypoints.data.detach().cpu().numpy()  # (N, K, 3)

        # Determine team labels for current boxes (if models loaded)
        team_labels = None
        if team_infer is not None and boxes:
            try:
                team_labels = team_infer.predict_labels(frame, boxes, central_ratio=float(args.central_ratio))
            except Exception as e:
                print(f"[WARN] Team inference error at frame {frame_idx}: {e}")
                team_labels = None

        # Prepare visualization
        vis = frame  # draw in-place to avoid extra copies

        # SAM2 segmentation using YOLO boxes (throttled)
        if sam2 is not None and boxes and (frame_idx % max(1, args.sam_every) == 0):
            try:
                obj_ids = list(range(len(boxes)))
                sam2.add_box_prompts(vis, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes)
                masks_by_id = sam2.segment_frame(vis, frame_idx)
                # Overlay masks with transparency
                overlay = vis.copy()
                for oid, mask in masks_by_id.items():
                    if mask is None:
                        continue
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.sum() == 0:
                        continue
                    color = color_for_team(team_labels[oid] if (team_labels and oid < len(team_labels)) else -1)
                    # Fill the mask area with team color on overlay
                    overlay[mask > 0] = color
                    # Draw contour outline
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, color, thickness=2)
                # Blend overlay with original frame (alpha=0.4 means 40% color, 60% original)
                alpha = 0.25
                cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
                sam_frames += 1
            except Exception as e:
                print(f"[WARN] SAM2 frame error at {frame_idx}: {e}")

        # Draw poses with team color if available
        if kpts is not None and len(kpts) > 0:
            # Try to align pose order to boxes via confidence sorting/indices used above
            for i in range(kpts.shape[0]):
                pts = kpts[i]
                tcol = color_for_team(team_labels[i] if (team_labels and i < len(team_labels)) else -1)
                # skeleton
                for a, b in COCO_SKELETON:
                    if a < pts.shape[0] and b < pts.shape[0]:
                        xa, ya, sa = pts[a]
                        xb, yb, sb = pts[b]
                        if sa > 0.05 and sb > 0.05:
                            cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), tcol, 2)
                # joints
                for j in range(pts.shape[0]):
                    x, y, s = pts[j]
                    if s > 0.05:
                        cv2.circle(vis, (int(x), int(y)), 3, tcol, -1, lineType=cv2.LINE_AA)

        # HUD
        hud = f"Frame: {frame_idx}"
        if team_infer is not None:
            hud += " | teams:on"
        cv2.putText(vis, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

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

    # Build per-run metrics and optionally write JSON
    try:
        metrics = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "status": "ok",
            "source": str(args.source),
            "out": str(out_path),  # Auto-generated
            "device": device,
            "half": bool(args.half and device == "cuda"),
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "sam_enabled": bool(not args.no_sam and (sam2 is not None)),
            "sam_model": (str(args.sam2) if not args.no_sam else None),
            "sam_every": int(args.sam_every),
            "sam_topk": int(args.sam_topk),
            "sam_reinit": int(args.sam_reinit),
            "empty_cache_interval": int(args.empty_cache_interval),
            "pose_model": str(args.pose_model),
            "src_fps": float(src_fps),
            "video_size": [int(W), int(H)],
            "frames_processed": int(frame_idx),
            "sam_frames": int(sam_frames),
            "time_sec": float(dt),
            "avg_fps": (float(frame_idx) / max(1e-6, float(dt))),
            "teams": bool(team_infer is not None),
        }
        if args.metrics_json:
            mpath = Path(args.metrics_json)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            with mpath.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write metrics JSON: {e}")

    print(f"Saved: {out_path} | frames={frame_idx} | time={dt:.1f}s | fps~{frame_idx / max(1e-6, dt):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
