# python
"""
Build training set for Unsupervised Team Clustering using:
- 1 FPS frame sampling from one or more videos
- Player detection via RF-DETR-S (MMDetection). Optional YOLO fallback.
- Central-crop of each detection to reduce noise
- SigLIP Vision embeddings for each crop
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Transformers (SigLIP)
from transformers import AutoImageProcessor


@dataclass
class DetResult:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    label: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser("Build dataset: 1 FPS -> RF-DETR-S -> central crop -> SigLIP embeddings")

    # Inputs
    ap.add_argument("--videos-dir", type=str, default=str(root / "videos_all" / "CAR_vs_NYR"),
                    help="Directory with input videos (*.mp4)")
    ap.add_argument("--glob", type=str, default="*.mp4", help="Glob to match videos inside --videos-dir")
    ap.add_argument("--videos", type=str, nargs="*", default=None,
                    help="Explicit list of video files (overrides --videos-dir/--glob)")

    # Detection (MMDetection RF-DETR-S)
    ap.add_argument("--det-config", type=str, default="",
                    help="MMDetection config path for RF-DETR-S")
    ap.add_argument("--det-weights", type=str, default="",
                    help="Checkpoint .pth for RF-DETR-S")
    ap.add_argument("--det-score-thr", type=float, default=0.30, help="Score threshold for detections")
    ap.add_argument("--person-class-name", type=str, default="person",
                    help="Class name to treat as player/person in the detector's classes")

    # YOLO fallback
    ap.add_argument("--yolo-fallback", action="store_true", help="Enable YOLO fallback if MMDetection is unavailable")
    ap.add_argument("--yolo-model", type=str, default=str(root / "yolo11n.pt"),
                    help="Ultralytics YOLO model path/name for fallback")

    # Sampling
    ap.add_argument("--fps", type=float, default=1.0, help="Target frames per second to sample (default 1 FPS)")
    ap.add_argument("--max-seconds", type=int, default=0, help="Limit seconds per video (0=all)")

    # Central crop
    ap.add_argument("--central-ratio", type=float, default=0.6,
                    help="Fraction of bbox width/height to keep around center (0<r<=1)")
    ap.add_argument("--min-crop-size", type=int, default=32, help="Discard crops smaller than this (pixels)")

    # SigLIP
    ap.add_argument("--siglip", type=str, default="google/siglip-base-patch16-224",
                    help="SigLIP Vision model with projection (HF id)")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")

    # Device
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")

    # Outputs
    ap.add_argument("--out-dir", type=str, default=str(root / "videos_all_processed" / "training_set"),
                    help="Output directory root")
    ap.add_argument("--save-crops", action="store_true", help="Save crop images to disk")

    # Misc
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def time_len_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0.1 or fps > 240.0:
        fps = 30.0
    if frames <= 0:
        # fallback: try CAP_PROP_POS_MSEC after reading
        return 0.0
    return frames / fps


def iter_sampled_frames(video_path: Path, fps_target: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_sec = time_len_seconds(cap)
    if total_sec <= 0:
        # fallback to step-by-step read with fps estimate
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if src_fps <= 0.1 or src_fps > 240.0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / max(1e-6, fps_target))))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx % step) == 0:
                sec = idx / src_fps
                yield int(idx), float(sec), frame
            idx += 1
        cap.release()
        return

    sec = 0.0
    # Cap to end
    while sec <= total_sec + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        # Retrieve actual frame index after seek
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        yield frame_idx, float(sec), frame
        sec += 1.0 / max(1e-6, fps_target)
    cap.release()


# ---------------- Detection wrappers ----------------
class RFDETRDetector:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda", person_class_name: str = "person"):
        # Lazy import via importlib to avoid static import errors
        import importlib
        try:
            mmdet_apis = importlib.import_module("mmdet.apis")
            init_detector = getattr(mmdet_apis, "init_detector")
        except Exception as e:
            raise RuntimeError("MMDetection not available. Install mmdet and mmcv or use --yolo-fallback.") from e
        if not (config and checkpoint):
            raise ValueError("--det-config and --det-weights are required for RF-DETR-S")
        self.model = init_detector(config, checkpoint, device=device)
        # Attempt to get class mapping
        self.classes = None
        try:
            self.classes = tuple(self.model.dataset_meta.get('classes', ()))
        except Exception:
            self.classes = None
        if self.classes and person_class_name in self.classes:
            self.person_id = self.classes.index(person_class_name)
        else:
            # Common COCO fallback: person id 0
            self.person_id = 0

    def detect(self, img: np.ndarray, score_thr: float = 0.3) -> List[DetResult]:
        import importlib
        try:
            mmdet_apis = importlib.import_module("mmdet.apis")
            inference_detector = getattr(mmdet_apis, "inference_detector")
        except Exception as e:
            raise RuntimeError("MMDetection not available at inference time.") from e
        data_sample = inference_detector(self.model, img)
        # mmdet>=3 returns DetDataSample, extract pred_instances
        try:
            pred = data_sample.pred_instances
            bboxes = pred.bboxes.tensor.detach().cpu().numpy() if hasattr(pred.bboxes, 'tensor') else pred.bboxes.detach().cpu().numpy()
            scores = pred.scores.detach().cpu().numpy()
            labels = pred.labels.detach().cpu().numpy()
        except Exception:
            # Older format or unexpected; try generic
            try:
                result = data_sample  # could be tuple
                if isinstance(result, (list, tuple)):
                    result = result[0]
                bboxes = result[:, :4]
                scores = result[:, 4]
                labels = result[:, 5].astype(int)
            except Exception as e:
                raise RuntimeError(f"Unsupported detector output: {type(data_sample)} | {e}")

        out: List[DetResult] = []
        for bb, sc, lb in zip(bboxes, scores, labels):
            if sc < score_thr:
                continue
            if int(lb) != int(self.person_id):
                continue
            x1, y1, x2, y2 = map(float, bb[:4])
            # Skip invalid boxes
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            out.append(DetResult((x1, y1, x2, y2), float(sc), int(lb)))
        return out


class YOLOFallbackDetector:
    def __init__(self, model_name: str, device: str = "cuda"):
        # Lazy import ultralytics
        try:
            from ultralytics import YOLO as YOLO_U
        except Exception as e:
            raise RuntimeError("Ultralytics not available.") from e
        self.model = YOLO_U(model_name)
        self.device = device
        if device == "cuda":
            self.model.to("cuda")

    def detect(self, img: np.ndarray, score_thr: float = 0.3) -> List[DetResult]:
        res = self.model.predict(img, imgsz=640, conf=score_thr, device=self.device, verbose=False)[0]
        out: List[DetResult] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        conf = res.boxes.conf.detach().cpu().numpy()
        cls = res.boxes.cls.detach().cpu().numpy().astype(int)
        # COCO person id is 0 in YOLO models
        for bb, sc, lb in zip(xyxy, conf, cls):
            if sc < score_thr:
                continue
            if int(lb) != 0:
                continue
            out.append(DetResult((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), float(sc), int(lb)))
        return out


# ---------------- Cropping and embeddings ----------------

def central_crop_from_bbox(img: np.ndarray, bbox: Tuple[float, float, float, float], ratio: float) -> np.ndarray | None:
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


class SigLIPEmbedder:
    def __init__(self, model_id: str, device: str = "cuda"):
        # Use SiglipVisionModel and take pooled_output as embedding
        from transformers.models.siglip import SiglipVisionModel  # type: ignore
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id)
        self.device = device
        if device == "cuda":
            self.model.to("cuda")
        self.model.eval()

    @torch.inference_mode()
    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        # Convert to PIL RGB
        imgs: List[Image.Image] = []
        for c in crops:
            if c is None or c.size == 0:
                continue
            if c.ndim == 2:
                c = np.stack([c, c, c], axis=-1)
            if c.shape[2] == 4:
                c = c[:, :, :3]
            # BGR -> RGB
            c = c[:, :, ::-1]
            imgs.append(Image.fromarray(c))
        inputs = self.processor(images=imgs, return_tensors="pt")
        px = inputs["pixel_values"].to(self.device)
        out = self.model(pixel_values=px)
        # Prefer pooled_output; fallback to mean of last_hidden_state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            x = out.last_hidden_state  # [B, N, C]
            emb = x.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1)
        return emb.detach().cpu().numpy().astype(np.float32)


# ---------------- Main pipeline ----------------

def build_for_video(video_path: Path,
                    detector,
                    embedder: SigLIPEmbedder,
                    args: argparse.Namespace,
                    out_root: Path) -> Dict[str, Any]:
    video_stem = video_path.stem
    out_dir = out_root / video_stem
    ensure_dir(out_dir)
    crops_dir = out_dir / "crops"
    if args.save_crops:
        ensure_dir(crops_dir)

    index_rows: List[List[Any]] = []
    embeddings: List[np.ndarray] = []

    seconds_limit = args.max_seconds if args.max_seconds and args.max_seconds > 0 else None
    pbar = tqdm(desc=f"{video_stem}", unit="sec", total=seconds_limit)

    sec_count = 0
    batch_crops: List[np.ndarray] = []
    batch_meta: List[Tuple[int, float, Tuple[int, int, int, int], float, int]] = []  # frame_idx, sec, bbox_ints, score, local_id
    next_crop_id = 0

    for frame_idx, sec, frame in iter_sampled_frames(video_path, args.fps):
        if seconds_limit is not None and sec > seconds_limit + 1e-6:
            break

        # Detect players
        try:
            dets: List[DetResult] = detector.detect(frame, score_thr=args.det_score_thr)
        except Exception as e:
            raise RuntimeError(f"Detection failed on {video_path} @ {sec:.2f}s: {e}")

        # Prepare crops for this second
        for det_i, det in enumerate(dets):
            crop = central_crop_from_bbox(frame, det.bbox, args.central_ratio)
            if crop is None:
                continue
            ch, cw = crop.shape[:2]
            if ch < args.min_crop_size or cw < args.min_crop_size:
                continue
            # Stash for batch embedding
            batch_crops.append(crop)
            bx = (int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3]))
            batch_meta.append((int(frame_idx), float(sec), bx, float(det.score), int(det.label)))

            if len(batch_crops) >= args.batch:
                embs = embedder.embed_batch(batch_crops)
                embeddings.append(embs)
                # Save crops and index rows
                for k, meta in enumerate(batch_meta):
                    fidx, s, bb, sc, lb = meta
                    crop_id = next_crop_id
                    next_crop_id += 1
                    crop_name = f"{fidx:07d}_{crop_id:05d}.jpg"
                    crop_rel = None
                    if args.save_crops:
                        crop_path = crops_dir / crop_name
                        # OpenCV expects BGR arrays; batch_crops[k] is BGR already
                        c = batch_crops[k]
                        cv2.imwrite(str(crop_path), c)
                        crop_rel = str(Path("crops") / crop_name)
                    index_rows.append([len(index_rows), fidx, f"{s:.3f}", *bb, f"{sc:.4f}", lb, crop_rel if crop_rel else ""])
                batch_crops.clear()
                batch_meta.clear()

        sec_count += 1
        pbar.update(1)

    # Flush any remaining
    if batch_crops:
        embs = embedder.embed_batch(batch_crops)
        embeddings.append(embs)
        for k, meta in enumerate(batch_meta):
            fidx, s, bb, sc, lb = meta
            crop_id = next_crop_id
            next_crop_id += 1
            crop_name = f"{fidx:07d}_{crop_id:05d}.jpg"
            crop_rel = None
            if args.save_crops:
                crop_path = crops_dir / crop_name
                # OpenCV expects BGR arrays; batch_crops[k] is BGR already
                cv2.imwrite(str(crop_path), batch_crops[k])
                crop_rel = str(Path("crops") / crop_name)
            index_rows.append([len(index_rows), fidx, f"{s:.3f}", *bb, f"{sc:.4f}", lb, crop_rel if crop_rel else ""])
        batch_crops.clear()
        batch_meta.clear()

    pbar.close()

    # Concatenate and save
    if embeddings:
        E = np.concatenate(embeddings, axis=0)
    else:
        E = np.zeros((0, 1024), dtype=np.float32)  # unknown dim; will be replaced if empty

    # If we had no crops, ensure correct dim by probing model config
    if E.shape[0] == 0:
        try:
            # small probe using a 224x224 dummy
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            probe = embedder.embed_batch([dummy])
            E = np.zeros((0, int(probe.shape[1])), dtype=np.float32)
        except Exception:
            pass

    np.save(out_dir / "embeddings.npy", E)

    # Write index.csv
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "frame_idx", "time_s", "x1", "y1", "x2", "y2", "score", "label", "crop_relpath"])
        writer.writerows(index_rows)

    return {
        "video": str(video_path),
        "stem": video_stem,
        "seconds_processed": int(sec_count),
        "num_crops": int(len(index_rows)),
        "embeddings_shape": list(E.shape),
    }


def main() -> int:
    args = parse_args()

    # Seed and device
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Discover videos
    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        videos = sorted(Path(args.videos_dir).glob(args.glob))
    videos = [v for v in videos if v.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if not videos:
        print(f"[ERROR] No videos found. Checked: --videos {args.videos} or {args.videos_dir} / {args.glob}")
        return 1

    # Init detector: try RF-DETR if provided; otherwise YOLO fallback if enabled
    detector = None
    det_name = None
    if args.det_config and args.det_weights:
        try:
            detector = RFDETRDetector(args.det_config, args.det_weights, device=device, person_class_name=args.person_class_name)
            det_name = "rf-detr-s"
        except Exception as e:
            print(f"[WARN] RF-DETR init failed: {e}")
    if detector is None and args.yolo_fallback:
        try:
            detector = YOLOFallbackDetector(args.yolo_model, device=device)
            det_name = "yolo-fallback"
        except Exception as e:
            print(f"[WARN] YOLO fallback init failed: {e}")
    if detector is None:
        print("[ERROR] No detector available. Provide --det-config/--det-weights for RF-DETR-S or enable --yolo-fallback.")
        return 1

    # Init embedder
    embedder = SigLIPEmbedder(args.siglip, device=device)

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    manifest_path = out_root / "_manifest.json"
    manifest: Dict[str, Any] = {
        "detector": det_name,
        "siglip": args.siglip,
        "central_ratio": float(args.central_ratio),
        "det_score_thr": float(args.det_score_thr),
        "videos": [],
    }

    for v in videos:
        try:
            info = build_for_video(v, detector, embedder, args, out_root)
        except Exception as e:
            print(f"[WARN] Failed on {v}: {e}")
            continue
        manifest["videos"].append(info)
        # Save/append manifest incrementally
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Output in: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
