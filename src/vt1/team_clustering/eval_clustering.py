# python
r"""
Evaluate team clustering models on images or sampled video frames.
- Detect players with YOLO (person class)
- Central-crop each detection
- Compute SigLIP embeddings, reduce with UMAP, predict KMeans label
- Save annotated images and a JSON summary with quick metrics

Usage examples (Windows cmd):

  # Evaluate on a folder of images
  python -m vt1.team_clustering.eval_clustering ^
    --images-dir some\images ^
    --team-models outputs\team_clustering ^
    --yolo-model models\yolo11n.pt --conf 0.3

  # Evaluate by sampling frames from a video (every 30 frames)
  python -m vt1.team_clustering.eval_clustering ^
    --video D:\WORK\VT1\data_hockey.mp4 ^
    --frame-step 30 ^
    --max-frames 200 ^
    --team-models outputs\team_clustering
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch

# External deps (installed in the project):
from ultralytics import YOLO

from vt1.config import settings

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

# Colors (BGR)
TEAM_COLORS = [
    (255, 0, 0),  # team 0 - blue-ish
    (0, 165, 255),  # team 1 - orange-ish
    (50, 205, 50),  # team 2 - green
    (255, 105, 180),  # team 3 - pink
    (255, 215, 0),  # team 4 - gold
]


def color_for_team(label: int) -> Tuple[int, int, int]:
    if label is None or label < 0:
        return (200, 200, 200)
    return TEAM_COLORS[label % len(TEAM_COLORS)]


def parse_args() -> argparse.Namespace:
    cfg = settings()
    root = cfg.repo_root
    local_base = cfg.team_output_dir
    ap = argparse.ArgumentParser("Evaluate team clustering models")
    src = ap.add_argument_group("Sources")
    src.add_argument(
        "--images-dir", type=str, default="", help="Directory of test images"
    )
    src.add_argument(
        "--glob",
        type=str,
        default="*.jpg;*.png;*.jpeg;*.JPG;*.PNG;*.JPEG",
        help="Semicolon-separated patterns for images",
    )
    src.add_argument(
        "--video",
        type=str,
        default=str(cfg.default_video_source),
        help="Optional: sample frames from a video file",
    )
    src.add_argument(
        "--frame-step",
        type=int,
        default=int(cfg.eval_frame_step),
        help="Take 1 frame every N frames when reading a video",
    )
    src.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames sampled (0=all)"
    )

    mdl = ap.add_argument_group("Models")
    mdl.add_argument(
        "--team-models",
        type=str,
        default=str(cfg.team_models_dir),
        help="Folder with umap.pkl and kmeans.pkl (default: config team_models_dir)",
    )
    mdl.add_argument(
        "--siglip", type=str, default=str(cfg.siglip_model), help="SigLIP model id"
    )
    mdl.add_argument(
        "--yolo-model",
        type=str,
        default=str(cfg.yolo_model),
        help="YOLO detection model path/id",
    )

    det = ap.add_argument_group("Detection")
    det.add_argument(
        "--imgsz", type=int, default=int(cfg.yolo_imgsz), help="YOLO inference size"
    )
    det.add_argument(
        "--conf",
        type=float,
        default=float(cfg.yolo_conf),
        help="YOLO confidence threshold",
    )
    det.add_argument(
        "--max-boxes",
        type=int,
        default=int(cfg.yolo_max_boxes),
        help="Max boxes per image/frame to annotate",
    )

    inf = ap.add_argument_group("Inference")
    inf.add_argument(
        "--central-ratio",
        type=float,
        default=float(cfg.central_ratio_default),
        help="Central crop ratio of bbox",
    )
    inf.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    out = ap.add_argument_group("Output")
    out.add_argument(
        "--out-dir", type=str, default=str(local_base), help="Output directory root"
    )
    out.add_argument("--show", action="store_true", help="Show previews in a window")
    out.add_argument(
        "--save-grid",
        action="store_true",
        help="Save a mosaic grid of annotated images",
    )
    out.add_argument(
        "--limit-images",
        type=int,
        default=int(cfg.eval_limit_images),
        help="Max annotated images to write",
    )

    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def central_crop_from_bbox(
    img: np.ndarray, bbox: Tuple[float, float, float, float], ratio: float
) -> Optional[np.ndarray]:
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


class ClusterTester:
    def __init__(
        self, models_dir: Path, yolo_model: str, siglip_id: str, device: str = "cuda"
    ):
        if not _HAS_JOBLIB:
            raise RuntimeError("joblib not available")
        models_dir = Path(models_dir)
        umap_p = models_dir / "umap.pkl"
        kmeans_p = models_dir / "kmeans.pkl"
        if not umap_p.exists() or not kmeans_p.exists():
            raise FileNotFoundError(
                f"Missing team models. Expected umap.pkl and kmeans.pkl in: {models_dir}.\n"
                f"Generate them via clustering (GUI: Clustering → Cluster → Save models) or CLI: \n"
                f"  python -m vt1.team_clustering.cluster_umap_kmeans --in-root {settings().team_output_dir} --out-dir {settings().team_output_dir} --save-models\n"
                f"Or point --team-models to a folder that contains these files."
            )
        self.reducer = joblib.load(umap_p)
        self.kmeans = joblib.load(kmeans_p)
        self.yolo = YOLO(yolo_model)
        self.device = device
        if device == "cuda":
            self.yolo.to("cuda")
        if not _HAS_SIGLIP:
            raise RuntimeError("transformers SigLIP not available")
        # Prefer fast processor when available to silence slow-processor warnings
        try:
            self.processor = AutoImageProcessor.from_pretrained(siglip_id, use_fast=True)  # type: ignore
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(siglip_id)
        self.siglip = SiglipVisionModel.from_pretrained(siglip_id)
        if device == "cuda":
            self.siglip.to("cuda")
        self.siglip.eval()

    @torch.inference_mode()
    def predict_labels_on_image(
        self,
        img_bgr: np.ndarray,
        imgsz: int,
        conf_thr: float,
        max_boxes: int,
        central_ratio: float,
    ) -> Tuple[np.ndarray, List[List[float]], List[int], List[float]]:
        # YOLO person detection
        res = self.yolo.predict(
            img_bgr, imgsz=imgsz, conf=conf_thr, device=self.device, verbose=False
        )[0]
        boxes: List[List[float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            conf = (
                res.boxes.conf.detach().cpu().numpy()
                if getattr(res.boxes, "conf", None) is not None
                else None
            )
            order = np.argsort(
                -(
                    conf
                    if conf is not None
                    else (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                )
            )
            keep = order[: min(len(order), max(0, int(max_boxes)))]
            boxes = xyxy[keep].tolist()
        if not boxes:
            return img_bgr, [], [], []
        # Build crops for SigLIP
        crops = []
        idxs = []
        for i, bb in enumerate(boxes):
            crop = central_crop_from_bbox(
                img_bgr, (bb[0], bb[1], bb[2], bb[3]), central_ratio
            )
            if crop is None:
                continue
            if crop.ndim == 2:
                crop = np.stack([crop, crop, crop], axis=-1)
            if crop.shape[2] == 4:
                crop = crop[:, :, :3]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            idxs.append(i)
        if not crops:
            return img_bgr, boxes, [], []
        # SigLIP embeddings
        px = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(
            self.device
        )
        out = self.siglip(pixel_values=px)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1).cpu().numpy()
        # UMAP -> KMeans
        Z = self.reducer.transform(emb)
        labels = self.kmeans.predict(Z).astype(int)
        # Distance margin to centroid as a pseudo-confidence
        centers = self.kmeans.cluster_centers_
        margins: List[float] = []
        for zi in Z:
            d = np.linalg.norm(centers - zi[None, :], axis=1)
            srt = np.sort(d)
            margin = float(srt[1] - srt[0]) if len(srt) >= 2 else float(srt[0])
            margins.append(margin)
        # Draw boxes
        annotated = img_bgr.copy()
        for j, i in enumerate(idxs):
            x1, y1, x2, y2 = map(int, boxes[i])
            lab = int(labels[j])
            col = color_for_team(lab)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            tag = f"T{lab} m={margins[j]:.2f}"
            cv2.putText(
                annotated,
                tag,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                tag,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                1,
                cv2.LINE_AA,
            )
        # Expand labels to align with 'boxes' order
        full_labels = [-1] * len(boxes)
        full_margins = [0.0] * len(boxes)
        for j, i in enumerate(idxs):
            full_labels[i] = int(labels[j])
            full_margins[i] = float(margins[j])
        return annotated, boxes, full_labels, full_margins


def read_images(images_dir: str, patterns: str) -> List[Path]:
    if not images_dir:
        return []
    root = Path(images_dir)
    pats = [p.strip() for p in patterns.split(";") if p.strip()]
    out: List[Path] = []
    for pat in pats:
        out.extend(sorted(root.glob(pat)))
    return [p for p in out if p.is_file()]


def sample_video_frames(
    video_path: str, frame_step: int, max_frames: int
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    idx = 0
    got = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, frame_step) == 0:
            frames.append(frame.copy())
            got += 1
            if max_frames > 0 and got >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def make_grid(
    images: List[np.ndarray], cols: int = 3, pad: int = 4
) -> Optional[np.ndarray]:
    if not images:
        return None
    Hs = [im.shape[0] for im in images]
    Ws = [im.shape[1] for im in images]
    H = int(np.median(Hs))
    W = int(np.median(Ws))
    resized = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in images]
    rows = (len(resized) + cols - 1) // cols
    grid = np.full(
        (rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad, 3),
        32,
        dtype=np.uint8,
    )
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(resized):
                break
            y = r * H + (r + 1) * pad
            x = c * W + (c + 1) * pad
            grid[y : y + H, x : x + W] = resized[k]
            k += 1
    return grid


def main() -> int:
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / f"eval_{ts}"
    ensure_dir(out_root)
    print(f"[INFO] Output folder: {out_root}")

    # helper: robust save with fallback and logging
    def save_annotated(img: np.ndarray, path: Path) -> None:
        ok = cv2.imwrite(str(path), img)
        if not ok:
            # try PNG fallback
            alt = path.with_suffix(".png")
            ok2 = cv2.imwrite(str(alt), img)
            if ok2:
                print(
                    f"[WARN] Failed to write {path.name}, saved as {alt.name} instead"
                )
            else:
                print(f"[ERROR] Failed to write annotated image: {path}")
        else:
            # optional: verbose listing
            pass

    # Load tester
    try:
        tester = ClusterTester(
            models_dir=Path(args.team_models),
            yolo_model=args.yolo_model,
            siglip_id=args.siglip,
            device=device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to init models: {e}")
        return 1

    saved_count = 0

    # Collect inputs
    annotated_images: List[np.ndarray] = []
    per_image_stats: List[Dict[str, Any]] = []

    # From images dir
    img_paths = read_images(args.images_dir, args.glob) if args.images_dir else []
    if args.images_dir:
        print(
            f"[INFO] Images dir set: {args.images_dir} | patterns: {args.glob} | matched: {len(img_paths)} file(s)"
        )
    if args.images_dir and not img_paths:
        print(
            f"[WARN] No images matched in {args.images_dir} for patterns: {args.glob}"
        )
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read image: {p}")
            continue
        ann, boxes, labels, margins = tester.predict_labels_on_image(
            img,
            imgsz=args.imgsz,
            conf_thr=args.conf,
            max_boxes=args.max_boxes,
            central_ratio=args.central_ratio,
        )
        # Save annotated
        out_path = out_root / f"{p.stem}_ann.jpg"
        save_annotated(ann, out_path)
        saved_count += 1
        annotated_images.append(ann)
        per_image_stats.append(
            {
                "source": str(p),
                "num_boxes": int(len(boxes)),
                "labels": labels,
                "margins": margins,
            }
        )
        if args.show:
            cv2.imshow("eval", ann)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        if args.limit_images and len(annotated_images) >= args.limit_images:
            break

    # From video frames
    if args.video:
        video_abs = str(Path(args.video).resolve())
        print(
            f"[INFO] Sampling video: {video_abs} | frame-step={int(args.frame_step)} | max-frames={int(args.max_frames)}"
        )
        frames = sample_video_frames(
            args.video, frame_step=int(args.frame_step), max_frames=int(args.max_frames)
        )
        print(f"[INFO] Sampled {len(frames)} frame(s) from video")
        if not frames:
            print(f"[WARN] No frames sampled from video: {args.video}")
        for i, fr in enumerate(frames):
            ann, boxes, labels, margins = tester.predict_labels_on_image(
                fr,
                imgsz=args.imgsz,
                conf_thr=args.conf,
                max_boxes=args.max_boxes,
                central_ratio=args.central_ratio,
            )
            out_path = out_root / f"video_{i:05d}_ann.jpg"
            save_annotated(ann, out_path)
            saved_count += 1
            annotated_images.append(ann)
            per_image_stats.append(
                {
                    "source": f"{args.video}#frame{i}",
                    "num_boxes": int(len(boxes)),
                    "labels": labels,
                    "margins": margins,
                }
            )
            if args.show:
                cv2.imshow("eval", ann)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if args.limit_images and len(annotated_images) >= args.limit_images:
                break

    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Aggregate metrics on this sample
    all_labels = []
    all_margins = []
    for s in per_image_stats:
        for lab in s.get("labels", []):
            if lab is not None and int(lab) >= 0:
                all_labels.append(int(lab))
        all_margins += [float(m) for m in s.get("margins", []) if m is not None]

    counts: Dict[int, int] = {}
    for lab in all_labels:
        counts[lab] = counts.get(lab, 0) + 1

    # Try silhouette placeholder (see note)
    silhouette = None

    # Grid mosaic
    if args.save_grid and annotated_images:
        grid = make_grid(annotated_images[: min(len(annotated_images), 12)], cols=3)
        if grid is not None:
            cv2.imwrite(str(out_root / "mosaic.jpg"), grid)

    summary = {
        "total_images": int(len(per_image_stats)),
        "total_detections": int(sum(s.get("num_boxes", 0) for s in per_image_stats)),
        "label_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "avg_margin": (float(np.mean(all_margins)) if all_margins else None),
        "silhouette_umap": silhouette,  # not computed here to keep eval fast
        "out_dir": str(out_root),
        "saved_files": int(saved_count),
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {saved_count} annotated image(s) to: {out_root}")
    if saved_count == 0:
        print(
            "[HINT] No images saved. Check that --images-dir or --video is set, patterns match files, and YOLO detected frames were read."
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
