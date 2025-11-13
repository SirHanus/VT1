# python
"""
Audit and visualize the training data produced by build_training_set.py.
- Scans per-video folders under --in-root (default: team_clustering/clustering)
- Loads index.csv and, if available, crops/*.jpg referenced by crop_relpath
- Writes per-video mosaics and a summary.json with basic stats

Usage (Windows cmd):
  python offline_pipeline\team_clustering\audit_training_set.py ^
    --in-root offline_pipeline\team_clustering\clustering ^
    --per-video 32 --save-grid

If crops were not saved during dataset build, re-run build_training_set with --save-crops to enable visual audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

from vt1.config import settings

try:
    from vt1.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = settings()
    ap = argparse.ArgumentParser("Audit training set crops and stats")
    ap.add_argument(
        "--in-root",
        type=str,
        default=str(cfg.team_output_dir),
        help="Root folder containing per-video subfolders with index.csv (and crops/ if saved)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(cfg.team_output_dir),
        help="Output directory root",
    )
    ap.add_argument(
        "--per-video",
        type=int,
        default=24,
        help="Max crops per video to include in mosaic",
    )
    ap.add_argument(
        "--save-grid", action="store_true", help="Save per-video grid mosaics"
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_grid(
    images: List[np.ndarray], cols: int = 6, pad: int = 4
) -> np.ndarray | None:
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
    if args.seed:
        np.random.seed(args.seed)

    out_root = Path(args.out_dir) / f"audit_{time.strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(out_root)

    in_root = Path(args.in_root)
    if not in_root.exists():
        logger.error(f"--in-root does not exist: {in_root}")
        return 1

    videos = [p for p in in_root.iterdir() if p.is_dir()]
    if not videos:
        logger.error(f"No per-video folders found under {in_root}")
        return 1

    global_stats: Dict[str, Any] = {
        "in_root": str(in_root.resolve()),
        "videos": [],
        "total_crops": 0,
        "with_crops_saved": 0,
    }

    for vdir in sorted(videos):
        idx_p = vdir / "index.csv"
        crops_dir = vdir / "crops"
        if not idx_p.exists():
            continue
        with idx_p.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header, data = rows[0], rows[1:]
        # columns: [idx, frame_idx, time_s, x1,y1,x2,y2, score, label, crop_relpath]
        crop_col = (
            header.index("crop_relpath")
            if "crop_relpath" in header
            else (len(header) - 1)
        )
        sizes = []
        images = []
        crop_paths: List[Path] = []
        for r in data:
            rel = r[crop_col].strip() if len(r) > crop_col else ""
            if not rel:
                continue
            cp = vdir / rel
            if not cp.exists():
                continue
            crop_paths.append(cp)
        global_stats["total_crops"] += len(crop_paths)
        if crop_paths:
            global_stats["with_crops_saved"] += 1
        # Sample per-video
        if args.per_video > 0 and len(crop_paths) > args.per_video:
            idxs = np.random.choice(len(crop_paths), size=args.per_video, replace=False)
            crop_paths = [crop_paths[i] for i in idxs]
        # Load
        for cp in crop_paths:
            img = cv2.imread(str(cp))
            if img is None:
                continue
            sizes.append([img.shape[1], img.shape[0]])
            images.append(img)
        # Save grid
        if args.save_grid and images:
            grid = make_grid(images, cols=6)
            if grid is not None:
                out_img = out_root / f"{vdir.name}_grid.jpg"
                cv2.imwrite(str(out_img), grid)
        # Per-video stats
        vstats = {
            "video_stem": vdir.name,
            "num_rows": len(data),
            "num_crops_found": len(images),
            "crops_dir": str(crops_dir) if crops_dir.exists() else None,
            "avg_crop_size": (
                np.asarray(sizes).mean(axis=0).tolist() if sizes else None
            ),  # [W, H]
        }
        global_stats["videos"].append(vstats)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Audit done. Output: {out_root}")
    logger.info(json.dumps(global_stats, indent=2))
    if args.show_hint and global_stats.get("with_crops_saved", 0) == 0:
        logger.warning(
            "No crops saved. Re-run build_training_set.py with --save-crops to enable visual audit."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
