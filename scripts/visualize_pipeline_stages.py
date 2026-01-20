"""
Create a side-by-side visualization showing three stages of the pipeline:
1. YOLO Pose Detection
2. SAM2 Segmentation
3. Team Clustering

For a single frame from the video.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vt1.config import settings
from vt1.pipeline.sam_general import SAM2VideoWrapper
from vt1.pipeline.sam_offline import TeamClusteringInfer

# COCO keypoint skeleton pairs (17-keypoint format)
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 11),
    (6, 12),
]

# Team colors (BGR)
TEAM_COLORS = [
    (255, 0, 0),  # Team 0 - Blue
    (0, 165, 255),  # Team 1 - Orange
    (50, 205, 50),  # Team 2 - Green
    (255, 105, 180),  # Team 3 - Pink
    (255, 215, 0),  # Team 4 - Gold
]


def color_for_team(label: int) -> Tuple[int, int, int]:
    if label is None or label < 0:
        return (200, 200, 200)
    return TEAM_COLORS[label % len(TEAM_COLORS)]


def draw_pose_keypoints(
    img: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)
):
    """Draw pose keypoints and skeleton on image."""
    if keypoints is None or len(keypoints) == 0:
        return img

    img_out = img.copy()
    kpts = keypoints  # Shape: (K, 3) where K=17 for COCO format

    # Draw skeleton lines
    for i, j in COCO_SKELETON:
        if i < len(kpts) and j < len(kpts):
            pt1 = kpts[i]
            pt2 = kpts[j]
            if pt1[2] > 0.5 and pt2[2] > 0.5:  # confidence threshold
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(img_out, (x1, y1), (x2, y2), color, 2)

    # Draw keypoints
    for kpt in kpts:
        if kpt[2] > 0.5:  # confidence threshold
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(img_out, (x, y), 4, (255, 0, 0), -1)

    return img_out


def create_stage1_yolo(
    frame: np.ndarray, yolo_model, imgsz: int, conf: float
) -> Tuple[np.ndarray, List, np.ndarray]:
    """Stage 1: YOLO Pose Detection"""
    img_yolo = frame.copy()

    # Run YOLO
    with torch.inference_mode():
        res = yolo_model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]

    boxes = []
    keypoints_all = None

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        boxes = xyxy.tolist()

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    if (
        hasattr(res, "keypoints")
        and res.keypoints is not None
        and len(res.keypoints) > 0
    ):
        keypoints_all = res.keypoints.data.detach().cpu().numpy()  # Shape: (N, K, 3)
        for kpts in keypoints_all:
            img_yolo = draw_pose_keypoints(img_yolo, kpts, (0, 255, 0))

    return img_yolo, boxes, keypoints_all


def create_stage2_sam(
    frame: np.ndarray, boxes: List, sam2_wrapper, frame_idx: int = 0
) -> np.ndarray:
    """Stage 2: SAM2 Segmentation"""
    img_sam = frame.copy()

    if not boxes or sam2_wrapper is None:
        return img_sam

    # Add box prompts to SAM2
    obj_ids = list(range(len(boxes)))
    sam2_wrapper.add_box_prompts(frame, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes)

    # Run SAM2
    masks_by_id = sam2_wrapper.segment_frame(frame, frame_idx)

    if masks_by_id is None or len(masks_by_id) == 0:
        return img_sam

    # Highly distinguishable color palette (BGR)
    distinguishable_colors = [
        (255, 0, 0),  # Bright Blue
        (0, 255, 0),  # Bright Green
        (0, 0, 255),  # Bright Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Orange
        (255, 128, 0),  # Sky Blue
        (0, 255, 128),  # Spring Green
        (128, 255, 0),  # Chartreuse
        (255, 0, 128),  # Rose
    ]

    # Create colored overlay
    overlay = img_sam.copy()

    for idx, (obj_id, mask) in enumerate(masks_by_id.items()):
        if mask is None:
            continue
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        # Use distinguishable color from palette
        color = distinguishable_colors[idx % len(distinguishable_colors)]
        overlay[mask > 0] = color

    # Blend with original
    img_sam = cv2.addWeighted(img_sam, 0.5, overlay, 0.5, 0)

    return img_sam


def create_stage3_teams(
    frame: np.ndarray,
    boxes: List,
    keypoints_all: np.ndarray,
    sam2_wrapper,
    team_infer: Optional[TeamClusteringInfer],
    central_ratio: float,
    frame_idx: int = 0,
) -> np.ndarray:
    """Stage 3: Team Clustering"""
    img_teams = frame.copy()

    if not boxes:
        return img_teams

    # Get SAM masks
    masks_by_id = None
    if sam2_wrapper is not None:
        # Note: We need to re-add prompts since segment_frame was already called in stage 2
        # In a real scenario, we'd cache the masks, but for visualization we'll regenerate
        obj_ids = list(range(len(boxes)))
        # Create a new wrapper instance or reset to avoid conflicts
        # For now, we'll just get the masks from a fresh call
        try:
            masks_by_id = sam2_wrapper.segment_frame(frame, frame_idx)
        except:
            # If it fails, re-add prompts
            sam2_wrapper.add_box_prompts(
                frame, frame_idx, obj_ids=obj_ids, boxes_xyxy=boxes
            )
            masks_by_id = sam2_wrapper.segment_frame(frame, frame_idx)

    # Get team labels
    team_labels = [-1] * len(boxes)
    if team_infer is not None:
        try:
            team_labels = team_infer.predict_labels(frame, boxes, central_ratio)
        except Exception as e:
            print(f"Team inference failed: {e}")

    # Create overlay with team colors
    overlay = img_teams.copy()

    for i, (box, label) in enumerate(zip(boxes, team_labels)):
        team_color = color_for_team(label)

        # Apply mask if available
        if masks_by_id is not None and i in masks_by_id and masks_by_id[i] is not None:
            mask = masks_by_id[i]
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)
            if mask.sum() > 0:
                overlay[mask > 0] = team_color
        else:
            # Fallback: draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), team_color, 2)

    # Blend with original
    img_teams = cv2.addWeighted(img_teams, 0.5, overlay, 0.5, 0)

    # Draw keypoints on top
    if keypoints_all is not None:
        for i, kpts in enumerate(keypoints_all):
            if i < len(team_labels):
                team_color = color_for_team(team_labels[i])
                img_teams = draw_pose_keypoints(img_teams, kpts, team_color)

    return img_teams


def create_pipeline_visualization(
    video_path: str,
    frame_number: int,
    yolo_model_path: str,
    sam2_model_id: str,
    team_models_dir: str,
    siglip_model: str,
    imgsz: int = 640,
    conf: float = 0.3,
    central_ratio: float = 0.6,
    output_dir: str = "outputs/pipeline_viz",
    frame_step: int = 1,
    max_frames: int = 0,
):
    """Create the full 3-panel visualization for multiple frames."""

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 1

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")

    # Determine which frames to process
    if max_frames > 0:
        end_frame = min(frame_number + max_frames * frame_step, total_frames)
    else:
        end_frame = total_frames

    frames_to_process = list(range(frame_number, end_frame, frame_step))
    print(f"Will process {len(frames_to_process)} frames (every {frame_step} frame(s))")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Read first frame to get dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return 1

    print(f"Frame shape: {first_frame.shape}")

    print(f"Frame shape: {first_frame.shape}")

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    if device == "cuda":
        yolo_model.to("cuda")

    # Load SAM2
    print("Loading SAM2 model...")
    sam2_wrapper = None
    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        sam2_wrapper = SAM2VideoWrapper(
            model_id=sam2_model_id, device=device, dtype=dtype
        )
    except Exception as e:
        print(f"Warning: SAM2 init failed: {e}")

    # Load team clustering
    print("Loading team clustering models...")
    team_infer = None
    try:
        team_infer = TeamClusteringInfer(
            models_dir=Path(team_models_dir), siglip_id=siglip_model, device=device
        )
    except Exception as e:
        print(f"Warning: Team clustering init failed: {e}")

    # Process each frame
    print(f"\nProcessing frames...")
    from tqdm import tqdm

    for frame_idx in tqdm(frames_to_process, desc="Processing frames"):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}, skipping...")
            continue

        # IMPORTANT: Reset SAM2 state for each new frame to prevent state accumulation
        if sam2_wrapper is not None:
            try:
                # Try to reset the state if method exists
                if hasattr(sam2_wrapper, "reset_state"):
                    sam2_wrapper.reset_state()
                else:
                    # Recreate the wrapper to ensure clean state
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    sam2_wrapper = SAM2VideoWrapper(
                        model_id=sam2_model_id, device=device, dtype=dtype
                    )
            except Exception as e:
                print(f"Warning: Could not reset SAM2 state: {e}")
                # Try recreating as fallback
                try:
                    dtype = torch.float16 if device == "cuda" else torch.float32
                    sam2_wrapper = SAM2VideoWrapper(
                        model_id=sam2_model_id, device=device, dtype=dtype
                    )
                except Exception as e2:
                    print(f"Error: Could not recreate SAM2 wrapper: {e2}")

        # Create each stage
        img_yolo, boxes, keypoints_all = create_stage1_yolo(
            frame, yolo_model, imgsz, conf
        )
        img_sam = create_stage2_sam(frame, boxes, sam2_wrapper, frame_idx=0)
        img_teams = create_stage3_teams(
            frame,
            boxes,
            keypoints_all,
            sam2_wrapper,
            team_infer,
            central_ratio,
            frame_idx=0,
        )

        # Create 3-panel figure with custom layout
        fig = plt.figure(figsize=(20, 12))

        # Create custom grid using GridSpec: 2 rows, 2 columns
        # Top row: YOLO (left) and SAM2 (right)
        # Bottom row: Team Clustering spanning both columns
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15)

        ax1 = fig.add_subplot(gs[0, 0])  # Top-left
        ax2 = fig.add_subplot(gs[0, 1])  # Top-right
        ax3 = fig.add_subplot(gs[1, :])  # Bottom row, spanning both columns

        ax1.imshow(cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB))
        ax1.set_title("1. YOLO Pose Detection", fontsize=16, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(img_sam, cv2.COLOR_BGR2RGB))
        ax2.set_title("2. SAM2 Segmentation", fontsize=16, fontweight="bold")
        ax2.axis("off")

        ax3.imshow(cv2.cvtColor(img_teams, cv2.COLOR_BGR2RGB))
        ax3.set_title("3. Team Clustering", fontsize=16, fontweight="bold")
        ax3.axis("off")

        # Adjust spacing to make it tighter
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Save with frame number in filename
        frame_output = output_path / f"frame_{frame_idx:06d}.png"
        plt.savefig(frame_output, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory

    cap.release()
    print(f"✓ Saved {len(frames_to_process)} visualizations to: {output_path}")

    return 0


def main():
    cfg = settings()

    parser = argparse.ArgumentParser(description="Visualize pipeline stages for frames")
    parser.add_argument(
        "--video",
        type=str,
        default=r"D:\WORK\VT1\highres_data\highres_hockey.mp4",
        help="Path to video file",
    )
    parser.add_argument(
        "--frame", type=int, default=100, help="Starting frame number to visualize"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="Process every nth frame (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of frames to process (0=all from start frame, default: 10)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=str(cfg.pose_model),
        help="Path to YOLO pose model",
    )
    parser.add_argument(
        "--sam2-model",
        type=str,
        default="facebook/sam2-hiera-large",
        help="SAM2 model ID",
    )
    parser.add_argument(
        "--team-models",
        type=str,
        default=str(cfg.team_models_dir),
        help="Directory with team clustering models",
    )
    parser.add_argument(
        "--siglip", type=str, default=str(cfg.siglip_model), help="SigLIP model ID"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--central-ratio",
        type=float,
        default=0.6,
        help="Central crop ratio for team inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pipeline_viz",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    return create_pipeline_visualization(
        video_path=args.video,
        frame_number=args.frame,
        yolo_model_path=args.yolo_model,
        sam2_model_id=args.sam2_model,
        team_models_dir=args.team_models,
        siglip_model=args.siglip,
        imgsz=args.imgsz,
        conf=args.conf,
        central_ratio=args.central_ratio,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    sys.exit(main())
