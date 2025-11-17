"""
Hockey player dataset extraction for YOLO11x-pose fine-tuning.

This module extracts frames from hockey videos, detects all players with pose estimation,
and prepares a dataset in YOLO format for fine-tuning pose estimation models.

Usage:
    python -m vt1.finetuning.extract_dataset --max-players-per-video 100
    # Review images in output directory
    python -m vt1.finetuning.extract_dataset --export
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from vt1.config import settings
except ImportError:
    settings = None

try:
    from vt1.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)


class HockeyPoseDatasetExtractor:
    """Extract and organize hockey player images with pose annotations for YOLO fine-tuning."""

    def __init__(
        self,
        output_dir: str = "hockey_pose_dataset",
        model_path: str = "yolo11x-pose.pt",
        detection_conf: float = 0.5,
        min_keypoints: int = 5,
        videos_dir: str | None = None,
        load_model: bool = True,
    ):
        """
        Initialize the dataset extractor.

        Args:
            output_dir: Directory to save the dataset
            model_path: Path to YOLO pose model for initial detection
            detection_conf: Minimum confidence threshold for detections
            min_keypoints: Minimum number of visible keypoints to include detection
            videos_dir: Directory containing hockey videos (default: videos_all/)
        """
        self.output_dir = Path(output_dir)
        # Load model only when requested (full-frames mode doesn't need the model)
        self.model = YOLO(model_path) if load_model else None
        self.detection_conf = detection_conf
        self.min_keypoints = min_keypoints

        # Set videos directory - check parameter, then environment, then default
        if videos_dir:
            self.videos_dir = Path(videos_dir)
        else:
            # Try to find videos_all in project root
            cfg = settings() if settings is not None else None
            if cfg:
                self.videos_dir = cfg.repo_root / "videos_all"
            else:
                self.videos_dir = Path("videos_all")

        self.setup_directories()

    def setup_directories(self):
        """Create necessary directory structure for YOLO pose dataset."""
        # Main dataset structure
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        for split in ["train", "val"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Review directory for all players
        self.review_dir = self.output_dir / "review"
        (self.review_dir / "players").mkdir(parents=True, exist_ok=True)
        (self.review_dir / "rejected").mkdir(parents=True, exist_ok=True)

        # Metadata directory
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self, video_path: str, frame_interval: int = 30, max_frames: int | None = None
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to hockey video
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract (None for all)

        Returns:
            List of (frame_number, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video: {total_frames} frames @ {fps:.2f} fps")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        return frames

    def detect_players_with_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect players in frame with pose estimation.

        Args:
            frame: Input frame

        Returns:
            List of detections with bounding boxes, keypoints, and crops
        """
        if self.model is None:
            # Model not loaded (e.g. full-frames mode) - skip detection
            logger.debug("No model loaded, skipping detection")
            return []

        results = self.model(frame, conf=self.detection_conf, verbose=False)
        detections = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes
            keypoints = result.keypoints

            for i in range(len(boxes)):
                box = boxes[i]
                kpts = keypoints[i] if i < len(keypoints) else None

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Filter by size (to remove small/distant players)
                width = x2 - x1
                height = y2 - y1
                if width < 40 or height < 80:
                    continue

                # Extract keypoints if available
                keypoints_data = None
                visible_count = 0

                if kpts is not None:
                    # YOLO pose format: [17, 3] where each row is [x, y, visibility]
                    kpts_array = kpts.data.cpu().numpy().squeeze()
                    if len(kpts_array.shape) == 2 and kpts_array.shape[0] > 0:
                        keypoints_data = kpts_array
                        # Count visible keypoints (visibility > 0.5)
                        visible_count = np.sum(kpts_array[:, 2] > 0.5)

                # Skip if too few keypoints are visible
                if visible_count < self.min_keypoints:
                    continue

                # Crop player from frame
                crop = frame[y1:y2, x1:x2].copy()

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "crop": crop,
                        "keypoints": keypoints_data,
                        "visible_keypoints": visible_count,
                    }
                )

        return detections

    def save_detection_for_review(
        self,
        detection: Dict,
        frame_idx: int,
        det_idx: int,
        video_name: str,
    ) -> Path:
        """
        Save individual detection for manual review.

        Args:
            detection: Player detection
            frame_idx: Frame index
            det_idx: Detection index within frame
            video_name: Source video name

        Returns:
            Path to saved image
        """
        # Save all players in players folder
        folder = "players"

        # Create filename with metadata
        filename = f"{video_name}_f{frame_idx:06d}_d{det_idx:02d}.jpg"
        save_path = self.review_dir / folder / filename

        # Save crop
        cv2.imwrite(str(save_path), detection["crop"])

        return save_path

    def save_frame_with_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        frame_idx: int,
        video_name: str,
    ):
        """
        Save annotated frame showing all detections.

        Args:
            frame: Original frame
            detections: Player detections
            frame_idx: Frame index
            video_name: Source video name
        """
        annotated = frame.copy()
        color = (0, 255, 0)  # Green for all players

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{det['visible_keypoints']} kpts"
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Draw keypoints if available
            if det["keypoints"] is not None:
                kpts = det["keypoints"]
                for kpt in kpts:
                    x, y, vis = kpt
                    if vis > 0.5:  # Only draw visible keypoints
                        cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

        # Save annotated frame
        frame_path = self.metadata_dir / f"{video_name}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), annotated)

    def process_video(
        self,
        video_path: str,
        frame_interval: int = 30,
        max_players: int = 100,
        min_players_per_frame: int = 2,
    ):
        """
        Process entire video: extract frames, detect players with pose.

        Args:
            video_path: Path to hockey video
            frame_interval: Extract every Nth frame
            max_players: Maximum number of players to extract from this video
            min_players_per_frame: Minimum players needed to process frame
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        logger.info("=" * 60)
        logger.info(f"Processing: {video_name}")
        logger.info("=" * 60)

        # Extract frames
        frames = self.extract_frames(str(video_path), frame_interval, max_frames=None)
        logger.info(f"Extracted {len(frames)} frames at interval={frame_interval}")

        # Process each frame until we have enough players
        total_detections = 0
        frame_metadata = []
        idx = 0

        for idx, (frame_num, frame) in enumerate(frames):
            if total_detections >= max_players:
                logger.info(f"Reached max players ({max_players}), stopping.")
                break

            detections = self.detect_players_with_pose(frame)

            if len(detections) >= min_players_per_frame:
                # Save annotated frame
                self.save_frame_with_annotations(frame, detections, idx, video_name)

                # Save individual detections for review
                saved_paths = []
                players_saved_this_frame = 0

                for det_idx, det in enumerate(detections):
                    if total_detections >= max_players:
                        break

                    path = self.save_detection_for_review(det, idx, det_idx, video_name)
                    saved_paths.append(str(path.relative_to(self.output_dir)))
                    total_detections += 1
                    players_saved_this_frame += 1

                # Record metadata
                frame_metadata.append(
                    {
                        "frame_idx": idx,
                        "frame_num": frame_num,
                        "detections": players_saved_this_frame,
                        "saved_images": saved_paths,
                    }
                )

            # Progress update
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Frame {idx + 1}/{len(frames)} - Total players extracted: {total_detections}/{max_players}"
                )

        # Save metadata
        metadata = {
            "video_name": video_name,
            "video_path": str(video_path),
            "frame_interval": frame_interval,
            "frames_processed": idx + 1,
            "frames_with_detections": len(frame_metadata),
            "total_detections": total_detections,
            "max_players_limit": max_players,
            "frames": frame_metadata,
        }

        metadata_path = self.metadata_dir / f"{video_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"Summary for {video_name}:")
        logger.info(f"  Frames processed: {idx + 1}")
        logger.info(f"  Frames with detections: {len(frame_metadata)}")
        logger.info(f"  Total players extracted: {total_detections}/{max_players}")
        logger.info(f"  Metadata saved: {metadata_path}")
        logger.info("=" * 60)

        return total_detections

    def process_all_videos(
        self,
        players_per_video: int = 100,
        frame_interval: int = 30,
        min_players_per_frame: int = 2,
    ):
        """
        Process all videos in the videos_all directory.

        Args:
            players_per_video: Number of players to extract from each video
            frame_interval: Extract every Nth frame
            min_players_per_frame: Minimum players needed to process frame
        """
        if not self.videos_dir.exists():
            logger.error(f"Videos directory not found: {self.videos_dir}")
            return

        # Find all video files recursively
        video_files = []
        for ext in ["*.mp4", "*.avi", "*.MP4", "*.AVI"]:
            video_files.extend(self.videos_dir.rglob(ext))

        if not video_files:
            msg = f"No video files found in {self.videos_dir}"
            print(msg)
            logger.error(msg)
            return

        print("=" * 60)
        print(f"Found {len(video_files)} videos in {self.videos_dir}")
        print(f"Will extract {players_per_video} players from each video")
        print("=" * 60)

        logger.info(
            f"Found {len(video_files)} videos, extracting {players_per_video} players each"
        )

        total_players = 0
        processed_videos = 0

        for video_file in video_files:
            try:
                players_extracted = self.process_video(
                    str(video_file),
                    frame_interval=frame_interval,
                    max_players=players_per_video,
                    min_players_per_frame=min_players_per_frame,
                )
                total_players += players_extracted
                processed_videos += 1
            except Exception as e:
                error_msg = f"Error processing {video_file}: {e}"
                print(error_msg)
                logger.error(error_msg)
                continue

        print("=" * 60)
        print("EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"  Videos processed: {processed_videos}/{len(video_files)}")
        print(f"  Total players extracted: {total_players}")
        print(
            f"  Average per video: {total_players/processed_videos:.1f}"
            if processed_videos > 0
            else "  No videos processed"
        )
        print(f"  Review directory: {self.review_dir}/players/")
        print("=" * 60)

        logger.info(
            f"EXTRACTION COMPLETE: {total_players} players from {processed_videos} videos"
        )

    def process_all_videos_save_frames(
        self,
        frame_interval: int = 30,
        max_frames_per_video: int | None = None,
        train_split: float = 0.8,
    ):
        """
        Save full frames from all videos into the dataset (train/val split).

        This mode ignores player-based extraction and any detection-related options.

        Args:
            frame_interval: Extract every Nth frame
            max_frames_per_video: Maximum frames to extract per video (None for all)
            train_split: Fraction of images to assign to train (rest -> val)
        """
        if not self.videos_dir.exists():
            logger.error(f"Videos directory not found: {self.videos_dir}")
            return

        # Find video files
        video_files = []
        for ext in ["*.mp4", "*.avi", "*.MP4", "*.AVI"]:
            video_files.extend(self.videos_dir.rglob(ext))

        if not video_files:
            logger.error(f"No video files found in {self.videos_dir}")
            return

        print("=" * 60)
        print(f"Found {len(video_files)} videos in {self.videos_dir}")
        print(f"Saving full frames to dataset (no detection) in {self.output_dir}")
        print("=" * 60)

        # Temp folder to collect frames before splitting
        tmp_dir = self.output_dir / "tmp_full_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        total_saved = 0

        for video_idx, video_file in enumerate(video_files, 1):
            video_name = video_file.stem
            print(f"[{video_idx}/{len(video_files)}] Processing: {video_name}")
            logger.info(f"Processing (full frames): {video_name}")
            frames = self.extract_frames(
                str(video_file), frame_interval, max_frames=max_frames_per_video
            )

            video_frame_count = 0
            for frame_num, frame in frames:
                filename = f"{video_name}_frame_{frame_num:06d}.jpg"
                path = tmp_dir / filename
                success = cv2.imwrite(str(path), frame)

                # Only add to list if write was successful
                if success and path.exists():
                    saved_paths.append(path)
                    total_saved += 1
                    video_frame_count += 1
                else:
                    logger.warning(f"Failed to write frame: {path}")

                # Log progress every 50 frames
                if video_frame_count % 50 == 0:
                    print(f"  Saved {video_frame_count} frames from {video_name}...")

            print(
                f"  ✓ Saved {video_frame_count} frames from {video_name} (total: {total_saved})"
            )

        if total_saved == 0:
            logger.error("No frames were extracted.")
            return

        # Shuffle and split
        print(f"\nShuffling and splitting {len(saved_paths)} frames...")
        np.random.shuffle(saved_paths)
        split_idx = int(len(saved_paths) * train_split)
        train_list = saved_paths[:split_idx]
        val_list = saved_paths[split_idx:]

        # Ensure dataset image folders exist
        (self.images_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.images_dir / "val").mkdir(parents=True, exist_ok=True)

        # Move files into train/val with error handling
        print(f"Moving {len(train_list)} frames to train...")
        moved_train = 0
        for p in train_list:
            if not p.exists():
                logger.warning(f"File does not exist, skipping: {p}")
                continue
            dest = self.images_dir / "train" / p.name
            try:
                shutil.move(str(p), str(dest))
                moved_train += 1
            except Exception as e:
                logger.error(f"Failed to move {p} to {dest}: {e}")

        print(f"Moving {len(val_list)} frames to val...")
        moved_val = 0
        for p in val_list:
            if not p.exists():
                logger.warning(f"File does not exist, skipping: {p}")
                continue
            dest = self.images_dir / "val" / p.name
            try:
                shutil.move(str(p), str(dest))
                moved_val += 1
            except Exception as e:
                logger.error(f"Failed to move {p} to {dest}: {e}")

        # Remove tmp dir
        try:
            # List remaining files for debugging
            remaining = list(tmp_dir.glob("*"))
            if remaining:
                logger.warning(
                    f"{len(remaining)} files remain in tmp_dir, attempting cleanup..."
                )
                for f in remaining:
                    try:
                        f.unlink()
                    except Exception:
                        pass
            tmp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Could not remove tmp directory: {e}")

        print("=" * 60)
        print("Full-frame dataset saved!")
        print(f"  Total images extracted: {total_saved}")
        print(f"  Train: {moved_train}/{len(train_list)}")
        print(f"  Val: {moved_val}/{len(val_list)}")
        print(f"  Image folders: {self.images_dir}")
        print("=" * 60)

        # Write a minimal dataset.yaml to help later steps
        yaml_content = f"""# Full-frame dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val

names:
  0: hockey_player
"""
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        logger.info(
            f"Full-frame dataset exported: {total_saved} images to {self.images_dir}"
        )

    def export_for_label_studio(
        self,
        frame_interval: int = 30,
        max_frames_per_video: int = 50,
        include_predictions: bool = True,
    ):
        """
        Export full frames with detections for Label Studio import.

        Args:
            frame_interval: Extract every Nth frame
            max_frames_per_video: Maximum frames to export per video
            include_predictions: Include bounding boxes and keypoints as pre-annotations
        """
        logger.info("=" * 60)
        logger.info("Exporting frames for Label Studio")
        logger.info("=" * 60)

        # Create Label Studio export directory
        ls_dir = self.output_dir / "label_studio"
        ls_images_dir = ls_dir / "images"
        ls_images_dir.mkdir(parents=True, exist_ok=True)

        # Find all video files
        if not self.videos_dir.exists():
            logger.error(f"Videos directory not found: {self.videos_dir}")
            return

        video_files = []
        for ext in ["*.mp4", "*.avi", "*.MP4", "*.AVI"]:
            video_files.extend(self.videos_dir.rglob(ext))

        if not video_files:
            logger.error(f"No video files found in {self.videos_dir}")
            return

        logger.info(f"Found {len(video_files)} videos")

        # Storage for Label Studio tasks
        ls_tasks = []
        total_frames_exported = 0

        for video_file in video_files:
            video_name = video_file.stem
            logger.info(f"Processing {video_name}...")

            # Extract frames
            frames = self.extract_frames(
                str(video_file), frame_interval, max_frames=max_frames_per_video
            )

            for frame_idx, (frame_num, frame) in enumerate(frames):
                # Save full frame
                img_filename = f"{video_name}_frame_{frame_num:06d}.jpg"
                img_path = ls_images_dir / img_filename
                cv2.imwrite(str(img_path), frame)

                # Create Label Studio task
                task = {
                    "data": {
                        "image": f"/data/local-files/?d=label_studio/images/{img_filename}"
                    },
                    "meta": {
                        "video": video_name,
                        "frame_number": frame_num,
                        "frame_index": frame_idx,
                    },
                }

                # Add predictions if requested
                if include_predictions:
                    detections = self.detect_players_with_pose(frame)

                    if detections:
                        h, w = frame.shape[:2]
                        predictions = []

                        for det_idx, det in enumerate(detections):
                            x1, y1, x2, y2 = det["bbox"]

                            # Convert to Label Studio format (percentage)
                            bbox_x = (x1 / w) * 100
                            bbox_y = (y1 / h) * 100
                            bbox_width = ((x2 - x1) / w) * 100
                            bbox_height = ((y2 - y1) / h) * 100

                            # Create prediction with bounding box
                            prediction = {
                                "id": f"pred_{det_idx}",
                                "type": "rectanglelabels",
                                "value": {
                                    "x": bbox_x,
                                    "y": bbox_y,
                                    "width": bbox_width,
                                    "height": bbox_height,
                                    "rotation": 0,
                                    "rectanglelabels": ["player"],
                                },
                                "from_name": "label",
                                "to_name": "image",
                                "original_width": w,
                                "original_height": h,
                            }

                            # Add keypoints if available
                            if det["keypoints"] is not None:
                                keypoints_data = []
                                kpts = det["keypoints"]

                                for kp_idx, kpt in enumerate(kpts):
                                    x, y, vis = kpt
                                    if vis > 0.5:  # Only include visible keypoints
                                        keypoints_data.append(
                                            {
                                                "x": (x / w) * 100,
                                                "y": (y / h) * 100,
                                                "keypointlabels": [f"kp_{kp_idx}"],
                                            }
                                        )

                                if keypoints_data:
                                    prediction["keypoints"] = keypoints_data

                            predictions.append(prediction)

                        task["predictions"] = [{"result": predictions}]

                ls_tasks.append(task)
                total_frames_exported += 1

            logger.info(
                f"  Exported {len(frames)} frames from {video_name} (total: {total_frames_exported})"
            )

        # Save Label Studio import file
        ls_json_path = ls_dir / "import.json"
        with open(ls_json_path, "w") as f:
            json.dump(ls_tasks, f, indent=2)

        # Create Label Studio config template
        ls_config = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="player" background="green"/>
    <Label value="referee" background="blue"/>
    <Label value="goalie" background="red"/>
  </RectangleLabels>
  <KeyPointLabels name="kp-label" toName="image">
    <Label value="kp_0" background="#FFA39E"/>
    <Label value="kp_1" background="#FFA39E"/>
    <Label value="kp_2" background="#FFA39E"/>
    <Label value="kp_3" background="#FFA39E"/>
    <Label value="kp_4" background="#FFA39E"/>
    <Label value="kp_5" background="#D3F261"/>
    <Label value="kp_6" background="#D3F261"/>
    <Label value="kp_7" background="#95DE64"/>
    <Label value="kp_8" background="#95DE64"/>
    <Label value="kp_9" background="#5CDBD3"/>
    <Label value="kp_10" background="#5CDBD3"/>
    <Label value="kp_11" background="#69C0FF"/>
    <Label value="kp_12" background="#69C0FF"/>
    <Label value="kp_13" background="#B37FEB"/>
    <Label value="kp_14" background="#B37FEB"/>
    <Label value="kp_15" background="#FF85C0"/>
    <Label value="kp_16" background="#FF85C0"/>
  </KeyPointLabels>
</View>"""

        config_path = ls_dir / "labeling_config.xml"
        with open(config_path, "w") as f:
            f.write(ls_config)

        # Create README
        readme_content = f"""# Label Studio Import

## Files Generated:
- `images/`: Full frame images ({total_frames_exported} images)
- `import.json`: Label Studio import file with pre-annotations
- `labeling_config.xml`: Label Studio labeling interface configuration

## Import Instructions:

1. Create a new project in Label Studio
2. Go to Settings > Labeling Interface
3. Copy content from `labeling_config.xml` into the configuration
4. Go to Settings > Cloud Storage
5. Add a local file storage pointing to: `{ls_images_dir.absolute()}`
6. Import data from `import.json`

## Pre-annotations:
- Bounding boxes are included for all detected players
- Keypoints are included where visible
- Review and correct annotations as needed

## Videos processed: {len(video_files)}
## Total frames exported: {total_frames_exported}
"""

        readme_path = ls_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        print("=" * 60)
        print("Label Studio export complete!")
        print("=" * 60)
        print(f"  Output directory: {ls_dir}")
        print(f"  Frames exported: {total_frames_exported}")
        print(f"  Import file: {ls_json_path}")
        print(f"  Config file: {config_path}")
        print(f"\nSee {readme_path} for import instructions")
        print("=" * 60)

        logger.info(f"Label Studio export: {total_frames_exported} frames to {ls_dir}")

    def export_yolo_dataset(self, train_split: float = 0.8):
        """
        Export final dataset in YOLO pose format after manual review.

        The YOLO pose format for each image is:
        class x_center y_center width height kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v

        Args:
            train_split: Fraction of data for training (rest goes to validation)
        """
        logger.info("=" * 60)
        logger.info("Exporting YOLO Pose Dataset")
        logger.info("=" * 60)

        # Collect all player images
        all_player_images = list((self.review_dir / "players").glob("*.jpg"))

        logger.info(f"Found {len(all_player_images)} player images")

        if len(all_player_images) == 0:
            logger.error("No images found in review/players/ folder!")
            logger.error("Please run extraction first")
            return

        # Shuffle and assign single class (0 = hockey_player)
        all_images = [(img, 0) for img in all_player_images]
        np.random.shuffle(all_images)

        # Split train/val
        split_idx = int(len(all_images) * train_split)
        train_data = all_images[:split_idx]
        val_data = all_images[split_idx:]

        logger.info(f"Train: {len(train_data)} images")
        logger.info(f"Val: {len(val_data)} images")

        # Process train and val splits
        skipped = {"train": 0, "val": 0}
        for data, split in [(train_data, "train"), (val_data, "val")]:
            logger.info(f"Processing {split} split...")
            saved_count = 0

            for idx, (img_path, class_id) in enumerate(data):
                # Re-run pose detection on the crop to get keypoints
                img_data = cv2.imread(str(img_path))
                if img_data is None:
                    logger.warning(f"  Could not read {img_path}, skipping")
                    skipped[split] += 1
                    continue

                h, w = img_data.shape[:2]

                # Run pose detection with lower threshold for better capture
                results = self.model(img_data, conf=0.2, verbose=False)

                # Only save if we detected valid keypoints
                has_valid_keypoints = False
                kpts_normalized = None

                if (
                    len(results) > 0
                    and results[0].keypoints is not None
                    and len(results[0].keypoints) > 0
                ):
                    # Get the first detection (should be the only one)
                    kpts = results[0].keypoints[0].data.cpu().numpy().squeeze()

                    # Ensure we have 17 keypoints with at least some visible
                    if len(kpts) == 17:
                        # Check how many keypoints are visible
                        visible_kpts = np.sum(kpts[:, 2] > 0.5)
                        if visible_kpts >= self.min_keypoints:
                            # Normalize keypoints to crop size
                            kpts_normalized = kpts.copy()
                            kpts_normalized[:, 0] /= w  # normalize x
                            kpts_normalized[:, 1] /= h  # normalize y
                            has_valid_keypoints = True

                if not has_valid_keypoints:
                    # Skip this image - no valid keypoints detected
                    skipped[split] += 1
                    continue

                # Copy image
                dest_img = self.images_dir / split / img_path.name
                shutil.copy(img_path, dest_img)

                # Create YOLO annotation
                ann_path = self.labels_dir / split / f"{img_path.stem}.txt"

                with open(ann_path, "w") as f:
                    # YOLO format: class x_center y_center width height kpts...
                    # For a crop, the bbox is the entire image
                    line_parts = [
                        str(class_id),
                        "0.5",
                        "0.5",
                        "1.0",
                        "1.0",
                    ]

                    # Add keypoints (x, y, visibility for each of 17 points)
                    for kpt in kpts_normalized:
                        x, y, v = kpt
                        line_parts.extend([f"{x:.6f}", f"{y:.6f}", f"{int(v)}"])

                    f.write(" ".join(line_parts) + "\n")

                saved_count += 1

            # Show final count for this split
            print(
                f"  [{split}] [OK] Completed: {saved_count} images saved, {skipped[split]} skipped"
            )
            logger.info(f"{split} split: {saved_count} saved, {skipped[split]} skipped")

        # Create dataset.yaml
        yaml_content = f"""# Hockey Players Pose Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: hockey_player

# Keypoints
kpt_shape: [17, 3]  # 17 keypoints, 3 values (x, y, visibility)

# Keypoint flip indices for left-right augmentation (COCO format)
# Maps each keypoint to its horizontally flipped counterpart
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
"""

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print("=" * 60)
        print("Dataset exported successfully!")
        print(f"  Dataset YAML: {yaml_path}")
        print(
            f"  Training images: {len(train_data) - skipped['train']} (skipped {skipped['train']})"
        )
        print(
            f"  Validation images: {len(val_data) - skipped['val']} (skipped {skipped['val']})"
        )
        print("\nTo train YOLO11x-pose:")
        print(f"  yolo pose train data={yaml_path} model=yolo11x-pose.pt epochs=100")
        print("=" * 60)

        logger.info(
            f"Dataset exported: {len(train_data) - skipped['train']} train, {len(val_data) - skipped['val']} val images"
        )


def main():
    """CLI entry point for dataset extraction."""
    parser = argparse.ArgumentParser(
        description="Extract hockey player dataset with pose for YOLO fine-tuning"
    )

    # Load config if available
    cfg = settings() if settings is not None else None

    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file or directory of videos",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        help="Directory containing all videos to process (e.g., videos_all/)",
    )
    parser.add_argument(
        "--max-players-per-video",
        type=int,
        default=100,
        help="Maximum players to extract from each video (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hockey_pose_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x-pose.pt",
        help="YOLO pose model path",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Extract every Nth frame",
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=5,
        help="Minimum visible keypoints to include detection",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export dataset to YOLO format (after extraction)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio",
    )
    parser.add_argument(
        "--export-label-studio",
        action="store_true",
        help="Export full frames with pre-annotations for Label Studio",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=50,
        help="Maximum frames to export per video for Label Studio (default: 50)",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Don't include bounding box predictions in Label Studio export",
    )
    parser.add_argument(
        "--full-frames",
        action="store_true",
        help="Save full video frames as dataset (ignores player extraction)",
    )

    args = parser.parse_args()

    # If full-frames is requested we don't need to load the model or use
    # any player-based extraction options. Warn if user supplied conflicting
    # options and instantiate extractor without loading the model.
    if args.full_frames:
        print("NOTE: --full-frames is set, ignoring player extraction options.")
        ignored_opts = []
        if args.detection_conf != 0.5:
            ignored_opts.append("--detection-conf")
        if args.min_keypoints != 5:
            ignored_opts.append("--min-keypoints")
        if args.max_players_per_video != 100:
            ignored_opts.append("--max-players-per-video")

        if ignored_opts:
            logger.warning(
                "The following options are ignored when --full-frames is set: %s",
                ", ".join(ignored_opts),
            )

    extractor = HockeyPoseDatasetExtractor(
        output_dir=args.output_dir,
        model_path=args.model,
        detection_conf=args.detection_conf,
        min_keypoints=args.min_keypoints,
        videos_dir=args.videos_dir,
        load_model=not args.full_frames,
    )

    if args.full_frames:
        # Save full frames from videos
        print("Saving full frames from videos into dataset...")
        extractor.process_all_videos_save_frames(
            frame_interval=args.frame_interval,
            max_frames_per_video=args.max_frames_per_video,
            train_split=args.train_split,
        )
    elif args.export:
        # Export reviewed dataset
        extractor.export_yolo_dataset(train_split=args.train_split)
    elif args.export_label_studio:
        # Export full frames for Label Studio
        extractor.export_for_label_studio(
            frame_interval=args.frame_interval,
            max_frames_per_video=args.max_frames_per_video,
            include_predictions=not args.no_predictions,
        )
    elif args.videos_dir:
        # Process all videos in specified directory
        extractor.process_all_videos(
            players_per_video=args.max_players_per_video,
            frame_interval=args.frame_interval,
        )

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print(f"1. Review images in: {extractor.review_dir}/players/")
        print("2. Delete poor quality/incorrect images")
        print("3. Run with --export to create YOLO dataset:")
        print(
            f"   python -m vt1.finetuning.extract_dataset --export --output-dir {args.output_dir}"
        )
        print("=" * 60 + "\n")
    elif args.video:
        # Process video(s)
        video_path = Path(args.video)

        if video_path.is_file():
            # Single video
            extractor.process_video(
                str(video_path),
                frame_interval=args.frame_interval,
                max_players=args.max_players_per_video,
            )
        elif video_path.is_dir():
            # Directory of videos
            video_files = list(video_path.glob("*.mp4")) + list(
                video_path.glob("*.avi")
            )
            logger.info(f"Found {len(video_files)} videos in {video_path}")

            total_extracted = 0
            for video_file in video_files:
                total_extracted += extractor.process_video(
                    str(video_file),
                    frame_interval=args.frame_interval,
                    max_players=args.max_players_per_video,
                )

            logger.info(f"Total players extracted from all videos: {total_extracted}")
        else:
            logger.error(f"{video_path} is not a valid file or directory")
            return

        logger.info("=" * 60)
        logger.info("NEXT STEPS:")
        logger.info("=" * 60)
        logger.info(f"1. Review images in: {extractor.review_dir}/players/")
        logger.info("2. Delete poor quality/incorrect images")
        logger.info("3. Run with --export to create YOLO dataset:")
        logger.info(
            f"   python -m vt1.finetuning.extract_dataset --export --output-dir {args.output_dir}"
        )
        logger.info("=" * 60)
    else:
        parser.print_help()
        logger.info("\nExamples:")
        logger.info("  # Process all videos in videos_all/ (100 players each)")
        logger.info(
            "  python -m vt1.finetuning.extract_dataset --videos-dir videos_all --max-players-per-video 100"
        )
        logger.info("\n  # Process specific video")
        logger.info(
            "  python -m vt1.finetuning.extract_dataset --video path/to/video.mp4 --max-players-per-video 50"
        )
        logger.info("\n  # Export full frames for Label Studio (with bounding boxes)")
        logger.info(
            "  python -m vt1.finetuning.extract_dataset --export-label-studio --max-frames-per-video 50"
        )
        logger.info("\n  # Export full frames for Label Studio (without predictions)")
        logger.info(
            "  python -m vt1.finetuning.extract_dataset --export-label-studio --no-predictions"
        )
        logger.info("\n  # Export dataset after review")
        logger.info("  python -m vt1.finetuning.extract_dataset --export")


if __name__ == "__main__":
    main()
