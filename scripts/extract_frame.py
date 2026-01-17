#!/usr/bin/env python3
"""
Quick command-line tool to extract specific frames from the highres hockey video.

Usage:
    python extract_frame.py 100                    # Extract frame 100
    python extract_frame.py 100 200                # Extract frames 100-200
    python extract_frame.py 100 200 --step 10      # Extract every 10th frame from 100-200
"""
import argparse
import sys
from pathlib import Path

import cv2

# Fixed constants
VIDEO_PATH = Path(
    r"D:\WORK\VT1\outputs\20260109_highres_hockey_pose_sam_cuda_img1280_c40_se1_sk12_sr60_teams.mp4"
)
OUTPUT_DIR = Path("outputs/extracted_frames")


def extract_single_frame(video_path: Path, frame_num: int, output_dir: Path) -> bool:
    """Extract a single frame from the video."""
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_num < 0 or frame_num >= total_frames:
        print(f"Error: Frame {frame_num} is out of range (0-{total_frames-1})")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    timestamp = frame_num / fps if fps > 0 else 0
    print(f"✓ Extracted frame {frame_num} (t={timestamp:.2f}s) -> {output_path}")
    return True


def extract_frame_range(
    video_path: Path, start_frame: int, end_frame: int, step: int, output_dir: Path
) -> bool:
    """Extract a range of frames from the video."""
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print(
            f"Error: Invalid frame range {start_frame}-{end_frame} (valid: 0-{total_frames-1})"
        )
        cap.release()
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    for frame_num in range(start_frame, end_frame + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_num}, skipping...")
            continue

        output_path = output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted_count += 1

        if extracted_count % 10 == 0:
            timestamp = frame_num / fps if fps > 0 else 0
            print(
                f"  Extracted {extracted_count} frames... (current: {frame_num}, t={timestamp:.2f}s)"
            )

    cap.release()

    print(
        f"✓ Extracted {extracted_count} frames ({start_frame}-{end_frame}, step={step}) -> {output_dir}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific frames from the highres hockey video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 100                    # Extract frame 100
  %(prog)s 100 200                # Extract frames 100-200 (inclusive)
  %(prog)s 100 200 --step 10      # Extract every 10th frame from 100-200
  %(prog)s 500 1000 --output ./my_frames  # Extract to custom output directory
        """,
    )

    parser.add_argument(
        "frame_start", type=int, help="Frame number to extract (or start of range)"
    )
    parser.add_argument(
        "frame_end",
        type=int,
        nargs="?",
        help="End frame number (optional, for range extraction)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Extract every Nth frame in range (default: 1)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Get the video path relative to the script's parent directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    video_path = project_root / VIDEO_PATH
    output_dir = (
        project_root / args.output if not args.output.is_absolute() else args.output
    )

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()

    if args.frame_end is None:
        # Single frame extraction
        success = extract_single_frame(video_path, args.frame_start, output_dir)
    else:
        # Range extraction
        success = extract_frame_range(
            video_path, args.frame_start, args.frame_end, args.step, output_dir
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
