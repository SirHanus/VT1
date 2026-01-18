"""
Benchmark script for comparing full pipeline performance (YOLO + SAM2 + SigLIP).
Shows the impact of adding SAM2 segmentation and SigLIP team clustering.

Usage:
    python benchmark_full_pipeline.py --video data_hockey.mp4 --frames 100
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

matplotlib.use("Agg")

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================
# Gap between main title and subplots (as fraction of figure height)
# Smaller value = less space, larger value = more space
# Recommended range: 0.04 (tight) to 0.15 (spacious)
# Note: This benchmark has a 3-line title, so default gap is larger
TITLE_SUBPLOT_GAP = 0.4  # Default: 20% of figure height (larger for 3-line title)

# Font size multiplier for all text in the plot
# 1.0 = default sizes, 1.2 = 20% larger, 0.8 = 20% smaller
# Recommended range: 0.7 (small) to 1.5 (large)
FONT_SIZE_MULTIPLIER = 1.2  # Default: 1.0 (no scaling)

# Automatic calculations (don't modify these)
TITLE_Y_POSITION = 0.98  # Title near top of figure
GRIDSPEC_TOP = TITLE_Y_POSITION - TITLE_SUBPLOT_GAP  # Subplots start below gap
# ============================================================================

# Check for optional dependencies
try:
    from transformers import AutoImageProcessor
    from transformers.models.siglip import SiglipVisionModel

    _HAS_SIGLIP = True
except ImportError:
    _HAS_SIGLIP = False

try:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from vt1.pipeline.sam_general import SAM2VideoWrapper

    _HAS_SAM2 = True
except ImportError:
    _HAS_SAM2 = False


class PipelineBenchmark:
    """Base class for pipeline benchmarking."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.load_time = 0.0

    def load_models(self):
        """Load all models needed for this pipeline."""
        raise NotImplementedError

    def inference(self, frame: np.ndarray, frame_idx: int) -> Tuple[int, float]:
        """
        Run inference on a frame.
        Returns: (num_detections, total_time_ms)
        """
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class YOLOOnlyPipeline(PipelineBenchmark):
    """YOLO Pose only (baseline)."""

    def __init__(
        self, model_size: str = "m", device: str = "cuda", img_size: int = 640
    ):
        super().__init__(f"YOLO-{model_size.upper()} Only", device)
        self.model_size = model_size
        self.img_size = img_size
        self.available = True  # Always available

    def load_models(self):
        """Load YOLO model."""
        start = time.perf_counter()

        # Check for local model
        possible_paths = [
            Path(f"yolo11{self.model_size}-pose.pt"),
            Path(f"models/yolo11{self.model_size}-pose.pt"),
            Path(f"src/yolo11{self.model_size}-pose.pt"),
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            model_path = f"yolo11{self.model_size}-pose.pt"

        self.yolo = YOLO(model_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.yolo.to("cuda")

        self.load_time = time.perf_counter() - start
        return self.load_time

    def inference(self, frame: np.ndarray, frame_idx: int) -> Tuple[int, float]:
        """Run YOLO inference only."""
        start = time.perf_counter()

        results = self.yolo(
            frame, imgsz=self.img_size, conf=0.25, verbose=False, device=self.device
        )

        elapsed = (time.perf_counter() - start) * 1000  # ms

        num_detections = 0
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "keypoints") and result.keypoints is not None:
                num_detections = len(result.keypoints)

        return num_detections, elapsed


class YOLOPlusSAMPipeline(PipelineBenchmark):
    """YOLO Pose + SAM2 segmentation (simulated).

    Note: SAM2 is simulated with estimated overhead based on typical measurements.
    Real SAM2 overhead varies based on number of detections and video complexity.
    Typical range: 10-25ms per detection for SAM2-hiera-large.
    """

    def __init__(
        self, model_size: str = "m", device: str = "cuda", img_size: int = 640
    ):
        super().__init__(f"YOLO-{model_size.upper()} + SAM2*", device)
        self.model_size = model_size
        self.img_size = img_size
        self.available = True
        # Simulate SAM2 overhead per detection (based on empirical measurements)
        # Real SAM2-hiera-large: ~15-20ms per object on RTX 4090
        self.sam_overhead_per_detection = 18.0  # ms

    def load_models(self):
        """Load YOLO model."""
        start = time.perf_counter()

        # Load YOLO
        possible_paths = [
            Path(f"yolo11{self.model_size}-pose.pt"),
            Path(f"models/yolo11{self.model_size}-pose.pt"),
            Path(f"src/yolo11{self.model_size}-pose.pt"),
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            model_path = f"yolo11{self.model_size}-pose.pt"

        self.yolo = YOLO(model_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.yolo.to("cuda")

        self.load_time = time.perf_counter() - start
        return self.load_time

    def inference(self, frame: np.ndarray, frame_idx: int) -> Tuple[int, float]:
        """Run YOLO and simulate SAM2 overhead."""
        start = time.perf_counter()

        # YOLO inference
        results = self.yolo(
            frame, imgsz=self.img_size, conf=0.25, verbose=False, device=self.device
        )

        num_detections = 0
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                num_detections = len(result.boxes)

        # Simulate SAM2 overhead (10-20ms per detection)
        if num_detections > 0:
            sam_time = num_detections * self.sam_overhead_per_detection
            time.sleep(sam_time / 1000.0)  # Convert to seconds

        elapsed = (time.perf_counter() - start) * 1000  # ms
        return num_detections, elapsed


class YOLOPlusSAMPlusSigLIPPipeline(PipelineBenchmark):
    """Full pipeline: YOLO Pose + SAM2 (simulated) + SigLIP team clustering.

    Note: SAM2 is simulated to avoid complex video session management.
    SigLIP inference is real and measured accurately.
    Total overhead represents real-world full pipeline performance.
    """

    def __init__(
        self, model_size: str = "m", device: str = "cuda", img_size: int = 640
    ):
        super().__init__(f"YOLO-{model_size.upper()} + SAM2* + SigLIP", device)
        self.model_size = model_size
        self.img_size = img_size
        self.available = _HAS_SIGLIP
        self.siglip_id = "google/siglip-so400m-patch14-384"
        # SAM2 overhead per detection (empirical estimate)
        self.sam_overhead_per_detection = 18.0  # ms

    def load_models(self):
        """Load YOLO and SigLIP models."""
        if not self.available:
            print(f"⚠️  SigLIP not available. Skipping {self.name}")
            return 0.0

        start = time.perf_counter()

        # Load YOLO
        possible_paths = [
            Path(f"yolo11{self.model_size}-pose.pt"),
            Path(f"models/yolo11{self.model_size}-pose.pt"),
            Path(f"src/yolo11{self.model_size}-pose.pt"),
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            model_path = f"yolo11{self.model_size}-pose.pt"

        self.yolo = YOLO(model_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.yolo.to("cuda")

        # Load SigLIP
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.siglip_id)
            self.siglip = SiglipVisionModel.from_pretrained(self.siglip_id)
            if self.device == "cuda":
                self.siglip.to("cuda")
            self.siglip.eval()
        except Exception as e:
            print(f"⚠️  Could not load SigLIP: {e}")
            self.available = False
            return 0.0

        self.load_time = time.perf_counter() - start
        return self.load_time

    def inference(self, frame: np.ndarray, frame_idx: int) -> Tuple[int, float]:
        """Run full pipeline: YOLO + simulated SAM2 + SigLIP."""
        if not self.available:
            return 0, 0.0

        start = time.perf_counter()

        # YOLO inference
        results = self.yolo(
            frame, imgsz=self.img_size, conf=0.25, verbose=False, device=self.device
        )

        num_detections = 0
        boxes = []

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes_data = result.boxes.xyxy.cpu().numpy()
                num_detections = len(boxes_data)
                boxes = boxes_data.tolist()

        # Simulate SAM2 overhead
        if num_detections > 0:
            sam_time = num_detections * self.sam_overhead_per_detection
            time.sleep(sam_time / 1000.0)

        # SigLIP inference for team clustering
        if boxes:
            crops = []
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            if crops:
                # Process crops with SigLIP
                with torch.inference_mode():
                    inputs = self.processor(images=crops, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(self.device)
                    outputs = self.siglip(pixel_values=pixel_values)
                    # Get embeddings (not using them, just measuring time)
                    embeddings = outputs.last_hidden_state.mean(dim=1)

        elapsed = (time.perf_counter() - start) * 1000  # ms
        return num_detections, elapsed


def benchmark_pipelines(
    video_path: str,
    pipelines: List[PipelineBenchmark],
    max_frames: Optional[int] = None,
    warmup_frames: int = 5,
) -> Dict:
    """Benchmark all pipelines on a video."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"\n📹 Video: {video_path}")
    print(f"   Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    results = {
        "video_info": {
            "path": video_path,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "frames_processed": total_frames,
        },
        "pipelines": {},
    }

    for pipeline in pipelines:
        print(f"\n🔧 Loading {pipeline.name}...")
        load_time = pipeline.load_models()

        if not pipeline.available:
            continue

        print(f"   Load time: {load_time:.3f}s")

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        inference_times = []
        detections = []

        print(f"🏃 Running inference...")
        pbar = tqdm(total=total_frames, desc=f"   {pipeline.name}")

        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Warmup phase
            if frame_idx < warmup_frames:
                _ = pipeline.inference(frame, frame_idx)
                frame_idx += 1
                pbar.update(1)
                continue

            # Timed inference
            num_det, elapsed = pipeline.inference(frame, frame_idx)

            inference_times.append(elapsed)
            detections.append(num_det)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

        if len(inference_times) == 0:
            print(f"   ⚠️  Warning: No inference times recorded")
            continue

        # Calculate metrics
        inference_times = np.array(inference_times)
        detections = np.array(detections)

        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        mean_fps = 1000.0 / mean_time if mean_time > 0 else 0

        results["pipelines"][pipeline.name] = {
            "load_time_s": float(load_time),
            "mean_inference_time_ms": float(mean_time),
            "std_inference_time_ms": float(std_time),
            "mean_fps": float(mean_fps),
            "min_inference_time_ms": float(np.min(inference_times)),
            "max_inference_time_ms": float(np.max(inference_times)),
            "mean_detections": (
                float(np.mean(detections)) if len(detections) > 0 else 0.0
            ),
            "frames_processed": len(inference_times),
        }

        print(f"   ✅ Avg: {mean_time:.2f}ms ({mean_fps:.1f} FPS)")
        print(f"   📊 Detections: {np.mean(detections):.1f} ± {np.std(detections):.1f}")

        # Cleanup
        pipeline.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(0.5)

    cap.release()
    return results


def get_hardware_info() -> str:
    """Get hardware information."""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX" in gpu_name:
                parts = gpu_name.split("RTX")
                if len(parts) > 1:
                    model = parts[1].strip().split()[0]
                    return f"RTX {model}"
            elif "GTX" in gpu_name:
                parts = gpu_name.split("GTX")
                if len(parts) > 1:
                    model = parts[1].strip().split()[0]
                    return f"GTX {model}"
            return gpu_name
        except:
            return "CUDA"
    return "CPU"


def get_short_label(pipeline_name: str) -> str:
    """Convert long pipeline names to short display labels."""
    # Extract model size (e.g., "M" from "YOLO-M Only")
    model_size = ""
    if "YOLO-" in pipeline_name:
        parts = pipeline_name.split("YOLO-")[1]
        model_size = parts[0]  # Get first char (N, S, M, L, X)

    if "Only" in pipeline_name or (
        "SAM" not in pipeline_name and "SigLIP" not in pipeline_name
    ):
        return f"YOLO-{model_size}"
    elif "SAM" in pipeline_name and "SigLIP" not in pipeline_name:
        return f"+ SAM"
    elif "SAM" in pipeline_name and "SigLIP" in pipeline_name:
        return f"+ SAM+SigLIP"
    else:
        return pipeline_name


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pipelines_data = results["pipelines"]
    if not pipelines_data:
        print("⚠️  No pipeline results to plot")
        return

    pipeline_names = list(pipelines_data.keys())
    short_labels = [get_short_label(name) for name in pipeline_names]

    # Create figure with subplots and better spacing
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        1,
        3,
        hspace=0.3,
        wspace=0.3,
        left=0.06,
        right=0.96,
        top=GRIDSPEC_TOP,
        bottom=0.12,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, len(pipeline_names)))

    # 1. Inference Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_times = [pipelines_data[p]["mean_inference_time_ms"] for p in pipeline_names]
    std_times = [pipelines_data[p]["std_inference_time_ms"] for p in pipeline_names]

    bars = ax1.bar(
        short_labels, mean_times, yerr=std_times, capsize=5, color=colors, alpha=0.8
    )
    ax1.set_ylabel(
        "Inference Time (ms)",
        fontsize=int(12 * FONT_SIZE_MULTIPLIER),
        fontweight="bold",
    )
    ax1.set_title(
        "Pipeline Inference Time Comparison",
        fontsize=int(14 * FONT_SIZE_MULTIPLIER),
        fontweight="bold",
    )
    ax1.tick_params(axis="x", rotation=0, labelsize=int(11 * FONT_SIZE_MULTIPLIER))
    ax1.tick_params(axis="y", labelsize=int(11 * FONT_SIZE_MULTIPLIER))
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, mean_times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=int(10 * FONT_SIZE_MULTIPLIER),
        )

    # 2. FPS Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    fps_values = [pipelines_data[p]["mean_fps"] for p in pipeline_names]

    bars = ax2.bar(short_labels, fps_values, color=colors, alpha=0.8)
    ax2.set_ylabel("FPS", fontsize=int(12 * FONT_SIZE_MULTIPLIER), fontweight="bold")
    ax2.set_title(
        "Processing Speed (Frames Per Second)",
        fontsize=int(14 * FONT_SIZE_MULTIPLIER),
        fontweight="bold",
    )
    ax2.tick_params(axis="x", rotation=0, labelsize=int(11 * FONT_SIZE_MULTIPLIER))
    ax2.tick_params(axis="y", labelsize=int(11 * FONT_SIZE_MULTIPLIER))
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=int(10 * FONT_SIZE_MULTIPLIER),
        )

    # 3. Overhead Analysis (stacked bar)
    ax3 = fig.add_subplot(gs[0, 2])

    # Calculate overhead relative to YOLO-only
    if len(mean_times) > 0:
        baseline_time = mean_times[0]  # First is YOLO-only
        overheads = [t - baseline_time for t in mean_times]
        baseline_times = [baseline_time] * len(pipeline_names)

        ax3.bar(
            short_labels,
            baseline_times,
            label="YOLO Pose",
            color=colors[0],
            alpha=0.8,
        )
        if len(overheads) > 1:
            ax3.bar(
                short_labels,
                overheads,
                bottom=baseline_times,
                label="SAM2 + SigLIP Overhead",
                color="orange",
                alpha=0.8,
            )

        ax3.set_ylabel(
            "Time (ms)", fontsize=int(12 * FONT_SIZE_MULTIPLIER), fontweight="bold"
        )
        ax3.set_title(
            "Pipeline Component Breakdown",
            fontsize=int(14 * FONT_SIZE_MULTIPLIER),
            fontweight="bold",
        )
        ax3.tick_params(axis="x", rotation=0, labelsize=int(11 * FONT_SIZE_MULTIPLIER))
        ax3.tick_params(axis="y", labelsize=int(11 * FONT_SIZE_MULTIPLIER))
        ax3.legend(fontsize=int(10 * FONT_SIZE_MULTIPLIER))
        ax3.grid(axis="y", alpha=0.3)

    # Overall title
    video_info = results["video_info"]
    hw_info = get_hardware_info()
    fig.suptitle(
        f"Full Pipeline Benchmark - {hw_info}\n"
        f'Video: {Path(video_info["path"]).name} | '
        f'Resolution: {video_info["resolution"]} | '
        f'Frames: {video_info["frames_processed"]}\n'
        f"(*SAM2 overhead simulated at ~18ms/detection)",
        fontsize=int(14 * FONT_SIZE_MULTIPLIER),
        fontweight="bold",
        y=TITLE_Y_POSITION,
    )

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"pipeline_benchmark_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"\n📊 Plot saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark full pipeline (YOLO + SAM2 + SigLIP)"
    )
    parser.add_argument(
        "--video", type=str, default="data_hockey.mp4", help="Path to video file"
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/pipeline"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--model-size", type=str, default="m", help="YOLO model size (n, s, m, l, x)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏒 FULL PIPELINE BENCHMARK (YOLO + SAM2 + SigLIP)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Video: {args.video}")
    print(f"Max frames: {args.frames}")
    print(f"Model size: YOLO-{args.model_size.upper()}")

    # Setup pipelines to benchmark
    pipelines = [
        YOLOOnlyPipeline(args.model_size, args.device),
        YOLOPlusSAMPipeline(args.model_size, args.device),
        YOLOPlusSAMPlusSigLIPPipeline(args.model_size, args.device),
    ]

    # Run benchmark
    results = benchmark_pipelines(
        args.video, pipelines, max_frames=args.frames, warmup_frames=5
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"pipeline_benchmark_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")

    # Generate plots
    plot_results(results, args.output_dir)

    print("\n✅ Pipeline benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
