"""
Benchmark script for comparing pose estimation models performance.
Compares YOLO Pose models (currently implemented) with placeholders for other frameworks.

Usage:
    python benchmark_pose_models.py --video data_hockey.mp4 --frames 300
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

matplotlib.use("Agg")  # Non-interactive backend


class PoseModelBenchmark:
    """Base class for pose estimation model benchmarking."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.model = None
        self.load_time = 0.0

    def load_model(self):
        """Load the model and measure loading time."""
        raise NotImplementedError

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on a frame.
        Returns: (num_detections, confidence_score)
        """
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources and clear memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete

            # Optional: Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass  # Some PyTorch versions may not have these


class YOLOPoseBenchmark(PoseModelBenchmark):
    """YOLO Pose model benchmark."""

    def __init__(
        self, model_size: str = "n", device: str = "cuda", img_size: int = 640
    ):
        super().__init__(f"YOLO-{model_size.upper()}-Pose", device)
        self.model_size = model_size
        self.img_size = img_size
        self.model_path = None

    def load_model(self):
        """Load YOLO pose model."""
        start = time.perf_counter()

        # Check for local model files
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
            # Will download from ultralytics
            model_path = f"yolo11{self.model_size}-pose.pt"

        self.model = YOLO(model_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")

        self.load_time = time.perf_counter() - start
        return self.load_time

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run YOLO pose inference."""
        results = self.model(
            frame, imgsz=self.img_size, conf=0.25, verbose=False, device=self.device
        )

        num_detections = 0
        avg_conf = 0.0

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "keypoints") and result.keypoints is not None:
                num_detections = len(result.keypoints)
                if hasattr(result.boxes, "conf"):
                    confs = result.boxes.conf.cpu().numpy()
                    avg_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0

        return num_detections, avg_conf


class MediaPipePoseBenchmark(PoseModelBenchmark):
    """MediaPipe Pose benchmark (placeholder - requires mediapipe installation)."""

    def __init__(self, device: str = "cpu"):
        super().__init__("MediaPipe-Pose", "cpu")  # MediaPipe runs on CPU
        self.available = False

    def load_model(self):
        """Load MediaPipe pose model."""
        try:
            import mediapipe as mp

            self.available = True
            start = time.perf_counter()
            self.mp_pose = mp.solutions.pose
            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.load_time = time.perf_counter() - start
            return self.load_time
        except ImportError:
            print(f"⚠️  MediaPipe not installed. Skipping {self.name}")
            return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run MediaPipe inference."""
        if not self.available:
            return 0, 0.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_frame)

        num_detections = 1 if results.pose_landmarks else 0
        avg_conf = 0.0

        if results.pose_landmarks:
            # MediaPipe provides per-landmark visibility scores
            visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
            avg_conf = float(np.mean(visibilities))

        return num_detections, avg_conf


class OpenVINOPoseBenchmark(PoseModelBenchmark):
    """OpenVINO MoveNet benchmark (placeholder)."""

    def __init__(self, device: str = "cpu"):
        super().__init__("OpenVINO-MoveNet", device)
        self.available = False

    def load_model(self):
        """Load OpenVINO model."""
        try:
            from openvino.runtime import Core

            self.available = True
            start = time.perf_counter()
            # Placeholder - would need actual model file
            self.load_time = time.perf_counter() - start
            print(f"⚠️  OpenVINO model files not found. Skipping {self.name}")
            self.available = False
            return self.load_time
        except ImportError:
            print(f"⚠️  OpenVINO not installed. Skipping {self.name}")
            return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run OpenVINO inference."""
        return 0, 0.0


class TRTPoseBenchmark(PoseModelBenchmark):
    """TRT Pose benchmark (placeholder)."""

    def __init__(self, device: str = "cuda"):
        super().__init__("TRT-Pose", device)
        self.available = False

    def load_model(self):
        """Load TRT Pose model."""
        print(f"⚠️  TRT Pose not implemented. Skipping {self.name}")
        return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run TRT Pose inference."""
        return 0, 0.0


class MMPoseBenchmark(PoseModelBenchmark):
    """MMPose benchmark (placeholder)."""

    def __init__(self, device: str = "cuda"):
        super().__init__("MMPose", device)
        self.available = False

    def load_model(self):
        """Load MMPose model."""
        try:
            import mmpose

            self.available = True
            print(f"⚠️  MMPose model configuration needed. Skipping {self.name}")
            self.available = False
            return 0.0
        except ImportError:
            print(f"⚠️  MMPose not installed. Skipping {self.name}")
            return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run MMPose inference."""
        return 0, 0.0


def reset_memory():
    """Reset memory and clear caches between benchmark runs."""
    print("   🧹 Clearing memory...")

    # Force garbage collection multiple times for thorough cleanup
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset memory stats for clean benchmarking
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass  # Not available in all PyTorch versions

    # Small delay to ensure cleanup completes
    time.sleep(0.5)


def benchmark_video(
    video_path: str,
    models: List[PoseModelBenchmark],
    max_frames: Optional[int] = None,
    warmup_frames: int = 10,
    reset_memory_between_runs: bool = True,
) -> Dict:
    """
    Benchmark all models on a video.

    Args:
        video_path: Path to video file
        models: List of model benchmarks to test
        max_frames: Maximum frames to process (None = all)
        warmup_frames: Number of warmup frames before timing
        reset_memory_between_runs: Whether to reset memory between model runs

    Returns:
        Dictionary with benchmark results
    """
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
        "models": {},
    }

    for model_bench in models:
        print(f"\n🔧 Loading {model_bench.name}...")
        load_time = model_bench.load_model()

        if not hasattr(model_bench, "available"):
            model_bench.available = True

        if not model_bench.available:
            continue

        print(f"   Load time: {load_time:.3f}s")

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        inference_times = []
        detections = []
        confidences = []

        print(f"🏃 Running inference...")
        pbar = tqdm(total=total_frames, desc=f"   {model_bench.name}")

        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Warmup phase
            if frame_idx < warmup_frames:
                _ = model_bench.inference(frame)
                frame_idx += 1
                pbar.update(1)
                continue

            # Timed inference
            start = time.perf_counter()
            num_det, conf = model_bench.inference(frame)
            elapsed = time.perf_counter() - start

            inference_times.append(elapsed)
            detections.append(num_det)
            confidences.append(conf)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # Calculate metrics
        inference_times = np.array(inference_times)
        detections = np.array(detections)
        confidences = (
            np.array(confidences[warmup_frames:]) if confidences else np.array([])
        )

        if len(inference_times) == 0:
            print(f"   ⚠️  Warning: No inference times recorded (frames < warmup)")
            continue

        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        mean_fps = 1.0 / mean_time if mean_time > 0 else 0

        results["models"][model_bench.name] = {
            "load_time_s": float(load_time),
            "mean_inference_time_ms": float(mean_time * 1000),
            "std_inference_time_ms": float(std_time * 1000),
            "mean_fps": float(mean_fps),
            "min_inference_time_ms": float(np.min(inference_times) * 1000),
            "max_inference_time_ms": float(np.max(inference_times) * 1000),
            "mean_detections": (
                float(np.mean(detections)) if len(detections) > 0 else 0.0
            ),
            "std_detections": float(np.std(detections)) if len(detections) > 0 else 0.0,
            "mean_confidence": (
                float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            ),
            "frames_processed": len(inference_times),
        }

        print(f"   ✅ Avg: {mean_time*1000:.2f}ms ({mean_fps:.1f} FPS)")
        if len(detections) > 0:
            print(
                f"   📊 Detections: {np.mean(detections):.1f} ± {np.std(detections):.1f}"
            )
        else:
            print(f"   📊 Detections: 0.0 ± 0.0")

        # Cleanup
        model_bench.cleanup()

        # Reset memory before next model (if enabled)
        if reset_memory_between_runs:
            reset_memory()

    cap.release()
    return results


def get_hardware_info() -> str:
    """Get hardware information for plot title."""

    # Check for CUDA
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            # Simplify GPU name
            if "RTX" in gpu_name:
                # Extract model number after RTX
                parts = gpu_name.split("RTX")
                if len(parts) > 1:
                    model = parts[1].strip().split()[0]
                    return f"RTX {model}"
            elif "GTX" in gpu_name:
                # Extract model number after GTX
                parts = gpu_name.split("GTX")
                if len(parts) > 1:
                    model = parts[1].strip().split()[0]
                    return f"GTX {model}"
            # Fallback: return full GPU name
            return gpu_name
        except Exception as e:
            # If anything fails, return generic CUDA
            return "CUDA"
    else:
        return "CPU"


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_data = results["models"]
    if not models_data:
        print("⚠️  No model results to plot")
        return

    model_names = list(models_data.keys())

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Inference Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_times = [models_data[m]["mean_inference_time_ms"] for m in model_names]
    std_times = [models_data[m]["std_inference_time_ms"] for m in model_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    bars = ax1.bar(
        model_names, mean_times, yerr=std_times, capsize=5, color=colors, alpha=0.8
    )
    ax1.set_ylabel("Inference Time (ms)", fontsize=12, fontweight="bold")
    ax1.set_title("Mean Inference Time per Frame", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, mean_times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. FPS Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    fps_values = [models_data[m]["mean_fps"] for m in model_names]
    bars = ax2.bar(model_names, fps_values, color=colors, alpha=0.8)
    ax2.set_ylabel("FPS", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Processing Speed (Frames Per Second)", fontsize=14, fontweight="bold"
    )
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Detection Count
    ax3 = fig.add_subplot(gs[1, 0])
    det_means = [models_data[m]["mean_detections"] for m in model_names]
    det_stds = [models_data[m]["std_detections"] for m in model_names]
    bars = ax3.bar(
        model_names, det_means, yerr=det_stds, capsize=5, color=colors, alpha=0.8
    )
    ax3.set_ylabel("Number of Detections", fontsize=12, fontweight="bold")
    ax3.set_title("Average Detections per Frame", fontsize=14, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, det_means):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Confidence Scores
    ax4 = fig.add_subplot(gs[1, 1])
    confidences = [models_data[m]["mean_confidence"] for m in model_names]
    bars = ax4.bar(model_names, confidences, color=colors, alpha=0.8)
    ax4.set_ylabel("Confidence Score", fontsize=12, fontweight="bold")
    ax4.set_title("Average Detection Confidence", fontsize=14, fontweight="bold")
    ax4.set_ylim([0, 1.0])
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, confidences):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. Model Load Time
    ax5 = fig.add_subplot(gs[2, 0])
    load_times = [models_data[m]["load_time_s"] for m in model_names]
    bars = ax5.bar(model_names, load_times, color=colors, alpha=0.8)
    ax5.set_ylabel("Load Time (seconds)", fontsize=12, fontweight="bold")
    ax5.set_title("Model Load Time", fontsize=14, fontweight="bold")
    ax5.tick_params(axis="x", rotation=45)
    ax5.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, load_times):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 6. Summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("tight")
    ax6.axis("off")

    table_data = []
    headers = ["Model", "FPS", "Time (ms)", "Detections"]
    for name in model_names:
        m = models_data[name]
        table_data.append(
            [
                name,
                f"{m['mean_fps']:.1f}",
                f"{m['mean_inference_time_ms']:.1f}",
                f"{m['mean_detections']:.1f}",
            ]
        )

    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["lightgray"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Overall title
    video_info = results["video_info"]
    hw_info = get_hardware_info()
    fig.suptitle(
        f"Pose Estimation Model Benchmark - {hw_info}\n"
        f'Video: {Path(video_info["path"]).name} | '
        f'Resolution: {video_info["resolution"]} | '
        f'Frames: {video_info["frames_processed"]}',
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"pose_benchmark_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pose estimation models on hockey video"
    )
    parser.add_argument(
        "--video", type=str, default="data_hockey.mp4", help="Path to video file"
    )
    parser.add_argument(
        "--frames", type=int, default=300, help="Maximum number of frames to process"
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
        default=Path("benchmark_results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--yolo-sizes",
        nargs="+",
        default=["n", "x"],
        help="YOLO model sizes to test (n, s, m, l, x)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Image size for YOLO inference"
    )
    parser.add_argument(
        "--include-mediapipe",
        action="store_true",
        help="Include MediaPipe benchmark (requires mediapipe package)",
    )
    parser.add_argument(
        "--no-memory-reset",
        action="store_true",
        help="Disable memory reset between model runs (may cause memory buildup)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏒 POSE ESTIMATION MODEL BENCHMARK")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Video: {args.video}")
    print(f"Max frames: {args.frames}")
    print(f"Memory reset: {'Disabled' if args.no_memory_reset else 'Enabled'}")

    # Setup models to benchmark
    models = []

    # YOLO models (different sizes)
    for size in args.yolo_sizes:
        models.append(YOLOPoseBenchmark(size, args.device, args.img_size))

    # MediaPipe (optional)
    if args.include_mediapipe:
        models.append(MediaPipePoseBenchmark("cpu"))

    # Placeholders for future implementation
    # models.append(OpenVINOPoseBenchmark(args.device))
    # models.append(TRTPoseBenchmark(args.device))
    # models.append(MMPoseBenchmark(args.device))

    # Run benchmark
    results = benchmark_video(
        args.video,
        models,
        max_frames=args.frames,
        warmup_frames=10,
        reset_memory_between_runs=not args.no_memory_reset,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"benchmark_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")

    # Generate plots
    plot_results(results, args.output_dir)

    print("\n✅ Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
