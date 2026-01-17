"""
Comprehensive benchmark comparison script.
Runs multiple benchmarks with different configurations and generates comparison plots.

Usage:
    python benchmark_comprehensive.py --video data_hockey.mp4 --frames 300
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def run_benchmark(
    video: str,
    frames: int,
    yolo_sizes: List[str],
    img_size: int,
    device: str,
    output_dir: Path,
) -> Path:
    """Run a single benchmark configuration."""
    cmd = [
        "python",
        "benchmark_pose_models.py",
        "--video",
        video,
        "--frames",
        str(frames),
        "--yolo-sizes",
        *yolo_sizes,
        "--img-size",
        str(img_size),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]

    print(f"\n🔧 Running benchmark: img_size={img_size}, models={yolo_sizes}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Benchmark failed: {result.stderr}")
        return None

    # Find the generated JSON file (most recent)
    json_files = list(output_dir.glob("benchmark_*.json"))
    if json_files:
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        return latest
    return None


def load_benchmark_results(json_paths: List[Path]) -> Dict:
    """Load multiple benchmark JSON files."""
    results = []
    for path in json_paths:
        if path and path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                results.append({"path": path, "data": data})
    return results


def plot_comparison(results: List[Dict], output_path: Path):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.35, top=0.90, bottom=0.05, left=0.05, right=0.98
    )

    # Organize data
    all_configs = []
    for r in results:
        data = r["data"]
        for model_name, metrics in data["models"].items():
            all_configs.append(
                {
                    "model": model_name,
                    "img_size": "Unknown",  # Could extract from path
                    "fps": metrics["mean_fps"],
                    "inference_ms": metrics["mean_inference_time_ms"],
                    "detections": metrics["mean_detections"],
                    "confidence": metrics["mean_confidence"],
                    "load_time": metrics["load_time_s"],
                }
            )

    if not all_configs:
        print("⚠️  No data to plot")
        return

    # Sort by model name
    all_configs.sort(key=lambda x: x["model"])

    model_names = [c["model"] for c in all_configs]
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

    # 1. FPS Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    fps_values = [c["fps"] for c in all_configs]
    bars = ax1.barh(model_names, fps_values, color=colors, alpha=0.8)
    ax1.set_xlabel("FPS", fontsize=11, fontweight="bold")
    ax1.set_title("Processing Speed Comparison", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, fps_values)):
        ax1.text(val, i, f" {val:.1f}", va="center", fontsize=8)

    # 2. Inference Time
    ax2 = fig.add_subplot(gs[0, 1])
    inf_times = [c["inference_ms"] for c in all_configs]
    bars = ax2.barh(model_names, inf_times, color=colors, alpha=0.8)
    ax2.set_xlabel("Inference Time (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Inference Time per Frame", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, inf_times)):
        ax2.text(val, i, f" {val:.1f}", va="center", fontsize=8)

    # 3. Detections
    ax3 = fig.add_subplot(gs[0, 2])
    detections = [c["detections"] for c in all_configs]
    bars = ax3.barh(model_names, detections, color=colors, alpha=0.8)
    ax3.set_xlabel("Detections per Frame", fontsize=11, fontweight="bold")
    ax3.set_title("Average Detection Count", fontsize=13, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, detections)):
        ax3.text(val, i, f" {val:.1f}", va="center", fontsize=8)

    # 4. Confidence
    ax4 = fig.add_subplot(gs[1, 0])
    confidences = [c["confidence"] for c in all_configs]
    bars = ax4.barh(model_names, confidences, color=colors, alpha=0.8)
    ax4.set_xlabel("Confidence Score", fontsize=11, fontweight="bold")
    ax4.set_title("Average Detection Confidence", fontsize=13, fontweight="bold")
    ax4.set_xlim([0, 1.0])
    ax4.grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, confidences)):
        ax4.text(val, i, f" {val:.3f}", va="center", fontsize=8)

    # 5. FPS vs Confidence Scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(
        fps_values,
        confidences,
        s=200,
        c=range(len(model_names)),
        cmap="tab20",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )

    for i, name in enumerate(model_names):
        ax5.annotate(
            name,
            (fps_values[i], confidences[i]),
            fontsize=8,
            ha="right",
            va="bottom",
            xytext=(-5, 5),
            textcoords="offset points",
        )

    ax5.set_xlabel("FPS", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Confidence Score", fontsize=11, fontweight="bold")
    ax5.set_title("Speed vs Confidence Trade-off", fontsize=13, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # 6. Load Time
    ax6 = fig.add_subplot(gs[2, 0])
    load_times = [c["load_time"] for c in all_configs]
    bars = ax6.bar(range(len(model_names)), load_times, color=colors, alpha=0.8)
    ax6.set_ylabel("Load Time (s)", fontsize=11, fontweight="bold")
    ax6.set_title("Model Initialization Time", fontsize=13, fontweight="bold")
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax6.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, load_times):
        height = bar.get_height()
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 7. Efficiency Score (custom metric)
    ax7 = fig.add_subplot(gs[2, 1])
    # Efficiency = (FPS * Detections * Confidence) / 1000
    efficiency = [
        (fps_values[i] * detections[i] * confidences[i]) / 100
        for i in range(len(model_names))
    ]
    bars = ax7.bar(range(len(model_names)), efficiency, color=colors, alpha=0.8)
    ax7.set_ylabel("Efficiency Score", fontsize=11, fontweight="bold")
    ax7.set_title("Overall Efficiency Metric", fontsize=13, fontweight="bold")
    ax7.set_xticks(range(len(model_names)))
    ax7.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax7.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax7.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Overall title
    fig.suptitle(
        "Comprehensive Pose Estimation Model Benchmark\n"
        "Performance Comparison Across Multiple Configurations",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Comprehensive plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive pose model benchmarks"
    )
    parser.add_argument("--video", type=str, default="data_hockey.mp4")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip running benchmarks, only plot existing results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏒 COMPREHENSIVE POSE ESTIMATION BENCHMARK")
    print("=" * 70)

    json_files = []

    if not args.skip_runs:
        # Configuration 1: Compare all YOLO sizes at 640
        result1 = run_benchmark(
            args.video,
            args.frames,
            ["n", "s", "m", "l", "x"],
            640,
            args.device,
            args.output_dir / "config1_sizes",
        )
        if result1:
            json_files.append(result1)

        # Configuration 2: Compare image sizes with medium model
        for img_size in [320, 480, 640, 960]:
            result = run_benchmark(
                args.video,
                args.frames,
                ["m"],
                img_size,
                args.device,
                args.output_dir / f"config2_imgsize_{img_size}",
            )
            if result:
                json_files.append(result)
    else:
        # Load existing results
        json_files = list(args.output_dir.glob("**/benchmark_*.json"))
        json_files.sort()

    if not json_files:
        print("❌ No benchmark results found")
        return

    print(f"\n📁 Found {len(json_files)} benchmark result files")

    # Load and plot comparison
    results = load_benchmark_results(json_files)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"comprehensive_comparison_{timestamp}.png"

    plot_comparison(results, output_path)

    print("\n✅ Comprehensive benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
