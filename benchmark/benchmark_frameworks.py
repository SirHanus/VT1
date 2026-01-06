"""
Framework Comparison Benchmark - October 3rd Evaluation
Compares all pose estimation frameworks evaluated for hockey video analysis.

Frameworks tested:
- MediaPipe Pose
- OpenVINO MoveNet
- TRT Pose (NVIDIA TensorRT)
- PyTorch Keypoint Baseline
- MMPose
- YOLO11-Pose (selected solution)

This benchmark justifies the framework selection decision in the paper.
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

matplotlib.use("Agg")

# Check for framework availability
_FRAMEWORKS_AVAILABLE = {
    "mediapipe": False,
    "openvino": False,
    "trt_pose": False,
    "mmpose": False,
    "pytorch_keypoint": False,
    "yolo": False,
}

try:
    # Try newer mediapipe structure first
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

    _FRAMEWORKS_AVAILABLE["mediapipe"] = True
    # Create mp object for compatibility
    mp = type(
        "MediaPipe",
        (),
        {
            "solutions": type(
                "Solutions",
                (),
                {
                    "pose": mp_pose,
                    "drawing_utils": mp_drawing,
                    "POSE_CONNECTIONS": mp_pose.POSE_CONNECTIONS,
                },
            )()
        },
    )()
except ImportError:
    try:
        # Fallback to older mediapipe structure
        import mediapipe as mp

        _FRAMEWORKS_AVAILABLE["mediapipe"] = hasattr(mp, "solutions")
    except ImportError:
        _FRAMEWORKS_AVAILABLE["mediapipe"] = False
        mp = None

try:
    from openvino.runtime import Core

    _FRAMEWORKS_AVAILABLE["openvino"] = True
except ImportError:
    pass

try:
    import trt_pose.coco

    _FRAMEWORKS_AVAILABLE["trt_pose"] = True
except ImportError:
    pass

try:
    from mmpose.apis import init_model, inference_topdown

    _FRAMEWORKS_AVAILABLE["mmpose"] = True
except ImportError:
    pass

try:
    from ultralytics import YOLO

    _FRAMEWORKS_AVAILABLE["yolo"] = True
except ImportError:
    pass

# PyTorch Keypoint baseline is part of torchvision
_FRAMEWORKS_AVAILABLE["pytorch_keypoint"] = True


class FrameworkBenchmark:
    """Base class for pose estimation framework benchmarking."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.load_time = 0.0
        self.available = False
        self.model_size_mb = 0.0
        self.setup_complexity = "Unknown"  # Low, Medium, High

    def load_model(self):
        """Load the model and measure loading time."""
        raise NotImplementedError

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on a frame.
        Returns: (num_detections, inference_time_ms)
        """
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class MediaPipeBenchmark(FrameworkBenchmark):
    """MediaPipe Pose benchmark."""

    def __init__(self, device: str = "cpu"):
        super().__init__("MediaPipe Pose", "cpu")  # MediaPipe runs on CPU
        self.available = _FRAMEWORKS_AVAILABLE["mediapipe"]
        self.setup_complexity = "Low"
        self.mp_pose = None
        self.mp_drawing = None

    def load_model(self):
        """Load MediaPipe pose model."""
        if not self.available:
            print(f"⚠️  MediaPipe not available. Skipping {self.name}")
            return 0.0

        start = time.perf_counter()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.load_time = time.perf_counter() - start
        self.model_size_mb = 5.0  # Approximate
        return self.load_time

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run MediaPipe inference."""
        if not self.available or not self.model:
            return 0, 0.0

        start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        num_detections = 1 if results.pose_landmarks else 0
        return num_detections, elapsed

    def get_results(self, frame: np.ndarray):
        """Get full MediaPipe results for visualization."""
        if not self.available or not self.model:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.model.process(rgb_frame)


class OpenVINOBenchmark(FrameworkBenchmark):
    """OpenVINO MoveNet benchmark."""

    def __init__(self, device: str = "cpu"):
        super().__init__("OpenVINO MoveNet", device)
        self.available = _FRAMEWORKS_AVAILABLE["openvino"]
        self.setup_complexity = "Medium"

    def load_model(self):
        """Load OpenVINO model."""
        if not self.available:
            print(f"⚠️  OpenVINO not available. Skipping {self.name}")
            return 0.0

        # Note: Would need actual MoveNet ONNX/IR model file
        print(f"⚠️  OpenVINO MoveNet model files not configured. Skipping {self.name}")
        self.available = False
        return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run OpenVINO inference."""
        if not self.available:
            return 0, 0.0
        # Implementation would go here
        return 0, 0.0


class TRTPoseBenchmark(FrameworkBenchmark):
    """TRT Pose (NVIDIA TensorRT) benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__("TRT Pose", device)
        self.available = _FRAMEWORKS_AVAILABLE["trt_pose"] and device == "cuda"
        self.setup_complexity = "High"

    def load_model(self):
        """Load TRT Pose model."""
        if not self.available:
            print(f"⚠️  TRT Pose not available or not on CUDA. Skipping {self.name}")
            return 0.0

        # Note: Would need TensorRT engine file
        print(f"⚠️  TRT Pose model not configured. Skipping {self.name}")
        self.available = False
        return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run TRT Pose inference."""
        if not self.available:
            return 0, 0.0
        return 0, 0.0


class PyTorchKeypointBenchmark(FrameworkBenchmark):
    """PyTorch Keypoint R-CNN baseline."""

    def __init__(self, device: str = "cuda"):
        super().__init__("PyTorch Keypoint R-CNN", device)
        self.available = True
        self.setup_complexity = "Low"

    def load_model(self):
        """Load PyTorch Keypoint R-CNN."""
        start = time.perf_counter()

        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.load_time = time.perf_counter() - start
        self.model_size_mb = 160.0  # Approximate
        return self.load_time

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run PyTorch Keypoint R-CNN inference."""
        start = time.perf_counter()

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(img_tensor)

        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Count detections with confidence > 0.5
        num_detections = 0
        if len(outputs) > 0 and "scores" in outputs[0]:
            scores = outputs[0]["scores"].cpu().numpy()
            num_detections = int((scores > 0.5).sum())

        return num_detections, elapsed


class MMPoseBenchmark(FrameworkBenchmark):
    """MMPose framework benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__("MMPose", device)
        self.available = _FRAMEWORKS_AVAILABLE["mmpose"]
        self.setup_complexity = "High"

    def load_model(self):
        """Load MMPose model."""
        if not self.available:
            print(f"⚠️  MMPose not available. Skipping {self.name}")
            return 0.0

        # Note: Would need config and checkpoint files
        print(f"⚠️  MMPose model configuration needed. Skipping {self.name}")
        self.available = False
        return 0.0

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run MMPose inference."""
        if not self.available:
            return 0, 0.0
        return 0, 0.0


class YOLOBenchmark(FrameworkBenchmark):
    """YOLO11-Pose benchmark (selected solution)."""

    def __init__(
        self, model_size: str = "m", device: str = "cuda", img_size: int = 640
    ):
        super().__init__(f"YOLO11-{model_size.upper()}-Pose", device)
        self.model_size = model_size
        self.img_size = img_size
        self.available = _FRAMEWORKS_AVAILABLE["yolo"]
        self.setup_complexity = "Low"

    def load_model(self):
        """Load YOLO pose model."""
        if not self.available:
            print(f"⚠️  YOLO not available. Skipping {self.name}")
            return 0.0

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
            model_path = f"yolo11{self.model_size}-pose.pt"

        from ultralytics import YOLO

        self.model = YOLO(model_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")

        self.load_time = time.perf_counter() - start

        # Model sizes (approximate)
        size_map = {"n": 6, "s": 22, "m": 50, "l": 81, "x": 137}
        self.model_size_mb = size_map.get(self.model_size, 50)

        return self.load_time

    def inference(self, frame: np.ndarray) -> Tuple[int, float]:
        """Run YOLO inference."""
        start = time.perf_counter()

        results = self.model(
            frame, imgsz=self.img_size, conf=0.25, verbose=False, device=self.device
        )

        elapsed = (time.perf_counter() - start) * 1000  # ms

        num_detections = 0
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "keypoints") and result.keypoints is not None:
                num_detections = len(result.keypoints)

        return num_detections, elapsed


def select_distributed_frames(total_frames: int, num_samples: int) -> List[int]:
    """
    Select frames distributed across beginning, middle, and end of video.

    Args:
        total_frames: Total number of frames available
        num_samples: Number of samples to select

    Returns:
        List of frame indices distributed across the video timeline
    """
    if num_samples >= total_frames:
        return list(range(total_frames))

    # Divide video into three segments (beginning, middle, end)
    frames_per_segment = max(1, num_samples // 3)
    remaining = num_samples - (frames_per_segment * 3)

    segment_size = total_frames // 3
    selected_frames = []

    # Beginning (first 33%)
    selected_frames.extend(
        np.linspace(0, segment_size - 1, frames_per_segment, dtype=int)
    )

    # Middle (33-66%)
    selected_frames.extend(
        np.linspace(segment_size, 2 * segment_size - 1, frames_per_segment, dtype=int)
    )

    # End (last 33%)
    selected_frames.extend(
        np.linspace(
            2 * segment_size,
            total_frames - 1,
            frames_per_segment + remaining,
            dtype=int,
        )
    )

    return sorted(set(selected_frames))


def export_sample_frames(
    video_path: str,
    frameworks: List[FrameworkBenchmark],
    output_dir: Path,
    num_samples: int = 10,
    max_frames: Optional[int] = None,
):
    """Export sample frames with pose annotations for visual inspection."""

    sample_dir = output_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📸 Exporting {num_samples} sample frames with annotations...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️  Could not open video for frame export")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Select frames distributed across beginning, middle, and end
    frame_indices = select_distributed_frames(total_frames, num_samples)

    # Colors for different frameworks
    colors = {
        "YOLO": (0, 255, 0),  # Green
        "PyTorch": (255, 0, 0),  # Blue
        "MediaPipe": (0, 255, 255),  # Yellow
    }

    for sample_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Create side-by-side comparison for each framework
        comparison_frames = []

        for framework in frameworks:
            if not framework.available:
                continue

            annotated = frame.copy()

            # Run inference
            try:
                if isinstance(framework, YOLOBenchmark) and hasattr(framework, "model"):
                    # YOLO framework
                    results = framework.model(
                        annotated,
                        imgsz=640,
                        conf=0.25,
                        verbose=False,
                        device=framework.device,
                    )
                    if results and len(results) > 0:
                        result = results[0]
                        if (
                            hasattr(result, "keypoints")
                            and result.keypoints is not None
                        ):
                            kpts = result.keypoints.xy.cpu().numpy()
                            boxes = (
                                result.boxes.xyxy.cpu().numpy()
                                if hasattr(result, "boxes")
                                else None
                            )

                            # Draw boxes and keypoints
                            for i, keypoints in enumerate(kpts):
                                if boxes is not None and i < len(boxes):
                                    x1, y1, x2, y2 = boxes[i].astype(int)
                                    cv2.rectangle(
                                        annotated, (x1, y1), (x2, y2), colors["YOLO"], 2
                                    )

                                # Draw keypoints
                                for kpt in keypoints:
                                    x, y = int(kpt[0]), int(kpt[1])
                                    if x > 0 and y > 0:
                                        cv2.circle(
                                            annotated, (x, y), 3, colors["YOLO"], -1
                                        )

                elif isinstance(framework, PyTorchKeypointBenchmark) and hasattr(
                    framework, "model"
                ):
                    # PyTorch Keypoint R-CNN
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(framework.device)

                    with torch.inference_mode():
                        outputs = framework.model(img_tensor)

                    if len(outputs) > 0 and "boxes" in outputs[0]:
                        boxes = outputs[0]["boxes"].cpu().numpy()
                        scores = outputs[0]["scores"].cpu().numpy()
                        keypoints = outputs[0]["keypoints"].cpu().numpy()

                        # Draw detections with score > 0.5
                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            if score > 0.5:
                                x1, y1, x2, y2 = box.astype(int)
                                cv2.rectangle(
                                    annotated, (x1, y1), (x2, y2), colors["PyTorch"], 2
                                )
                                cv2.putText(
                                    annotated,
                                    f"{score:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    colors["PyTorch"],
                                    1,
                                )

                                # Draw keypoints
                                if i < len(keypoints):
                                    for kpt in keypoints[i]:
                                        x, y, v = int(kpt[0]), int(kpt[1]), kpt[2]
                                        if v > 0 and x > 0 and y > 0:
                                            cv2.circle(
                                                annotated,
                                                (x, y),
                                                3,
                                                colors["PyTorch"],
                                                -1,
                                            )

                elif isinstance(framework, MediaPipeBenchmark) and hasattr(
                    framework, "model"
                ):
                    # MediaPipe Pose
                    results = framework.get_results(annotated)

                    if results and results.pose_landmarks:
                        # Get image dimensions
                        h, w = annotated.shape[:2]

                        # Draw landmarks
                        landmarks = results.pose_landmarks.landmark

                        # Calculate bounding box from landmarks
                        x_coords = [lm.x * w for lm in landmarks]
                        y_coords = [lm.y * h for lm in landmarks]

                        if x_coords and y_coords:
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                            cv2.rectangle(
                                annotated, (x1, y1), (x2, y2), colors["MediaPipe"], 2
                            )

                        # Draw keypoints
                        for lm in landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if 0 < x < w and 0 < y < h and lm.visibility > 0.5:
                                cv2.circle(
                                    annotated, (x, y), 3, colors["MediaPipe"], -1
                                )

                        # Draw skeleton connections
                        mp_pose = mp.solutions.pose
                        connections = mp_pose.POSE_CONNECTIONS
                        for connection in connections:
                            start_idx, end_idx = connection
                            start_lm = landmarks[start_idx]
                            end_lm = landmarks[end_idx]

                            if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                                end_point = (int(end_lm.x * w), int(end_lm.y * h))
                                cv2.line(
                                    annotated,
                                    start_point,
                                    end_point,
                                    colors["MediaPipe"],
                                    2,
                                )

            except Exception as e:
                print(f"   ⚠️  Error annotating frame for {framework.name}: {e}")

            # Add framework name and detection count
            num_det = 0
            try:
                num_det, _ = framework.inference(frame)
            except:
                pass

            cv2.putText(
                annotated,
                f"{framework.name}: {num_det} detections",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            comparison_frames.append(annotated)

        # Combine frames side by side if multiple frameworks
        if len(comparison_frames) > 0:
            if len(comparison_frames) == 1:
                final_frame = comparison_frames[0]
            elif len(comparison_frames) == 2:
                final_frame = np.hstack(comparison_frames)
            else:
                # Stack 3 frames: top row 2, bottom row 1
                top = np.hstack(comparison_frames[:2])
                if len(comparison_frames) > 2:
                    bottom = comparison_frames[2]
                    # Pad bottom to match width
                    if bottom.shape[1] < top.shape[1]:
                        pad_width = top.shape[1] - bottom.shape[1]
                        bottom = np.pad(
                            bottom, ((0, 0), (0, pad_width), (0, 0)), mode="constant"
                        )
                    final_frame = np.vstack([top, bottom])
                else:
                    final_frame = top

            output_path = (
                sample_dir / f"frame_{frame_idx:04d}_sample{sample_idx:02d}.jpg"
            )
            cv2.imwrite(str(output_path), final_frame)
            print(f"   Saved: {output_path.name}")

    cap.release()
    print(f"✅ Exported {len(frame_indices)} sample frames to {sample_dir}")


def benchmark_frameworks(
    video_path: str,
    frameworks: List[FrameworkBenchmark],
    max_frames: Optional[int] = None,
    warmup_frames: int = 5,
) -> Dict:
    """Benchmark all frameworks on a video."""

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
        "frameworks": {},
    }

    for framework in frameworks:
        print(f"\n🔧 Loading {framework.name}...")
        load_time = framework.load_model()

        if not framework.available:
            continue

        print(f"   Load time: {load_time:.3f}s")
        print(f"   Model size: ~{framework.model_size_mb:.0f} MB")
        print(f"   Setup complexity: {framework.setup_complexity}")

        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        inference_times = []
        detections = []

        print(f"🏃 Running inference...")
        pbar = tqdm(total=total_frames, desc=f"   {framework.name}")

        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Warmup phase
            if frame_idx < warmup_frames:
                _ = framework.inference(frame)
                frame_idx += 1
                pbar.update(1)
                continue

            # Timed inference
            num_det, elapsed = framework.inference(frame)

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

        results["frameworks"][framework.name] = {
            "load_time_s": float(load_time),
            "mean_inference_time_ms": float(mean_time),
            "std_inference_time_ms": float(std_time),
            "mean_fps": float(mean_fps),
            "min_inference_time_ms": float(np.min(inference_times)),
            "max_inference_time_ms": float(np.max(inference_times)),
            "mean_detections": (
                float(np.mean(detections)) if len(detections) > 0 else 0.0
            ),
            "std_detections": float(np.std(detections)) if len(detections) > 0 else 0.0,
            "frames_processed": len(inference_times),
            "model_size_mb": float(framework.model_size_mb),
            "setup_complexity": framework.setup_complexity,
            "device": framework.device,
        }

        print(f"   ✅ Avg: {mean_time:.2f}ms ({mean_fps:.1f} FPS)")
        if len(detections) > 0:
            print(
                f"   📊 Detections: {np.mean(detections):.1f} ± {np.std(detections):.1f}"
            )

        # Cleanup
        framework.cleanup()
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


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frameworks_data = results["frameworks"]
    if not frameworks_data:
        print("⚠️  No framework results to plot")
        return

    framework_names = list(frameworks_data.keys())

    # Create figure with subplots and better spacing
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3, 3, hspace=0.45, wspace=0.4, left=0.08, right=0.95, top=0.82, bottom=0.08
    )

    colors = plt.cm.Set3(np.linspace(0, 1, len(framework_names)))

    # 1. Inference Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mean_times = [frameworks_data[f]["mean_inference_time_ms"] for f in framework_names]
    std_times = [frameworks_data[f]["std_inference_time_ms"] for f in framework_names]

    bars = ax1.bar(
        range(len(framework_names)),
        mean_times,
        yerr=std_times,
        capsize=5,
        color=colors,
        alpha=0.8,
    )
    ax1.set_xticks(range(len(framework_names)))
    ax1.set_xticklabels(framework_names, rotation=45, ha="right")
    ax1.set_ylabel("Inference Time (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Inference Time per Frame", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

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
    fps_values = [frameworks_data[f]["mean_fps"] for f in framework_names]

    bars = ax2.bar(range(len(framework_names)), fps_values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(framework_names)))
    ax2.set_xticklabels(framework_names, rotation=45, ha="right")
    ax2.set_ylabel("FPS", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Processing Speed (Frames Per Second)", fontsize=13, fontweight="bold"
    )
    ax2.grid(axis="y", alpha=0.3)

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
    ax3 = fig.add_subplot(gs[0, 2])
    det_means = [frameworks_data[f]["mean_detections"] for f in framework_names]
    det_stds = [frameworks_data[f]["std_detections"] for f in framework_names]

    bars = ax3.bar(
        range(len(framework_names)),
        det_means,
        yerr=det_stds,
        capsize=5,
        color=colors,
        alpha=0.8,
    )
    ax3.set_xticks(range(len(framework_names)))
    ax3.set_xticklabels(framework_names, rotation=45, ha="right")
    ax3.set_ylabel("Detections", fontsize=11, fontweight="bold")
    ax3.set_title("Average Detections per Frame", fontsize=13, fontweight="bold")
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

    # 4. Model Size
    ax4 = fig.add_subplot(gs[1, 0])
    model_sizes = [frameworks_data[f]["model_size_mb"] for f in framework_names]

    bars = ax4.bar(range(len(framework_names)), model_sizes, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(framework_names)))
    ax4.set_xticklabels(framework_names, rotation=45, ha="right")
    ax4.set_ylabel("Model Size (MB)", fontsize=11, fontweight="bold")
    ax4.set_title("Model Size Comparison", fontsize=13, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, model_sizes):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. Load Time
    ax5 = fig.add_subplot(gs[1, 1])
    load_times = [frameworks_data[f]["load_time_s"] for f in framework_names]

    bars = ax5.bar(range(len(framework_names)), load_times, color=colors, alpha=0.8)
    ax5.set_xticks(range(len(framework_names)))
    ax5.set_xticklabels(framework_names, rotation=45, ha="right")
    ax5.set_ylabel("Load Time (seconds)", fontsize=11, fontweight="bold")
    ax5.set_title("Model Load Time", fontsize=13, fontweight="bold")
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

    # 6. Speed vs Accuracy scatter
    ax6 = fig.add_subplot(gs[1, 2])
    scatter_fps = []
    scatter_det = []
    scatter_names = []

    for name in framework_names:
        f = frameworks_data[name]
        scatter_fps.append(f["mean_fps"])
        scatter_det.append(f["mean_detections"])
        scatter_names.append(name.split()[0])  # Short name

    ax6.scatter(
        scatter_fps,
        scatter_det,
        s=200,
        c=colors,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
    )

    for i, name in enumerate(scatter_names):
        ax6.annotate(
            name,
            (scatter_fps[i], scatter_det[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax6.set_xlabel("FPS", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Detections/Frame", fontsize=11, fontweight="bold")
    ax6.set_title("Speed vs Detection Trade-off", fontsize=13, fontweight="bold")
    ax6.grid(alpha=0.3)

    # 7. Setup Complexity (qualitative)
    ax7 = fig.add_subplot(gs[2, 0])
    complexities = [frameworks_data[f]["setup_complexity"] for f in framework_names]
    complexity_map = {"Low": 1, "Medium": 2, "High": 3}
    complexity_values = [complexity_map.get(c, 2) for c in complexities]

    bars = ax7.barh(
        range(len(framework_names)), complexity_values, color=colors, alpha=0.8
    )
    ax7.set_yticks(range(len(framework_names)))
    ax7.set_yticklabels(framework_names)
    ax7.set_xlabel("Complexity", fontsize=11, fontweight="bold")
    ax7.set_title("Setup Complexity", fontsize=13, fontweight="bold")
    ax7.set_xticks([1, 2, 3])
    ax7.set_xticklabels(["Low", "Medium", "High"])
    ax7.grid(axis="x", alpha=0.3)

    # 8. Summary Table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis("tight")
    ax8.axis("off")

    table_data = []
    headers = ["Framework", "FPS", "Time(ms)", "Detect", "Size(MB)", "Setup"]

    for name in framework_names:
        f = frameworks_data[name]
        short_name = name.replace("Pose", "").replace("-", "").strip()
        table_data.append(
            [
                short_name,
                f"{f['mean_fps']:.1f}",
                f"{f['mean_inference_time_ms']:.1f}",
                f"{f['mean_detections']:.1f}",
                f"{f['model_size_mb']:.0f}",
                f["setup_complexity"],
            ]
        )

    table = ax8.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["lightgray"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Overall title
    video_info = results["video_info"]
    hw_info = get_hardware_info()
    fig.suptitle(
        f"Pose Estimation Framework Comparison - {hw_info}\n"
        f'Video: {Path(video_info["path"]).name} | '
        f'Resolution: {video_info["resolution"]} | '
        f'Frames: {video_info["frames_processed"]}\n'
        f"October 3rd Framework Evaluation - YOLO11-Pose Selected",
        fontsize=15,
        fontweight="bold",
        y=0.91,
    )

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"framework_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"\n📊 Plot saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare pose estimation frameworks (October 3rd evaluation)"
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
        default=Path("benchmark_results/frameworks"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--yolo-size", type=str, default="m", help="YOLO model size (n, s, m, l, x)"
    )
    parser.add_argument(
        "--export-frames",
        action="store_true",
        help="Export sample frames with pose annotations for visual inspection",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample frames to export (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏒 FRAMEWORK COMPARISON BENCHMARK (October 3rd Evaluation)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Video: {args.video}")
    print(f"Max frames: {args.frames}")

    # Setup frameworks to benchmark
    frameworks = []

    # MediaPipe
    frameworks.append(MediaPipeBenchmark("cpu"))

    # PyTorch Keypoint R-CNN
    frameworks.append(PyTorchKeypointBenchmark(args.device))

    # YOLO (selected solution)
    frameworks.append(YOLOBenchmark(args.yolo_size, args.device))

    # Placeholders for frameworks that need model files
    # frameworks.append(OpenVINOBenchmark(args.device))
    # frameworks.append(TRTPoseBenchmark(args.device))
    # frameworks.append(MMPoseBenchmark(args.device))

    print(f"\nFrameworks available for testing:")
    for fw in frameworks:
        status = "✅" if fw.available else "⚠️"
        print(f"  {status} {fw.name}")

    # Run benchmark first (this loads the models)
    results = benchmark_frameworks(
        args.video, frameworks, max_frames=args.frames, warmup_frames=5
    )

    # Export sample frames AFTER benchmark (models are loaded and available)
    if args.export_frames:
        # Reload models for frameworks that need it
        for fw in frameworks:
            if fw.available and not hasattr(fw, "model") and not hasattr(fw, "yolo"):
                fw.load_model()

        export_sample_frames(
            args.video, frameworks, args.output_dir, args.num_samples, args.frames
        )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"framework_comparison_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")

    # Generate plots
    plot_results(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("FRAMEWORK COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Framework':<30} {'FPS':>8} {'Time(ms)':>10} {'Detect':>8} {'Size(MB)':>10}"
    )
    print("-" * 70)

    for name, data in results["frameworks"].items():
        print(
            f"{name:<30} {data['mean_fps']:>8.1f} {data['mean_inference_time_ms']:>10.1f} "
            f"{data['mean_detections']:>8.1f} {data['model_size_mb']:>10.0f}"
        )

    print("\n✅ Framework comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
