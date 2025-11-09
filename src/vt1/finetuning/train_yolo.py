"""
Training script for YOLO pose models.
Called by the GUI to train models with proper argument handling.
"""

import sys
import warnings
from pathlib import Path

from ultralytics import YOLO


def count_dataset_images(data_yaml: str) -> tuple[int, int]:
    """Count training and validation images from dataset.yaml"""
    import yaml

    try:
        with open(data_yaml, "r") as f:
            data = yaml.safe_load(f)

        base_path = Path(data.get("path", "."))
        train_dir = base_path / data.get("train", "images/train")
        val_dir = base_path / data.get("val", "images/val")

        train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0

        return train_count, val_count
    except Exception:
        return 0, 0


def recommend_model(train_count: int) -> str:
    """Recommend model size based on dataset size."""
    if train_count < 200:
        return "yolo11n-pose.pt (nano - for very small datasets)"
    elif train_count < 500:
        return "yolo11s-pose.pt (small)"
    elif train_count < 1000:
        return "yolo11m-pose.pt (medium)"
    else:
        return "yolo11l-pose.pt or yolo11x-pose.pt (large/xlarge)"


def main():
    # Suppress some warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Parse arguments
    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            kwargs[key] = value

    # Extract model parameter
    model_name = kwargs.pop("model", "yolo11n-pose.pt")
    data_yaml = kwargs.get("data", "")

    # Check dataset size and provide recommendations
    train_count = 0
    val_count = 0
    total = 0

    if data_yaml and Path(data_yaml).exists():
        train_count, val_count = count_dataset_images(data_yaml)
        total = train_count + val_count

        print(f"\n{'='*60}")
        print(f"Dataset Info:")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
        print(f"  Total: {total}")
        print(f"{'='*60}\n")

        if total < 500:
            print("⚠️  WARNING: Small dataset detected!")
            print(f"   Current dataset: {total} images")
            print(f"   Recommended minimum: 500+ images")
            print(f"   Recommended model: {recommend_model(train_count)}")
            print(f"   Current model: {model_name}")

            if "x-pose" in model_name.lower() or "l-pose" in model_name.lower():
                print("\n   ⚠️  Large models (x/l) may overfit on small datasets!")
                print("   Consider using yolo11n-pose.pt or yolo11s-pose.pt instead.\n")

            if total < 200:
                print("\n   ⚠️  Dataset is very small - training may fail!")
                print("   Extract more data before training for best results.\n")

            print("   Continuing training anyway...\n")

    # Create model and train
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training with parameters: {kwargs}")

    try:
        results = model.train(**kwargs)
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print("=" * 60)

        # Print results location
        if hasattr(results, "save_dir"):
            print(f"Results saved to: {results.save_dir}")
            print(f"Best model: {results.save_dir}/weights/best.pt")

        return 0

    except RuntimeError as e:
        error_msg = str(e)

        # Handle specific known errors
        if (
            "fitness collapse" in error_msg.lower()
            or "inplace update" in error_msg.lower()
        ):
            print("\n" + "=" * 60)
            print("❌ TRAINING FAILED: Fitness Collapse Detected")
            print("=" * 60)
            print("\nThis error occurs when the model fails to learn properly.")
            print("\nCommon causes:")
            print("  1. Dataset too small for the model size")
            print("  2. Poor quality keypoint annotations")
            print("  3. All pose keypoints have low visibility")
            print("  4. Model too large for dataset")
            print("\n🔧 Recommended solutions:")
            print(
                "  1. Extract MORE data (current: {} images, aim for 500-1000+)".format(
                    total if total > 0 else "unknown"
                )
            )
            print("  2. Use a SMALLER model:")
            if train_count > 0:
                print(f"     → Recommended: {recommend_model(train_count)}")
            else:
                print("     → Try: yolo11n-pose.pt or yolo11s-pose.pt")
            print("  3. Verify dataset quality:")
            print("     → Go to Tab 1, increase --max-players-per-video")
            print("     → Lower --min-keypoints threshold (try 3 instead of 5)")
            print("     → Decrease --frame-interval (try 15 instead of 30)")
            print("  4. Review extracted images:")
            print("     → Delete poor quality images before export")
            print("     → Ensure keypoints are visible on players")
            print("\nCurrent configuration:")
            print(f"  Dataset: {total} images ({train_count} train, {val_count} val)")
            print(f"  Model: {model_name}")
            print("=" * 60)
            return 1
        else:
            # Re-raise other runtime errors
            raise

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ Unexpected error during training:")
        print("=" * 60)
        print(str(e))
        print("\nFull traceback:")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 60)
        sys.exit(1)
