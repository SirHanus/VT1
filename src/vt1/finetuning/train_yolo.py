"""
Training script for YOLO pose models.
Called by the GUI to train models with proper argument handling.
"""

import logging
import sys
import warnings
from pathlib import Path

from ultralytics import YOLO

try:
    from vt1.config import settings
    from vt1.logger import get_logger
except ImportError:
    settings = None
    get_logger = None

# Initialize logger
if get_logger:
    logger = get_logger(__name__)
else:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)


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

    # Set project directory to runs folder in project root (same level as models/ and outputs/)
    # If not already set, configure it to be at the project root
    if "project" not in kwargs:
        logger.info("=" * 60)
        logger.info("Configuring YOLO training output directory...")
        logger.info("=" * 60)

        cfg = settings() if settings is not None else None
        if cfg:
            runs_dir = cfg.runs_dir
            logger.info("Using config system:")
            logger.info(f"  Project root: {cfg.repo_root}")
            logger.info(f"  Runs directory: {runs_dir}")
        else:
            # Fallback: try to find project root by looking for pyproject.toml
            logger.warning("Config not available, using fallback method...")
            current = Path(__file__).resolve()
            logger.debug(f"  Current file: {current}")

            for parent in [current] + list(current.parents):
                logger.debug(f"  Checking: {parent}")
                if (parent / "pyproject.toml").exists():
                    runs_dir = parent / "runs"
                    logger.info(f"Found project root: {parent}")
                    logger.info(f"  Runs directory: {runs_dir}")
                    break
            else:
                runs_dir = Path("runs").resolve()
                logger.warning(f"Could not find project root, using: {runs_dir}")

        kwargs["project"] = str(runs_dir)
        logger.info(f"Setting YOLO project parameter to: {kwargs['project']}")

        # Ensure the runs directory exists
        runs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Runs directory created/verified: {runs_dir.absolute()}")
        logger.info("=" * 60)

    # Check dataset size and provide recommendations
    train_count = 0
    val_count = 0
    total = 0

    if data_yaml and Path(data_yaml).exists():
        train_count, val_count = count_dataset_images(data_yaml)
        total = train_count + val_count

        logger.info("=" * 60)
        logger.info("Dataset Info:")
        logger.info(f"  Training images: {train_count}")
        logger.info(f"  Validation images: {val_count}")
        logger.info(f"  Total: {total}")
        logger.info("=" * 60)

        if total < 500:
            logger.warning("Small dataset detected!")
            logger.warning(f"  Current dataset: {total} images")
            logger.warning(f"  Recommended minimum: 500+ images")
            logger.warning(f"  Recommended model: {recommend_model(train_count)}")
            logger.warning(f"  Current model: {model_name}")

            if "x-pose" in model_name.lower() or "l-pose" in model_name.lower():
                logger.warning("  Large models (x/l) may overfit on small datasets!")
                logger.warning(
                    "  Consider using yolo11n-pose.pt or yolo11s-pose.pt instead."
                )

            if total < 200:
                logger.warning("  Dataset is very small - training may fail!")
                logger.warning("  Extract more data before training for best results.")

            logger.warning("  Continuing training anyway...")

    # Create model and train
    logger.info(f"Loading model: {model_name}")
    model = YOLO(model_name)

    logger.info("=" * 60)
    logger.info("Starting YOLO Pose Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Training images: {train_count}")
    logger.info(f"Validation images: {val_count}")
    logger.info(f"Epochs: {kwargs.get('epochs', 'default')}")
    logger.info(f"Batch size: {kwargs.get('batch', 'auto')}")
    logger.info(f"Image size: {kwargs.get('imgsz', 640)}")
    logger.info(f"Project dir: {kwargs.get('project', 'runs')}")
    logger.info("=" * 60)
    logger.info("Training in progress... (this may take a while)")
    logger.info("=" * 60)

    try:
        results = model.train(**kwargs)
        logger.info("=" * 60)
        logger.info("[OK] Training complete!")
        logger.info("=" * 60)

        # Print results location
        if hasattr(results, "save_dir"):
            logger.info(f"Results saved to: {results.save_dir}")
            logger.info(f"Best model: {results.save_dir}/weights/best.pt")
            logger.info(f"Last model: {results.save_dir}/weights/last.pt")

        return 0

    except RuntimeError as e:
        error_msg = str(e)

        # Handle specific known errors
        if (
            "fitness collapse" in error_msg.lower()
            or "inplace update" in error_msg.lower()
        ):
            logger.error("=" * 60)
            logger.error("TRAINING FAILED: Fitness Collapse Detected")
            logger.error("=" * 60)
            logger.error("This error occurs when the model fails to learn properly.")
            logger.error("")
            logger.error("Common causes:")
            logger.error("  1. Dataset too small for the model size")
            logger.error("  2. Poor quality keypoint annotations")
            logger.error("  3. All pose keypoints have low visibility")
            logger.error("  4. Model too large for dataset")
            logger.error("")
            logger.error("Recommended solutions:")
            logger.error(
                f"  1. Extract MORE data (current: {total if total > 0 else 'unknown'} images, aim for 500-1000+)"
            )
            logger.error("  2. Use a SMALLER model:")
            if train_count > 0:
                logger.error(f"     -> Recommended: {recommend_model(train_count)}")
            else:
                logger.error("     -> Try: yolo11n-pose.pt or yolo11s-pose.pt")
            logger.error("  3. Verify dataset quality:")
            logger.error("     -> Go to Tab 1, increase --max-players-per-video")
            logger.error("     -> Lower --min-keypoints threshold (try 3 instead of 5)")
            logger.error("     -> Decrease --frame-interval (try 15 instead of 30)")
            logger.error("  4. Review extracted images:")
            logger.error("     -> Delete poor quality images before export")
            logger.error("     -> Ensure keypoints are visible on players")
            logger.error("")
            logger.error("Current configuration:")
            logger.error(
                f"  Dataset: {total} images ({train_count} train, {val_count} val)"
            )
            logger.error(f"  Model: {model_name}")
            logger.error("=" * 60)
            return 1
        else:
            # Re-raise other runtime errors
            raise

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("=" * 60)
        print("[ERROR] Unexpected error during training:")
        print("=" * 60)
        print(str(e))
        print("")
        print("Full traceback:")
        import traceback

        traceback.print_exc()
        print("=" * 60)

        logger.error("Unexpected error during training", exc_info=True)
        sys.exit(1)
