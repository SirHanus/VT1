"""
Fine-tuning utilities for YOLO models on hockey-specific data.

This package provides tools to extract, organize, and prepare hockey player
datasets for fine-tuning YOLO models, particularly YOLO11x-pose.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vt1.finetuning.extract_dataset import HockeyPoseDatasetExtractor

__all__ = ["HockeyPoseDatasetExtractor"]


def __getattr__(name):
    """Lazy import to avoid importing modules before runpy executes them."""
    if name == "HockeyPoseDatasetExtractor":
        from vt1.finetuning.extract_dataset import HockeyPoseDatasetExtractor

        return HockeyPoseDatasetExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
