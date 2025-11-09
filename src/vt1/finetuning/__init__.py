"""
Fine-tuning utilities for YOLO models on hockey-specific data.

This package provides tools to extract, organize, and prepare hockey player
datasets for fine-tuning YOLO models, particularly YOLO11x-pose.
"""

from vt1.finetuning.extract_dataset import HockeyPoseDatasetExtractor

__all__ = ["HockeyPoseDatasetExtractor"]
