"""
Fine-tuning Tab for VT1 GUI.

Provides interface for:
1. Extracting hockey player dataset from videos
2. Exporting dataset in YOLO format
3. Training YOLO pose models
4. Managing fine-tuned models
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Matplotlib imports for training metrics chart
import matplotlib
from PyQt6 import QtCore, QtGui, QtWidgets

matplotlib.use("Qt5Agg")  # Use Qt5Agg backend for PyQt6 compatibility
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from vt1.config import settings


class TrainingMetricsCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for real-time training metrics visualization."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Configure the plot
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.set_title("Training & Validation Metrics")
        self.axes.grid(True, alpha=0.3)
        self.axes.legend()

        # Initialize empty lines
        (self.train_box_line,) = self.axes.plot(
            [], [], "b-", label="Train Box Loss", linewidth=2
        )
        (self.train_pose_line,) = self.axes.plot(
            [], [], "r-", label="Train Pose Loss", linewidth=2
        )
        (self.val_box_line,) = self.axes.plot(
            [], [], "b--", label="Val Box Loss", linewidth=2
        )
        (self.val_pose_line,) = self.axes.plot(
            [], [], "r--", label="Val Pose Loss", linewidth=2
        )

        self.axes.legend()
        self.fig.tight_layout()

    def update_plot(self, epochs, train_box, train_pose, val_box, val_pose):
        """Update the plot with new data."""
        if not epochs:
            return

        # Update data
        self.train_box_line.set_data(epochs, train_box)
        self.train_pose_line.set_data(epochs, train_pose)

        if val_box:
            self.val_box_line.set_data(epochs[: len(val_box)], val_box)
        if val_pose:
            self.val_pose_line.set_data(epochs[: len(val_pose)], val_pose)

        # Adjust axes limits
        self.axes.relim()
        self.axes.autoscale_view()

        # Redraw
        self.draw()

    def clear_plot(self):
        """Clear the plot data."""
        self.train_box_line.set_data([], [])
        self.train_pose_line.set_data([], [])
        self.val_box_line.set_data([], [])
        self.val_pose_line.set_data([], [])
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()


class FinetuningTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Process tracking
        self.extract_proc: Optional[QtCore.QProcess] = None
        self.export_proc: Optional[QtCore.QProcess] = None
        self.ls_proc: Optional[QtCore.QProcess] = None
        self.train_proc: Optional[QtCore.QProcess] = None

        # Training metrics tracking
        self.train_epochs = []
        self.train_box_loss = []
        self.train_pose_loss = []
        self.val_box_loss = []
        self.val_pose_loss = []
        # Optional canvas for training metrics (may be None if not created)
        self.train_metrics_canvas = None

        # Build UI
        vlay = QtWidgets.QVBoxLayout(self)

        self.disclaimer_lbl = QtWidgets.QLabel(
            "<b>Disclaimer:</b> Fine-tuning is under development and may have side effects."
        )
        self.disclaimer_lbl.setWordWrap(True)
        self.disclaimer_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.disclaimer_lbl.setStyleSheet(
            "QLabel { color: #b71c1c; background-color: #ffebee; "
            "border: 1px solid #f44336; padding: 8px; border-radius: 4px; }"
        )
        vlay.addWidget(self.disclaimer_lbl)

        self.tabs = QtWidgets.QTabWidget()
        vlay.addWidget(self.tabs)

        # Add tabs
        self.tabs.addTab(self._make_extract_tab(), "1. Extract & Format Dataset")
        self.tabs.addTab(self._make_train_tab(), "2. Train Model")
        self.tabs.addTab(self._make_manage_tab(), "3. Manage Models")

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    # ========== TAB 1: EXTRACT DATASET ==========
    def _make_extract_tab(self) -> QtWidgets.QWidget:
        """Tab for extracting player images from videos."""
        widget = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(widget)

        # Configuration group
        config_group = QtWidgets.QGroupBox("Extraction Configuration")
        form = QtWidgets.QFormLayout()

        # Output directory
        self.extract_output_ed = QtWidgets.QLineEdit()
        self.extract_output_btn = QtWidgets.QPushButton("Browse...")
        self.extract_output_btn.clicked.connect(self._browse_extract_output)
        form.addRow(
            "Output Directory:",
            self._hrow(self.extract_output_ed, self.extract_output_btn),
        )

        # Videos directory
        self.extract_videos_ed = QtWidgets.QLineEdit()
        self.extract_videos_btn = QtWidgets.QPushButton("Browse...")
        self.extract_videos_btn.clicked.connect(self._browse_videos_dir)
        form.addRow(
            "Videos Directory:",
            self._hrow(self.extract_videos_ed, self.extract_videos_btn),
        )

        # Pose model
        self.extract_model_ed = QtWidgets.QLineEdit()
        self.extract_model_btn = QtWidgets.QPushButton("Browse...")
        self.extract_model_btn.clicked.connect(self._browse_pose_model)
        form.addRow(
            "Pose Model:", self._hrow(self.extract_model_ed, self.extract_model_btn)
        )

        # Players per video
        self.extract_players_sb = QtWidgets.QSpinBox()
        self.extract_players_sb.setRange(10, 10000)
        self.extract_players_sb.setSingleStep(50)
        self.extract_players_sb.setToolTip(
            "Maximum number of players to extract from each video"
        )
        form.addRow("Players per Video:", self.extract_players_sb)

        # Frame interval
        self.extract_interval_sb = QtWidgets.QSpinBox()
        self.extract_interval_sb.setRange(1, 300)
        self.extract_interval_sb.setToolTip(
            "Extract every Nth frame (lower = more data)"
        )
        form.addRow("Frame Interval:", self.extract_interval_sb)

        # Detection confidence
        self.extract_conf_dsb = QtWidgets.QDoubleSpinBox()
        self.extract_conf_dsb.setRange(0.1, 1.0)
        self.extract_conf_dsb.setSingleStep(0.05)
        self.extract_conf_dsb.setDecimals(2)
        self.extract_conf_dsb.setToolTip("Minimum confidence for player detection")
        form.addRow("Detection Confidence:", self.extract_conf_dsb)

        # Minimum keypoints
        self.extract_keypoints_sb = QtWidgets.QSpinBox()
        self.extract_keypoints_sb.setRange(1, 17)
        self.extract_keypoints_sb.setToolTip(
            "Minimum visible keypoints required per player"
        )
        form.addRow("Min Keypoints:", self.extract_keypoints_sb)

        # Full-frames mode (save full video frames as dataset)
        self.extract_full_frames_cb = QtWidgets.QCheckBox(
            "Save Full Video Frames (disable detection/player extraction)"
        )
        self.extract_full_frames_cb.setToolTip(
            "When checked, saves full frames as the dataset and disables detection/player extraction options"
        )
        self.extract_full_frames_cb.setChecked(True)
        self.extract_full_frames_cb.toggled.connect(self._on_full_frames_toggled)
        form.addRow("Full Frames:", self.extract_full_frames_cb)

        config_group.setLayout(form)
        vlay.addWidget(config_group)

        # Status group
        status_group = QtWidgets.QGroupBox("Extraction Status")
        status_lay = QtWidgets.QVBoxLayout()

        self.extract_progress = QtWidgets.QProgressBar()
        self.extract_progress.setRange(0, 100)
        status_lay.addWidget(self.extract_progress)

        self.extract_status_lbl = QtWidgets.QLabel("Ready")
        status_lay.addWidget(self.extract_status_lbl)

        self.extract_log = QtWidgets.QPlainTextEdit()
        self.extract_log.setReadOnly(True)
        self.extract_log.setMaximumBlockCount(1000)
        self.extract_log.setFont(QtGui.QFont("Courier New", 9))
        status_lay.addWidget(self.extract_log)

        status_group.setLayout(status_lay)
        vlay.addWidget(status_group)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.extract_run_btn = QtWidgets.QPushButton("Extract Dataset")
        self.extract_run_btn.clicked.connect(self._run_extract)
        self.extract_stop_btn = QtWidgets.QPushButton("Stop")
        self.extract_stop_btn.setEnabled(False)
        self.extract_stop_btn.clicked.connect(self._stop_extract)
        self.extract_open_btn = QtWidgets.QPushButton("Open Review Folder")
        self.extract_open_btn.clicked.connect(self._open_review_folder)

        btn_row.addWidget(self.extract_run_btn)
        btn_row.addWidget(self.extract_stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.extract_open_btn)
        vlay.addLayout(btn_row)

        # ========== Format data for YOLO section ==========
        export_group = QtWidgets.QGroupBox("Format data for YOLO")
        export_vlay = QtWidgets.QVBoxLayout()

        info = QtWidgets.QLabel(
            "After extracting and reviewing images (in review/players/ folder), "
            "export to YOLO pose format for training. Delete poor quality images before formatting."
        )

        info.setWordWrap(True)
        info.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #e3f2fd;   /* light info background */
                border-radius: 4px;
                color: #0b192f;              /* dark text for contrast */
                font-size: 11pt;             /* slightly larger text */
            }
        """
        )
        export_vlay.addWidget(info)

        # Configuration
        export_form = QtWidgets.QFormLayout()

        # Train/val split
        self.export_split_dsb = QtWidgets.QDoubleSpinBox()
        self.export_split_dsb.setRange(0.5, 0.95)
        self.export_split_dsb.setSingleStep(0.05)
        self.export_split_dsb.setDecimals(2)
        self.export_split_dsb.setToolTip(
            "Fraction of data for training (rest goes to validation)"
        )
        export_form.addRow("Train Split:", self.export_split_dsb)

        export_vlay.addLayout(export_form)

        # Export status
        self.export_progress = QtWidgets.QProgressBar()
        self.export_progress.setRange(0, 100)
        export_vlay.addWidget(self.export_progress)

        self.export_status_lbl = QtWidgets.QLabel("Ready")
        export_vlay.addWidget(self.export_status_lbl)

        self.export_log = QtWidgets.QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMaximumBlockCount(500)
        self.export_log.setFont(QtGui.QFont("Courier New", 9))
        self.export_log.setMaximumHeight(150)
        export_vlay.addWidget(self.export_log)

        # Export buttons
        export_btn_row = QtWidgets.QHBoxLayout()
        self.export_run_btn = QtWidgets.QPushButton("Format data for YOLO")
        self.export_run_btn.clicked.connect(self._run_export)
        self.export_stop_btn = QtWidgets.QPushButton("Stop")
        self.export_stop_btn.setEnabled(False)
        self.export_stop_btn.clicked.connect(self._stop_export)
        self.export_open_btn = QtWidgets.QPushButton("Open Dataset Folder")
        self.export_open_btn.clicked.connect(self._open_dataset_folder)

        export_btn_row.addWidget(self.export_run_btn)
        export_btn_row.addWidget(self.export_stop_btn)
        export_btn_row.addStretch()
        export_btn_row.addWidget(self.export_open_btn)
        export_vlay.addLayout(export_btn_row)

        export_group.setLayout(export_vlay)
        vlay.addWidget(export_group)

        # ========== Label Studio Export section ==========
        ls_group = QtWidgets.QGroupBox("Export for Label Studio (Under Development)")
        ls_group.setEnabled(False)  # Disable entire section
        ls_vlay = QtWidgets.QVBoxLayout()

        ls_info = QtWidgets.QLabel("⚠️ UNDER DEVELOPMENT ⚠️\n\n")
        ls_info.setWordWrap(True)
        ls_info.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #f5f5f5;   /* light gray background */
                border-radius: 4px;
                color: #757575;              /* gray text */
                font-size: 11pt;             /* slightly larger text */
            }
        """
        )
        ls_vlay.addWidget(ls_info)

        # Configuration
        ls_form = QtWidgets.QFormLayout()

        # Max frames per video
        self.ls_max_frames_sb = QtWidgets.QSpinBox()
        self.ls_max_frames_sb.setRange(10, 500)
        self.ls_max_frames_sb.setSingleStep(10)
        self.ls_max_frames_sb.setValue(50)
        self.ls_max_frames_sb.setToolTip("Maximum number of frames to export per video")
        ls_form.addRow("Max Frames/Video:", self.ls_max_frames_sb)

        # Include predictions checkbox
        self.ls_predictions_cb = QtWidgets.QCheckBox(
            "Include Bounding Boxes & Keypoints"
        )
        self.ls_predictions_cb.setChecked(True)
        self.ls_predictions_cb.setToolTip(
            "Include detected bounding boxes and keypoints as pre-annotations"
        )
        ls_form.addRow("Pre-annotations:", self.ls_predictions_cb)

        ls_vlay.addLayout(ls_form)

        # Export status
        self.ls_progress = QtWidgets.QProgressBar()
        self.ls_progress.setRange(0, 100)
        ls_vlay.addWidget(self.ls_progress)

        self.ls_status_lbl = QtWidgets.QLabel("Ready")
        ls_vlay.addWidget(self.ls_status_lbl)

        self.ls_log = QtWidgets.QPlainTextEdit()
        self.ls_log.setReadOnly(True)
        self.ls_log.setMaximumBlockCount(500)
        self.ls_log.setFont(QtGui.QFont("Courier New", 9))
        self.ls_log.setMaximumHeight(150)
        ls_vlay.addWidget(self.ls_log)

        # Label Studio export buttons
        ls_btn_row = QtWidgets.QHBoxLayout()
        self.ls_run_btn = QtWidgets.QPushButton("Export for Label Studio")
        self.ls_run_btn.clicked.connect(self._run_label_studio_export)
        self.ls_stop_btn = QtWidgets.QPushButton("Stop")
        self.ls_stop_btn.setEnabled(False)
        self.ls_stop_btn.clicked.connect(self._stop_label_studio_export)
        self.ls_open_btn = QtWidgets.QPushButton("Open Label Studio Folder")
        self.ls_open_btn.clicked.connect(self._open_label_studio_folder)

        ls_btn_row.addWidget(self.ls_run_btn)
        ls_btn_row.addWidget(self.ls_stop_btn)
        ls_btn_row.addStretch()
        ls_btn_row.addWidget(self.ls_open_btn)
        ls_vlay.addLayout(ls_btn_row)

        ls_group.setLayout(ls_vlay)
        vlay.addWidget(ls_group)

        # Load defaults
        self._load_extract_defaults()

        return widget

    # ========== TAB 2: TRAIN MODEL ==========
    def _make_train_tab(self) -> QtWidgets.QWidget:
        """Tab for training YOLO pose model."""
        widget = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(widget)

        # Info
        info = QtWidgets.QLabel(
            "Train a YOLO11 pose model on your hockey player dataset.\n"
            "Make sure you've formatted the dataset for YOLO first (Tab 1)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                background-color: #e3f2fd;   /* light info background */
                border-radius: 4px;
                color: #0b192f;              /* dark text for contrast */
                font-size: 11pt;             /* slightly larger text */
            }
        """
        )
        vlay.addWidget(info)

        # Configuration
        config_group = QtWidgets.QGroupBox("Training Configuration")
        form = QtWidgets.QFormLayout()

        # Dataset YAML
        self.train_data_ed = QtWidgets.QLineEdit()
        self.train_data_btn = QtWidgets.QPushButton("Browse...")
        self.train_data_btn.clicked.connect(self._browse_dataset_yaml)
        form.addRow(
            "Dataset YAML:", self._hrow(self.train_data_ed, self.train_data_btn)
        )

        # Base model
        self.train_model_cb = QtWidgets.QComboBox()
        self.train_model_cb.addItems(
            [
                "yolo11n-pose.pt",
                "yolo11s-pose.pt",
                "yolo11m-pose.pt",
                "yolo11l-pose.pt",
                "yolo11x-pose.pt",
            ]
        )
        self.train_model_cb.setToolTip(
            "Base model to fine-tune (n=nano, s=small, m=medium, l=large, x=xlarge)"
        )
        form.addRow("Base Model:", self.train_model_cb)

        # Epochs
        self.train_epochs_sb = QtWidgets.QSpinBox()
        self.train_epochs_sb.setRange(1, 1000)
        self.train_epochs_sb.setToolTip("Number of training epochs")
        form.addRow("Epochs:", self.train_epochs_sb)

        # Batch size
        self.train_batch_sb = QtWidgets.QSpinBox()
        self.train_batch_sb.setRange(1, 128)
        self.train_batch_sb.setToolTip("Batch size (lower if out of memory)")
        form.addRow("Batch Size:", self.train_batch_sb)

        # Image size
        self.train_imgsz_sb = QtWidgets.QSpinBox()
        self.train_imgsz_sb.setRange(320, 1280)
        self.train_imgsz_sb.setSingleStep(32)
        self.train_imgsz_sb.setToolTip("Image size for training")
        form.addRow("Image Size:", self.train_imgsz_sb)

        # Device
        self.train_device_cb = QtWidgets.QComboBox()
        self.train_device_cb.addItems(["cuda", "cpu", "0", "1", "2", "3"])
        self.train_device_cb.setToolTip("Device for training (cuda/cpu or GPU index)")
        form.addRow("Device:", self.train_device_cb)

        # Model name
        self.train_name_ed = QtWidgets.QLineEdit()
        self.train_name_ed.setPlaceholderText("hockey_pose_v1")
        self.train_name_ed.setToolTip(
            "Name for this training run (saved in runs/pose/<name>)"
        )
        form.addRow("Run Name:", self.train_name_ed)

        # Patience
        self.train_patience_sb = QtWidgets.QSpinBox()
        self.train_patience_sb.setRange(10, 500)
        self.train_patience_sb.setToolTip(
            "Early stopping patience (epochs without improvement)"
        )
        form.addRow("Patience:", self.train_patience_sb)

        config_group.setLayout(form)
        vlay.addWidget(config_group)

        # Status
        status_group = QtWidgets.QGroupBox("Training Status")
        status_lay = QtWidgets.QVBoxLayout()

        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.setRange(0, 100)
        status_lay.addWidget(self.train_progress)

        self.train_status_lbl = QtWidgets.QLabel("Ready")
        status_lay.addWidget(self.train_status_lbl)

        # Add training metrics chart
        # self.train_metrics_canvas = TrainingMetricsCanvas(
        #     self, width=8, height=3, dpi=100
        # )
        # status_lay.addWidget(self.train_metrics_canvas)
        self.train_log = QtWidgets.QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(1000)
        self.train_log.setFont(QtGui.QFont("Courier New", 9))
        status_lay.addWidget(self.train_log)

        status_group.setLayout(status_lay)
        vlay.addWidget(status_group)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.train_run_btn = QtWidgets.QPushButton("Start Training")
        self.train_run_btn.clicked.connect(self._run_train)
        self.train_stop_btn = QtWidgets.QPushButton("Stop")
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.clicked.connect(self._stop_train)
        self.train_tensorboard_btn = QtWidgets.QPushButton("Open TensorBoard")
        self.train_tensorboard_btn.clicked.connect(self._open_tensorboard)

        btn_row.addWidget(self.train_run_btn)
        btn_row.addWidget(self.train_stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.train_tensorboard_btn)
        vlay.addLayout(btn_row)

        # Load defaults
        self._load_train_defaults()

        return widget

    # ========== TAB 4: MANAGE MODELS ==========
    def _make_manage_tab(self) -> QtWidgets.QWidget:
        """Tab for managing fine-tuned models."""
        widget = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(widget)

        # Info
        info = QtWidgets.QLabel(
            "Manage your fine-tuned models. Copy trained models to the models directory for easy access."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { padding: 10px; background-color: #fff3e0; border-radius: 5px; }"
        )
        vlay.addWidget(info)

        # Model list
        list_group = QtWidgets.QGroupBox("Trained Models")
        list_lay = QtWidgets.QVBoxLayout()

        self.model_list = QtWidgets.QListWidget()
        self.model_list.setToolTip("Double-click to open model directory")
        self.model_list.itemDoubleClicked.connect(self._open_model_dir)
        list_lay.addWidget(self.model_list)

        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh List")
        refresh_btn.clicked.connect(self._refresh_model_list)
        list_lay.addWidget(refresh_btn)

        list_group.setLayout(list_lay)
        vlay.addWidget(list_group)

        # Actions
        actions_group = QtWidgets.QGroupBox("Model Actions")
        actions_lay = QtWidgets.QVBoxLayout()

        copy_group = QtWidgets.QHBoxLayout()
        self.model_source_ed = QtWidgets.QLineEdit()
        self.model_source_ed.setPlaceholderText("Path to trained model (best.pt)")
        self.model_source_btn = QtWidgets.QPushButton("Browse...")
        self.model_source_btn.clicked.connect(self._browse_trained_model)
        copy_group.addWidget(QtWidgets.QLabel("Model File:"))
        copy_group.addWidget(self.model_source_ed)
        copy_group.addWidget(self.model_source_btn)
        actions_lay.addLayout(copy_group)

        name_group = QtWidgets.QHBoxLayout()
        self.model_dest_ed = QtWidgets.QLineEdit()
        self.model_dest_ed.setPlaceholderText("hockey-pose-v1.pt")
        name_group.addWidget(QtWidgets.QLabel("Save As:"))
        name_group.addWidget(self.model_dest_ed)
        actions_lay.addLayout(name_group)

        copy_btn = QtWidgets.QPushButton("Copy Model to Models Directory")
        copy_btn.clicked.connect(self._copy_model)
        actions_lay.addWidget(copy_btn)

        actions_group.setLayout(actions_lay)
        vlay.addWidget(actions_group)

        # Open folders
        folders_group = QtWidgets.QGroupBox("Quick Access")
        folders_lay = QtWidgets.QHBoxLayout()

        open_runs_btn = QtWidgets.QPushButton("Open Training Runs Folder")
        open_runs_btn.clicked.connect(self._open_runs_folder)
        open_models_btn = QtWidgets.QPushButton("Open Models Folder")
        open_models_btn.clicked.connect(self._open_models_folder)

        folders_lay.addWidget(open_runs_btn)
        folders_lay.addWidget(open_models_btn)

        folders_group.setLayout(folders_lay)
        vlay.addWidget(folders_group)

        vlay.addStretch()

        # Initial refresh
        self._refresh_model_list()

        return widget

    # ========== HELPER METHODS ==========
    def _hrow(self, *widgets) -> QtWidgets.QWidget:
        """Create horizontal layout with widgets."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            lay.addWidget(widget)
        return w

    def _quote(self, s: str) -> str:
        """Quote string if it contains spaces."""
        if " " in s:
            return f'"{s}"'
        return s

    # ========== LOAD DEFAULTS ==========
    def _load_extract_defaults(self):
        """Load default values for extraction tab."""
        cfg = settings()
        self.extract_output_ed.setText(str(cfg.finetuning_output_dir))
        self.extract_videos_ed.setText(str(cfg.repo_root / "videos_all"))
        self.extract_model_ed.setText(str(cfg.pose_model))
        self.extract_players_sb.setValue(cfg.finetuning_max_players_per_video)
        self.extract_interval_sb.setValue(cfg.finetuning_frame_interval)
        self.extract_conf_dsb.setValue(cfg.finetuning_detection_conf)
        self.extract_keypoints_sb.setValue(cfg.finetuning_min_keypoints)
        # Ensure full-frames checkbox default (unchecked)
        try:
            self.extract_full_frames_cb.setChecked(False)
        except Exception:
            pass
        # Also load export defaults
        self.export_split_dsb.setValue(cfg.finetuning_train_split)

    def _load_train_defaults(self):
        """Load default values for training tab."""
        cfg = settings()
        dataset_yaml = cfg.finetuning_output_dir / "dataset.yaml"
        self.train_data_ed.setText(str(dataset_yaml))
        self.train_model_cb.setCurrentText(
            "yolo11n-pose.pt"
        )  # Default to nano for small datasets
        self.train_epochs_sb.setValue(cfg.finetuning_epochs)
        self.train_batch_sb.setValue(cfg.finetuning_batch)
        self.train_imgsz_sb.setValue(cfg.finetuning_imgsz)
        self.train_device_cb.setCurrentText("cuda")
        self.train_name_ed.setText("hockey_pose_v1")
        self.train_patience_sb.setValue(50)

    def _on_full_frames_toggled(self, checked: bool):
        """Enable/disable detection/player-related fields when full-frames is toggled.

        When full-frames is ON we disable model, players-per-video, detection confidence,
        and min-keypoints inputs because they are not used in full-frame extraction.
        """
        widgets = [
            self.extract_model_ed,
            self.extract_model_btn,
            self.extract_players_sb,
            self.extract_conf_dsb,
            self.extract_keypoints_sb,
        ]

        for w in widgets:
            try:
                w.setEnabled(not checked)
            except Exception:
                pass

        # Update status label for clarity
        if checked:
            self.extract_status_lbl.setText(
                "Full-frames mode: detection/player extraction disabled"
            )
        else:
            self.extract_status_lbl.setText("Ready")

    # ========== BROWSE METHODS ==========
    def _browse_extract_output(self):
        """Browse for extraction output directory."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.extract_output_ed.text()
        )
        if path:
            self.extract_output_ed.setText(path)

    def _browse_videos_dir(self):
        """Browse for videos directory."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Videos Directory", self.extract_videos_ed.text()
        )
        if path:
            self.extract_videos_ed.setText(path)

    def _browse_pose_model(self):
        """Browse for pose model file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Pose Model",
            str(settings().models_dir),
            "Model Files (*.pt *.onnx);;All Files (*.*)",
        )
        if path:
            self.extract_model_ed.setText(path)

    def _browse_dataset_yaml(self):
        """Browse for dataset.yaml file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Dataset YAML",
            self.train_data_ed.text(),
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
        )
        if path:
            self.train_data_ed.setText(path)

    def _browse_trained_model(self):
        """Browse for trained model file."""
        start_dir = str(self._repo_root() / "runs" / "pose")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Trained Model",
            start_dir,
            "Model Files (*.pt);;All Files (*.*)",
        )
        if path:
            self.model_source_ed.setText(path)

    # ========== EXTRACTION METHODS ==========
    def _run_extract(self):
        """Run dataset extraction."""
        if self.extract_proc is not None:
            QtWidgets.QMessageBox.warning(
                self, "Already Running", "Stop the current extraction first."
            )
            return

        # Build command
        args = [
            "-m",
            "vt1.finetuning.extract_dataset",
            "--videos-dir",
            self.extract_videos_ed.text(),
            "--output-dir",
            self.extract_output_ed.text(),
            "--frame-interval",
            str(self.extract_interval_sb.value()),
        ]

        # If full-frames mode is selected, instruct CLI to save full frames and
        # skip detection/player-specific arguments. Otherwise include detection args.
        if (
            getattr(self, "extract_full_frames_cb", None)
            and self.extract_full_frames_cb.isChecked()
        ):
            args.extend(["--full-frames"])
        else:
            # include detection/player options
            args.extend(
                [
                    "--model",
                    self.extract_model_ed.text(),
                    "--max-players-per-video",
                    str(self.extract_players_sb.value()),
                    "--detection-conf",
                    str(self.extract_conf_dsb.value()),
                    "--min-keypoints",
                    str(self.extract_keypoints_sb.value()),
                ]
            )

        self._start_process(
            args,
            self.extract_proc,
            self.extract_log,
            self.extract_status_lbl,
            self.extract_progress,
            self.extract_run_btn,
            self.extract_stop_btn,
            self._on_extract_finished,
        )

    def _stop_extract(self):
        """Stop extraction process."""
        self._stop_process(
            self.extract_proc,
            self.extract_log,
            self.extract_status_lbl,
            self.extract_progress,
            self.extract_run_btn,
            self.extract_stop_btn,
        )
        self.extract_proc = None

    def _on_extract_finished(self, code: int, status):
        """Handle extraction process finished."""
        self.extract_proc = None
        self.extract_run_btn.setEnabled(True)
        self.extract_stop_btn.setEnabled(False)
        self.extract_progress.setRange(0, 100)
        self.extract_progress.setValue(100 if code == 0 else 0)

        if code == 0:
            self.extract_status_lbl.setText("Extraction complete!")
            QtWidgets.QMessageBox.information(
                self,
                "Complete",
                "Dataset extraction complete!\n\n"
                "Review images in the review/players folder, delete poor quality images, "
                "then use 'Format data for YOLO' below to prepare the dataset for training.",
            )
        else:
            self.extract_status_lbl.setText(f"Extraction failed (code {code})")

    # ========== EXPORT METHODS ==========
    def _run_export(self):
        """Run dataset export to YOLO format."""
        if self.export_proc is not None:
            QtWidgets.QMessageBox.warning(
                self, "Already Running", "Stop the current export first."
            )
            return

        args = [
            "-m",
            "vt1.finetuning.extract_dataset",
            "--export",
            "--output-dir",
            self.extract_output_ed.text(),  # Use the same output dir as extraction
            "--train-split",
            str(self.export_split_dsb.value()),
        ]

        self._start_process(
            args,
            self.export_proc,
            self.export_log,
            self.export_status_lbl,
            self.export_progress,
            self.export_run_btn,
            self.export_stop_btn,
            self._on_export_finished,
        )

    def _stop_export(self):
        """Stop export process."""
        self._stop_process(
            self.export_proc,
            self.export_log,
            self.export_status_lbl,
            self.export_progress,
            self.export_run_btn,
            self.export_stop_btn,
        )
        self.export_proc = None

    def _on_export_finished(self, code: int, status):
        """Handle export process finished."""
        self.export_proc = None
        self.export_run_btn.setEnabled(True)
        self.export_stop_btn.setEnabled(False)
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(100 if code == 0 else 0)

        if code == 0:
            self.export_status_lbl.setText("Export complete!")
            QtWidgets.QMessageBox.information(
                self,
                "Complete",
                "Dataset exported successfully!\n\n"
                "Proceed to Tab 2 to train a model on this dataset.",
            )
        else:
            self.export_status_lbl.setText(f"Export failed (code {code})")

    # ========== LABEL STUDIO EXPORT METHODS ==========
    def _run_label_studio_export(self):
        """Run Label Studio export."""
        if self.ls_proc is not None:
            QtWidgets.QMessageBox.warning(
                self, "Already Running", "Stop the current export first."
            )
            return

        # Build command
        args = [
            "-m",
            "vt1.finetuning.extract_dataset",
            "--export-label-studio",
            "--videos-dir",
            self.extract_videos_ed.text(),
            "--output-dir",
            self.extract_output_ed.text(),
            "--max-frames-per-video",
            str(self.ls_max_frames_sb.value()),
            "--frame-interval",
            str(self.extract_interval_sb.value()),
        ]

        # Add --no-predictions flag if checkbox is unchecked
        if not self.ls_predictions_cb.isChecked():
            args.append("--no-predictions")

        self._start_process(
            args,
            self.ls_proc,
            self.ls_log,
            self.ls_status_lbl,
            self.ls_progress,
            self.ls_run_btn,
            self.ls_stop_btn,
            self._on_label_studio_export_finished,
        )

    def _stop_label_studio_export(self):
        """Stop Label Studio export process."""
        self._stop_process(
            self.ls_proc,
            self.ls_log,
            self.ls_status_lbl,
            self.ls_progress,
            self.ls_run_btn,
            self.ls_stop_btn,
        )
        self.ls_proc = None

    def _on_label_studio_export_finished(self, code: int, status):
        """Handle Label Studio export process finished."""
        self.ls_proc = None
        self.ls_run_btn.setEnabled(True)
        self.ls_stop_btn.setEnabled(False)
        self.ls_progress.setRange(0, 100)
        self.ls_progress.setValue(100 if code == 0 else 0)

        if code == 0:
            self.ls_status_lbl.setText("Label Studio export complete!")

            # Get the label studio folder path
            output_dir = Path(self.extract_output_ed.text())
            ls_dir = output_dir / "label_studio"
            readme_path = ls_dir / "README.md"

            msg = "Label Studio export complete!\n\n"
            if readme_path.exists():
                msg += f"See {readme_path.name} for import instructions."
            else:
                msg += "Files exported to label_studio/ folder."

            QtWidgets.QMessageBox.information(
                self,
                "Complete",
                msg,
            )
        else:
            self.ls_status_lbl.setText(f"Export failed (code {code})")

    def _open_label_studio_folder(self):
        """Open Label Studio export folder in file explorer."""
        output_dir = Path(self.extract_output_ed.text())
        ls_dir = output_dir / "label_studio"

        if not ls_dir.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Folder Not Found",
                f"Label Studio export folder not found:\n{ls_dir}\n\n"
                "Run 'Export for Label Studio' first.",
            )
            return

        self._open_folder(ls_dir)

    # ========== TRAINING METHODS ==========
    def _run_train(self):
        """Run model training."""
        if self.train_proc is not None:
            QtWidgets.QMessageBox.warning(
                self, "Already Running", "Stop the current training first."
            )
            return

        # Check if dataset exists
        data_yaml = Path(self.train_data_ed.text())
        if not data_yaml.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Dataset Not Found",
                f"Dataset YAML not found: {data_yaml}\n\n"
                "Please format the dataset for YOLO first (Tab 1).",
            )
            return

        # Clear previous training metrics
        self.train_epochs.clear()
        self.train_box_loss.clear()
        self.train_pose_loss.clear()
        self.val_box_loss.clear()
        self.val_pose_loss.clear()
        if getattr(self, "train_metrics_canvas", None):
            try:
                self.train_metrics_canvas.clear_plot()
            except Exception:
                pass

        # Build command using custom training script
        model = self.train_model_cb.currentText()
        name = self.train_name_ed.text() or "hockey_pose"

        # Use our custom training script that properly invokes YOLO
        args = [
            "-m",
            "vt1.finetuning.train_yolo",
            f"data={self.train_data_ed.text()}",
            f"model={model}",
            f"epochs={self.train_epochs_sb.value()}",
            f"batch={self.train_batch_sb.value()}",
            f"imgsz={self.train_imgsz_sb.value()}",
            f"device={self.train_device_cb.currentText()}",
            f"name={name}",
            f"patience={self.train_patience_sb.value()}",
        ]

        self._start_process(
            args,
            self.train_proc,
            self.train_log,
            self.train_status_lbl,
            self.train_progress,
            self.train_run_btn,
            self.train_stop_btn,
            self._on_train_finished,
        )

    def _stop_train(self):
        """Stop training process."""
        self._stop_process(
            self.train_proc,
            self.train_log,
            self.train_status_lbl,
            self.train_progress,
            self.train_run_btn,
            self.train_stop_btn,
        )
        self.train_proc = None

    def _on_train_finished(self, code: int, status):
        """Handle training process finished."""
        self.train_proc = None
        self.train_run_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(100 if code == 0 else 0)

        if code == 0:
            self.train_status_lbl.setText("Training complete!")
            QtWidgets.QMessageBox.information(
                self,
                "Training Complete",
                "Model training complete! ✅\n\n"
                "Find the trained model in runs/pose/<name>/weights/best.pt\n\n"
                "Use Tab 4 to copy the model to the models directory.",
            )
            self._refresh_model_list()
        else:
            self.train_status_lbl.setText(f"Training failed (code {code})")

            # Check if it's a fitness collapse error
            log_text = self.train_log.toPlainText()
            if (
                "fitness collapse" in log_text.lower()
                or "inplace update" in log_text.lower()
            ):
                msg = QtWidgets.QMessageBox(self)
                msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msg.setWindowTitle("Training Failed - Fitness Collapse")
                msg.setText("❌ Training failed due to fitness collapse.")
                msg.setInformativeText(
                    "This typically happens when:\n"
                    "• Dataset is too small for the model\n"
                    "• Keypoint quality is poor\n\n"
                    "See the log output for detailed recommendations."
                )
                msg.setDetailedText(
                    "Recommended solutions:\n\n"
                    "1. Extract MORE data:\n"
                    "   → Go to Tab 1 (Extract Dataset)\n"
                    "   → Increase 'Players per Video' to 300-500\n"
                    "   → Lower 'Frame Interval' to 15\n"
                    "   → Lower 'Min Keypoints' to 3\n"
                    "   → Click 'Extract Dataset'\n\n"
                    "2. Use a SMALLER model:\n"
                    "   → Change model from yolo11x-pose to yolo11n-pose\n"
                    "   → Or use yolo11s-pose for small datasets\n\n"
                    "3. Verify dataset quality:\n"
                    "   → Review extracted images in review/players/\n"
                    "   → Delete blurry or poor quality images\n"
                    "   → Re-export the dataset (Tab 2)\n\n"
                    "Target: 500-1000+ images for reliable training"
                )
                msg.exec()
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Training Failed",
                    f"Training failed with code {code}.\n\n"
                    "Check the log output for details.",
                )

    # ========== PROCESS MANAGEMENT ==========
    def _start_process(
        self,
        args,
        proc_var,
        log,
        status_lbl,
        progress,
        run_btn,
        stop_btn,
        finished_callback,
    ):
        """Start a subprocess."""
        frozen = getattr(sys, "frozen", False)
        # Existing args expected like ["-m", "vt1.finetuning.extract_dataset", ...]
        if frozen and len(args) >= 2 and args[0] == "-m":
            module = args[1]
            mod_args = args[2:]
            cmd = sys.executable
            launch_args = ["--module-run", module, *mod_args]
        else:
            py = sys.executable or "python"
            cmd = py
            launch_args = args

        proc = QtCore.QProcess(self)

        # Set environment
        env = QtCore.QProcessEnvironment.systemEnvironment()
        src_path = str(self._repo_root() / "src")
        existing = env.value("PYTHONPATH", "")
        sep = ";" if os.name == "nt" else ":"
        env.insert("PYTHONPATH", src_path + (sep + existing if existing else ""))
        proc.setProcessEnvironment(env)

        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(self._repo_root()))
        print(
            f"[GUI] Starting process: {cmd} {' '.join(self._quote(a) for a in launch_args)}"
        )
        # Connect signals
        proc.readyReadStandardOutput.connect(
            lambda: self._on_process_output(proc, log, status_lbl, progress)
        )
        proc.finished.connect(lambda code, status: finished_callback(code, status))

        # Clear log and update UI
        log.clear()
        status_lbl.setText("Starting...")
        progress.setRange(0, 0)  # Indeterminate

        # Log command
        log.appendPlainText(
            f"[GUI] Command: {cmd} {' '.join(self._quote(a) for a in launch_args)}\n"
        )

        # Start process
        try:
            proc.start(cmd, launch_args)
            if not proc.waitForStarted(5000):
                raise RuntimeError("Failed to start process")
        except Exception as e:
            log.appendPlainText(f"[GUI] Failed to start: {e}")
            QtWidgets.QMessageBox.critical(self, "Failed", str(e))
            return

        # Update reference and buttons
        if proc_var is self.extract_proc:
            self.extract_proc = proc
        elif proc_var is self.export_proc:
            self.export_proc = proc
        elif proc_var is self.train_proc:
            self.train_proc = proc

        run_btn.setEnabled(False)
        stop_btn.setEnabled(True)
        status_lbl.setText("Running...")

    def _stop_process(self, proc, log, status_lbl, progress, run_btn, stop_btn):
        """Stop a subprocess."""
        if proc is not None:
            log.appendPlainText("[GUI] Terminating...")
            proc.terminate()
            if not proc.waitForFinished(3000):
                log.appendPlainText("[GUI] Killing...")
                proc.kill()
                proc.waitForFinished(2000)

        run_btn.setEnabled(True)
        stop_btn.setEnabled(False)
        progress.setRange(0, 100)
        progress.setValue(0)
        status_lbl.setText("Stopped")

    def _on_process_output(self, proc, log, status_lbl, progress):
        """Handle process output with enhanced progress tracking."""
        if proc is None:
            return

        data = proc.readAllStandardOutput()
        text = bytes(data.data()).decode("utf-8", "ignore")

        # Append to log
        log.appendPlainText(text.rstrip())
        log.verticalScrollBar().setValue(log.verticalScrollBar().maximum())

        # === EXTRACTION PROGRESS ===
        # Look for "Processing: <video_name>"
        if "Processing:" in text and "=" in text:
            video_match = re.search(r"Processing:\s+(.+)", text)
            if video_match:
                video_name = video_match.group(1).strip()
                status_lbl.setText(f"Extracting from: {video_name}")

        # Look for "Extracted N frames"
        extracted_match = re.search(r"Extracted\s+(\d+)\s+frames", text)
        if extracted_match:
            frames = extracted_match.group(1)
            status_lbl.setText(f"Extracted {frames} frames, detecting players...")

        # Look for "Frame X/Y" during detection
        frame_match = re.search(r"Frame\s+(\d+)/(\d+)", text)
        if frame_match:
            current = int(frame_match.group(1))
            total = int(frame_match.group(2))
            progress.setRange(0, total)
            progress.setValue(current)
            status_lbl.setText(f"Detecting players: frame {current}/{total}")

        # Look for "Total players extracted: X/Y"
        players_match = re.search(r"Total players extracted:\s+(\d+)/(\d+)", text)
        if players_match:
            extracted = int(players_match.group(1))
            max_players = int(players_match.group(2))
            pct = (extracted / max_players * 100) if max_players > 0 else 0
            progress.setRange(0, max_players)
            progress.setValue(extracted)
            status_lbl.setText(
                f"Extracted {extracted}/{max_players} players ({pct:.0f}%)"
            )

        # Look for "Videos processed: X/Y"
        videos_match = re.search(r"Videos processed:\s+(\d+)/(\d+)", text)
        if videos_match:
            current = int(videos_match.group(1))
            total = int(videos_match.group(2))
            progress.setRange(0, total)
            progress.setValue(current)
            status_lbl.setText(f"Processed {current}/{total} videos")

        # Look for "EXTRACTION COMPLETE"
        if "EXTRACTION COMPLETE" in text:
            status_lbl.setText("✓ Extraction complete!")
            progress.setValue(progress.maximum())

        # === EXPORT PROGRESS ===
        # Look for "Exporting YOLO Pose Dataset"
        if "Exporting YOLO Pose Dataset" in text:
            status_lbl.setText("Starting YOLO dataset export...")
            progress.setRange(0, 0)  # Indeterminate

        # Look for "Found N player images"
        found_match = re.search(r"Found\s+(\d+)\s+player images", text)
        if found_match:
            count = found_match.group(1)
            status_lbl.setText(f"Found {count} images to export")

        # Look for "Train: N images"
        train_match = re.search(r"Train:\s+(\d+)\s+images", text)
        if train_match:
            count = train_match.group(1)
            status_lbl.setText(f"Preparing {count} training images...")

        # Look for "Processing train/val split"
        if "Processing train split" in text:
            status_lbl.setText("Exporting training split...")
            progress.setRange(0, 0)
        elif "Processing val split" in text:
            status_lbl.setText("Exporting validation split...")

        # Look for "Processed X/Y images (N skipped)"
        export_prog_match = re.search(
            r"Processed\s+(\d+)/(\d+)\s+images\s+\((\d+)\s+skipped\)", text
        )
        if export_prog_match:
            current = int(export_prog_match.group(1))
            total = int(export_prog_match.group(2))
            skipped = int(export_prog_match.group(3))
            progress.setRange(0, total)
            progress.setValue(current)
            status_lbl.setText(
                f"Exporting: {current}/{total} images ({skipped} skipped)"
            )

        # Look for "Dataset exported successfully"
        if "Dataset exported successfully" in text or "YOLO dataset created" in text:
            status_lbl.setText("✓ Dataset export complete!")
            progress.setValue(progress.maximum())

        # === TRAINING PROGRESS ===
        # Look for "Epoch X/Y" (YOLO training)
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", text)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            progress.setRange(0, total)
            progress.setValue(current)
            status_lbl.setText(f"Training: epoch {current}/{total}")

        # Look for epoch metrics (loss values)
        loss_match = re.search(r"box_loss:\s+([\d.]+).*?pose_loss:\s+([\d.]+)", text)
        if loss_match and epoch_match:
            box_loss = float(loss_match.group(1))
            pose_loss = float(loss_match.group(2))
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            status_lbl.setText(
                f"Epoch {current}/{total} - box: {box_loss:.3f}, pose: {pose_loss:.3f}"
            )

            # Update training metrics for chart
            # if current not in self.train_epochs:
            #     self.train_epochs.append(current)
            #     self.train_box_loss.append(box_loss)
            #     self.train_pose_loss.append(pose_loss)
            #
            #     # Update the chart
            #     self.train_metrics_canvas.update_plot(
            #         self.train_epochs,
            #         self.train_box_loss,
            #         self.train_pose_loss,
            #         self.val_box_loss,
            #         self.val_pose_loss,
            #     )

        # Look for validation metrics
        val_match = re.search(
            r"val/box_loss:\s+([\d.]+).*?val/pose_loss:\s+([\d.]+)", text
        )
        if val_match:
            val_box = float(val_match.group(1))
            val_pose = float(val_match.group(2))
            status_lbl.setText(f"Validation - box: {val_box:.3f}, pose: {val_pose:.3f}")

            # Update validation metrics for chart
            # self.val_box_loss.append(val_box)
            # self.val_pose_loss.append(val_pose)
            #
            # # Update the chart
            # self.train_metrics_canvas.update_plot(
            #     self.train_epochs,
            #     self.train_box_loss,
            #     self.train_pose_loss,
            #     self.val_box_loss,
            #     self.val_pose_loss,
            # )

        # Look for "Training complete" or "Results saved"
        if "Training complete" in text or "Results saved to" in text:
            status_lbl.setText("✓ Training complete!")
            progress.setValue(progress.maximum())

        # Look for early stopping
        if "Stopping training early" in text or "EarlyStopping" in text:
            status_lbl.setText("⚠ Training stopped early (patience reached)")

        # Look for fitness collapse warnings
        if "fitness collapse" in text.lower():
            status_lbl.setText("❌ Training failed: fitness collapse")

        # === GENERAL PROGRESS ===
        # Look for generic "X% complete" or similar
        pct_match = re.search(r"(\d+)%", text)
        if pct_match and "complete" in text.lower():
            pct = int(pct_match.group(1))
            progress.setRange(0, 100)
            progress.setValue(pct)

    # ========== FOLDER OPENING METHODS ==========
    def _open_review_folder(self):
        """Open review folder in file explorer."""
        review_dir = Path(self.extract_output_ed.text()) / "review" / "players"
        self._open_folder(review_dir)

    def _open_dataset_folder(self):
        """Open dataset folder in file explorer."""
        dataset_dir = Path(self.extract_output_ed.text())
        self._open_folder(dataset_dir)

    def _open_runs_folder(self):
        """Open training runs folder."""
        runs_dir = self._repo_root() / "runs" / "pose"
        self._open_folder(runs_dir)

    def _open_models_folder(self):
        """Open models folder."""
        self._open_folder(settings().finetuning_models_dir)

    def _open_folder(self, path: Path):
        """Open folder in system file explorer."""
        import subprocess
        import platform

        path.mkdir(parents=True, exist_ok=True)

        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Cannot Open", f"Failed to open folder:\n{e}"
            )

    def _open_tensorboard(self):
        """Open TensorBoard in browser."""
        runs_dir = self._repo_root() / "runs"
        QtWidgets.QMessageBox.information(
            self,
            "TensorBoard",
            f"To view training progress with TensorBoard, run:\n\n"
            f"tensorboard --logdir {runs_dir}\n\n"
            f"Then open http://localhost:6006 in your browser.",
        )

    # ========== MODEL MANAGEMENT ==========
    def _refresh_model_list(self):
        """Refresh the list of trained models."""
        self.model_list.clear()
        runs_dir = self._repo_root() / "runs" / "pose"

        if not runs_dir.exists():
            return

        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                best_pt = run_dir / "weights" / "best.pt"
                if best_pt.exists():
                    self.model_list.addItem(f"{run_dir.name} → {best_pt}")

    def _open_model_dir(self, item):
        """Open model directory when double-clicked."""
        text = item.text()
        run_name = text.split(" → ")[0]
        model_dir = self._repo_root() / "runs" / "pose" / run_name
        self._open_folder(model_dir)

    def _copy_model(self):
        """Copy trained model to models directory."""
        source = Path(self.model_source_ed.text())
        dest_name = self.model_dest_ed.text()

        if not source.exists():
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", f"Model file not found:\n{source}"
            )
            return

        if not dest_name:
            QtWidgets.QMessageBox.warning(
                self, "No Name", "Please enter a destination filename."
            )
            return

        if not dest_name.endswith(".pt"):
            dest_name += ".pt"

        dest = settings().finetuning_models_dir / dest_name

        try:
            import shutil

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            QtWidgets.QMessageBox.information(
                self, "Success", f"Model copied to:\n{dest}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to copy model:\n{e}")
