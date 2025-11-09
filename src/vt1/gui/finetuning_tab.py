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

from PyQt6 import QtCore, QtGui, QtWidgets

from vt1.config import settings


class FinetuningTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Process tracking
        self.extract_proc: Optional[QtCore.QProcess] = None
        self.export_proc: Optional[QtCore.QProcess] = None
        self.train_proc: Optional[QtCore.QProcess] = None

        # Build UI
        vlay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        vlay.addWidget(self.tabs)

        # Add tabs
        self.tabs.addTab(self._make_extract_tab(), "1. Extract Dataset")
        self.tabs.addTab(self._make_export_tab(), "2. Export YOLO")
        self.tabs.addTab(self._make_train_tab(), "3. Train Model")
        self.tabs.addTab(self._make_manage_tab(), "4. Manage Models")

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

        # Videos directory (for --process-all)
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

        # Load defaults
        self._load_extract_defaults()

        return widget

    # ========== TAB 2: EXPORT YOLO ==========
    def _make_export_tab(self) -> QtWidgets.QWidget:
        """Tab for exporting dataset to YOLO format."""
        widget = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(widget)

        # Info label
        info = QtWidgets.QLabel(
            "Export the extracted dataset to YOLO pose format after reviewing and cleaning images.\n"
            "Review images in: <output_dir>/review/players/\n"
            "Delete poor quality images before exporting."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { padding: 10px; background-color: #e3f2fd; border-radius: 5px; }"
        )
        vlay.addWidget(info)

        # Configuration
        config_group = QtWidgets.QGroupBox("Export Configuration")
        form = QtWidgets.QFormLayout()

        # Dataset directory (same as extraction output)
        self.export_dataset_ed = QtWidgets.QLineEdit()
        self.export_dataset_btn = QtWidgets.QPushButton("Browse...")
        self.export_dataset_btn.clicked.connect(self._browse_export_dataset)
        form.addRow(
            "Dataset Directory:",
            self._hrow(self.export_dataset_ed, self.export_dataset_btn),
        )

        # Train/val split
        self.export_split_dsb = QtWidgets.QDoubleSpinBox()
        self.export_split_dsb.setRange(0.5, 0.95)
        self.export_split_dsb.setSingleStep(0.05)
        self.export_split_dsb.setDecimals(2)
        self.export_split_dsb.setToolTip(
            "Fraction of data for training (rest goes to validation)"
        )
        form.addRow("Train Split:", self.export_split_dsb)

        config_group.setLayout(form)
        vlay.addWidget(config_group)

        # Status
        status_group = QtWidgets.QGroupBox("Export Status")
        status_lay = QtWidgets.QVBoxLayout()

        self.export_progress = QtWidgets.QProgressBar()
        self.export_progress.setRange(0, 100)
        status_lay.addWidget(self.export_progress)

        self.export_status_lbl = QtWidgets.QLabel("Ready")
        status_lay.addWidget(self.export_status_lbl)

        self.export_log = QtWidgets.QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMaximumBlockCount(1000)
        self.export_log.setFont(QtGui.QFont("Courier New", 9))
        status_lay.addWidget(self.export_log)

        status_group.setLayout(status_lay)
        vlay.addWidget(status_group)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.export_run_btn = QtWidgets.QPushButton("Export to YOLO Format")
        self.export_run_btn.clicked.connect(self._run_export)
        self.export_stop_btn = QtWidgets.QPushButton("Stop")
        self.export_stop_btn.setEnabled(False)
        self.export_stop_btn.clicked.connect(self._stop_export)
        self.export_open_btn = QtWidgets.QPushButton("Open Dataset Folder")
        self.export_open_btn.clicked.connect(self._open_dataset_folder)

        btn_row.addWidget(self.export_run_btn)
        btn_row.addWidget(self.export_stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.export_open_btn)
        vlay.addLayout(btn_row)

        # Load defaults
        self._load_export_defaults()

        return widget

    # ========== TAB 3: TRAIN MODEL ==========
    def _make_train_tab(self) -> QtWidgets.QWidget:
        """Tab for training YOLO pose model."""
        widget = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(widget)

        # Info
        info = QtWidgets.QLabel(
            "Train a YOLO11 pose model on your hockey player dataset.\n"
            "Make sure you've exported the dataset first (Tab 2)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "QLabel { padding: 10px; background-color: #e8f5e9; border-radius: 5px; }"
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

    def _load_export_defaults(self):
        """Load default values for export tab."""
        cfg = settings()
        self.export_dataset_ed.setText(str(cfg.finetuning_output_dir))
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

    def _browse_export_dataset(self):
        """Browse for dataset directory to export."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", self.export_dataset_ed.text()
        )
        if path:
            self.export_dataset_ed.setText(path)

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
            "--process-all",
            "--output-dir",
            self.extract_output_ed.text(),
            "--model",
            self.extract_model_ed.text(),
            "--max-players-per-video",
            str(self.extract_players_sb.value()),
            "--frame-interval",
            str(self.extract_interval_sb.value()),
            "--detection-conf",
            str(self.extract_conf_dsb.value()),
            "--min-keypoints",
            str(self.extract_keypoints_sb.value()),
        ]

        # Update videos_dir in extractor
        os.environ["VT1_VIDEOS_DIR"] = self.extract_videos_ed.text()

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
                "then proceed to Tab 2 to export the dataset.",
            )
        else:
            self.extract_status_lbl.setText(f"Extraction failed (code {code})")

    # ========== EXPORT METHODS ==========
    def _run_export(self):
        """Run dataset export."""
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
            self.export_dataset_ed.text(),
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
                "Proceed to Tab 3 to train a model on this dataset.",
            )
        else:
            self.export_status_lbl.setText(f"Export failed (code {code})")

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
                "Please export the dataset first (Tab 2).",
            )
            return

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
        py = sys.executable or "python"
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
            f"[GUI] Command: {py} {' '.join(self._quote(a) for a in args)}\n"
        )

        # Start process
        try:
            proc.start(py, args)
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
        """Handle process output."""
        if proc is None:
            return

        data = proc.readAllStandardOutput()
        text = bytes(data.data()).decode("utf-8", "ignore")

        # Append to log
        log.appendPlainText(text.rstrip())
        log.verticalScrollBar().setValue(log.verticalScrollBar().maximum())

        # Try to extract progress info
        # Look for patterns like "Epoch X/Y" or "X/Y images"
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", text)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            progress.setRange(0, total)
            progress.setValue(current)
            status_lbl.setText(f"Training epoch {current}/{total}")

        # Look for "Processed X/Y"
        proc_match = re.search(r"Processed\s+(\d+)/(\d+)", text)
        if proc_match:
            current = int(proc_match.group(1))
            total = int(proc_match.group(2))
            progress.setRange(0, total)
            progress.setValue(current)

    # ========== FOLDER OPENING METHODS ==========
    def _open_review_folder(self):
        """Open review folder in file explorer."""
        review_dir = Path(self.extract_output_ed.text()) / "review" / "players"
        self._open_folder(review_dir)

    def _open_dataset_folder(self):
        """Open dataset folder in file explorer."""
        dataset_dir = Path(self.export_dataset_ed.text())
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
