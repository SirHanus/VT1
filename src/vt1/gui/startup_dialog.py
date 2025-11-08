"""
Startup dialog offering to run the full training workflow with defaults.
"""
from __future__ import annotations
import sys
from typing import Optional

from PyQt6 import QtCore, QtWidgets, QtGui

from vt1.config import settings


class StartupDialog(QtWidgets.QDialog):
    """Dialog shown on first GUI launch to optionally run full training workflow."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.user_choice: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("VT1 - First Run Setup")
        self.setMinimumWidth(600)

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header = QtWidgets.QLabel("<h2>Welcome to VT1!</h2>")
        layout.addWidget(header)

        # Info text
        info = QtWidgets.QLabel(
            "<p>Would you like to run the <b>complete training workflow</b> with default settings?</p>"
            "<p>This will:</p>"
            "<ol>"
            "<li><b>Build training set</b> from videos (extract player crops + embeddings)</li>"
            "<li><b>Cluster</b> using UMAP + KMeans (create team classification models)</li>"
            "<li><b>Run pipeline demo</b> on first video with team coloring enabled</li>"
            "</ol>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Requirements box
        req_group = QtWidgets.QGroupBox("Requirements")
        req_layout = QtWidgets.QVBoxLayout(req_group)

        cfg = settings()
        videos_dir = cfg.training_videos_dir
        videos_exist = videos_dir.exists()

        req_text = (
            f"<b>Training videos directory:</b><br>"
            f"<code>{videos_dir}</code><br><br>"
            f"<b>Status:</b> {'✓ Found' if videos_exist else '✗ Not found'}<br><br>"
            f"<b>Expected files:</b> Video files matching pattern <code>{cfg.videos_glob}</code><br><br>"
            f"<b>Models will be saved to:</b><br>"
            f"<code>{cfg.team_models_dir}</code>"
        )

        req_label = QtWidgets.QLabel(req_text)
        req_label.setWordWrap(True)
        if not videos_exist:
            # Warning style: amber background with dark text
            req_label.setStyleSheet(
                "QLabel { "
                "background-color: #ffc107; "
                "color: #000000; "
                "padding: 10px; "
                "border: 2px solid #ff9800; "
                "border-radius: 4px; "
                "font-weight: bold; "
                "}"
            )
        else:
            # Success style: green background with dark text
            req_label.setStyleSheet(
                "QLabel { "
                "background-color: #4caf50; "
                "color: #ffffff; "
                "padding: 10px; "
                "border: 2px solid #388e3c; "
                "border-radius: 4px; "
                "font-weight: bold; "
                "}"
            )

        req_layout.addWidget(req_label)
        layout.addWidget(req_group)

        # Estimated time
        time_label = QtWidgets.QLabel(
            "<p><i>⏱ Estimated time: 5-30 minutes depending on dataset size and hardware</i></p>"
        )
        time_label.setWordWrap(True)
        layout.addWidget(time_label)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_run = QtWidgets.QPushButton("Run Training Workflow")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self._on_run)
        if not videos_exist:
            self.btn_run.setEnabled(False)
            self.btn_run.setToolTip(f"Videos directory not found: {videos_dir}")

        self.btn_skip = QtWidgets.QPushButton("Skip (I'll configure manually)")
        self.btn_skip.clicked.connect(self._on_skip)

        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_exit.clicked.connect(self._on_exit)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_skip)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_exit)

        layout.addLayout(btn_layout)

    def _on_run(self):
        self.user_choice = "run"
        self.accept()

    def _on_skip(self):
        self.user_choice = "skip"
        self.accept()

    def _on_exit(self):
        self.user_choice = "exit"
        self.reject()


class TrainingWorkflowRunner(QtWidgets.QWidget):
    """Widget that runs the full training workflow (build → cluster) with progress feedback."""

    finished = QtCore.pyqtSignal(bool, str)  # success, message

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.proc: Optional[QtCore.QProcess] = None
        self.current_step = 0
        self.steps = ["build", "cluster", "pipeline"]
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Running Training Workflow")
        self.setMinimumSize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)

        # Progress label
        self.lbl_step = QtWidgets.QLabel("Step 1/3: Building training set...")
        self.lbl_step.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.lbl_step)

        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress)

        # Log output
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        layout.addWidget(self.log, 1)

        # Cancel button
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def start(self):
        """Start the workflow."""
        self.current_step = 0
        self._run_step(0)

    def _run_step(self, step_idx: int):
        """Run a specific step."""
        if step_idx >= len(self.steps):
            # All steps complete
            self._on_complete(True)
            return

        step_name = self.steps[step_idx]
        self.lbl_step.setText(f"Step {step_idx + 1}/{len(self.steps)}: {self._step_label(step_name)}...")

        cfg = settings()

        if step_name == "build":
            module = "vt1.team_clustering.build_training_set"
            args = [
                "--videos-dir", str(cfg.training_videos_dir),
                "--glob", cfg.videos_glob,
                "--fps", str(cfg.build_fps),
                "--out-dir", str(cfg.team_output_dir),
                "--min-crop-size", str(cfg.build_min_crop_size),
                "--batch", str(cfg.build_batch_size),  # Fixed: --batch not --batch-size
                "--central-ratio", str(cfg.central_ratio_default),
                "--siglip", str(cfg.siglip_model),
                "--device", "cuda",
                "--yolo-fallback",  # safer default
                "--yolo-model", str(cfg.yolo_model),
            ]
        elif step_name == "cluster":
            module = "vt1.team_clustering.cluster_umap_kmeans"
            args = [
                "--in-root", str(cfg.team_output_dir),
                "--out-dir", str(cfg.team_output_dir),
                "--models-dir", str(cfg.team_models_dir),
                "--k", str(cfg.cluster_k),
                "--umap-dim", str(cfg.umap_dim),
                "--umap-neighbors", str(cfg.umap_neighbors),
                "--umap-metric", cfg.umap_metric,
                "--umap-min-dist", str(cfg.umap_min_dist),
                "--seed", str(cfg.random_seed),
                "--save-models",
            ]
        elif step_name == "pipeline":
            # Run pipeline demo on first video found in training_videos_dir
            from pathlib import Path
            videos_dir = Path(cfg.training_videos_dir)
            video_files = sorted(videos_dir.glob(cfg.videos_glob))

            if not video_files:
                self._log(f"[ERROR] No videos found in {videos_dir} matching {cfg.videos_glob}\n")
                self._on_complete(False)
                return

            first_video = video_files[0]
            self._log(f"[INFO] Running pipeline demo on: {first_video.name}\n\n")

            module = "vt1.pipeline.sam_offline"
            args = [
                "--source", str(first_video),
                "--pose-model", str(cfg.pose_model),
                "--device", "cuda",
                "--imgsz", str(cfg.yolo_imgsz),
                "--conf", str(cfg.yolo_conf),
                "--max-frames", "300",  # Limit to 300 frames for demo (10 sec at 30fps)
                "--no-sam",  # Disable SAM for faster demo
                "--team-models", str(cfg.team_models_dir),
                "--siglip", str(cfg.siglip_model),
                "--central-ratio", str(cfg.central_ratio_default),
                "--out-dir", str(cfg.pipeline_output_dir),
            ]
        else:
            self._log(f"[ERROR] Unknown step: {step_name}\n")
            self._on_complete(False)
            return

        self._log(f"\n{'=' * 60}\n")
        self._log(f"[STEP {step_idx + 1}] {self._step_label(step_name)}\n")
        self._log(f"{'=' * 60}\n")

        py = sys.executable or "python"
        full_args = ["-m", module] + args

        self._log(f"[CMD] {py} {' '.join(full_args)}\n\n")

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(lambda code, status: self._on_step_finished(step_idx, code, status))

        self.proc.start(py, full_args)
        if not self.proc.waitForStarted(5000):
            self._log("[ERROR] Failed to start process\n")
            self._on_complete(False)

    def _step_label(self, step_name: str) -> str:
        labels = {
            "build": "Building training set (extract crops + embeddings)",
            "cluster": "Clustering (UMAP + KMeans, save models)",
            "pipeline": "Running pipeline demo with team coloring",
        }
        return labels.get(step_name, step_name)

    def _on_output(self):
        if self.proc:
            data = self.proc.readAllStandardOutput()
            text = bytes(data.data()).decode("utf-8", errors="ignore")
            self._log(text)

    def _log(self, text: str):
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _on_step_finished(self, step_idx: int, code: int, status):
        step_name = self.steps[step_idx]
        if code != 0:
            self._log(f"\n[ERROR] {self._step_label(step_name)} failed with exit code {code}\n")
            self._on_complete(False)
            return

        self._log(f"\n[SUCCESS] {self._step_label(step_name)} completed\n")

        # Move to next step
        self._run_step(step_idx + 1)

    def _on_complete(self, success: bool):
        self.proc = None
        self.btn_cancel.setText("Close")
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)

        if success:
            cfg = settings()
            self.lbl_step.setText("✓ Training workflow complete!")
            msg = (
                f"Models saved to:\n"
                f"  {cfg.team_models_dir}\n\n"
                f"Demo video saved to:\n"
                f"  {cfg.pipeline_output_dir}\n\n"
                f"You can now use the Pipeline tab with team coloring enabled."
            )
            self._log(f"\n{'=' * 60}\n")
            self._log(f"[SUCCESS] Training workflow complete!\n")
            self._log(f"{'=' * 60}\n\n")
            self._log(msg + "\n")
            self.finished.emit(True, msg)
        else:
            self.lbl_step.setText("✗ Training workflow failed")
            msg = "Check the log above for details."
            self.finished.emit(False, msg)

    def _on_cancel(self):
        if self.proc:
            self._log("\n[INFO] Cancelling...\n")
            self.proc.terminate()
            if not self.proc.waitForFinished(3000):
                self.proc.kill()
            self.proc = None
            self._on_complete(False)
        else:
            self.close()


def show_startup_dialog(parent: Optional[QtWidgets.QWidget] = None) -> Optional[str]:
    """
    Show startup dialog and return user choice.

    Returns:
        'run' if user wants to run workflow
        'skip' if user wants to skip
        'exit' if user wants to exit
        None if dialog was cancelled
    """
    dialog = StartupDialog(parent)
    result = dialog.exec()
    if result == QtWidgets.QDialog.DialogCode.Accepted:
        return dialog.user_choice
    return "exit"


def run_training_workflow(parent: Optional[QtWidgets.QWidget] = None) -> bool:
    """
    Run the training workflow in a modal window.

    Returns:
        True if successful, False otherwise
    """
    runner = TrainingWorkflowRunner(parent)
    runner.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    runner.show()
    runner.start()

    # Wait for completion
    loop = QtCore.QEventLoop()
    success = [False]  # mutable container for closure

    def on_finished(ok: bool, msg: str):
        success[0] = ok
        loop.quit()

    runner.finished.connect(on_finished)
    loop.exec()

    return success[0]
