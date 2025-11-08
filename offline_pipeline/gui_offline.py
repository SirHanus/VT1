# python
# Simple PyQt GUI to run offline_pipeline/sam_offline.py with selectable options and show progress
from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from typing import Optional
# Add typing-time stub imports for static analyzers
from typing import TYPE_CHECKING

from PyQt6 import QtCore as QtCore, QtGui as QtGui, QtWidgets as QtWidgets  # type: ignore

# Try multiple Qt bindings for portability
QT_BINDINGS_TRIED = []

for mod_name in ("PyQt6", "PySide6", "PyQt5", "PySide2"):
    try:
        if mod_name == "PyQt6":
            from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
        elif mod_name == "PySide6":
            from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
        elif mod_name == "PyQt5":
            from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
        elif mod_name == "PySide2":
            from PySide2 import QtCore, QtGui, QtWidgets  # type: ignore
        QT_BINDINGS_TRIED.append(mod_name)
        break
    except Exception as e:
        QT_BINDINGS_TRIED.append(f"{mod_name} (fail: {e})")
        QtCore = QtGui = QtWidgets = None

if QtWidgets is None:
    msg = (
        "No Qt bindings found. Please install one of: PyQt6, PySide6, PyQt5, or PySide2.\n"
        "Example (Windows):\n"
        "  pip install PyQt6\n"
        "or pip install PyQt5\n"
        f"Tried: {', '.join(QT_BINDINGS_TRIED)}\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)
# Hint to static analyzers: at this point, these are available
assert QtWidgets is not None and QtCore is not None and QtGui is not None


class SamOfflineGUI(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 + YOLO Pose - Offline Pipeline GUI")
        self.resize(1000, 700)

        self.proc: Optional[QtCore.QProcess] = None
        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.total_frames: Optional[int] = None
        self._progress_pattern_total = re.compile(r"Processing.*?(\d+)\s*/\s*(\d+)")
        self._progress_pattern_it = re.compile(r"Processing.*?(\d+)\s*it")

        self._build_ui()
        self._wire_events()
        self._fill_defaults()

    def _build_ui(self):
        root_layout = QtWidgets.QVBoxLayout(self)

        # Form for options
        form = QtWidgets.QFormLayout()

        # Paths and core options
        self.ed_source = QtWidgets.QLineEdit()
        self.ed_source.setToolTip("Video source path (--source)")
        self.btn_source = QtWidgets.QPushButton("Browse…")
        self.btn_source.setToolTip("Browse for video file")
        self._h_source = self._hrow(self.ed_source, self.btn_source)
        form.addRow("Video source (--source)", self._h_source)


        self.ed_pose_model = QtWidgets.QLineEdit()
        self.ed_pose_model.setToolTip("Ultralytics YOLO pose model path/name (--pose-model)")
        self.btn_pose_model = QtWidgets.QPushButton("Browse…")
        self.btn_pose_model.setToolTip("Browse for pose model (.pt/.onnx)")
        self._h_pose = self._hrow(self.ed_pose_model, self.btn_pose_model)
        form.addRow("Pose model (--pose-model)", self._h_pose)

        self.ed_sam2 = QtWidgets.QLineEdit()
        self.ed_sam2.setToolTip("HF SAM2 model id (--sam2)")
        form.addRow("SAM2 id (--sam2)", self.ed_sam2)

        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.setToolTip("Device selection: 'cuda' or 'cpu' (--device)")
        self.cb_device.addItems(["cuda", "cpu"])
        form.addRow("Device (--device)", self.cb_device)

        self.sb_imgsz = QtWidgets.QSpinBox()
        self.sb_imgsz.setToolTip("YOLO inference size (--imgsz)")
        self.sb_imgsz.setRange(64, 4096)
        self.sb_imgsz.setSingleStep(32)
        form.addRow("Image size (--imgsz)", self.sb_imgsz)

        self.dsb_conf = QtWidgets.QDoubleSpinBox()
        self.dsb_conf.setToolTip("YOLO confidence threshold (--conf)")
        self.dsb_conf.setRange(0.0, 1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setDecimals(3)
        form.addRow("Confidence (--conf)", self.dsb_conf)

        self.sb_max_frames = QtWidgets.QSpinBox()
        self.sb_max_frames.setToolTip("Process at most N frames (0 = all) (--max-frames)")
        self.sb_max_frames.setRange(0, 2_000_000)
        form.addRow("Max frames (--max-frames)", self.sb_max_frames)

        # Flags
        self.cb_show = QtWidgets.QCheckBox("Show live window (--show)")
        self.cb_show.setToolTip("Show a live window while processing")
        self.cb_no_sam = QtWidgets.QCheckBox("Disable SAM (--no-sam)")
        self.cb_no_sam.setToolTip("Disable SAM segmentation (faster, only pose)")
        self.cb_half = QtWidgets.QCheckBox("Use FP16 on CUDA (--half)")
        self.cb_half.setToolTip("Use FP16 precision for YOLO on CUDA to save memory / increase speed")
        flags_row = self._hrow(self.cb_show, self.cb_no_sam, self.cb_half)
        form.addRow("Flags", flags_row)

        # SAM performance controls
        self.sb_sam_every = QtWidgets.QSpinBox()
        self.sb_sam_every.setToolTip("Run SAM every N frames (1 = every frame) (--sam-every)")
        self.sb_sam_every.setRange(1, 10_000)
        form.addRow("SAM every N frames (--sam-every)", self.sb_sam_every)

        self.sb_sam_topk = QtWidgets.QSpinBox()
        self.sb_sam_topk.setToolTip("Limit SAM to top-K boxes per frame (--sam-topk)")
        self.sb_sam_topk.setRange(0, 1000)
        form.addRow("SAM top-K (--sam-topk)", self.sb_sam_topk)

        self.sb_sam_reinit = QtWidgets.QSpinBox()
        self.sb_sam_reinit.setToolTip("Re-init SAM2 every N frames (0 = never) (--sam-reinit)")
        self.sb_sam_reinit.setRange(0, 1_000_000)
        form.addRow("SAM reinit interval (--sam-reinit)", self.sb_sam_reinit)

        self.sb_empty_cache = QtWidgets.QSpinBox()
        self.sb_empty_cache.setToolTip("Call torch.cuda.empty_cache() every N frames on CUDA (0 = never) (--empty-cache-interval)")
        self.sb_empty_cache.setRange(0, 1_000_000)
        form.addRow("Empty CUDA cache every N (--empty-cache-interval)", self.sb_empty_cache)

        # Metrics
        self.ed_metrics = QtWidgets.QLineEdit()
        self.ed_metrics.setToolTip("Write per-run metrics JSON to this path if set (--metrics-json)")
        self.btn_metrics = QtWidgets.QPushButton("Browse…")
        self.btn_metrics.setToolTip("Select metrics JSON output path")
        self._h_metrics = self._hrow(self.ed_metrics, self.btn_metrics)
        form.addRow("Metrics JSON (--metrics-json, optional)", self._h_metrics)

        # Team clustering
        self.ed_team_models = QtWidgets.QLineEdit()
        self.ed_team_models.setToolTip("Directory containing umap.pkl and kmeans.pkl (--team-models)")
        self.btn_team_models = QtWidgets.QPushButton("Browse dir…")
        self.btn_team_models.setToolTip("Select team clustering models directory")
        self._h_team_models = self._hrow(self.ed_team_models, self.btn_team_models)
        form.addRow("Team models dir (--team-models)", self._h_team_models)

        self.ed_siglip = QtWidgets.QLineEdit()
        self.ed_siglip.setToolTip("SigLIP model id for vision embeddings (--siglip)")
        form.addRow("SigLIP id (--siglip)", self.ed_siglip)

        self.dsb_central_ratio = QtWidgets.QDoubleSpinBox()
        self.dsb_central_ratio.setToolTip("Central crop ratio for team inference (--central-ratio)")
        self.dsb_central_ratio.setRange(0.0, 1.0)
        self.dsb_central_ratio.setSingleStep(0.05)
        self.dsb_central_ratio.setDecimals(3)
        form.addRow("Central crop ratio (--central-ratio)", self.dsb_central_ratio)

        self.cb_disable_team = QtWidgets.QCheckBox("Disable team coloring (--disable-team)")
        self.cb_disable_team.setToolTip("Disable team coloring even if models are present")
        form.addRow("Team flag", self.cb_disable_team)

        root_layout.addLayout(form)

        # Run controls
        run_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setToolTip("Start processing with current parameters")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setToolTip("Terminate the running process")
        self.btn_open_outputs = QtWidgets.QPushButton("Open outputs folder")
        self.btn_open_outputs.setToolTip("Open the auto-generated outputs directory")
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_stop)
        run_row.addStretch(1)
        run_row.addWidget(self.btn_open_outputs)
        root_layout.addLayout(run_row)

        # Progress + log
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_status = QtWidgets.QLabel("Idle")
        root_layout.addWidget(self.progress)
        root_layout.addWidget(self.lbl_status)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        root_layout.addWidget(self.txt_log, 1)

    def _hrow(self, *widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for it in widgets:
            lay.addWidget(it)
        return w

    def _wire_events(self):
        self.btn_source.clicked.connect(self._pick_source)
        self.btn_pose_model.clicked.connect(self._pick_pose_model)
        self.btn_metrics.clicked.connect(self._pick_metrics)
        self.btn_team_models.clicked.connect(self._pick_team_models)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open_outputs.clicked.connect(self._open_outputs_folder)

    def _fill_defaults(self):
        # Mirror defaults in sam_offline.py
        root = Path(__file__).resolve().parents[1]
        default_source = root / "data_hockey.mp4"
        default_pose_model = root / "hockeypose_1" / "yolo11x-pose.pt"
        default_team_models = root / "offline_pipeline" / "team_clustering" / "clustering"

        self.ed_source.setText(str(default_source))
        # self.ed_out.setText("")  # Removed output path field
        self.ed_pose_model.setText(str(default_pose_model))
        self.ed_sam2.setText("facebook/sam2-hiera-large")
        self.cb_device.setCurrentText("cuda")
        self.sb_imgsz.setValue(640)
        self.dsb_conf.setValue(0.25)
        self.sb_max_frames.setValue(0)

        self.cb_show.setChecked(False)
        self.cb_no_sam.setChecked(False)
        self.cb_half.setChecked(False)

        self.sb_sam_every.setValue(1)
        self.sb_sam_topk.setValue(5)
        self.sb_sam_reinit.setValue(0)
        self.sb_empty_cache.setValue(25)

        self.ed_metrics.setText("")
        self.ed_team_models.setText(str(default_team_models))
        self.ed_siglip.setText("google/siglip-base-patch16-224")
        self.dsb_central_ratio.setValue(0.6)
        self.cb_disable_team.setChecked(False)

        # If CUDA likely unavailable, default to CPU (best-effort, avoid heavy torch import)
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
            self.cb_device.setCurrentText("cpu")

    # --- File pickers
    def _pick_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video file",
                                                        str(Path(self.ed_source.text() or ".").resolve().parent),
                                                        "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if path:
            self.ed_source.setText(path)

    def _pick_pose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select pose model",
                                                        str(Path(self.ed_pose_model.text() or ".").resolve().parent),
                                                        "Model Files (*.pt *.pth *.onnx);;All Files (*)")
        if path:
            self.ed_pose_model.setText(path)

    def _pick_metrics(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select metrics JSON path",
                                                        str(Path(self.ed_metrics.text() or ".").resolve()),
                                                        "JSON (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.ed_metrics.setText(path)

    def _pick_team_models(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select team models directory",
                                                          str(Path(self.ed_team_models.text() or ".").resolve()))
        if path:
            self.ed_team_models.setText(path)

    # --- Run/Stop
    def _on_run(self):
        if self.proc is not None:
            QtWidgets.QMessageBox.warning(self, "Already running", "A process is already running. Stop it first.")
            return

        source = self.ed_source.text().strip()
        if not source:
            QtWidgets.QMessageBox.warning(self, "Source required", "Please pick a video source file.")
            return
        if not Path(source).exists():
            QtWidgets.QMessageBox.warning(self, "Source missing", f"Source file does not exist:\n{source}")
            return

        args = self._build_args()

        self.txt_log.clear()
        self.progress.setRange(0, 0)  # indefinite until we know total
        self.progress.setValue(0)
        self.lbl_status.setText("Starting…")
        self.total_frames = None

        # Launch sam_offline.py via QProcess
        python_exe = sys.executable or "python"
        script_path = str(Path(__file__).resolve().parent / "sam_offline.py")
        program = python_exe
        full_args = [script_path] + args

        self.proc = QtCore.QProcess(self)
        # Use separate channels so we can parse stdout reliably
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)

        # Working dir = project root to keep relative paths familiar
        self.proc.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))

        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)

        try:
            self.proc.start(program, full_args)
            started = self.proc.waitForStarted(5000)
            if not started:
                raise RuntimeError("Failed to start process")
        except Exception as e:
            self._append_log(f"[GUI] Failed to start: {e}\n")
            self._cleanup_proc()
            QtWidgets.QMessageBox.critical(self, "Failed to start", str(e))
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Running…")
        self._append_log("[GUI] Command: {} {}\n".format(program, " ".join(self._quote(a) for a in full_args)))

    def _on_stop(self):
        if self.proc is not None:
            self._append_log("[GUI] Terminating process…\n")
            self.proc.terminate()
            if not self.proc.waitForFinished(3000):
                self._append_log("[GUI] Killing process…\n")
                self.proc.kill()
                self.proc.waitForFinished(2000)
        self._cleanup_proc()

    def _on_stdout(self):
        if self.proc is None:
            return
        data = self.proc.readAllStandardOutput()
        try:
            text = bytes(data.data()).decode("utf-8", errors="ignore")
        except Exception:
            text = str(bytes(data.data()), errors="ignore")
        # Use streaming handler to collapse carriage-return progress updates
        self._handle_text_stream(text)

    def _on_stderr(self):
        if self.proc is None:
            return
        data = self.proc.readAllStandardError()
        try:
            text = bytes(data.data()).decode("utf-8", errors="ignore")
        except Exception:
            text = str(bytes(data.data()), errors="ignore")
        self._handle_text_stream(text)

    def _on_finished(self, exitCode: int, exitStatus: QtCore.QProcess.ExitStatus):
        self._append_log(
            f"[GUI] Finished with code={exitCode}, status={exitStatus.name if hasattr(exitStatus, 'name') else exitStatus}\n")
        self._cleanup_proc()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if exitCode == 0 else 0)
        self.lbl_status.setText("Done" if exitCode == 0 else "Failed")

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError):
        self._append_log(f"[GUI] Process error: {err}\n")

    def _cleanup_proc(self):
        self.proc = None

    def _append_log(self, s: str):
        self.txt_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.txt_log.insertPlainText(s)
        self.txt_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # New helper to process text chunks with potential \r-based progress updates
    def _handle_text_stream(self, text: str):
        # tqdm overwrites a single line using carriage returns; splitting lets us detect updates
        segments = text.split('\r')
        for seg in segments:
            if not seg:
                continue
            # If progress handled, skip regular log append
            if self._parse_progress(seg):
                continue
            self._append_log(seg)

    def _log_progress_line(self, line: str):
        # Replace last progress line instead of appending endlessly
        doc = self.txt_log.document()
        last_block = doc.lastBlock()
        if last_block.text().startswith('[Progress]'):
            cursor = self.txt_log.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cursor.select(QtGui.QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            # Remove leftover newline if present
            cursor.deletePreviousChar()
        # Append updated progress line
        self.txt_log.appendPlainText(f"[Progress] {line.strip()}")

    def _parse_progress(self, chunk: str) -> bool:
        # Return True if chunk was a progress update (suppresses normal logging)
        m = self._progress_pattern_total.search(chunk)
        if m:
            try:
                cur = int(m.group(1))
                total = int(m.group(2))
                self.total_frames = total
                self.progress.setRange(0, total)
                self.progress.setValue(cur)
                self.lbl_status.setText(f"Processing {cur}/{total} frames")
                self._log_progress_line(f"{cur}/{total}")
                return True
            except Exception:
                pass
        m2 = self._progress_pattern_it.search(chunk)
        if m2:
            try:
                cur = int(m2.group(1))
                self.progress.setRange(0, 0)
                self.lbl_status.setText(f"Processing {cur} frames…")
                self._log_progress_line(f"{cur} frames")
                return True
            except Exception:
                pass
        if "[ERROR]" in chunk:
            self.lbl_status.setText("Error encountered – see log")
        return False

    def _build_args(self) -> list[str]:
        args: list[str] = []

        def add(name: str, val: Optional[str]):
            if val is None:
                return
            v = str(val).strip()
            if v == "":
                return
            args.extend([name, v])

        # Required/positional-like
        add("--source", self.ed_source.text())
        add("--pose-model", self.ed_pose_model.text())
        add("--sam2", self.ed_sam2.text())
        add("--device", self.cb_device.currentText())
        add("--imgsz", str(self.sb_imgsz.value()))
        add("--conf", str(self.dsb_conf.value()))

        # Optional outputs
        if self.sb_max_frames.value() > 0:
            add("--max-frames", str(self.sb_max_frames.value()))
        if self.cb_show.isChecked():
            args.append("--show")
        if self.cb_no_sam.isChecked():
            args.append("--no-sam")
        if self.cb_half.isChecked():
            args.append("--half")

        add("--sam-every", str(self.sb_sam_every.value()))
        add("--sam-topk", str(self.sb_sam_topk.value()))
        add("--sam-reinit", str(self.sb_sam_reinit.value()))
        add("--empty-cache-interval", str(self.sb_empty_cache.value()))

        if self.ed_metrics.text().strip():
            add("--metrics-json", self.ed_metrics.text())

        add("--team-models", self.ed_team_models.text())
        add("--siglip", self.ed_siglip.text())
        add("--central-ratio", str(self.dsb_central_ratio.value()))
        if self.cb_disable_team.isChecked():
            args.append("--disable-team")

        return args

    def _open_outputs_folder(self):
        # Use the same outputs dir as sam_offline: offline_pipeline/outputs
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir)
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])

    @staticmethod
    def _quote(s: str) -> str:
        if " " in s or "\t" in s:
            if sys.platform.startswith("win"):
                # Escape internal quotes
                return '"' + s.replace('"', '\\"') + '"'
            else:
                return '"' + s.replace('"', '\\"') + '"'
        return s


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = SamOfflineGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
