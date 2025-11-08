from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from vt1.config import settings


class PipelineTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.proc: Optional[QtCore.QProcess] = None
        self.total_frames: Optional[int] = None
        self._progress_pattern_total = re.compile(r"Processing.*?(\d+)\s*/\s*(\d+)")
        self._progress_pattern_it = re.compile(r"Processing.*?(\d+)\s*it")
        self._build_ui();
        self._wire_events();
        self._fill_defaults()

    def _repo_root(self) -> Path:
        # vt1/gui -> vt1 -> repo_root
        return Path(__file__).resolve().parents[2]

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.ed_source = QtWidgets.QLineEdit();
        self.ed_source.setToolTip("Video source path (--source)")
        self.btn_source = QtWidgets.QPushButton("Browse…");
        self.btn_source.setToolTip("Browse for video file")
        form.addRow("Video source (--source)", self._hrow(self.ed_source, self.btn_source))
        self.ed_pose_model = QtWidgets.QLineEdit();
        self.ed_pose_model.setToolTip("Ultralytics YOLO pose model path/name (--pose-model)")
        self.btn_pose_model = QtWidgets.QPushButton("Browse…");
        self.btn_pose_model.setToolTip("Browse for pose model (.pt/.onnx)")
        form.addRow("Pose model (--pose-model)", self._hrow(self.ed_pose_model, self.btn_pose_model))
        self.ed_sam2 = QtWidgets.QLineEdit();
        self.ed_sam2.setToolTip("HF SAM2 model id (--sam2)")
        form.addRow("SAM2 id (--sam2)", self.ed_sam2)
        self.cb_device = QtWidgets.QComboBox();
        self.cb_device.addItems(["cuda", "cpu"]);
        self.cb_device.setToolTip("Device selection: 'cuda' or 'cpu' (--device)")
        form.addRow("Device (--device)", self.cb_device)
        self.sb_imgsz = QtWidgets.QSpinBox();
        self.sb_imgsz.setRange(64, 4096);
        self.sb_imgsz.setSingleStep(32);
        self.sb_imgsz.setToolTip("YOLO inference size (--imgsz)")
        form.addRow("Image size (--imgsz)", self.sb_imgsz)
        self.dsb_conf = QtWidgets.QDoubleSpinBox();
        self.dsb_conf.setRange(0.0, 1.0);
        self.dsb_conf.setSingleStep(0.01);
        self.dsb_conf.setDecimals(3);
        self.dsb_conf.setToolTip("YOLO confidence threshold (--conf)")
        form.addRow("Confidence (--conf)", self.dsb_conf)
        self.sb_max_frames = QtWidgets.QSpinBox();
        self.sb_max_frames.setRange(0, 2_000_000);
        self.sb_max_frames.setToolTip("Process at most N frames (0=all) (--max-frames)")
        form.addRow("Max frames (--max-frames)", self.sb_max_frames)
        self.cb_show = QtWidgets.QCheckBox("Show live window (--show)");
        self.cb_show.setToolTip("Show a live window while processing")
        self.cb_no_sam = QtWidgets.QCheckBox("Disable SAM (--no-sam)");
        self.cb_no_sam.setToolTip("Disable SAM segmentation (faster, only pose)")
        self.cb_half = QtWidgets.QCheckBox("Use FP16 on CUDA (--half)");
        self.cb_half.setToolTip("Use FP16 precision (CUDA only)")
        form.addRow("Flags", self._hrow(self.cb_show, self.cb_no_sam, self.cb_half))
        self.sb_sam_every = QtWidgets.QSpinBox();
        self.sb_sam_every.setRange(1, 10_000);
        self.sb_sam_every.setToolTip("Run SAM every N frames (--sam-every)")
        form.addRow("SAM every (--sam-every)", self.sb_sam_every)
        self.sb_sam_topk = QtWidgets.QSpinBox();
        self.sb_sam_topk.setRange(0, 1000);
        self.sb_sam_topk.setToolTip("Limit SAM to top-K boxes (--sam-topk)")
        form.addRow("SAM top-K (--sam-topk)", self.sb_sam_topk)
        self.sb_sam_reinit = QtWidgets.QSpinBox();
        self.sb_sam_reinit.setRange(0, 1_000_000);
        self.sb_sam_reinit.setToolTip("Re-init SAM every N frames (0=never) (--sam-reinit)")
        form.addRow("SAM reinit (--sam-reinit)", self.sb_sam_reinit)
        self.sb_empty_cache = QtWidgets.QSpinBox();
        self.sb_empty_cache.setRange(0, 1_000_000);
        self.sb_empty_cache.setToolTip(
            "Call torch.cuda.empty_cache() every N frames on CUDA (0=never) (--empty-cache-interval)")
        form.addRow("Empty CUDA cache every N (--empty-cache-interval)", self.sb_empty_cache)
        self.ed_metrics = QtWidgets.QLineEdit();
        self.ed_metrics.setToolTip("Write per-run metrics JSON to this path if set (--metrics-json)")
        self.btn_metrics = QtWidgets.QPushButton("Browse…");
        self.btn_metrics.setToolTip("Select metrics JSON output path")
        form.addRow("Metrics JSON (--metrics-json, optional)", self._hrow(self.ed_metrics, self.btn_metrics))
        self.ed_team_models = QtWidgets.QLineEdit();
        self.ed_team_models.setToolTip("Directory containing umap.pkl and kmeans.pkl (--team-models)")
        self.btn_team_models = QtWidgets.QPushButton("Browse dir…");
        self.btn_team_models.setToolTip("Select team clustering models directory")
        form.addRow("Team models dir (--team-models)", self._hrow(self.ed_team_models, self.btn_team_models))
        self.ed_siglip = QtWidgets.QLineEdit();
        self.ed_siglip.setToolTip("SigLIP model id (--siglip)")
        form.addRow("SigLIP id (--siglip)", self.ed_siglip)
        self.dsb_central_ratio = QtWidgets.QDoubleSpinBox();
        self.dsb_central_ratio.setRange(0.0, 1.0);
        self.dsb_central_ratio.setSingleStep(0.05);
        self.dsb_central_ratio.setDecimals(3);
        self.dsb_central_ratio.setToolTip("Central crop ratio (--central-ratio)")
        form.addRow("Central crop ratio (--central-ratio)", self.dsb_central_ratio)
        self.cb_disable_team = QtWidgets.QCheckBox("Disable team coloring (--disable-team)");
        self.cb_disable_team.setToolTip("Disable team coloring even if models are present")
        form.addRow("Team flag", self.cb_disable_team)
        lay.addLayout(form)
        run_row = QtWidgets.QHBoxLayout();
        self.btn_run = QtWidgets.QPushButton("Run Pipeline");
        self.btn_stop = QtWidgets.QPushButton("Stop");
        self.btn_stop.setEnabled(False);
        self.btn_open_outputs = QtWidgets.QPushButton("Open outputs folder")
        run_row.addWidget(self.btn_run);
        run_row.addWidget(self.btn_stop);
        run_row.addStretch(1);
        run_row.addWidget(self.btn_open_outputs)
        lay.addLayout(run_row)
        self.progress = QtWidgets.QProgressBar();
        self.progress.setRange(0, 100);
        self.progress.setValue(0)
        self.lbl_status = QtWidgets.QLabel("Idle");
        lay.addWidget(self.progress);
        lay.addWidget(self.lbl_status)
        self.txt_log = QtWidgets.QPlainTextEdit();
        self.txt_log.setReadOnly(True);
        lay.addWidget(self.txt_log, 1)

    def _wire_events(self):
        self.btn_source.clicked.connect(self._pick_source)
        self.btn_pose_model.clicked.connect(self._pick_pose_model)
        self.btn_metrics.clicked.connect(self._pick_metrics)
        self.btn_team_models.clicked.connect(self._pick_team_models)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open_outputs.clicked.connect(self._open_outputs_folder)

    def _fill_defaults(self):
        cfg = settings()
        root = cfg.repo_root
        self.ed_source.setText(str(root / "data_hockey.mp4"))
        self.ed_pose_model.setText(str(cfg.pose_model))
        self.ed_sam2.setText("facebook/sam2-hiera-large")
        self.cb_device.setCurrentText("cuda")
        self.sb_imgsz.setValue(int(cfg.yolo_imgsz))
        self.dsb_conf.setValue(float(cfg.yolo_conf))
        self.sb_max_frames.setValue(0)
        self.cb_show.setChecked(False); self.cb_no_sam.setChecked(False); self.cb_half.setChecked(False)
        self.sb_sam_every.setValue(int(cfg.sam_every))
        self.sb_sam_topk.setValue(int(cfg.sam_topk))
        self.sb_sam_reinit.setValue(int(cfg.sam_reinit))
        self.sb_empty_cache.setValue(int(cfg.empty_cache_interval))
        self.ed_metrics.setText("")
        self.ed_team_models.setText(str(cfg.team_models_dir))
        self.ed_siglip.setText(str(cfg.siglip_model))
        self.dsb_central_ratio.setValue(float(cfg.central_ratio_default))
        self.cb_disable_team.setChecked(False)
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1": self.cb_device.setCurrentText("cpu")

    def _pick_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video file",
                                                        str(Path(self.ed_source.text() or ".").resolve().parent),
                                                        "Video Files (*.mp4 *.avi *.mkv);;All Files (*)")
        if path: self.ed_source.setText(path)

    def _pick_pose_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select pose model",
                                                        str(Path(self.ed_pose_model.text() or ".").resolve().parent),
                                                        "Model Files (*.pt *.pth *.onnx);;All Files (*)")
        if path: self.ed_pose_model.setText(path)

    def _pick_metrics(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select metrics JSON path",
                                                        str(Path(self.ed_metrics.text() or ".").resolve()),
                                                        "JSON (*.json)")
        if path:
            if not path.lower().endswith(".json"): path += ".json"
            self.ed_metrics.setText(path)

    def _pick_team_models(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select team models directory",
                                                          str(Path(self.ed_team_models.text() or ".").resolve()))
        if path: self.ed_team_models.setText(path)

    def _on_run(self):
        if self.proc is not None:
            QtWidgets.QMessageBox.warning(self, "Already running", "A process is already running. Stop it first.")
            return
        src = self.ed_source.text().strip()
        if not src or not Path(src).exists():
            QtWidgets.QMessageBox.warning(self, "Source required", "Please pick a valid video source file.")
            return
        args = self._build_args()
        self.txt_log.clear();
        self.progress.setRange(0, 0);
        self.progress.setValue(0);
        self.lbl_status.setText("Starting…");
        self.total_frames = None

        self.proc = QtCore.QProcess(self)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        src_path = str(self._repo_root() / "src")
        existing = env.value("PYTHONPATH", "")
        sep = ";" if os.name == "nt" else ":"
        env.insert("PYTHONPATH", src_path + (sep + existing if existing else ""))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        self.proc.setWorkingDirectory(str(self._repo_root()))
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)

        # Choose how to launch the pipeline
        frozen = getattr(sys, 'frozen', False)
        cmd: list[str]
        if frozen:
            # Look for a sibling vt1-pipeline.exe next to the GUI executable
            exe_dir = Path(sys.executable).resolve().parent
            pipeline_exe = exe_dir / ("vt1-pipeline.exe" if os.name == 'nt' else "vt1-pipeline")
            if pipeline_exe.exists():
                cmd = [str(pipeline_exe), *args]
                self._append_log("[GUI] Command: {}\n".format(" ".join(self._quote(a) for a in cmd)))
                try:
                    self.proc.start(cmd[0], cmd[1:])
                    if not self.proc.waitForStarted(5000):
                        raise RuntimeError("Failed to start pipeline exe")
                except Exception as e:
                    self._append_log(f"[GUI] Failed to start: {e}\n"); self._cleanup_proc();
                    QtWidgets.QMessageBox.critical(self, "Failed to start", str(e)); return
            else:
                # Fallback: try a system Python to run the module
                py = "python"
                full_args = ["-m", "vt1.pipeline.sam_offline", *args]
                self._append_log("[GUI] Command: {} {}\n".format(py, " ".join(self._quote(a) for a in full_args)))
                try:
                    self.proc.start(py, full_args)
                    if not self.proc.waitForStarted(5000): raise RuntimeError("Failed to start process")
                except Exception as e:
                    self._append_log(f"[GUI] Failed to start: {e}\n"); self._cleanup_proc();
                    QtWidgets.QMessageBox.critical(self, "Failed to start", str(e)); return
        else:
            # Dev mode: run module vt1.pipeline.sam_offline
            py = sys.executable or "python"
            full_args = ["-m", "vt1.pipeline.sam_offline", *args]
            self._append_log("[GUI] Command: {} {}\n".format(py, " ".join(self._quote(a) for a in full_args)))
            try:
                self.proc.start(py, full_args)
                if not self.proc.waitForStarted(5000): raise RuntimeError("Failed to start process")
            except Exception as e:
                self._append_log(f"[GUI] Failed to start: {e}\n"); self._cleanup_proc();
                QtWidgets.QMessageBox.critical(self, "Failed to start", str(e)); return

        self.btn_run.setEnabled(False);
        self.btn_stop.setEnabled(True);
        self.lbl_status.setText("Running…")

    def _on_stop(self):
        if self.proc is not None:
            self._append_log("[GUI] Terminating process…\n");
            self.proc.terminate()
            if not self.proc.waitForFinished(3000): self._append_log(
                "[GUI] Killing process…\n"); self.proc.kill(); self.proc.waitForFinished(2000)
        self._cleanup_proc()

    def _on_stdout(self):
        if self.proc is None: return
        data = self.proc.readAllStandardOutput();
        text = bytes(data.data()).decode("utf-8", errors="ignore");
        self._handle_text_stream(text)

    def _on_stderr(self):
        if self.proc is None: return
        data = self.proc.readAllStandardError();
        text = bytes(data.data()).decode("utf-8", errors="ignore");
        self._handle_text_stream(text)

    def _on_finished(self, code: int, status: QtCore.QProcess.ExitStatus):
        self._append_log(f"[GUI] Finished with code={code}, status={status}\n");
        self._cleanup_proc();
        self.btn_run.setEnabled(True);
        self.btn_stop.setEnabled(False)
        self.progress.setRange(0, 100);
        self.progress.setValue(100 if code == 0 else 0);
        self.lbl_status.setText("Done" if code == 0 else "Failed")

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError):
        self._append_log(f"[GUI] Process error: {err}\n")

    def _cleanup_proc(self):
        self.proc = None

    def _append_log(self, s: str):
        self.txt_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.txt_log.insertPlainText(s)
        self.txt_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def _handle_text_stream(self, text: str):
        for seg in text.split('\r'):
            if not seg: continue
            if self._parse_progress(seg): continue
            self._append_log(seg)

    def _log_progress_line(self, line: str):
        doc = self.txt_log.document();
        last_block = doc.lastBlock()
        if last_block.text().startswith('[Progress]'):
            cursor = self.txt_log.textCursor();
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cursor.select(QtGui.QTextCursor.SelectionType.BlockUnderCursor);
            cursor.removeSelectedText();
        self.txt_log.appendPlainText(f"[Progress] {line.strip()}")

    def _parse_progress(self, chunk: str) -> bool:
        m = self._progress_pattern_total.search(chunk)
        if m:
            try:
                cur = int(m.group(1));
                total = int(m.group(2));
                self.total_frames = total
                self.progress.setRange(0, total);
                self.progress.setValue(cur)
                self.lbl_status.setText(f"Processing {cur}/{total} frames");
                self._log_progress_line(f"{cur}/{total}");
                return True
            except Exception:
                pass
        m2 = self._progress_pattern_it.search(chunk)
        if m2:
            try:
                cur = int(m2.group(1));
                self.progress.setRange(0, 0);
                self.lbl_status.setText(f"Processing {cur} frames…");
                self._log_progress_line(f"{cur} frames");
                return True
            except Exception:
                pass
        if "[ERROR]" in chunk: self.lbl_status.setText("Error encountered – see log")
        return False

    def _build_args(self) -> list[str]:
        args: list[str] = []

        def add(name: str, val: Optional[str]):
            if val is None: return
            v = str(val).strip();
            if v == "": return
            args.extend([name, v])

        add("--source", self.ed_source.text())
        add("--pose-model", self.ed_pose_model.text())
        add("--sam2", self.ed_sam2.text())
        add("--device", self.cb_device.currentText())
        add("--imgsz", str(self.sb_imgsz.value()))
        add("--conf", str(self.dsb_conf.value()))
        if self.sb_max_frames.value() > 0: args.append("--max-frames"); args.append(str(self.sb_max_frames.value()))
        if self.cb_show.isChecked(): args.append("--show")
        if self.cb_no_sam.isChecked(): args.append("--no-sam")
        if self.cb_half.isChecked(): args.append("--half")
        add("--sam-every", str(self.sb_sam_every.value()))
        add("--sam-topk", str(self.sb_sam_topk.value()))
        add("--sam-reinit", str(self.sb_sam_reinit.value()))
        add("--empty-cache-interval", str(self.sb_empty_cache.value()))
        if self.ed_metrics.text().strip(): add("--metrics-json", self.ed_metrics.text())
        add("--team-models", self.ed_team_models.text())
        add("--siglip", self.ed_siglip.text())
        add("--central-ratio", str(self.dsb_central_ratio.value()))
        if self.cb_disable_team.isChecked(): args.append("--disable-team")
        return args

    def _open_outputs_folder(self):
        out_dir = settings().pipeline_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        p = str(out_dir)
        if sys.platform.startswith("win"): os.startfile(p)  # type: ignore
        elif sys.platform == "darwin": import subprocess; subprocess.Popen(["open", p])
        else: import subprocess; subprocess.Popen(["xdg-open", p])

    @staticmethod
    def _hrow(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget();
        lay = QtWidgets.QHBoxLayout(w);
        lay.setContentsMargins(0, 0, 0, 0)
        for it in widgets: lay.addWidget(it)
        return w

    @staticmethod
    def _quote(s: str) -> str:
        if (" " in s) or ("\t" in s): return '"' + s.replace('"', '\\"') + '"'
        return s
