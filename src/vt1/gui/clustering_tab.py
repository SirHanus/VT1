from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from vt1.config import settings

class ClusteringTab(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        vlay = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget(); vlay.addWidget(self.tabs)
        self.tabs.addTab(self._make_build_tab(), "Build Set")
        self.tabs.addTab(self._make_cluster_tab(), "Cluster")
        self.tabs.addTab(self._make_audit_tab(), "Audit")
        self.tabs.addTab(self._make_eval_tab(), "Evaluate")

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    # QProcess helpers
    def _start(self, module_name: str, args: list[str], widgets: dict):
        if widgets.get('proc') is not None:
            QtWidgets.QMessageBox.warning(self, 'Already running', 'Stop the current process first.')
            return
        py = sys.executable or 'python'
        proc = QtCore.QProcess(self)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        src_path = str(self._repo_root() / 'src')
        existing = env.value('PYTHONPATH', '')
        sep = ';' if os.name == 'nt' else ':'
        env.insert('PYTHONPATH', src_path + (sep + existing if existing else ''))
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        proc.setWorkingDirectory(str(self._repo_root()))
        widgets['proc'] = proc
        widgets['log'].clear(); widgets['status'].setText('Starting…'); widgets['progress'].setRange(0,0)
        proc.readyReadStandardOutput.connect(lambda w=widgets: self._on_stdout(w))
        proc.readyReadStandardError.connect(lambda w=widgets: self._on_stderr(w))
        proc.finished.connect(lambda code, status, w=widgets: self._on_finished(code, status, w))
        full_args = ['-m', module_name, *args]
        widgets['log'].appendPlainText('[GUI] Command: {} {}'.format(py, ' '.join(self._quote(a) for a in full_args)))
        try:
            proc.start(py, full_args)
            if not proc.waitForStarted(5000): raise RuntimeError('Failed to start process')
        except Exception as e:
            widgets['log'].appendPlainText(f'[GUI] Failed to start: {e}'); widgets['proc']=None
            QtWidgets.QMessageBox.critical(self, 'Failed', str(e)); return
        widgets['run'].setEnabled(False); widgets['stop'].setEnabled(True)

    def _stop(self, widgets: dict):
        p = widgets.get('proc')
        if p is not None:
            widgets['log'].appendPlainText('[GUI] Terminating…')
            p.terminate()
            if not p.waitForFinished(3000): widgets['log'].appendPlainText('[GUI] Killing…'); p.kill(); p.waitForFinished(2000)
        widgets['proc'] = None
        widgets['run'].setEnabled(True); widgets['stop'].setEnabled(False)
        widgets['progress'].setRange(0,100); widgets['progress'].setValue(0)
        widgets['status'].setText('Stopped')

    def _on_stdout(self, widgets: dict):
        p = widgets.get('proc');
        if p is None: return
        data = p.readAllStandardOutput(); text = bytes(data.data()).decode('utf-8','ignore')
        self._handle_stream(text, widgets)
    def _on_stderr(self, widgets: dict):
        p = widgets.get('proc');
        if p is None: return
        data = p.readAllStandardError(); text = bytes(data.data()).decode('utf-8','ignore')
        self._handle_stream(text, widgets)
    def _handle_stream(self, text: str, widgets: dict):
        for seg in text.split('\r'):
            if not seg: continue
            if any(k in seg for k in ('Processing','Done','Complete','Finished','Output')):
                widgets['status'].setText(seg.strip()[:160])
            widgets['log'].appendPlainText(seg)
    def _on_finished(self, code: int, status: QtCore.QProcess.ExitStatus, widgets: dict):
        widgets['log'].appendPlainText(f'[GUI] Finished code={code}')
        widgets['proc'] = None
        widgets['run'].setEnabled(True); widgets['stop'].setEnabled(False)
        widgets['progress'].setRange(0,100); widgets['progress'].setValue(100 if code==0 else 0)
        widgets['status'].setText('Done' if code==0 else 'Failed')

    @staticmethod
    def _row(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        for it in widgets: lay.addWidget(it)
        return w
    @staticmethod
    def _quote(s: str) -> str:
        if (" " in s) or ("\t" in s): return '"' + s.replace('"','\\"') + '"'
        return s

    # Build tab
    def _make_build_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w); form = QtWidgets.QFormLayout()
        cfg = settings(); root = cfg.repo_root
        ed_videos_dir = QtWidgets.QLineEdit(str(root / 'videos_all' / 'CAR_vs_NYR')); ed_videos_dir.setToolTip('Directory with input videos (*.mp4) (--videos-dir)')
        btn_videos_dir = QtWidgets.QPushButton('Browse…'); btn_videos_dir.setToolTip('Pick videos directory'); btn_videos_dir.clicked.connect(lambda: self._pick_dir_into(ed_videos_dir))
        form.addRow('Videos dir (--videos-dir)', self._row(ed_videos_dir, btn_videos_dir))
        ed_glob = QtWidgets.QLineEdit('*.mp4'); ed_glob.setToolTip('Glob to match videos inside --videos-dir (--glob)')
        form.addRow('Glob (--glob)', ed_glob)
        ed_det_config = QtWidgets.QLineEdit(); ed_det_config.setToolTip('MMDetection config path for RF-DETR-S (--det-config)')
        btn_det_config = QtWidgets.QPushButton('Browse…'); btn_det_config.setToolTip('Pick RF-DETR-S config (.py)'); btn_det_config.clicked.connect(lambda: self._pick_file_into(ed_det_config, 'Config (*.py);;All (*)'))
        form.addRow('Det config (--det-config)', self._row(ed_det_config, btn_det_config))
        ed_det_weights = QtWidgets.QLineEdit(); ed_det_weights.setToolTip('Checkpoint .pth for RF-DETR-S (--det-weights)')
        btn_det_weights = QtWidgets.QPushButton('Browse…'); btn_det_weights.setToolTip('Pick RF-DETR-S weights (.pth)'); btn_det_weights.clicked.connect(lambda: self._pick_file_into(ed_det_weights, 'Weights (*.pth);;All (*)'))
        form.addRow('Det weights (--det-weights)', self._row(ed_det_weights, btn_det_weights))
        ds_det_thr = QtWidgets.QDoubleSpinBox(); ds_det_thr.setRange(0.0,1.0); ds_det_thr.setSingleStep(0.01); ds_det_thr.setValue(0.30); ds_det_thr.setToolTip('Score threshold for detections (--det-score-thr)')
        form.addRow('Det score thr (--det-score-thr)', ds_det_thr)
        ed_person = QtWidgets.QLineEdit('person'); ed_person.setToolTip('Class name to treat as player/person (--person-class-name)')
        form.addRow('Person class (--person-class-name)', ed_person)
        cb_yolo_fallback = QtWidgets.QCheckBox('Enable YOLO fallback'); cb_yolo_fallback.setToolTip('Enable YOLO fallback if MMDetection is unavailable (--yolo-fallback)')
        form.addRow('YOLO fallback (--yolo-fallback)', cb_yolo_fallback)
        ed_yolo_model = QtWidgets.QLineEdit(str(cfg.yolo_model)); ed_yolo_model.setToolTip('Ultralytics YOLO model path/name for fallback (--yolo-model)')
        btn_yolo_model = QtWidgets.QPushButton('Browse…'); btn_yolo_model.setToolTip('Pick YOLO model (.pt/.onnx)'); btn_yolo_model.clicked.connect(lambda: self._pick_file_into(ed_yolo_model, 'Model (*.pt *.onnx);;All (*)'))
        form.addRow('YOLO model (--yolo-model)', self._row(ed_yolo_model, btn_yolo_model))
        ds_fps = QtWidgets.QDoubleSpinBox(); ds_fps.setRange(0.1,60.0); ds_fps.setSingleStep(0.1); ds_fps.setValue(1.0); ds_fps.setToolTip('Target frames per second to sample (default 1 FPS) (--fps)')
        form.addRow('FPS (--fps)', ds_fps)
        sb_max_seconds = QtWidgets.QSpinBox(); sb_max_seconds.setRange(0,36000); sb_max_seconds.setToolTip('Limit seconds per video (0=all) (--max-seconds)')
        form.addRow('Max seconds (--max-seconds)', sb_max_seconds)
        ds_central = QtWidgets.QDoubleSpinBox(); ds_central.setRange(0.05,1.0); ds_central.setSingleStep(0.05); ds_central.setValue(0.6); ds_central.setToolTip('Fraction of bbox width/height to keep around center (0<r<=1) (--central-ratio)')
        form.addRow('Central ratio (--central-ratio)', ds_central)
        sb_min_crop = QtWidgets.QSpinBox(); sb_min_crop.setRange(1,2048); sb_min_crop.setValue(32); sb_min_crop.setToolTip('Discard crops smaller than this (pixels) (--min-crop-size)')
        form.addRow('Min crop size (--min-crop-size)', sb_min_crop)
        ed_siglip = QtWidgets.QLineEdit('google/siglip-base-patch16-224'); ed_siglip.setToolTip('SigLIP Vision model with projection (HF id) (--siglip)')
        form.addRow('SigLIP (--siglip)', ed_siglip)
        sb_batch = QtWidgets.QSpinBox(); sb_batch.setRange(1,1024); sb_batch.setValue(64); sb_batch.setToolTip('Embedding batch size (--batch)')
        form.addRow('Batch size (--batch)', sb_batch)
        cb_device = QtWidgets.QComboBox(); cb_device.addItems(['cuda','cpu']); cb_device.setToolTip('Inference device (--device)')
        form.addRow('Device (--device)', cb_device)
        ed_out_dir = QtWidgets.QLineEdit(str(cfg.team_output_dir)); ed_out_dir.setToolTip('Output directory root (default: outputs/team_clustering) (--out-dir)')
        btn_out_dir = QtWidgets.QPushButton('Browse…'); btn_out_dir.setToolTip('Pick output folder'); btn_out_dir.clicked.connect(lambda: self._pick_dir_into(ed_out_dir))
        form.addRow('Out dir (--out-dir)', self._row(ed_out_dir, btn_out_dir))
        cb_save_crops = QtWidgets.QCheckBox('Save crop images'); cb_save_crops.setToolTip('Save crop images to disk (--save-crops)')
        form.addRow('Save crops (--save-crops)', cb_save_crops)
        sb_seed = QtWidgets.QSpinBox(); sb_seed.setRange(0,10_000); sb_seed.setToolTip('Random seed for reproducibility (--seed)')
        form.addRow('Seed (--seed)', sb_seed)
        cb_verbose = QtWidgets.QCheckBox('Verbose logging'); cb_verbose.setToolTip('Enable verbose logging (--verbose)')
        form.addRow('Verbose (--verbose)', cb_verbose)
        lay.addLayout(form)
        run = QtWidgets.QHBoxLayout(); btn_run = QtWidgets.QPushButton('Run Build'); btn_stop = QtWidgets.QPushButton('Stop'); btn_stop.setEnabled(False); status = QtWidgets.QLabel('Idle'); pb = QtWidgets.QProgressBar(); pb.setRange(0,0)
        run.addWidget(btn_run); run.addWidget(btn_stop); run.addWidget(status); run.addWidget(pb); lay.addLayout(run)
        log = QtWidgets.QPlainTextEdit(); log.setReadOnly(True); lay.addWidget(log,1)
        widgets={'proc':None,'run':btn_run,'stop':btn_stop,'status':status,'progress':pb,'log':log}
        btn_run.clicked.connect(lambda: self._start('vt1.team_clustering.build_training_set', self._args_build(
            ed_videos_dir.text(), ed_glob.text(), ed_det_config.text(), ed_det_weights.text(), ds_det_thr.value(), ed_person.text(),
            cb_yolo_fallback.isChecked(), ed_yolo_model.text(), ds_fps.value(), sb_max_seconds.value(), ds_central.value(), sb_min_crop.value(),
            ed_siglip.text(), sb_batch.value(), cb_device.currentText(), ed_out_dir.text(), cb_save_crops.isChecked(), sb_seed.value(), cb_verbose.isChecked()
        ), widgets))
        btn_stop.clicked.connect(lambda: self._stop(widgets))
        return w

    def _args_build(self, videos_dir, glob, det_config, det_weights, det_thr, person_cls, yolo_fallback, yolo_model, fps, max_seconds, central_ratio, min_crop, siglip, batch, device, out_dir, save_crops, seed, verbose):
        args=[]
        def add(f,v):
            if v is None: return
            s=str(v).strip();
            if s=='': return
            args.extend([f,s])
        add('--videos-dir', videos_dir); add('--glob', glob)
        if det_config: add('--det-config', det_config)
        if det_weights: add('--det-weights', det_weights)
        add('--det-score-thr', det_thr); add('--person-class-name', person_cls)
        if yolo_fallback: args.append('--yolo-fallback'); add('--yolo-model', yolo_model)
        add('--fps', fps); add('--max-seconds', max_seconds); add('--central-ratio', central_ratio); add('--min-crop-size', min_crop)
        add('--siglip', siglip); add('--batch', batch); add('--device', device); add('--out-dir', out_dir)
        if save_crops: args.append('--save-crops')
        add('--seed', seed)
        if verbose: args.append('--verbose')
        return [str(a) for a in args]

    # Cluster tab
    def _make_cluster_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w); form = QtWidgets.QFormLayout()
        base = settings().team_output_dir
        ed_in_root = QtWidgets.QLineEdit(str(base)); ed_in_root.setToolTip('Root directory with per-video embeddings (default: outputs/team_clustering) (--in-root)')
        btn_in_root = QtWidgets.QPushButton('Browse…'); btn_in_root.setToolTip('Pick input root'); btn_in_root.clicked.connect(lambda: self._pick_dir_into(ed_in_root))
        form.addRow('In root (--in-root)', self._row(ed_in_root, btn_in_root))
        ed_out_root = QtWidgets.QLineEdit(str(base)); ed_out_root.setToolTip('Output directory for clustering artifacts (--out-dir)')
        btn_out_root = QtWidgets.QPushButton('Browse…'); btn_out_root.setToolTip('Pick output dir'); btn_out_root.clicked.connect(lambda: self._pick_dir_into(ed_out_root))
        form.addRow('Out dir (--out-dir)', self._row(ed_out_root, btn_out_root))
        sb_k = QtWidgets.QSpinBox(); sb_k.setRange(2,32); sb_k.setValue(2); sb_k.setToolTip('Number of clusters for KMeans (e.g., 2 teams) (--k)')
        form.addRow('K (--k)', sb_k)
        sb_umap_dim = QtWidgets.QSpinBox(); sb_umap_dim.setRange(2,512); sb_umap_dim.setValue(16); sb_umap_dim.setToolTip('UMAP output dimensionality (--umap-dim)')
        form.addRow('UMAP dim (--umap-dim)', sb_umap_dim)
        sb_umap_neighbors = QtWidgets.QSpinBox(); sb_umap_neighbors.setRange(2,200); sb_umap_neighbors.setValue(15); sb_umap_neighbors.setToolTip('UMAP n_neighbors (--umap-neighbors)')
        form.addRow('UMAP neighbors (--umap-neighbors)', sb_umap_neighbors)
        ed_umap_metric = QtWidgets.QLineEdit('cosine'); ed_umap_metric.setToolTip('UMAP distance metric (--umap-metric)')
        form.addRow('UMAP metric (--umap-metric)', ed_umap_metric)
        ds_umap_min_dist = QtWidgets.QDoubleSpinBox(); ds_umap_min_dist.setRange(0.0,1.0); ds_umap_min_dist.setSingleStep(0.05); ds_umap_min_dist.setValue(0.1); ds_umap_min_dist.setToolTip('UMAP min_dist (lower=more compact clusters) (--umap-min-dist)')
        form.addRow('UMAP min_dist (--umap-min-dist)', ds_umap_min_dist)
        ed_reuse_umap = QtWidgets.QLineEdit(''); ed_reuse_umap.setToolTip('Path to a pre-trained umap.pkl to reuse (skip fitting) (--reuse-umap)')
        btn_reuse_umap = QtWidgets.QPushButton('Browse…'); btn_reuse_umap.setToolTip('Pick existing umap.pkl'); btn_reuse_umap.clicked.connect(lambda: self._pick_file_into(ed_reuse_umap, 'UMAP (*.pkl);;All (*)'))
        form.addRow('Reuse UMAP (--reuse-umap)', self._row(ed_reuse_umap, btn_reuse_umap))
        sb_limit = QtWidgets.QSpinBox(); sb_limit.setRange(0,10_000_000); sb_limit.setToolTip('Limit N total rows (0=all) (--limit)')
        form.addRow('Limit rows (--limit)', sb_limit)
        cb_plot = QtWidgets.QCheckBox('Save scatter plot'); cb_plot.setToolTip('Save a 2D scatter plot if possible (--plot)')
        form.addRow('Plot (--plot)', cb_plot)
        sb_seed2 = QtWidgets.QSpinBox(); sb_seed2.setRange(0,10_000); sb_seed2.setToolTip('Random seed (--seed)')
        form.addRow('Seed (--seed)', sb_seed2)
        cb_save_models = QtWidgets.QCheckBox('Persist UMAP & KMeans'); cb_save_models.setToolTip('Persist fitted UMAP and KMeans models (umap.pkl, kmeans.pkl) (--save-models)')
        form.addRow('Save models (--save-models)', cb_save_models)
        lay.addLayout(form)
        run = QtWidgets.QHBoxLayout(); btn_run = QtWidgets.QPushButton('Run Cluster'); btn_stop = QtWidgets.QPushButton('Stop'); btn_stop.setEnabled(False); status = QtWidgets.QLabel('Idle'); pb = QtWidgets.QProgressBar(); pb.setRange(0,0)
        run.addWidget(btn_run); run.addWidget(btn_stop); run.addWidget(status); run.addWidget(pb); lay.addLayout(run)
        log = QtWidgets.QPlainTextEdit(); log.setReadOnly(True); lay.addWidget(log,1)
        widgets={'proc':None,'run':btn_run,'stop':btn_stop,'status':status,'progress':pb,'log':log}
        btn_run.clicked.connect(lambda: self._start('vt1.team_clustering.cluster_umap_kmeans', self._args_cluster(
            ed_in_root.text(), ed_out_root.text(), sb_k.value(), sb_umap_dim.value(), sb_umap_neighbors.value(), ed_umap_metric.text(), ds_umap_min_dist.value(), ed_reuse_umap.text(), sb_limit.value(), cb_plot.isChecked(), sb_seed2.value(), cb_save_models.isChecked()
        ), widgets))
        btn_stop.clicked.connect(lambda: self._stop(widgets))
        return w

    def _args_cluster(self, in_root, out_root, k, dim, neighbors, metric, min_dist, reuse_umap, limit, plot, seed, save_models):
        args=[]
        def add(f,v):
            if v is None: return
            s=str(v).strip();
            if s=='': return
            args.extend([f,s])
        add('--in-root', in_root); add('--out-dir', out_root); add('--k', k); add('--umap-dim', dim); add('--umap-neighbors', neighbors); add('--umap-metric', metric); add('--umap-min-dist', min_dist)
        if reuse_umap: add('--reuse-umap', reuse_umap)
        add('--limit', limit)
        if plot: args.append('--plot')
        add('--seed', seed)
        if save_models: args.append('--save-models')
        return [str(a) for a in args]

    # Audit tab
    def _make_audit_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w); form = QtWidgets.QFormLayout()
        base = settings().team_output_dir
        ed_in_root = QtWidgets.QLineEdit(str(base)); ed_in_root.setToolTip('Root folder containing per-video subfolders with index.csv (and crops/ if saved) (--in-root)')
        btn_in_root = QtWidgets.QPushButton('Browse…'); btn_in_root.setToolTip('Pick input root'); btn_in_root.clicked.connect(lambda: self._pick_dir_into(ed_in_root))
        form.addRow('In root (--in-root)', self._row(ed_in_root, btn_in_root))
        ed_out_dir = QtWidgets.QLineEdit(str(base)); ed_out_dir.setToolTip('Output directory root (--out-dir)')
        btn_out_dir = QtWidgets.QPushButton('Browse…'); btn_out_dir.setToolTip('Pick output dir'); btn_out_dir.clicked.connect(lambda: self._pick_dir_into(ed_out_dir))
        form.addRow('Out dir (--out-dir)', self._row(ed_out_dir, btn_out_dir))
        sb_per_video = QtWidgets.QSpinBox(); sb_per_video.setRange(1,200); sb_per_video.setValue(24); sb_per_video.setToolTip('Max crops per video to include in mosaic (--per-video)')
        form.addRow('Per-video (--per-video)', sb_per_video)
        cb_save_grid = QtWidgets.QCheckBox('Save per-video mosaics'); cb_save_grid.setToolTip('Save per-video grid mosaics (--save-grid)')
        form.addRow('Save grid (--save-grid)', cb_save_grid)
        sb_seed = QtWidgets.QSpinBox(); sb_seed.setRange(0,10_000); sb_seed.setToolTip('Random seed for sampling (--seed)')
        form.addRow('Seed (--seed)', sb_seed)
        lay.addLayout(form)
        run = QtWidgets.QHBoxLayout(); btn_run = QtWidgets.QPushButton('Run Audit'); btn_stop = QtWidgets.QPushButton('Stop'); btn_stop.setEnabled(False); status = QtWidgets.QLabel('Idle'); pb = QtWidgets.QProgressBar(); pb.setRange(0,0)
        run.addWidget(btn_run); run.addWidget(btn_stop); run.addWidget(status); run.addWidget(pb); lay.addLayout(run)
        log = QtWidgets.QPlainTextEdit(); log.setReadOnly(True); lay.addWidget(log,1)
        widgets={'proc':None,'run':btn_run,'stop':btn_stop,'status':status,'progress':pb,'log':log}
        btn_run.clicked.connect(lambda: self._start('vt1.team_clustering.audit_training_set', self._args_audit(ed_in_root.text(), ed_out_dir.text(), sb_per_video.value(), cb_save_grid.isChecked(), sb_seed.value()), widgets))
        btn_stop.clicked.connect(lambda: self._stop(widgets))
        return w

    def _args_audit(self, in_root, out_dir, per_video, save_grid, seed):
        args=[]; args.extend(['--in-root', in_root]); args.extend(['--out-dir', out_dir]); args.extend(['--per-video', per_video])
        if save_grid: args.append('--save-grid'); args.extend(['--seed', seed])
        return [str(a) for a in args]

    # Evaluate tab
    def _make_eval_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w); form = QtWidgets.QFormLayout()
        cfg = settings(); root = cfg.repo_root
        ed_images_dir = QtWidgets.QLineEdit(''); ed_images_dir.setToolTip('Directory of test images (--images-dir)')
        btn_images_dir = QtWidgets.QPushButton('Browse…'); btn_images_dir.setToolTip('Pick images dir'); btn_images_dir.clicked.connect(lambda: self._pick_dir_into(ed_images_dir))
        form.addRow('Images dir (--images-dir)', self._row(ed_images_dir, btn_images_dir))
        ed_glob = QtWidgets.QLineEdit('*.jpg;*.png;*.jpeg;*.JPG;*.PNG;*.JPEG'); ed_glob.setToolTip('Semicolon-separated patterns for images (--glob)')
        form.addRow('Glob (--glob)', ed_glob)
        ed_video = QtWidgets.QLineEdit(str(root / 'data_hockey.mp4')); ed_video.setToolTip('Optional: sample frames from a video file (--video)')
        btn_video = QtWidgets.QPushButton('Browse…'); btn_video.setToolTip('Pick video file'); btn_video.clicked.connect(lambda: self._pick_file_into(ed_video, 'Video (*.mp4 *.avi *.mkv);;All (*)'))
        form.addRow('Video (--video)', self._row(ed_video, btn_video))
        sb_frame_step = QtWidgets.QSpinBox(); sb_frame_step.setRange(1,10_000); sb_frame_step.setValue(30); sb_frame_step.setToolTip('Take 1 frame every N frames when reading a video (--frame-step)')
        form.addRow('Frame step (--frame-step)', sb_frame_step)
        sb_max_frames = QtWidgets.QSpinBox(); sb_max_frames.setRange(0,100_000); sb_max_frames.setToolTip('Stop after N frames sampled (0=all) (--max-frames)')
        form.addRow('Max frames (--max-frames)', sb_max_frames)
        ed_team_models = QtWidgets.QLineEdit(str(cfg.team_models_dir)); ed_team_models.setToolTip('Folder with umap.pkl and kmeans.pkl (--team-models)')
        btn_team_models = QtWidgets.QPushButton('Browse dir…'); btn_team_models.setToolTip('Pick team models directory'); btn_team_models.clicked.connect(lambda: self._pick_dir_into(ed_team_models))
        form.addRow('Team models (--team-models)', self._row(ed_team_models, btn_team_models))
        ed_siglip = QtWidgets.QLineEdit('google/siglip-base-patch16-224'); ed_siglip.setToolTip('SigLIP model id (--siglip)')
        form.addRow('SigLIP (--siglip)', ed_siglip)
        ed_yolo_model = QtWidgets.QLineEdit(str(cfg.yolo_model)); ed_yolo_model.setToolTip('YOLO detection model path/id (--yolo-model)')
        btn_yolo_model = QtWidgets.QPushButton('Browse…'); btn_yolo_model.setToolTip('Pick YOLO model file'); btn_yolo_model.clicked.connect(lambda: self._pick_file_into(ed_yolo_model, 'Model (*.pt *.onnx);;All (*)'))
        form.addRow('YOLO model (--yolo-model)', self._row(ed_yolo_model, btn_yolo_model))
        sb_imgsz = QtWidgets.QSpinBox(); sb_imgsz.setRange(64,4096); sb_imgsz.setValue(640); sb_imgsz.setToolTip('YOLO inference size (--imgsz)')
        form.addRow('Image size (--imgsz)', sb_imgsz)
        ds_conf = QtWidgets.QDoubleSpinBox(); ds_conf.setRange(0.0,1.0); ds_conf.setSingleStep(0.01); ds_conf.setValue(0.30); ds_conf.setToolTip('YOLO confidence threshold (--conf)')
        form.addRow('Conf (--conf)', ds_conf)
        sb_max_boxes = QtWidgets.QSpinBox(); sb_max_boxes.setRange(1,100); sb_max_boxes.setValue(8); sb_max_boxes.setToolTip('Max boxes per image/frame to annotate (--max-boxes)')
        form.addRow('Max boxes (--max-boxes)', sb_max_boxes)
        ds_central = QtWidgets.QDoubleSpinBox(); ds_central.setRange(0.05,1.0); ds_central.setSingleStep(0.05); ds_central.setValue(0.6); ds_central.setToolTip('Central crop ratio of bbox (--central-ratio)')
        form.addRow('Central ratio (--central-ratio)', ds_central)
        cb_device = QtWidgets.QComboBox(); cb_device.addItems(['cuda','cpu']); cb_device.setToolTip('cuda or cpu (--device)')
        form.addRow('Device (--device)', cb_device)
        ed_out_dir = QtWidgets.QLineEdit(str(cfg.team_output_dir)); ed_out_dir.setToolTip('Output directory root (--out-dir)')
        btn_out_dir = QtWidgets.QPushButton('Browse…'); btn_out_dir.setToolTip('Pick output dir'); btn_out_dir.clicked.connect(lambda: self._pick_dir_into(ed_out_dir))
        form.addRow('Out dir (--out-dir)', self._row(ed_out_dir, btn_out_dir))
        cb_show = QtWidgets.QCheckBox('Show preview window'); cb_show.setToolTip('Show previews in a window (--show)')
        form.addRow('Show (--show)', cb_show)
        cb_save_grid = QtWidgets.QCheckBox('Save mosaic grid'); cb_save_grid.setToolTip('Save a mosaic grid of annotated images (--save-grid)')
        form.addRow('Save grid (--save-grid)', cb_save_grid)
        sb_limit_images = QtWidgets.QSpinBox(); sb_limit_images.setRange(1,10_000); sb_limit_images.setValue(50); sb_limit_images.setToolTip('Max annotated images to write (--limit-images)')
        form.addRow('Limit images (--limit-images)', sb_limit_images)
        lay.addLayout(form)
        run = QtWidgets.QHBoxLayout(); btn_run = QtWidgets.QPushButton('Run Eval'); btn_stop = QtWidgets.QPushButton('Stop'); btn_stop.setEnabled(False); status = QtWidgets.QLabel('Idle'); pb = QtWidgets.QProgressBar(); pb.setRange(0,0)
        run.addWidget(btn_run); run.addWidget(btn_stop); run.addWidget(status); run.addWidget(pb); lay.addLayout(run)
        log = QtWidgets.QPlainTextEdit(); log.setReadOnly(True); lay.addWidget(log,1)
        widgets={'proc':None,'run':btn_run,'stop':btn_stop,'status':status,'progress':pb,'log':log}
        btn_run.clicked.connect(lambda: self._start('vt1.team_clustering.eval_clustering', self._args_eval(
            ed_images_dir.text(), ed_glob.text(), ed_video.text(), sb_frame_step.value(), sb_max_frames.value(), ed_team_models.text(), ed_siglip.text(), ed_yolo_model.text(), sb_imgsz.value(), ds_conf.value(), sb_max_boxes.value(), ds_central.value(), cb_device.currentText(), ed_out_dir.text(), cb_show.isChecked(), cb_save_grid.isChecked(), sb_limit_images.value()
        ), widgets))
        btn_stop.clicked.connect(lambda: self._stop(widgets))
        return w

    def _args_eval(self, images_dir, glob, video, frame_step, max_frames, team_models, siglip, yolo_model, imgsz, conf, max_boxes, central_ratio, device, out_dir, show, save_grid, limit_images):
        args=[]
        def add(f,v):
            if v is None: return
            s=str(v).strip();
            if s=='': return
            args.extend([f,s])
        if images_dir: add('--images-dir', images_dir); add('--glob', glob)
        if video: add('--video', video); add('--frame-step', frame_step); add('--max-frames', max_frames)
        add('--team-models', team_models); add('--siglip', siglip); add('--yolo-model', yolo_model); add('--imgsz', imgsz); add('--conf', conf); add('--max-boxes', max_boxes); add('--central-ratio', central_ratio); add('--device', device); add('--out-dir', out_dir); add('--limit-images', limit_images)
        if show: args.append('--show')
        if save_grid: args.append('--save-grid')
        return [str(a) for a in args]

    # pickers
    def _pick_dir_into(self, le: QtWidgets.QLineEdit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select directory', le.text() or str(Path('.').resolve()))
        if path: le.setText(path)
    def _pick_file_into(self, le: QtWidgets.QLineEdit, filter_str: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select file', le.text() or str(Path('.').resolve()), filter_str)
        if path: le.setText(path)
