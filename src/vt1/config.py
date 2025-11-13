from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

try:  # optional .env support
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

_LOCK = threading.RLock()
_SETTINGS: Config | None = None


@dataclass(frozen=True)
class Config:
    repo_root: Path
    models_dir: Path
    pose_model: Path
    yolo_model: Path
    default_video_source: Path
    team_models_dir: Path
    pipeline_output_dir: Path
    team_output_dir: Path
    runs_dir: Path
    logs_dir: Path
    log_level: str
    # Shared defaults
    yolo_conf: float
    yolo_imgsz: int
    central_ratio_default: float
    siglip_model: str
    training_videos_dir: Path
    videos_glob: str
    build_fps: float
    build_min_crop_size: int
    build_batch_size: int
    det_score_thr_default: float
    cluster_k: int
    umap_dim: int
    umap_neighbors: int
    umap_metric: str
    umap_min_dist: float
    random_seed: int
    eval_frame_step: int
    eval_limit_images: int
    yolo_max_boxes: int
    # Pipeline SAM2 defaults
    sam_every: int
    sam_topk: int
    sam_reinit: int
    empty_cache_interval: int
    # Clustering behavior defaults
    save_models_default: bool
    # Fine-tuning defaults
    finetuning_output_dir: Path
    finetuning_models_dir: Path
    finetuning_frame_interval: int
    finetuning_max_players_per_video: int
    finetuning_detection_conf: float
    finetuning_min_keypoints: int
    finetuning_train_split: float
    finetuning_epochs: int
    finetuning_batch: int
    finetuning_imgsz: int

    def ensure_dirs(self) -> None:
        for p in [
            self.models_dir,
            self.team_models_dir,
            self.pipeline_output_dir,
            self.team_output_dir,
            self.runs_dir,
            self.logs_dir,
            self.finetuning_output_dir,
            self.finetuning_models_dir,
        ]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


def _repo_root() -> Path:
    """Get the repository root directory.
    In dev: finds pyproject.toml parent.
    When frozen: uses AppData/Local/vt1 as the 'root'.
    """
    import sys

    frozen = getattr(sys, "frozen", False)

    if frozen:
        # Running as exe: use AppData/Local/vt1 as root for all data
        import os

        app_data = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.local")))
        return app_data / "vt1"

    # Dev mode: find repo root via pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current file's grandparent (src/vt1 -> src -> repo_root)
    return current.parents[2]


def _read_toml(path: Path) -> Dict[str, Any]:
    """Read TOML file, return empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _env_override(key: str, current: Any) -> Any:
    env_key = "VT1_" + key.upper()
    val = os.getenv(env_key)
    return val if val is not None else current


def _coerce_paths(repo_root: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    def rp(v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (repo_root / p)

    out: Dict[str, Any] = {}
    out["repo_root"] = repo_root
    out["models_dir"] = rp(raw.get("models_dir", "models"))
    out["pose_model"] = rp(raw.get("pose_model", "models/yolo11x-pose.pt"))
    out["yolo_model"] = rp(raw.get("yolo_model", "models/yolo11n.pt"))
    out["default_video_source"] = rp(raw.get("default_video_source", "data_hockey.mp4"))
    out["team_models_dir"] = rp(raw.get("team_models_dir", "models/team_clustering"))
    out["pipeline_output_dir"] = rp(raw.get("pipeline_output_dir", "outputs"))
    out["team_output_dir"] = rp(raw.get("team_output_dir", "outputs/team_clustering"))
    out["runs_dir"] = rp(raw.get("runs_dir", "runs"))
    out["logs_dir"] = rp(raw.get("logs_dir", "logs"))
    out["log_level"] = str(raw.get("log_level", "INFO"))
    # Shared defaults coercion
    out["yolo_conf"] = float(raw.get("yolo_conf", 0.30))
    out["yolo_imgsz"] = int(raw.get("yolo_imgsz", 640))
    out["central_ratio_default"] = float(raw.get("central_ratio_default", 0.6))
    out["siglip_model"] = str(raw.get("siglip_model", "google/siglip-base-patch16-224"))
    out["training_videos_dir"] = rp(
        raw.get("training_videos_dir", "videos_all/CAR_vs_NYR")
    )
    out["videos_glob"] = str(raw.get("videos_glob", "*.mp4"))
    out["build_fps"] = float(raw.get("build_fps", 1.0))
    out["build_min_crop_size"] = int(raw.get("build_min_crop_size", 32))
    out["build_batch_size"] = int(raw.get("build_batch_size", 64))
    out["det_score_thr_default"] = float(raw.get("det_score_thr_default", 0.30))
    out["cluster_k"] = int(raw.get("cluster_k", 2))
    out["umap_dim"] = int(raw.get("umap_dim", 16))
    out["umap_neighbors"] = int(raw.get("umap_neighbors", 15))
    out["umap_metric"] = str(raw.get("umap_metric", "cosine"))
    out["umap_min_dist"] = float(raw.get("umap_min_dist", 0.1))
    out["random_seed"] = int(raw.get("random_seed", 0))
    out["eval_frame_step"] = int(raw.get("eval_frame_step", 30))
    out["eval_limit_images"] = int(raw.get("eval_limit_images", 50))
    out["yolo_max_boxes"] = int(raw.get("yolo_max_boxes", 8))
    # SAM2 defaults coercion
    out["sam_every"] = int(raw.get("sam_every", 1))
    out["sam_topk"] = int(raw.get("sam_topk", 10))
    out["sam_reinit"] = int(raw.get("sam_reinit", 0))
    out["empty_cache_interval"] = int(raw.get("empty_cache_interval", 0))
    # Clustering behavior defaults
    out["save_models_default"] = bool(raw.get("save_models_default", True))
    # Fine-tuning defaults
    out["finetuning_output_dir"] = rp(
        raw.get("finetuning_output_dir", "hockey_pose_dataset")
    )
    out["finetuning_models_dir"] = rp(
        raw.get("finetuning_models_dir", "models/finetuned")
    )
    out["finetuning_frame_interval"] = int(raw.get("finetuning_frame_interval", 30))
    out["finetuning_max_players_per_video"] = int(
        raw.get("finetuning_max_players_per_video", 100)
    )
    out["finetuning_detection_conf"] = float(raw.get("finetuning_detection_conf", 0.5))
    out["finetuning_min_keypoints"] = int(raw.get("finetuning_min_keypoints", 5))
    out["finetuning_train_split"] = float(raw.get("finetuning_train_split", 0.8))
    out["finetuning_epochs"] = int(raw.get("finetuning_epochs", 100))
    out["finetuning_batch"] = int(raw.get("finetuning_batch", 8))
    out["finetuning_imgsz"] = int(raw.get("finetuning_imgsz", 640))
    return out


def _load() -> Config:
    repo = _repo_root()
    defaults = _read_toml(repo / "config_defaults.toml")
    local = _read_toml(repo / "config_local.toml")
    merged: Dict[str, Any] = {**defaults, **local}
    # env overrides
    for k in list(merged.keys()):
        merged[k] = _env_override(k, merged[k])
    paths = _coerce_paths(repo, merged)
    cfg = Config(**paths)
    cfg.ensure_dirs()
    return cfg


def settings() -> Config:
    global _SETTINGS
    if _SETTINGS is None:
        with _LOCK:
            if _SETTINGS is None:
                _SETTINGS = _load()
    return _SETTINGS


def reload() -> Config:
    global _SETTINGS
    with _LOCK:
        _SETTINGS = _load()
        return _SETTINGS
