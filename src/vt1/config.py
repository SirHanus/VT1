from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os
import threading

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
    team_models_dir: Path
    pipeline_output_dir: Path
    team_output_dir: Path
    log_level: str

    def ensure_dirs(self) -> None:
        for p in [self.models_dir, self.team_models_dir, self.pipeline_output_dir, self.team_output_dir]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / 'pyproject.toml').exists():
            return parent
    return Path.cwd()


def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('rb') as f:
        return tomllib.load(f)


def _env_override(key: str, current: Any) -> Any:
    env_key = 'VT1_' + key.upper()
    val = os.getenv(env_key)
    return val if val is not None else current


def _coerce_paths(repo_root: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    def rp(v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (repo_root / p)
    out: Dict[str, Any] = {}
    out['repo_root'] = repo_root
    out['models_dir'] = rp(raw.get('models_dir', 'models'))
    out['pose_model'] = rp(raw.get('pose_model', 'models/yolo11x-pose.pt'))
    out['yolo_model'] = rp(raw.get('yolo_model', 'models/yolo11n.pt'))
    out['team_models_dir'] = rp(raw.get('team_models_dir', 'models/team_clustering'))
    out['pipeline_output_dir'] = rp(raw.get('pipeline_output_dir', 'outputs'))
    out['team_output_dir'] = rp(raw.get('team_output_dir', 'outputs/team_clustering'))
    out['log_level'] = str(raw.get('log_level', 'INFO'))
    return out


def _load() -> Config:
    repo = _repo_root()
    defaults = _read_toml(repo / 'config_defaults.toml')
    local = _read_toml(repo / 'config_local.toml')
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

