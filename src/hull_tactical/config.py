"""Configuration loading utilities (YAML/JSON) for Hull Tactical.

Design goals:
- Keep modules side-effect free: no implicit downloads/writes.
- Allow Kaggle/offline usage: if YAML deps/files are missing, fall back cleanly.
- Centralize experiment knobs so notebooks/scripts stop hard-coding dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .allocation import AllocationConfig

try:  # Optional dependency: Kaggle images usually include PyYAML.
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAS_YAML = False


DEFAULT_CONFIG_DIRNAME = "configs"
ENV_CONFIG_DIR = "HT_CONFIG_DIR"


@dataclass(frozen=True)
class HullLoadedConfigs:
    """Bundle of configs resolved from disk (possibly empty)."""

    config_dir: Path | None
    feature_cfg: dict[str, Any]
    intentional_cfg: dict[str, Any]
    lgb_params: dict[str, Any]
    run_cfg: dict[str, Any]
    allocation_cfg: AllocationConfig | None


def _repo_root_from_here() -> Path:
    # src/hull_tactical/config.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def pick_config_dir(config_dir: Path | str | None = None, *, env_var: str = ENV_CONFIG_DIR) -> Path | None:
    """Returns the first existing configs directory found.

    Resolution order:
    1) explicit `config_dir`
    2) env var `HT_CONFIG_DIR`
    3) repo-root `./configs`
    4) CWD `./configs`
    """
    if config_dir is not None:
        cand = Path(config_dir)
        return cand if cand.exists() else None

    env_val = os.environ.get(env_var)
    if env_val:
        cand = Path(env_val)
        if cand.exists():
            return cand

    cand = _repo_root_from_here() / DEFAULT_CONFIG_DIRNAME
    if cand.exists():
        return cand

    cand = Path.cwd() / DEFAULT_CONFIG_DIRNAME
    if cand.exists():
        return cand

    return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Loads a config dict from YAML or JSON.

    If the file doesn't exist or parsing fails, returns an empty dict.
    """
    try:
        if not path.exists():
            return {}
    except OSError:
        return {}

    try:
        if _HAS_YAML:
            data = yaml.safe_load(_read_text(path))  # type: ignore[union-attr]
        else:
            data = json.loads(_read_text(path))
    except Exception:
        return {}

    if data is None:
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def load_feature_configs(config_dir: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads `feature_cfg` and `intentional_cfg` from `features.yaml`."""
    cfg_dir = pick_config_dir(config_dir)
    if cfg_dir is None:
        return {}, {}
    raw = load_yaml_or_json(cfg_dir / "features.yaml")
    feature_cfg = raw.get("feature_cfg") if isinstance(raw.get("feature_cfg"), dict) else {}
    intentional_cfg = raw.get("intentional_cfg") if isinstance(raw.get("intentional_cfg"), dict) else {}
    return dict(feature_cfg), dict(intentional_cfg)


def load_lgb_params(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Loads LightGBM params from `lgb.yaml` (key: `best_params`)."""
    cfg_dir = pick_config_dir(config_dir)
    if cfg_dir is None:
        return {}
    raw = load_yaml_or_json(cfg_dir / "lgb.yaml")
    best = raw.get("best_params") if isinstance(raw.get("best_params"), dict) else raw
    return dict(best) if isinstance(best, dict) else {}


def load_run_config(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Loads run-level knobs (splits, seeds, defaults) from `run.yaml`."""
    cfg_dir = pick_config_dir(config_dir)
    if cfg_dir is None:
        return {}
    return load_yaml_or_json(cfg_dir / "run.yaml")


def allocation_config_from_dict(data: dict[str, Any] | None) -> AllocationConfig | None:
    """Best-effort conversion of a dict into AllocationConfig."""
    if not data:
        return None
    if not isinstance(data, dict):
        return None

    allowed = {f.name for f in AllocationConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in allowed}
    try:
        return AllocationConfig(**kwargs)
    except Exception:
        return None


def load_all_configs(config_dir: Path | str | None = None) -> HullLoadedConfigs:
    """Loads the full bundle (features/lgb/run) from disk (if present)."""
    cfg_dir = pick_config_dir(config_dir)
    feature_cfg, intentional_cfg = load_feature_configs(cfg_dir)
    lgb_params = load_lgb_params(cfg_dir)
    run_cfg = load_run_config(cfg_dir)
    allocation_cfg = allocation_config_from_dict(run_cfg.get("allocation") if isinstance(run_cfg, dict) else None)
    return HullLoadedConfigs(
        config_dir=cfg_dir,
        feature_cfg=feature_cfg,
        intentional_cfg=intentional_cfg,
        lgb_params=lgb_params,
        run_cfg=run_cfg,
        allocation_cfg=allocation_cfg,
    )


__all__ = [
    "AllocationConfig",
    "DEFAULT_CONFIG_DIRNAME",
    "ENV_CONFIG_DIR",
    "HullLoadedConfigs",
    "allocation_config_from_dict",
    "load_all_configs",
    "load_feature_configs",
    "load_lgb_params",
    "load_run_config",
    "load_yaml_or_json",
    "pick_config_dir",
]

