"""Model artifact serialization helpers.

Goal: allow a clear separation between:
- training (offline) that produces artifacts;
- inference (fast) that only loads artifacts.

Artifacts are kept JSON-based to stay Kaggle-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HullModelArtifact:
    """Minimal information needed to reproduce preprocessing and inference."""

    model_kind: str
    feature_set: str
    target_col: str
    model_files: list[str]

    # Raw feature columns (before preprocess_basic)
    feature_cols: list[str]
    # Final model feature columns (after preprocess_basic)
    model_feature_cols: list[str]
    # Per-feature median values (aligned to `feature_cols`)
    medians: dict[str, float]

    # Training policy (kept for reproducibility / consistent fit_ref usage)
    train_only_scored: bool = False
    weight_scored: float | None = None
    weight_unscored: float | None = None

    # Feature configuration used for feature generation.
    feature_cfg: dict[str, Any] | None = None
    intentional_cfg: dict[str, Any] | None = None

    # Allocation configuration (frozen mapping pred->alloc)
    alloc_k: float | None = None
    alloc_alpha: float | None = None
    allocation_cfg: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "feature_set": self.feature_set,
            "target_col": self.target_col,
            "model_files": list(self.model_files),
            "feature_cols": list(self.feature_cols),
            "model_feature_cols": list(self.model_feature_cols),
            "medians": {k: float(v) for k, v in self.medians.items()},
            "train_only_scored": bool(self.train_only_scored),
            "weight_scored": None if self.weight_scored is None else float(self.weight_scored),
            "weight_unscored": None if self.weight_unscored is None else float(self.weight_unscored),
            "feature_cfg": dict(self.feature_cfg) if isinstance(self.feature_cfg, dict) else None,
            "intentional_cfg": dict(self.intentional_cfg) if isinstance(self.intentional_cfg, dict) else None,
            "alloc_k": None if self.alloc_k is None else float(self.alloc_k),
            "alloc_alpha": None if self.alloc_alpha is None else float(self.alloc_alpha),
            "allocation_cfg": dict(self.allocation_cfg) if isinstance(self.allocation_cfg, dict) else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HullModelArtifact":
        return cls(
            model_kind=str(data.get("model_kind") or "lgb"),
            feature_set=str(data.get("feature_set") or "D_intentional"),
            target_col=str(data.get("target_col") or "target"),
            model_files=list(data.get("model_files") or []),
            feature_cols=list(data.get("feature_cols") or []),
            model_feature_cols=list(data.get("model_feature_cols") or []),
            medians={str(k): float(v) for k, v in (data.get("medians") or {}).items()},
            train_only_scored=bool(data.get("train_only_scored", False)),
            weight_scored=data.get("weight_scored"),
            weight_unscored=data.get("weight_unscored"),
            feature_cfg=data.get("feature_cfg"),
            intentional_cfg=data.get("intentional_cfg"),
            alloc_k=data.get("alloc_k"),
            alloc_alpha=data.get("alloc_alpha"),
            allocation_cfg=data.get("allocation_cfg"),
        )


def save_artifact(path: Path, artifact: HullModelArtifact) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_artifact(path: Path) -> HullModelArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid artifact JSON: expected object at {path}")
    return HullModelArtifact.from_dict(data)


__all__ = [
    "HullModelArtifact",
    "load_artifact",
    "save_artifact",
]
