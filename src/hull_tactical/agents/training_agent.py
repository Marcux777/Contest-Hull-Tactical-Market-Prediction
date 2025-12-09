"""Agent stub for training orchestration."""
from __future__ import annotations

from .. import pipeline


def run(config=None):
    # Config pode ser um dict simples; pipeline usa defaults se None
    data_dir = None if config is None else config.get("data_dir")
    feature_set = None if config is None else config.get("feature_set")
    feature_cfg = None if config is None else config.get("feature_cfg")
    intentional_cfg = None if config is None else config.get("intentional_cfg")
    return pipeline.train_pipeline(
        data_dir=data_dir,
        feature_set=feature_set,
        feature_cfg=feature_cfg,
        intentional_cfg=intentional_cfg,
    )
