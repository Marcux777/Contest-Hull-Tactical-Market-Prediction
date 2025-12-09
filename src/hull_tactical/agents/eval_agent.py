"""Agent stub for evaluation tasks."""
from __future__ import annotations

from .. import models


def run(df, feature_cols, target_col, n_splits=5):
    return models.time_cv_lightgbm(df, feature_cols, target_col, n_splits=n_splits)
