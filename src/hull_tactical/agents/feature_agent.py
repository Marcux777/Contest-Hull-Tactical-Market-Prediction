"""Agent stub for feature generation."""
from __future__ import annotations

from .. import features


def run(train_df, test_df=None, target_col: str = "target", feature_set: str | None = None, intentional_cfg: dict | None = None, fe_cfg: dict | None = None):
    return features.make_features(train_df, test_df=test_df, target_col=target_col, feature_set=feature_set, intentional_cfg=intentional_cfg, fe_cfg=fe_cfg)
