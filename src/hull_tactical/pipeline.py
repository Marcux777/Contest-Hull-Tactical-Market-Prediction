"""High-level pipelines tying data, features, and models together."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import data, features
from .models import FEATURE_CFG_DEFAULT, INTENTIONAL_CFG, default_config, train_full_and_predict_model


def train_pipeline(
    data_dir: Optional[Path] = None,
    feature_set: str | None = None,
    feature_cfg: dict | None = None,
    intentional_cfg: dict | None = None,
):
    """Minimal train pipeline using existing feature/model helpers."""
    cfg = default_config()
    df_train, df_test = data.load_raw_data(data_dir)
    fe_cfg = feature_cfg or cfg.feature_cfg or FEATURE_CFG_DEFAULT
    intent_cfg = intentional_cfg or cfg.intentional_cfg or INTENTIONAL_CFG
    cfg.feature_cfg = fe_cfg
    cfg.intentional_cfg = intent_cfg
    train_fe, test_fe, feature_cols, feature_sets, feature_used = features.make_features(
        df_train,
        test_df=df_test,
        target_col="target",
        feature_set=feature_set,
        intentional_cfg=intent_cfg,
        fe_cfg=fe_cfg,
    )
    allocations = train_full_and_predict_model(
        df_train,
        df_test,
        feature_cols,
        target_col="target",
        model_kind="lgb",
        params=cfg.best_params,
        alloc_k=None,
        alloc_alpha=1.0,
        intentional_cfg=intent_cfg,
        fe_cfg=fe_cfg,
        df_train_fe=train_fe,
        df_test_fe=test_fe,
        feature_set=feature_used,
        cfg=cfg,
    )
    return allocations


def make_submission_csv(path: Path, allocations, row_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = allocations.rename("allocation").to_frame()
    out_df.insert(0, "row_id", row_ids)
    out_df.to_csv(path, index=False)
    return path


__all__ = ["train_pipeline", "make_submission_csv"]
