"""High-level pipelines tying data, features, and models together."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import data, features
from .models import (
    BEST_PARAMS,
    FEATURE_CFG_DEFAULT,
    INTENTIONAL_CFG,
    train_full_and_predict_model,
)


def train_pipeline(
    data_dir: Optional[Path] = None,
    feature_set: str | None = None,
    feature_cfg: dict | None = None,
    intentional_cfg: dict | None = None,
):
    """Minimal train pipeline using existing feature/model helpers."""
    df_train, df_test = data.load_raw_data(data_dir)
    fe_cfg = FEATURE_CFG_DEFAULT if feature_cfg is None else feature_cfg
    intent_cfg = INTENTIONAL_CFG if intentional_cfg is None else intentional_cfg
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
        params=BEST_PARAMS,
        alloc_k=None,
        alloc_alpha=1.0,
        intentional_cfg=intent_cfg,
        fe_cfg=fe_cfg,
        df_train_fe=train_fe,
        df_test_fe=test_fe,
        feature_set=feature_used,
    )
    return allocations


def make_submission_csv(path: Path, allocations, row_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df = allocations.rename("allocation").to_frame()
    out_df.insert(0, "row_id", row_ids)
    out_df.to_csv(path, index=False)
    return path


__all__ = ["train_pipeline", "make_submission_csv"]
