"""High-level pipelines tying data, features, and models together."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import data, features
from . import models as m
from .models import FEATURE_CFG_DEFAULT, INTENTIONAL_CFG, default_config, train_full_and_predict_model


def train_pipeline(
    data_dir: Optional[Path] = None,
    feature_set: str | None = None,
    feature_cfg: dict | None = None,
    intentional_cfg: dict | None = None,
    df_train=None,
    df_test=None,
    cfg: m.HullConfig | None = None,
):
    """
    Minimal train pipeline using existing feature/model helpers.
    Se df_train/df_test forem fornecidos, são usados diretamente; caso contrário, carrega de data_dir.
    """
    cfg = cfg or default_config()
    if df_train is None or df_test is None:
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


def choose_best_training_variant(
    df,
    feature_cols,
    target_col: str,
    variants: list[dict] | None = None,
    cfg: m.HullConfig | None = None,
):
    """
    Roda CV para variantes full / weighted / scored_only e retorna a melhor pelo Sharpe médio.
    Cada variante é um dict com chaves: name, train_only_scored, weight_unscored.
    """
    cfg_use = m._resolve_cfg(cfg)
    if variants is None:
        variants = [
            {"name": "full", "train_only_scored": False, "weight_unscored": 1.0},
            {"name": "weighted_0.2", "train_only_scored": False, "weight_unscored": 0.2},
            {"name": "scored_only", "train_only_scored": True, "weight_unscored": None},
        ]
    results = []
    for variant in variants:
        metrics = m.time_cv_lightgbm(
            df,
            feature_cols,
            target_col,
            n_splits=4,
            val_frac=0.12,
            weight_scored=1.0,
            weight_unscored=variant.get("weight_unscored"),
            train_only_scored=variant.get("train_only_scored", False),
            cfg=cfg_use,
            log_prefix=f"[{variant.get('name','variant')}]",
        )
        summary = m.summarize_cv_metrics(metrics) or {}
        results.append(
            {
                "name": variant.get("name"),
                "train_only_scored": variant.get("train_only_scored", False),
                "weight_unscored": variant.get("weight_unscored"),
                "metrics": metrics,
                "summary": summary,
            }
        )
    best = None
    for res in results:
        sharpe_mean = res["summary"].get("sharpe_mean")
        if sharpe_mean is None or not (sharpe_mean == sharpe_mean):
            continue
        if best is None or sharpe_mean > best["summary"].get("sharpe_mean", float("-inf")):
            best = res
    return {"best": best, "all": results}


__all__ = ["train_pipeline", "make_submission_csv", "choose_best_training_variant"]


def run_time_cv(
    df_train,
    feature_set: str,
    target_col: str = "target",
    cfg: m.HullConfig | None = None,
    n_splits: int = 5,
    val_frac: float = 0.1,
):
    """
    Wrapper para rodar CV temporal com o feature set escolhido usando o pipeline de features oficial.
    Retorna a lista de métricas (uma por fold).
    """
    cfg_use = cfg or default_config()
    train_fe, _, feature_cols, feature_sets, feature_used = features.make_features(
        df_train,
        test_df=None,
        target_col=target_col,
        feature_set=feature_set,
        intentional_cfg=cfg_use.intentional_cfg,
        fe_cfg=cfg_use.feature_cfg,
    )
    cols = feature_sets.get(feature_used, feature_cols)
    return m.time_cv_lightgbm(train_fe, cols, target_col, n_splits=n_splits, val_frac=val_frac, cfg=cfg_use)
