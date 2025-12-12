"""
Modeling, CV, and allocation helpers shared by the Hull Tactical notebook.
Kept dependency-light so it can run inside Kaggle without extra installs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

try:
    import xgboost as xgb  # type: ignore

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor  # type: ignore

    HAS_CAT = True
except Exception:  # pragma: no cover
    HAS_CAT = False

from .features import (
    FEATURE_CFG_DEFAULT as HF_FEATURE_CFG_DEFAULT,
    INTENTIONAL_CFG as HF_INTENTIONAL_CFG,
    build_feature_sets,
    make_features,
    align_feature_frames,
    preprocess_basic,
)
from .allocation import AllocationConfig, apply_allocation_strategy, calibrate_global_scale
from . import config as ht_config

SEED = 42
ALLOC_K = 1.0
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0

MARKET_COL: str | None = None
RF_COL: str | None = None
IS_SCORED_COL: str | None = None

INTENTIONAL_CFG = dict(HF_INTENTIONAL_CFG)
FEATURE_CFG_DEFAULT = dict(HF_FEATURE_CFG_DEFAULT)

BEST_PARAMS = {
    "learning_rate": 0.010897827897948612,
    "num_leaves": 42,
    "min_data_in_leaf": 388,
    "feature_fraction": 0.7079417306723327,
    "bagging_fraction": 0.6681010468585586,
    "bagging_freq": 10,
    "lambda_l1": 0.9244442250838115,
    "lambda_l2": 0.16637075433315218,
    "objective": "regression",
    "metric": "rmse",
    "seed": SEED,
    "verbosity": -1,
}


@dataclass
class HullConfig:
    market_col: str | None = MARKET_COL
    rf_col: str | None = RF_COL
    is_scored_col: str | None = IS_SCORED_COL
    intentional_cfg: dict | None = None
    feature_cfg: dict | None = None
    best_params: dict | None = None
    alloc_k: float = ALLOC_K
    min_investment: float = MIN_INVESTMENT
    max_investment: float = MAX_INVESTMENT


def default_config(config_dir: Path | str | None = None) -> HullConfig:
    """Returns a default HullConfig.

    If `configs/` is available, its YAML/JSON values are merged on top of the
    in-code defaults (so notebooks/scripts can iterate by editing YAML only).
    """
    feature_cfg = dict(FEATURE_CFG_DEFAULT)
    intentional_cfg = dict(INTENTIONAL_CFG)
    best_params = dict(BEST_PARAMS)
    alloc_k = ALLOC_K
    min_investment = MIN_INVESTMENT
    max_investment = MAX_INVESTMENT

    loaded = ht_config.load_all_configs(config_dir)
    if loaded.feature_cfg:
        feature_cfg.update(loaded.feature_cfg)
    if loaded.intentional_cfg:
        intentional_cfg.update(loaded.intentional_cfg)
    if loaded.lgb_params:
        best_params.update(loaded.lgb_params)
    if isinstance(loaded.run_cfg, dict):
        if "alloc_k" in loaded.run_cfg:
            alloc_k = float(loaded.run_cfg["alloc_k"])
        if "min_investment" in loaded.run_cfg:
            min_investment = float(loaded.run_cfg["min_investment"])
        if "max_investment" in loaded.run_cfg:
            max_investment = float(loaded.run_cfg["max_investment"])

    return HullConfig(
        market_col=MARKET_COL,
        rf_col=RF_COL,
        is_scored_col=IS_SCORED_COL,
        intentional_cfg=intentional_cfg,
        feature_cfg=feature_cfg,
        best_params=best_params,
        alloc_k=alloc_k,
        min_investment=min_investment,
        max_investment=max_investment,
    )


def _resolve_cfg(cfg: HullConfig | None) -> HullConfig:
    return cfg if cfg is not None else default_config()


def set_data_columns(market_col: str | None = None, rf_col: str | None = None, is_scored_col: str | None = None) -> None:
    """Setter explícito para colunas-chave; evita depender de globais implícitas."""
    global MARKET_COL, RF_COL, IS_SCORED_COL
    MARKET_COL = market_col or MARKET_COL
    RF_COL = rf_col or RF_COL
    IS_SCORED_COL = is_scored_col or IS_SCORED_COL


def evaluate_baselines(train_df, feature_cols, target_col, cfg: HullConfig | None = None):
    """Baselines simples em split 80/20: alocação constante e modelo linear (Ridge)."""
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    tr_frac = int(len(train_df) * 0.8)
    tr = train_df.iloc[:tr_frac].copy()
    va = train_df.iloc[tr_frac:].copy()
    tr_aligned, va_aligned, cols_use = align_feature_frames(tr, va, feature_cols)
    tr_proc, keep_cols, medians = preprocess_basic(tr_aligned, cols_use)
    va_proc, _, _ = preprocess_basic(va_aligned, cols_use, ref_cols=keep_cols, ref_medians=medians)

    X_tr = tr_proc.drop(columns=[target_col], errors="ignore")
    X_va = va_proc.drop(columns=[target_col], errors="ignore")
    y_tr = tr[target_col]
    y_va = va[target_col]

    alloc_const = pd.Series(1.0, index=va.index)
    sharpe_const, const_details = adjusted_sharpe_score(
        va, alloc_const, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
    )

    lin = Ridge(alpha=1.0, random_state=SEED)
    lin.fit(X_tr, y_tr)
    pred_lin = lin.predict(X_va)
    best_k_lin, best_alpha_lin, _ = optimize_allocation_scale(
        pred_lin, va, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
    )
    alloc_lin = map_return_to_alloc(pred_lin, k=best_k_lin, intercept=best_alpha_lin)
    sharpe_lin, lin_details = adjusted_sharpe_score(
        va, pd.Series(alloc_lin, index=va.index), market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
    )
    rmse_lin = mean_squared_error(y_va, pred_lin) ** 0.5

    return {
        "const": {"sharpe": sharpe_const, "details": const_details},
        "ridge": {
            "sharpe": sharpe_lin,
            "rmse": rmse_lin,
            "best_k": best_k_lin,
            "best_alpha": best_alpha_lin,
            "details": lin_details,
        },
    }


def constant_allocation_cv(df, n_splits=5, val_frac=0.1, cfg: HullConfig | None = None):
    """Baseline: alocação constante (1.0) por fold temporal."""
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    splits = make_time_splits(df, date_col="date_id", n_splits=n_splits, val_frac=val_frac)
    metrics = []
    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_val = df.loc[mask_val].copy()
        if df_val.empty:
            continue
        alloc_const = pd.Series(1.0, index=df_val.index)
        sharpe_const, details = adjusted_sharpe_score(
            df_val, alloc_const, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
        )
        n_scored = int(df_val[is_scored_col].sum()) if is_scored_col and is_scored_col in df_val.columns else len(df_val)
        metrics.append(
            {
                "fold": i,
                "sharpe": sharpe_const,
                "n_val": len(df_val),
                "n_scored": n_scored,
                "strategy_vol": details.get("strategy_vol"),
            }
        )
    return metrics


def time_cv_lightgbm_fitref(
    df,
    feature_set_name,
    target_col,
    n_splits=4,
    val_frac=0.12,
    params_override=None,
    num_boost_round=200,
    weight_scored: float | None = None,
    weight_unscored: float | None = None,
    train_only_scored: bool = False,
    allocation_cfg: AllocationConfig | None = None,
    cfg: HullConfig | None = None,
):
    """CV temporal recalculando features por fold (winsor/clipping/z-score usando apenas o treino)."""
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    intent_cfg = cfg_resolved.intentional_cfg or INTENTIONAL_CFG
    fe_cfg = cfg_resolved.feature_cfg or FEATURE_CFG_DEFAULT
    splits = make_time_splits(df, n_splits=n_splits, val_frac=val_frac)
    metrics = []
    if not splits:
        return metrics
    params_use = dict(cfg_resolved.best_params or BEST_PARAMS)
    if params_override:
        params_use.update(params_override)
    params_use["metric"] = "rmse"
    params_use["objective"] = "regression"
    params_use["seed"] = SEED

    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored

    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr = df.loc[mask_tr].copy()
        df_val = df.loc[mask_val].copy()
        if df_val.empty or df_tr.empty:
            continue
        if train_only_scored and is_scored_col and is_scored_col in df_tr.columns:
            df_tr = df_tr.loc[df_tr[is_scored_col] == 1]
            if df_tr.empty:
                continue

        df_tr_fe, fs_tr = build_feature_sets(df_tr, target_col, intentional_cfg=intent_cfg, fe_cfg=fe_cfg, fit_ref=None)
        if df_tr_fe.columns.has_duplicates:
            df_tr_fe = df_tr_fe.loc[:, ~df_tr_fe.columns.duplicated(keep="last")]
        cols = fs_tr.get(feature_set_name, next(iter(fs_tr.values())))
        df_val_fe, _ = build_feature_sets(df_val, target_col, intentional_cfg=intent_cfg, fe_cfg=fe_cfg, fit_ref=df_tr_fe)
        if df_val_fe.columns.has_duplicates:
            df_val_fe = df_val_fe.loc[:, ~df_val_fe.columns.duplicated(keep="last")]
        df_val_fe = df_val_fe.reindex(columns=df_tr_fe.columns, fill_value=0)

        df_tr_proc, keep_cols, medians = preprocess_basic(df_tr_fe, cols)
        df_val_proc, _, _ = preprocess_basic(df_val_fe, cols, ref_cols=keep_cols, ref_medians=medians)
        X_tr = df_tr_proc
        y_tr = df_tr_fe[target_col]
        X_val = df_val_proc
        y_val = df_val_fe[target_col]

        model = lgb.LGBMRegressor(**params_use, n_estimators=num_boost_round, random_state=SEED)
        sample_weight = (
            make_sample_weight(
                df_tr_fe,
                weight_scored=weight_scored,
                weight_unscored=weight_unscored,
                is_scored_col=is_scored_col,
            )
            if use_weights
            else None
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        pred_val = model.predict(X_val)
        best_k, best_alpha, _ = optimize_allocation_scale(
            pred_val,
            df_val_fe,
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
            allocation_cfg=allocation_cfg,
        )
        if allocation_cfg is None:
            alloc_val = map_return_to_alloc(pred_val, k=best_k, intercept=best_alpha)
        else:
            alloc_val = apply_allocation_strategy(pred_val, df_val_fe, k=float(best_k), alpha=float(best_alpha), cfg=allocation_cfg)
        sharpe_val, details = adjusted_sharpe_score(
            df_val_fe, alloc_val, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
        )
        metrics.append(
            {
                "fold": i,
                "sharpe": sharpe_val,
                "best_k": best_k,
                "best_alpha": best_alpha,
                "n_scored": int(df_val_fe[is_scored_col].sum()) if is_scored_col and is_scored_col in df_val_fe.columns else len(df_val_fe),
                "strategy_vol": details.get("strategy_vol"),
            }
        )
    return metrics


def run_cv_preds_fitref(
    df: pd.DataFrame,
    feature_set_name: str,
    target_col: str,
    *,
    model_kind: str = "lgb",
    params_override: dict | None = None,
    n_splits: int = 4,
    val_frac: float = 0.12,
    num_boost_round: int = 200,
    seed: int | None = None,
    weight_scored: float | None = None,
    weight_unscored: float | None = None,
    train_only_scored: bool = False,
    allocation_cfg: AllocationConfig | None = None,
    keep_context_cols: tuple[str, ...] | None = ("date_id", "regime_std_20", "regime_high_vol"),
    cfg: HullConfig | None = None,
):
    """Gera predições OOF com features recalculadas por fold (fit_ref).

    Retorna:
    - metrics: métricas por fold (com calibração local de k/alpha no próprio fold).
    - pred_df: dataframe OOF com colunas (pred_return/alloc/target/market/rf/is_scored/fold/...).
    """
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    intent_cfg = cfg_resolved.intentional_cfg or INTENTIONAL_CFG
    fe_cfg = cfg_resolved.feature_cfg or FEATURE_CFG_DEFAULT
    splits = make_time_splits(df, n_splits=n_splits, val_frac=val_frac)
    if not splits:
        return [], pd.DataFrame()

    params_use = dict(cfg_resolved.best_params or BEST_PARAMS)
    if params_override:
        params_use.update(params_override)
    params_use["metric"] = "rmse"
    params_use["objective"] = "regression"
    seed_use = SEED if seed is None else seed

    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored

    metrics: list[dict] = []
    preds: list[pd.DataFrame] = []

    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr_raw = df.loc[mask_tr].copy()
        df_val_raw = df.loc[mask_val].copy()
        if df_tr_raw.empty or df_val_raw.empty:
            continue
        if train_only_scored and is_scored_col and is_scored_col in df_tr_raw.columns:
            df_tr_raw = df_tr_raw.loc[df_tr_raw[is_scored_col] == 1]
            if df_tr_raw.empty:
                continue

        df_tr_fe, fs_tr = build_feature_sets(df_tr_raw, target_col, intentional_cfg=intent_cfg, fe_cfg=fe_cfg, fit_ref=None)
        if df_tr_fe.columns.has_duplicates:
            df_tr_fe = df_tr_fe.loc[:, ~df_tr_fe.columns.duplicated(keep="last")]
        cols = fs_tr.get(feature_set_name, next(iter(fs_tr.values())))

        df_val_fe, _ = build_feature_sets(df_val_raw, target_col, intentional_cfg=intent_cfg, fe_cfg=fe_cfg, fit_ref=df_tr_fe)
        if df_val_fe.columns.has_duplicates:
            df_val_fe = df_val_fe.loc[:, ~df_val_fe.columns.duplicated(keep="last")]
        df_val_fe = df_val_fe.reindex(columns=df_tr_fe.columns, fill_value=0)

        df_tr_proc, keep_cols, medians = preprocess_basic(df_tr_fe, cols)
        df_val_proc, _, _ = preprocess_basic(df_val_fe, cols, ref_cols=keep_cols, ref_medians=medians)
        X_tr = df_tr_proc
        y_tr = df_tr_fe[target_col]
        X_val = df_val_proc
        y_val = df_val_fe[target_col]

        sample_weight = (
            make_sample_weight(
                df_tr_fe,
                weight_scored=weight_scored,
                weight_unscored=weight_unscored,
                is_scored_col=is_scored_col,
            )
            if use_weights
            else None
        )

        if model_kind == "lgb":
            model = lgb.LGBMRegressor(**params_use, n_estimators=num_boost_round, random_state=seed_use)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            pred = model.predict(X_val)
        elif model_kind == "ridge":
            alpha = params_override.get("alpha", 1.0) if params_override else 1.0
            model = Ridge(alpha=float(alpha), random_state=seed_use)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            pred = model.predict(X_val)
        elif model_kind == "xgb" and HAS_XGB:
            default = {
                "n_estimators": num_boost_round,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed_use,
                "objective": "reg:squarederror",
            }
            if params_override:
                default.update(params_override)
            model = xgb.XGBRegressor(**default)
            fit_kwargs = {"verbose": False}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(X_tr, y_tr, **fit_kwargs)
            pred = model.predict(X_val)
        elif model_kind == "cat" and HAS_CAT:
            default = {
                "depth": 6,
                "learning_rate": 0.05,
                "iterations": num_boost_round,
                "loss_function": "RMSE",
                "random_seed": seed_use,
                "verbose": False,
            }
            if params_override:
                default.update(params_override)
            model = CatBoostRegressor(**default)
            fit_kwargs = {"verbose": False}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(X_tr, y_tr, **fit_kwargs)
            pred = model.predict(X_val)
        else:
            continue

        best_k, best_alpha, _ = optimize_allocation_scale(
            pred,
            df_val_fe,
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
            allocation_cfg=allocation_cfg,
        )
        if allocation_cfg is None:
            alloc = map_return_to_alloc(pred, k=best_k, intercept=best_alpha)
        else:
            alloc = apply_allocation_strategy(pred, df_val_fe, k=float(best_k), alpha=float(best_alpha), cfg=allocation_cfg)

        sharpe_adj, details = adjusted_sharpe_score(
            df_val_fe,
            pd.Series(alloc, index=df_val_fe.index),
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
        )
        n_scored = int(df_val_fe[is_scored_col].sum()) if is_scored_col and is_scored_col in df_val_fe.columns else len(df_val_fe)
        if pd.notna(sharpe_adj) and np.isfinite(sharpe_adj):
            metrics.append(
                {
                    "fold": i,
                    "sharpe": float(sharpe_adj),
                    "best_alpha": float(best_alpha),
                    "best_k": float(best_k),
                    "n_val": int(len(df_val_fe)),
                    "n_scored": int(n_scored),
                    "strategy_vol": details.get("strategy_vol"),
                    "seed": int(seed_use),
                }
            )

        preds.append(
            pd.DataFrame(
                {
                    "alloc": alloc,
                    "pred_return": pred,
                    "target": y_val,
                    market_col or "forward_returns": df_val_fe.get(market_col) if market_col else df_val_fe.get("forward_returns"),
                    rf_col or "risk_free_rate": df_val_fe.get(rf_col) if rf_col else df_val_fe.get("risk_free_rate"),
                    is_scored_col or "is_scored": df_val_fe[is_scored_col] if is_scored_col and is_scored_col in df_val_fe.columns else 1,
                    "fold": i,
                    "seed": seed_use,
                    **{
                        c: df_val_fe[c]
                        for c in (keep_context_cols or ())
                        if c in df_val_fe.columns and c not in {"alloc", "pred_return", "target"}
                    },
                },
                index=df_val_fe.index,
            )
        )

    return metrics, (pd.concat(preds) if preds else pd.DataFrame())


def time_cv_lightgbm_fitref_oof(
    df: pd.DataFrame,
    feature_set_name: str,
    target_col: str,
    *,
    n_splits: int = 4,
    val_frac: float = 0.12,
    params_override: dict | None = None,
    num_boost_round: int = 200,
    weight_scored: float | None = None,
    weight_unscored: float | None = None,
    train_only_scored: bool = False,
    allocation_cfg: AllocationConfig | None = None,
    return_oof_df: bool = False,
    cfg: HullConfig | None = None,
):
    """CV fit_ref com calibração global de allocation em OOF concatenado."""
    _, pred_df = run_cv_preds_fitref(
        df,
        feature_set_name,
        target_col,
        model_kind="lgb",
        params_override=params_override,
        n_splits=n_splits,
        val_frac=val_frac,
        num_boost_round=num_boost_round,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        train_only_scored=train_only_scored,
        allocation_cfg=None,  # OOF global calibra depois; evita "melhor k por fold"
        cfg=cfg,
    )

    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL or "is_scored"

    scored = score_oof_predictions_with_allocation(
        pred_df,
        allocation_cfg=allocation_cfg or AllocationConfig(),
        pred_col="pred_return",
        market_col=market_col,
        rf_col=rf_col,
        is_scored_col=is_scored_col,
        fold_col="fold",
    )
    metrics = scored["metrics"]
    summary = summarize_cv_metrics(metrics) or {}
    out = {
        "metrics": metrics,
        "summary": summary,
        "oof_sharpe": scored.get("oof_sharpe"),
        "oof_details": scored.get("oof_details"),
        "best_k": getattr(scored.get("calibration"), "best_k", np.nan),
        "best_alpha": getattr(scored.get("calibration"), "best_alpha", np.nan),
        "search_results": getattr(scored.get("calibration"), "results", pd.DataFrame()),
    }
    if return_oof_df:
        out["oof_pred_df"] = pred_df.assign(alloc_calibrated=scored["alloc"])
    return out


def make_time_splits(df, date_col="date_id", n_splits=5, val_frac=0.1, min_train_frac=0.5):
    """Splits temporais por blocos contíguos de datas com janela de treino expandindo."""
    if date_col not in df.columns:
        return []
    dates = np.array(sorted(df[date_col].unique()))
    n_dates = len(dates)
    if n_dates < 3:
        return []
    val_size = max(1, int(n_dates * val_frac))
    min_train = max(1, int(n_dates * min_train_frac))

    splits = []
    if "is_scored" in df.columns:
        scored_per_date = df.groupby(date_col)["is_scored"].sum().reindex(dates, fill_value=0).to_numpy()
        total_scored = scored_per_date.sum()
        target_scored = total_scored / n_splits if total_scored > 0 else val_size
        idx = min_train
        for _ in range(n_splits):
            if idx >= n_dates:
                break
            accum = 0.0
            end = idx
            while end < n_dates and accum < target_scored:
                accum += scored_per_date[end]
                end += 1
            if end <= idx:
                end = min(n_dates, idx + val_size)
            val_start = idx
            val_end = min(end, n_dates)
            train_mask = df[date_col].isin(dates[:val_start])
            val_mask = df[date_col].isin(dates[val_start:val_end])
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                idx = val_end
                continue
            if df.loc[val_mask, "is_scored"].sum() == 0:
                idx = val_end
                continue
            splits.append((train_mask, val_mask))
            idx = val_end
        return splits
    else:
        for i in range(n_splits):
            train_end = min_train + i * val_size
            val_start = train_end
            val_end = val_start + val_size
            if val_end > n_dates:
                break
            train_mask = df[date_col].isin(dates[:train_end])
            val_mask = df[date_col].isin(dates[val_start:val_end])
            if val_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            splits.append((train_mask, val_mask))
        return splits


def compute_strategy_returns(pred_alloc, market_returns):
    return pred_alloc * market_returns


def map_return_to_alloc(pred_return, k=ALLOC_K, intercept=1.0):
    """Aplica allocation=clip(intercept + k * pred, MIN_INVESTMENT, MAX_INVESTMENT); regra base 1 + k*pred."""
    return np.clip(intercept + k * pred_return, MIN_INVESTMENT, MAX_INVESTMENT)


def adjusted_sharpe_score(
    df,
    allocation,
    market_col=None,
    rf_col=None,
    is_scored_col=None,
    trading_days_per_yr=252,
    clip_alloc=True,
):
    """Replica a métrica oficial (modified Sharpe) aplicada só em linhas is_scored==1."""
    market_use = market_col or MARKET_COL or "forward_returns"
    rf_use = rf_col or RF_COL or "risk_free_rate"
    scored_col = is_scored_col or IS_SCORED_COL

    required = [market_use, rf_use]
    if not all(col in df.columns for col in required):
        return np.nan, {}

    eval_df = df.copy()
    if scored_col and scored_col in eval_df.columns:
        eval_df = eval_df.loc[eval_df[scored_col] == 1].copy()
    if len(eval_df) == 0:
        return np.nan, {}

    alloc_series = pd.Series(allocation, index=df.index)
    if clip_alloc:
        alloc_series = alloc_series.clip(MIN_INVESTMENT, MAX_INVESTMENT)
    alloc_series = alloc_series.reindex(eval_df.index)

    eval_df["position"] = alloc_series
    strat_returns = eval_df[rf_use] * (1 - eval_df["position"]) + eval_df["position"] * eval_df[market_use]
    strategy_excess = strat_returns - eval_df[rf_use]
    strategy_excess_cum = (1 + strategy_excess).prod()
    strategy_mean_excess = strategy_excess_cum ** (1 / len(eval_df)) - 1
    strategy_std = strat_returns.std(ddof=0)

    if strategy_std <= 0:
        return -np.inf, {}

    sharpe = strategy_mean_excess / (strategy_std + 1e-12) * np.sqrt(trading_days_per_yr)
    strategy_vol = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    market_excess = eval_df[market_use] - eval_df[rf_use]
    market_excess_cum = (1 + market_excess).prod()
    market_mean_excess = market_excess_cum ** (1 / len(eval_df)) - 1
    market_std = eval_df[market_use].std(ddof=0)
    if market_std <= 0:
        return -np.inf, {}
    market_vol = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    excess_vol = max(0.0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0.0
    vol_penalty = 1 + excess_vol

    return_gap = max(0.0, (market_mean_excess - strategy_mean_excess) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    adjusted = sharpe / (vol_penalty * return_penalty)
    details = {
        "sharpe_raw": sharpe,
        "strategy_vol": strategy_vol,
        "market_vol": market_vol,
        "strategy_mean_excess": strategy_mean_excess,
        "market_mean_excess": market_mean_excess,
        "vol_penalty": vol_penalty,
        "return_penalty": return_penalty,
    }
    return float(min(adjusted, 1_000_000)), details


def optimize_allocation_scale(
    pred_returns,
    df_eval,
    k_grid=None,
    alpha_grid=None,
    market_col=None,
    rf_col=None,
    is_scored_col=None,
    allocation_cfg: AllocationConfig | None = None,
):
    """Busca em (k, alpha) para maximizar a métrica oficial no conjunto de avaliação.

    Se `allocation_cfg` for fornecido, otimiza o fluxo completo de allocation (regime/risk/smoothing).
    Caso contrário, cai para o mapeamento linear simples (alpha + k * pred, com clip).
    """
    if df_eval is None or len(df_eval) == 0:
        return np.nan, np.nan, np.nan

    market_use = market_col or MARKET_COL or "forward_returns"
    rf_use = rf_col or RF_COL or "risk_free_rate"
    scored_col = is_scored_col or IS_SCORED_COL

    if allocation_cfg is None:
        # Compatível com a regra base 1 + k*pred: sem regime/risk/smoothing por padrão.
        cfg = AllocationConfig(
            k_grid=np.linspace(0.0, 3.0, 61) if k_grid is None else k_grid,
            alpha_grid=np.asarray([0.8, 1.0, 1.2]) if alpha_grid is None else alpha_grid,
            high_vol_k_factor=1.0,
            risk_col=None,
            smooth_alpha=None,
            delta_cap=None,
            standardize_window=None,
            standardize_clip=None,
        )
    else:
        cfg = allocation_cfg
        if k_grid is not None or alpha_grid is not None:
            cfg = replace(cfg, k_grid=k_grid if k_grid is not None else cfg.k_grid, alpha_grid=alpha_grid if alpha_grid is not None else cfg.alpha_grid)

    pred_series = pd.Series(pred_returns, index=df_eval.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eval_df = df_eval.copy()
    pred_col = "__pred_return"
    while pred_col in eval_df.columns:
        pred_col = f"_{pred_col}"
    eval_df[pred_col] = pred_series

    calib = calibrate_global_scale(
        eval_df,
        cfg,
        pred_col=pred_col,
        market_col=market_use,
        rf_col=rf_use,
        is_scored_col=scored_col or "is_scored",
    )
    return calib.best_k, calib.best_alpha, calib.best_score


def score_oof_predictions_with_allocation(
    pred_df: pd.DataFrame,
    *,
    allocation_cfg: AllocationConfig | None = None,
    pred_col: str = "pred_return",
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str = "is_scored",
    fold_col: str = "fold",
):
    """Calibra (k, alpha) globalmente em OOF e reporta Sharpe por fold e no OOF inteiro."""
    if pred_df is None or pred_df.empty or pred_col not in pred_df.columns:
        return {
            "calibration": None,
            "oof_sharpe": np.nan,
            "oof_details": {},
            "metrics": [],
            "alloc": pd.Series(dtype=float),
        }

    market_use = market_col or MARKET_COL or "forward_returns"
    rf_use = rf_col or RF_COL or "risk_free_rate"
    cfg = allocation_cfg or AllocationConfig()

    calib = calibrate_global_scale(
        pred_df,
        cfg,
        pred_col=pred_col,
        market_col=market_use,
        rf_col=rf_use,
        is_scored_col=is_scored_col,
    )
    if not np.isfinite(calib.best_k) or not np.isfinite(calib.best_alpha):
        return {
            "calibration": calib,
            "oof_sharpe": np.nan,
            "oof_details": {},
            "metrics": [],
            "alloc": pd.Series(index=pred_df.index, dtype=float),
        }

    alloc = apply_allocation_strategy(
        pred_df[pred_col],
        pred_df,
        k=float(calib.best_k),
        alpha=float(calib.best_alpha),
        cfg=cfg,
    )
    oof_sharpe, oof_details = adjusted_sharpe_score(
        pred_df,
        alloc,
        market_col=market_use,
        rf_col=rf_use,
        is_scored_col=is_scored_col,
    )

    metrics: list[dict] = []
    if fold_col in pred_df.columns and pred_df[fold_col].notna().any():
        folds = sorted(pd.Series(pred_df[fold_col]).dropna().unique().tolist())
    else:
        folds = [None]

    for fold in folds:
        if fold is None:
            df_slice = pred_df
            alloc_slice = alloc
            fold_label = 0
        else:
            df_slice = pred_df.loc[pred_df[fold_col] == fold]
            alloc_slice = alloc.reindex(df_slice.index)
            fold_label = int(fold) if pd.notna(fold) else 0
        if df_slice.empty:
            continue
        sharpe, details = adjusted_sharpe_score(
            df_slice,
            alloc_slice,
            market_col=market_use,
            rf_col=rf_use,
            is_scored_col=is_scored_col,
        )
        const_sharpe, _ = adjusted_sharpe_score(
            df_slice,
            pd.Series(1.0, index=df_slice.index),
            market_col=market_use,
            rf_col=rf_use,
            is_scored_col=is_scored_col,
        )
        n_scored = int(df_slice[is_scored_col].sum()) if is_scored_col in df_slice.columns else len(df_slice)
        metrics.append(
            {
                "fold": fold_label,
                "sharpe": float(sharpe),
                "best_k": float(calib.best_k),
                "best_alpha": float(calib.best_alpha),
                "n_val": len(df_slice),
                "n_scored": n_scored,
                "strategy_vol": details.get("strategy_vol"),
                "const_sharpe": float(const_sharpe) if const_sharpe is not None else np.nan,
            }
        )

    return {
        "calibration": calib,
        "oof_sharpe": float(oof_sharpe),
        "oof_details": oof_details,
        "metrics": metrics,
        "alloc": alloc,
    }


def time_cv_lightgbm(
    df,
    feature_cols,
    target_col,
    n_splits=5,
    params_override=None,
    num_boost_round=200,
    early_stopping_rounds=20,
    val_frac=0.1,
    min_scored=20,
    weight_scored=None,
    weight_unscored=None,
    train_only_scored=False,
    log_prefix="",
    cfg: HullConfig | None = None,
):
    """
    Cross-val temporal com LightGBM.
    - weight_scored/weight_unscored: permitem ponderar linhas is_scored; se ambos None, treino sem pesos.
    - train_only_scored=True filtra o treino para usar apenas linhas is_scored==1 (útil para comparar variantes).
    """
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    splits = make_time_splits(df, date_col="date_id", n_splits=n_splits, val_frac=val_frac)
    metrics = []
    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored
    prefix = f"{log_prefix} " if log_prefix else ""
    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr = df.loc[mask_tr].copy()
        df_val = df.loc[mask_val].copy()
        if train_only_scored and is_scored_col and is_scored_col in df_tr.columns:
            df_tr = df_tr.loc[df_tr[is_scored_col] == 1]
            if df_tr.empty:
                print(f"{prefix}Fold {i}: treino ficou vazio após filtrar is_scored==1; pulando.")
                continue

        df_tr_aligned, df_val_aligned, cols_use = align_feature_frames(df_tr, df_val, feature_cols)
        df_tr_proc, keep_cols, medians = preprocess_basic(df_tr_aligned, cols_use)
        df_val_proc, _, _ = preprocess_basic(df_val_aligned, cols_use, ref_cols=keep_cols, ref_medians=medians)

        X_tr = df_tr_proc.drop(columns=[target_col], errors="ignore")
        y_tr = df_tr[target_col]
        X_val = df_val_proc.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]

        train_weight = make_sample_weight(df_tr_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored, is_scored_col=is_scored_col) if use_weights else None
        val_weight = make_sample_weight(df_val_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored, is_scored_col=is_scored_col) if use_weights else None
        train_ds = lgb.Dataset(X_tr, label=y_tr, weight=train_weight)
        val_ds = lgb.Dataset(X_val, label=y_val, weight=val_weight, reference=train_ds)
        params = dict(cfg_resolved.best_params or BEST_PARAMS)
        if params_override:
            params.update(params_override)
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )
        best_iter = model.best_iteration or model.current_iteration()
        pred_val = model.predict(X_val, num_iteration=best_iter)
        best_k, best_alpha, _ = optimize_allocation_scale(
            pred_val, df_val, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
        )
        alloc_val = map_return_to_alloc(pred_val, k=best_k, intercept=best_alpha)
        sharpe_adj, details = adjusted_sharpe_score(
            df_val,
            pd.Series(alloc_val, index=df_val.index),
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
        )

        n_val = len(X_val)
        n_scored = (
            int(df_val[is_scored_col].sum()) if is_scored_col and is_scored_col in df_val.columns else n_val
        )
        if n_scored < min_scored:
            print(
                f"Aviso: fold {i} tem poucas linhas is_scored ({n_scored}); considere ajustar n_splits/val_frac."
            )
        const_sharpe, _ = adjusted_sharpe_score(
            df_val, pd.Series(1.0, index=df_val.index), market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
        )
        if pd.isna(sharpe_adj) or sharpe_adj == -np.inf:
            print(f"Fold {i}: sem score válido; pulando.")
            continue

        metrics.append(
            {
                "fold": i,
                "sharpe": sharpe_adj,
                "best_k": best_k,
                "best_alpha": best_alpha,
                "best_iter": best_iter,
                "n_val": n_val,
                "n_scored": n_scored,
                "strategy_vol": details.get("strategy_vol"),
                "const_sharpe": const_sharpe,
            }
        )
    return metrics


def sanity_shuffle_test(df, feature_cols, target_col, n_splits=3):
    """Sanidade: embaralha o alvo e espera Sharpe ajustado ~0."""
    df_shuffled = df.copy()
    df_shuffled[target_col] = np.random.permutation(df_shuffled[target_col].values)
    metrics = time_cv_lightgbm(df_shuffled, feature_cols, target_col, n_splits=n_splits, num_boost_round=100, early_stopping_rounds=10)
    return metrics


def clipping_sensitivity(pred_returns, df_eval, k=None, alpha=1.0):
    """Compara Sharpe ajustado com e sem clipping da alocação para checar sensibilidade."""
    k_use = ALLOC_K if k is None else k
    alloc_raw = alpha + k_use * pred_returns
    alloc_raw = pd.Series(alloc_raw, index=df_eval.index)
    sharpe_clip, _ = adjusted_sharpe_score(df_eval, alloc_raw, clip_alloc=True)
    sharpe_no_clip, _ = adjusted_sharpe_score(df_eval, alloc_raw, clip_alloc=False)
    return {"clipped": sharpe_clip, "unclipped": sharpe_no_clip}


def summarize_cv_metrics(metrics):
    """Resumo padronizado de CV com Sharpe e n_scored por fold."""
    if not metrics:
        return None
    sharpe_vals = [m.get("sharpe") for m in metrics if m.get("sharpe") is not None]
    k_vals = [m.get("best_k", ALLOC_K) for m in metrics if m.get("best_k") is not None]
    alpha_vals = [m.get("best_alpha", 1.0) for m in metrics if m.get("best_alpha") is not None]
    n_scored_vals = [m.get("n_scored", np.nan) for m in metrics]
    const_vals = [m.get("const_sharpe") for m in metrics if m.get("const_sharpe") is not None]
    return {
        "sharpe_mean": float(np.mean(sharpe_vals)) if sharpe_vals else np.nan,
        "sharpe_std": float(np.std(sharpe_vals)) if sharpe_vals else np.nan,
        "k_median": float(np.median(k_vals)) if k_vals else np.nan,
        "alpha_median": float(np.median(alpha_vals)) if alpha_vals else np.nan,
        "n_scored_min": int(np.nanmin(n_scored_vals)) if n_scored_vals else 0,
        "n_scored_med": int(np.nanmedian(n_scored_vals)) if n_scored_vals else 0,
        "n_scored_max": int(np.nanmax(n_scored_vals)) if n_scored_vals else 0,
        "const_sharpe_mean": float(np.mean(const_vals)) if const_vals else np.nan,
        "folds": len(metrics),
    }


def make_sample_weight(df, weight_scored=1.0, weight_unscored=0.2, is_scored_col: str | None = None):
    weight_uns = weight_scored if weight_unscored is None else weight_unscored
    if is_scored_col is None and IS_SCORED_COL in df.columns:
        is_scored_col = IS_SCORED_COL
    if is_scored_col and is_scored_col in df.columns:
        return df[is_scored_col].map({1: weight_scored, 0: weight_uns}).fillna(weight_uns).to_numpy()
    return np.ones(len(df)) * weight_scored


def time_cv_lightgbm_weighted(
    df,
    feature_cols,
    target_col,
    n_splits=4,
    val_frac=0.12,
    params_override=None,
    num_boost_round=200,
    weight_scored=1.0,
    weight_unscored=0.2,
    cfg: HullConfig | None = None,
):
    """
    Conveniência para CV temporal ponderada por is_scored.
    Treino recebe pesos weight_scored/weight_unscored; avaliação continua pela métrica oficial (usa is_scored no slice).
    """
    metrics = time_cv_lightgbm(
        df,
        feature_cols,
        target_col,
        n_splits=n_splits,
        params_override=params_override,
        num_boost_round=num_boost_round,
        val_frac=val_frac,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        log_prefix="[weighted]",
        cfg=cfg,
    )
    return metrics


def stability_check(df, feature_cols, target_col, configs=((5, 0.1), (4, 0.15))):
    """Roda CV com configs alternativas para checar estabilidade de Sharpe e ordem de modelos."""
    results = []
    for n_splits, val_frac in configs:
        metrics = time_cv_lightgbm(
            df, feature_cols, target_col, n_splits=n_splits, val_frac=val_frac, num_boost_round=150, early_stopping_rounds=15
        )
        if metrics:
            summary = summarize_cv_metrics(metrics)
            summary.update({"n_splits": n_splits, "val_frac": val_frac})
            results.append(summary)
    return results


def run_cv_preds(
    df,
    feature_cols,
    target_col,
    model_kind="lgb",
    params=None,
    n_splits=5,
    val_frac=0.1,
    seed=None,
    weight_scored=None,
    weight_unscored=None,
    train_only_scored=False,
    cfg: HullConfig | None = None,
    keep_context_cols: tuple[str, ...] | None = ("date_id", "regime_std_20", "regime_high_vol"),
):
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    splits = make_time_splits(df, date_col="date_id", n_splits=n_splits, val_frac=val_frac)
    metrics, preds = [], []
    seed_use = SEED if seed is None else seed
    np.random.seed(seed_use)
    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored
    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr = df.loc[mask_tr].copy()
        df_val = df.loc[mask_val].copy()
        if train_only_scored and is_scored_col and is_scored_col in df_tr.columns:
            df_tr = df_tr.loc[df_tr[is_scored_col] == 1]
            if df_tr.empty:
                continue
        df_tr_aligned, df_val_aligned, cols_use = align_feature_frames(df_tr, df_val, feature_cols)
        df_tr_proc, keep_cols, medians = preprocess_basic(df_tr_aligned, cols_use)
        df_val_proc, _, _ = preprocess_basic(df_val_aligned, cols_use, ref_cols=keep_cols, ref_medians=medians)
        X_tr = df_tr_proc.drop(columns=[target_col], errors="ignore")
        y_tr = df_tr[target_col]
        X_val = df_val_proc.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]

        train_weight = make_sample_weight(df_tr_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored, is_scored_col=is_scored_col) if use_weights else None
        val_weight = make_sample_weight(df_val_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored, is_scored_col=is_scored_col) if use_weights else None
        if model_kind == "lgb":
            params_use = dict(cfg_resolved.best_params or BEST_PARAMS)
            if params:
                params_use.update(params)
            params_use["seed"] = seed_use
            train_ds = lgb.Dataset(X_tr, label=y_tr, weight=train_weight)
            val_ds = lgb.Dataset(X_val, label=y_val, weight=val_weight, reference=train_ds)
            model = lgb.train(
                params_use,
                train_ds,
                num_boost_round=200,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            pred = model.predict(X_val, num_iteration=model.best_iteration or model.current_iteration())
        elif model_kind == "ridge":
            alpha = params.get("alpha", 1.0) if params else 1.0
            model = Ridge(alpha=alpha, random_state=seed_use)
            model.fit(X_tr, y_tr, sample_weight=train_weight)
            pred = model.predict(X_val)
        elif model_kind == "xgb" and HAS_XGB:
            default = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed_use,
                "objective": "reg:squarederror",
            }
            if params:
                default.update(params)
            model = xgb.XGBRegressor(**default)
            fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
            if train_weight is not None:
                fit_kwargs["sample_weight"] = train_weight
            if val_weight is not None:
                fit_kwargs["eval_sample_weight"] = [val_weight]
            model.fit(X_tr, y_tr, **fit_kwargs)
            pred = model.predict(X_val)
        elif model_kind == "cat" and HAS_CAT:
            default = {
                "depth": 6,
                "learning_rate": 0.05,
                "iterations": 400,
                "loss_function": "RMSE",
                "random_seed": seed_use,
                "verbose": False,
            }
            if params:
                default.update(params)
            model = CatBoostRegressor(**default)
            fit_kwargs = {"eval_set": (X_val, y_val), "use_best_model": False, "verbose": False}
            if train_weight is not None:
                fit_kwargs["sample_weight"] = train_weight
            model.fit(X_tr, y_tr, **fit_kwargs)
            pred = model.predict(X_val)
        else:
            continue

        best_k, best_alpha, _ = optimize_allocation_scale(
            pred, df_val, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
        )
        alloc = map_return_to_alloc(pred, k=best_k, intercept=best_alpha)
        sharpe_adj, details = adjusted_sharpe_score(
            df_val,
            pd.Series(alloc, index=df_val.index),
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
        )
        n_scored = (
            int(df_val[is_scored_col].sum()) if is_scored_col and is_scored_col in df_val.columns else len(df_val)
        )
        if pd.isna(sharpe_adj) or sharpe_adj == -np.inf:
            continue
        metrics.append(
            {
                "fold": i,
                "sharpe": sharpe_adj,
                "best_alpha": best_alpha,
                "best_k": best_k,
                "n_val": len(X_val),
                "n_scored": n_scored,
                "strategy_vol": details.get("strategy_vol"),
                "seed": seed_use,
            }
        )
        preds.append(
            pd.DataFrame(
                {
                    "alloc": alloc,
                    "pred_return": pred,
                    "target": y_val,
                    "forward_returns": df_val.get(market_col),
                    "risk_free_rate": df_val.get(rf_col),
                    "is_scored": df_val[is_scored_col] if is_scored_col and is_scored_col in df_val.columns else 1,
                    "fold": i,
                    "seed": seed_use,
                    **{
                        c: df_val_aligned[c]
                        for c in (keep_context_cols or ())
                        if c in df_val_aligned.columns and c not in {"alloc", "pred_return", "target"}
                    },
                },
                index=X_val.index,
            )
        )
    return metrics, (pd.concat(preds) if preds else pd.DataFrame())


def calibrate_k_from_cv_preds(pred_df, k_grid=None, intercept=1.0):
    """Calibra k na regra allocation=clip(intercept + k*pred, 0, 2) usando Sharpe ajustado nas predições OOF."""
    if pred_df is None or pred_df.empty or "pred_return" not in pred_df.columns:
        return pd.DataFrame(), None
    if k_grid is None:
        k_grid = np.linspace(0.2, 2.0, 10)
    results = []
    for k in k_grid:
        alloc = map_return_to_alloc(pred_df["pred_return"], k=k, intercept=intercept)
        sharpe, details = adjusted_sharpe_score(
            pred_df,
            alloc,
            market_col=MARKET_COL,
            rf_col=RF_COL,
            is_scored_col="is_scored",
        )
        if pd.notna(sharpe) and np.isfinite(sharpe):
            results.append(
                {"k": float(k), "sharpe": float(sharpe), "strategy_vol": details.get("strategy_vol")}
            )
    res_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    if res_df.empty:
        return res_df, None
    best_row = res_df.iloc[0]
    best_k = float(best_row["k"])
    return res_df, best_k


def calibrate_k_alpha_from_cv_preds(
    pred_df: pd.DataFrame,
    *,
    allocation_cfg: AllocationConfig | None = None,
    k_grid=None,
    alpha_grid=None,
    pred_col: str = "pred_return",
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str = "is_scored",
):
    """Global calibration for (k, alpha) on concatenated OOF predictions."""
    cfg = allocation_cfg or AllocationConfig(k_grid=k_grid, alpha_grid=alpha_grid)
    market_use = market_col or MARKET_COL or "forward_returns"
    rf_use = rf_col or RF_COL or "risk_free_rate"
    res = calibrate_global_scale(
        pred_df,
        cfg,
        pred_col=pred_col,
        market_col=market_use,
        rf_col=rf_use,
        is_scored_col=is_scored_col,
    )
    return res.results, res.best_k, res.best_alpha, res.best_score


def expanding_holdout_eval(
    df,
    feature_cols,
    target,
    holdout_frac=0.12,
    train_only_scored=False,
    label="holdout",
    use_weights=False,
    weight_unscored=0.2,
    cfg: HullConfig | None = None,
):
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    if "date_id" in df.columns:
        df_sorted = df.sort_values("date_id")
    else:
        df_sorted = df.copy()
    n_hold = max(1, int(len(df_sorted) * holdout_frac))
    train_part = df_sorted.iloc[:-n_hold]
    holdout_part = df_sorted.iloc[-n_hold:]
    if train_only_scored and is_scored_col and is_scored_col in train_part.columns:
        train_part = train_part.loc[train_part[is_scored_col] == 1]
    if len(train_part) == 0 or len(holdout_part) == 0:
        return None

    train_aligned, holdout_aligned, cols_use = align_feature_frames(train_part, holdout_part, feature_cols)
    tr_proc, keep_cols, medians = preprocess_basic(train_aligned, cols_use)
    ho_proc, _, _ = preprocess_basic(holdout_aligned, cols_use, ref_cols=keep_cols, ref_medians=medians)
    X_tr = tr_proc.drop(columns=[target], errors="ignore")
    y_tr = train_part[target]
    X_ho = ho_proc.drop(columns=[target], errors="ignore")

    train_weight = make_sample_weight(train_aligned, weight_scored=1.0, weight_unscored=weight_unscored, is_scored_col=is_scored_col) if use_weights else None
    params = {"objective": "regression", "metric": "rmse", **(cfg_resolved.best_params or BEST_PARAMS)}
    model = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=train_weight), num_boost_round=200)
    pred_ho = model.predict(X_ho)
    best_k, best_alpha, _ = optimize_allocation_scale(pred_ho, holdout_part, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col)
    alloc_ho = map_return_to_alloc(pred_ho, k=best_k, intercept=best_alpha)
    sharpe_ho, details = adjusted_sharpe_score(
        holdout_part, alloc_ho, market_col=market_col, rf_col=rf_col, is_scored_col=is_scored_col
    )
    return {
        "label": label,
        "holdout_frac": holdout_frac,
        "train_only_scored": train_only_scored,
        "sharpe_holdout": sharpe_ho,
        "k_best": best_k,
        "alpha_best": best_alpha,
        "n_train": len(train_part),
        "n_holdout": len(holdout_part),
        "n_scored_holdout": int(holdout_part[is_scored_col].sum()) if is_scored_col and is_scored_col in holdout_part.columns else len(holdout_part),
        "weight_unscored": weight_unscored if use_weights else None,
    }


def choose_scored_strategy(results_df, cfg_lookup, fallback="weighted_0.2"):
    if results_df is None or results_df.empty:
        cfg = cfg_lookup.get(fallback, {"train_only_scored": False, "weight_unscored": 1.0})
        return {**cfg, "name": fallback, "holdout_combo": np.nan, "cv_mean": np.nan, "note": "fallback (sem métricas de comparação)"}
    eval_df = results_df.copy()
    eval_df["holdout_combo"] = eval_df[["holdout12_sharpe", "holdout15_sharpe"]].mean(axis=1)
    eval_df["holdout_combo"] = eval_df["holdout_combo"].fillna(-np.inf)
    eval_df["cv_sharpe_mean"] = eval_df["cv_sharpe_mean"].fillna(-np.inf)
    eval_df = eval_df.sort_values(by=["holdout_combo", "cv_sharpe_mean"], ascending=False)
    best = eval_df.iloc[0]
    cfg = cfg_lookup.get(best["config"], cfg_lookup.get(fallback, {"train_only_scored": False, "weight_unscored": 1.0}))
    note = f"holdout≈{best['holdout_combo']:.4f}, cv≈{best['cv_sharpe_mean']:.4f}, cfg={best['config']}"
    return {
        **cfg,
        "name": best["config"],
        "holdout_combo": float(best["holdout_combo"]),
        "cv_mean": float(best["cv_sharpe_mean"]),
        "note": note,
    }


def compute_sharpe_weights(metrics_map, floor=0.01):
    """Gera pesos normalizados a partir do Sharpe médio (clipped em 0)."""
    weights = {}
    for name, mets in metrics_map.items():
        if not mets:
            continue
        sharpe_mean = np.mean([m["sharpe"] for m in mets if m.get("sharpe") is not None])
        if np.isfinite(sharpe_mean):
            weights[name] = max(sharpe_mean, 0.0) + floor
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def blend_and_eval(pred_frames, df_ref, target_col, weights=None):
    """Blends allocation predictions and evaluates Sharpe on is_scored rows."""
    alloc_dfs = []
    oof_join = []
    for name, df_pred in pred_frames.items():
        if df_pred.empty:
            continue
        part = df_pred[df_pred["is_scored"] == 1][["alloc"]].rename(columns={"alloc": f"alloc_{name}"})
        alloc_dfs.append(part)
        oof_join.append(df_pred[["pred_return"]].rename(columns={"pred_return": f"pred_{name}"}))
    if not alloc_dfs:
        return None
    merged = pd.concat(alloc_dfs, axis=1, join="inner")
    pred_merged = pd.concat(oof_join, axis=1, join="inner") if oof_join else pd.DataFrame()
    alloc_cols = [c for c in merged.columns if c.startswith("alloc_")]
    merged["blend_mean"] = merged[alloc_cols].mean(axis=1)
    if weights:
        weight_map = {f"alloc_{k}": v for k, v in weights.items() if f"alloc_{k}" in alloc_cols}
        total_w = sum(weight_map.values())
        if total_w > 0:
            merged["blend_weighted"] = sum(weight_map[col] * merged[col] for col in weight_map) / total_w
    if len(alloc_cols) >= 2:
        ridge = Ridge(alpha=0.1, fit_intercept=True, random_state=SEED)
        ridge.fit(merged[alloc_cols], df_ref.loc[merged.index, target_col])
        merged["blend_stack_ridge"] = ridge.predict(merged[alloc_cols])
    stats = {}
    blend_cols = alloc_cols + ["blend_mean"]
    for extra_col in ["blend_weighted", "blend_stack_ridge"]:
        if extra_col in merged.columns:
            blend_cols.append(extra_col)
    for col in blend_cols:
        sharpe, details = adjusted_sharpe_score(
            df_ref.loc[merged.index],
            merged[col],
            market_col=MARKET_COL,
            rf_col=RF_COL,
            is_scored_col=IS_SCORED_COL,
        )
        stats[col] = {"sharpe": sharpe, "strategy_vol": details.get("strategy_vol")}
    return stats, pred_merged


def train_full_and_predict_model(
    df_train,
    df_test,
    feature_cols,
    target_col,
    model_kind="lgb",
    params=None,
    num_boost_round: int | None = None,
    alloc_k=None,
    alloc_alpha=1.0,
    intentional_cfg=None,
    fe_cfg=None,
    seed=None,
    train_only_scored=False,
    weight_scored=None,
    weight_unscored=None,
    df_train_fe=None,
    df_test_fe=None,
    feature_set=None,
    allocation_cfg: AllocationConfig | None = None,
    cfg: HullConfig | None = None,
):
    """Aplica pipeline de features compartilhado com a CV, treina modelo especificado e retorna alocação."""
    cfg_resolved = _resolve_cfg(cfg)
    market_col = cfg_resolved.market_col or MARKET_COL
    rf_col = cfg_resolved.rf_col or RF_COL
    is_scored_col = cfg_resolved.is_scored_col or IS_SCORED_COL
    seed_use = SEED if seed is None else seed
    np.random.seed(seed_use)
    prep = prepare_train_test_frames(
        df_train,
        df_test,
        feature_cols,
        target_col,
        intentional_cfg=intentional_cfg or cfg_resolved.intentional_cfg,
        fe_cfg=fe_cfg or cfg_resolved.feature_cfg,
        feature_set=feature_set,
        train_only_scored=train_only_scored,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        df_train_fe=df_train_fe,
        df_test_fe=df_test_fe,
        is_scored_col=is_scored_col,
    )

    if model_kind == "lgb":
        params_use = dict(cfg_resolved.best_params or BEST_PARAMS)
        if params:
            params_use.update(params)
        params_use["seed"] = seed_use
        num_round = int(num_boost_round if num_boost_round is not None else params_use.pop("num_boost_round", 400))
        model = lgb.LGBMRegressor(**params_use, n_estimators=num_round, random_state=seed_use)
        model.fit(prep["X_tr"], prep["y_tr"], sample_weight=prep["train_weight"])
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "ridge":
        alpha = params.get("alpha", 1.0) if params else 1.0
        model = Ridge(alpha=alpha, random_state=seed_use)
        model.fit(prep["X_tr"], prep["y_tr"], sample_weight=prep["train_weight"])
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "cat" and HAS_CAT:
        default = {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 500,
            "loss_function": "RMSE",
            "random_seed": seed_use,
            "verbose": False,
        }
        if params:
            default.update(params)
        model = CatBoostRegressor(**default)
        fit_kwargs = {"verbose": False}
        if prep["train_weight"] is not None:
            fit_kwargs["sample_weight"] = prep["train_weight"]
        model.fit(prep["X_tr"], prep["y_tr"], **fit_kwargs)
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "xgb" and HAS_XGB:
        default = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed_use,
            "objective": "reg:squarederror",
        }
        if params:
            default.update(params)
        model = xgb.XGBRegressor(**default)
        fit_kwargs = {"verbose": False}
        if prep["train_weight"] is not None:
            fit_kwargs["sample_weight"] = prep["train_weight"]
        model.fit(prep["X_tr"], prep["y_tr"], **fit_kwargs)
        pred_test = model.predict(prep["X_te"])
    else:
        raise ValueError(f"Modelo {model_kind} não suportado ou dependência ausente.")

    k_use = alloc_k if alloc_k is not None else cfg_resolved.alloc_k
    if allocation_cfg is None:
        alloc_test = map_return_to_alloc(pred_test, k=k_use, intercept=alloc_alpha)
        return pd.Series(alloc_test, index=prep["df_test_index"])

    alloc_series = apply_allocation_strategy(
        pred_test,
        prep["df_test_base"],
        k=float(k_use),
        alpha=float(alloc_alpha),
        cfg=allocation_cfg,
    )
    return alloc_series.reindex(prep["df_test_index"])


def train_full_and_predict_returns(
    df_train,
    df_test,
    feature_cols,
    target_col,
    model_kind="lgb",
    params=None,
    num_boost_round: int | None = None,
    intentional_cfg=None,
    fe_cfg=None,
    seed=None,
    train_only_scored=False,
    weight_scored=None,
    weight_unscored=None,
    df_train_fe=None,
    df_test_fe=None,
    feature_set=None,
    cfg: HullConfig | None = None,
):
    """Aplica o mesmo pipeline do treino final, mas retorna `pred_return` (sem aplicar allocation)."""
    cfg_resolved = _resolve_cfg(cfg)
    seed_use = SEED if seed is None else seed
    np.random.seed(seed_use)
    prep = prepare_train_test_frames(
        df_train,
        df_test,
        feature_cols,
        target_col,
        intentional_cfg=intentional_cfg or cfg_resolved.intentional_cfg,
        fe_cfg=fe_cfg or cfg_resolved.feature_cfg,
        feature_set=feature_set,
        train_only_scored=train_only_scored,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        df_train_fe=df_train_fe,
        df_test_fe=df_test_fe,
        is_scored_col=cfg_resolved.is_scored_col or IS_SCORED_COL,
    )

    if model_kind == "lgb":
        params_use = dict(cfg_resolved.best_params or BEST_PARAMS)
        if params:
            params_use.update(params)
        params_use["seed"] = seed_use
        num_round = int(num_boost_round if num_boost_round is not None else params_use.pop("num_boost_round", 400))
        model = lgb.LGBMRegressor(**params_use, n_estimators=num_round, random_state=seed_use)
        model.fit(prep["X_tr"], prep["y_tr"], sample_weight=prep["train_weight"])
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "ridge":
        alpha = params.get("alpha", 1.0) if params else 1.0
        model = Ridge(alpha=alpha, random_state=seed_use)
        model.fit(prep["X_tr"], prep["y_tr"], sample_weight=prep["train_weight"])
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "cat" and HAS_CAT:
        default = {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 500,
            "loss_function": "RMSE",
            "random_seed": seed_use,
            "verbose": False,
        }
        if params:
            default.update(params)
        model = CatBoostRegressor(**default)
        fit_kwargs = {"verbose": False}
        if prep["train_weight"] is not None:
            fit_kwargs["sample_weight"] = prep["train_weight"]
        model.fit(prep["X_tr"], prep["y_tr"], **fit_kwargs)
        pred_test = model.predict(prep["X_te"])
    elif model_kind == "xgb" and HAS_XGB:
        default = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed_use,
            "objective": "reg:squarederror",
        }
        if params:
            default.update(params)
        model = xgb.XGBRegressor(**default)
        fit_kwargs = {"verbose": False}
        if prep["train_weight"] is not None:
            fit_kwargs["sample_weight"] = prep["train_weight"]
        model.fit(prep["X_tr"], prep["y_tr"], **fit_kwargs)
        pred_test = model.predict(prep["X_te"])
    else:
        raise ValueError(f"Modelo {model_kind} não suportado ou dependência ausente.")

    return pd.Series(pred_test, index=prep["df_test_index"])


def prepare_train_test_frames(
    df_train,
    df_test,
    feature_cols,
    target_col,
    intentional_cfg=None,
    fe_cfg=None,
    feature_set=None,
    train_only_scored=False,
    weight_scored=None,
    weight_unscored=None,
    df_train_fe=None,
    df_test_fe=None,
    is_scored_col: str | None = None,
    cfg: HullConfig | None = None,
):
    """Prepara matrizes X/y de treino e teste com pipeline de features compartilhado."""
    cfg_resolved = _resolve_cfg(cfg)
    is_scored_col = is_scored_col or cfg_resolved.is_scored_col or IS_SCORED_COL
    fe_cfg_use = cfg_resolved.feature_cfg if fe_cfg is None else fe_cfg
    feature_cols_use = list(feature_cols) if feature_cols is not None else None
    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored
    df_train_raw = df_train.copy()
    df_train_base = df_train_fe.copy() if df_train_fe is not None else None
    df_test_base = df_test_fe.copy() if df_test_fe is not None else None

    needs_features = df_train_base is None or (df_test is not None and df_test_base is None)
    if not needs_features and feature_cols_use is not None:
        missing_train = [c for c in feature_cols_use if c not in df_train_base.columns]
        missing_test = [c for c in feature_cols_use if df_test_base is not None and c not in df_test_base.columns]
        needs_features = bool(missing_train or missing_test)

    if needs_features or feature_cols_use is None:
        gen_train_fe, gen_test_fe, cols_gen, feature_sets_gen, feature_set_gen = make_features(
            df_train_raw,
            test_df=df_test,
            target_col=target_col,
            feature_set=feature_set,
            intentional_cfg=intentional_cfg,
            fe_cfg=fe_cfg_use,
        )
        df_train_base = gen_train_fe
        df_test_base = gen_test_fe if gen_test_fe is not None else (df_test.copy() if df_test is not None else None)
        chosen_set = feature_set or feature_set_gen
        feature_cols_from_set = feature_sets_gen.get(chosen_set) if feature_sets_gen else None
        if feature_cols_use is None or needs_features:
            feature_cols_use = feature_cols_from_set or cols_gen

    if df_test_base is None and df_test is not None:
        df_test_base = df_test.copy()

    if train_only_scored and is_scored_col and is_scored_col in df_train_base.columns:
        mask_scored = df_train_base[is_scored_col] == 1
        df_train_base = df_train_base.loc[mask_scored]
        df_train_raw = df_train_raw.loc[mask_scored]

    df_train_base, df_test_base, feature_cols_aligned = align_feature_frames(df_train_base, df_test_base, feature_cols_use)

    df_train_proc, keep_cols, medians = preprocess_basic(df_train_base, feature_cols_aligned)
    df_test_proc, _, _ = preprocess_basic(df_test_base, feature_cols_aligned, ref_cols=keep_cols, ref_medians=medians)
    X_tr = df_train_proc.drop(columns=[target_col], errors="ignore")
    y_tr = df_train_raw.loc[df_train_proc.index, target_col]
    X_te = df_test_proc.drop(columns=[target_col], errors="ignore")
    train_weight = (
        make_sample_weight(
            df_train_raw.loc[df_train_proc.index],
            weight_scored=weight_scored,
            weight_unscored=weight_unscored,
            is_scored_col=is_scored_col,
        )
        if use_weights
        else None
    )
    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "df_train_base": df_train_base,
        "df_test_base": df_test_base,
        "df_test_index": df_test_proc.index,
        "feature_cols": feature_cols_aligned,
        "train_weight": train_weight,
    }


def add_exp_log(
    log,
    exp_id,
    feature_set,
    model,
    sharpe_mean,
    sharpe_std,
    n_splits=None,
    val_frac=None,
    params=None,
    sharpe_lb_public=np.nan,
    notes=None,
):
    log.append(
        {
            "exp_id": exp_id,
            "feature_set": feature_set,
            "model": model,
            "n_splits": n_splits,
            "val_frac": val_frac,
            "sharpe_cv_mean": sharpe_mean,
            "sharpe_cv_std": sharpe_std,
            "sharpe_lb_public": sharpe_lb_public,
            "params": params,
            "notes": notes,
        }
    )


__all__ = [
    "ALLOC_K",
    "MIN_INVESTMENT",
    "MAX_INVESTMENT",
    "SEED",
    "MARKET_COL",
    "RF_COL",
    "IS_SCORED_COL",
    "BEST_PARAMS",
    "INTENTIONAL_CFG",
    "FEATURE_CFG_DEFAULT",
    "evaluate_baselines",
    "constant_allocation_cv",
    "time_cv_lightgbm_fitref",
    "make_time_splits",
    "compute_strategy_returns",
    "map_return_to_alloc",
    "adjusted_sharpe_score",
    "optimize_allocation_scale",
    "score_oof_predictions_with_allocation",
    "time_cv_lightgbm",
    "sanity_shuffle_test",
    "clipping_sensitivity",
    "summarize_cv_metrics",
    "make_sample_weight",
    "time_cv_lightgbm_weighted",
    "stability_check",
    "run_cv_preds",
    "run_cv_preds_fitref",
    "calibrate_k_from_cv_preds",
    "calibrate_k_alpha_from_cv_preds",
    "time_cv_lightgbm_fitref_oof",
    "expanding_holdout_eval",
    "choose_scored_strategy",
    "compute_sharpe_weights",
    "blend_and_eval",
    "train_full_and_predict_returns",
    "train_full_and_predict_model",
    "prepare_train_test_frames",
    "add_exp_log",
    "set_data_columns",
]
