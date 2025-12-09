"""
Modeling, CV, and allocation helpers shared by the Hull Tactical notebook.
Kept dependency-light so it can run inside Kaggle without extra installs.
"""
from __future__ import annotations

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

from hull_features import (
    FEATURE_CFG_DEFAULT as HF_FEATURE_CFG_DEFAULT,
    INTENTIONAL_CFG as HF_INTENTIONAL_CFG,
    build_feature_sets,
    make_features,
    align_feature_frames,
    preprocess_basic,
)

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


def evaluate_baselines(train_df, feature_cols, target_col):
    """Baselines simples em split 80/20: alocação constante e modelo linear (Ridge)."""
    tr_frac = int(len(train_df) * 0.8)
    tr = train_df.iloc[:tr_frac].copy()
    va = train_df.iloc[tr_frac:].copy()
    tr_aligned, va_aligned, cols_use = align_feature_frames(tr, va, feature_cols)
    tr_proc, keep_cols = preprocess_basic(tr_aligned, cols_use)
    va_proc, _ = preprocess_basic(va_aligned, cols_use, ref_cols=keep_cols)

    X_tr = tr_proc.drop(columns=[target_col], errors="ignore")
    X_va = va_proc.drop(columns=[target_col], errors="ignore")
    y_tr = tr[target_col]
    y_va = va[target_col]

    alloc_const = pd.Series(1.0, index=va.index)
    sharpe_const, const_details = adjusted_sharpe_score(
        va, alloc_const, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
    )

    lin = Ridge(alpha=1.0, random_state=SEED)
    lin.fit(X_tr, y_tr)
    pred_lin = lin.predict(X_va)
    best_k_lin, best_alpha_lin, _ = optimize_allocation_scale(
        pred_lin, va, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
    )
    alloc_lin = map_return_to_alloc(pred_lin, k=best_k_lin, intercept=best_alpha_lin)
    sharpe_lin, lin_details = adjusted_sharpe_score(
        va, pd.Series(alloc_lin, index=va.index), market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
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


def constant_allocation_cv(df, n_splits=5, val_frac=0.1):
    """Baseline: alocação constante (1.0) por fold temporal."""
    splits = make_time_splits(df, date_col="date_id", n_splits=n_splits, val_frac=val_frac)
    metrics = []
    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_val = df.loc[mask_val].copy()
        if df_val.empty:
            continue
        alloc_const = pd.Series(1.0, index=df_val.index)
        sharpe_const, details = adjusted_sharpe_score(
            df_val, alloc_const, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
        )
        n_scored = int(df_val[IS_SCORED_COL].sum()) if IS_SCORED_COL and IS_SCORED_COL in df_val.columns else len(df_val)
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
):
    """CV temporal recalculando features por fold (winsor/clipping/z-score usando apenas o treino)."""
    splits = make_time_splits(df, n_splits=n_splits, val_frac=val_frac)
    metrics = []
    if not splits:
        return metrics
    params_use = dict(BEST_PARAMS)
    if params_override:
        params_use.update(params_override)
    params_use["metric"] = "rmse"
    params_use["objective"] = "regression"
    params_use["seed"] = SEED

    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr = df.loc[mask_tr].copy()
        df_val = df.loc[mask_val].copy()
        if df_val.empty or df_tr.empty:
            continue

        df_tr_fe, fs_tr = build_feature_sets(
            df_tr, target_col, intentional_cfg=INTENTIONAL_CFG, fe_cfg=FEATURE_CFG_DEFAULT, fit_ref=df_tr
        )
        if df_tr_fe.columns.has_duplicates:
            df_tr_fe = df_tr_fe.loc[:, ~df_tr_fe.columns.duplicated(keep="last")]
        cols = fs_tr.get(feature_set_name, next(iter(fs_tr.values())))
        df_val_fe, _ = build_feature_sets(
            df_val, target_col, intentional_cfg=INTENTIONAL_CFG, fe_cfg=FEATURE_CFG_DEFAULT, fit_ref=df_tr
        )
        if df_val_fe.columns.has_duplicates:
            df_val_fe = df_val_fe.loc[:, ~df_val_fe.columns.duplicated(keep="last")]
        df_val_fe = df_val_fe.reindex(columns=df_tr_fe.columns, fill_value=0)

        X_tr = df_tr_fe[cols]
        y_tr = df_tr_fe[target_col]
        X_val = df_val_fe[cols]
        y_val = df_val_fe[target_col]

        model = lgb.LGBMRegressor(**params_use, n_estimators=num_boost_round, random_state=SEED)
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)
        best_k, best_alpha, _ = optimize_allocation_scale(
            pred_val, df_val_fe, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
        )
        alloc_val = map_return_to_alloc(pred_val, k=best_k, intercept=best_alpha)
        sharpe_val, details = adjusted_sharpe_score(
            df_val_fe, alloc_val, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
        )
        metrics.append(
            {
                "fold": i,
                "sharpe": sharpe_val,
                "best_k": best_k,
                "best_alpha": best_alpha,
                "n_scored": int(df_val_fe[IS_SCORED_COL].sum()) if IS_SCORED_COL and IS_SCORED_COL in df_val_fe.columns else len(df_val_fe),
                "strategy_vol": details.get("strategy_vol"),
            }
        )
    return metrics


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
    pred_returns, df_eval, k_grid=None, alpha_grid=None, market_col=None, rf_col=None, is_scored_col=None
):
    """Busca simples em k e intercepto (alpha) para maximizar a métrica oficial no conjunto de validação."""
    if k_grid is None:
        k_grid = np.linspace(0.0, 2.5, 16)
    if alpha_grid is None:
        alpha_grid = [0.5, 1.0, 1.5]
    market_use = market_col or MARKET_COL or "forward_returns"
    rf_use = rf_col or RF_COL or "risk_free_rate"
    scored_col = is_scored_col or IS_SCORED_COL
    best_k = ALLOC_K
    best_alpha = 1.0
    best_score = -np.inf
    for alpha in alpha_grid:
        for k in k_grid:
            alloc = pd.Series(map_return_to_alloc(pred_returns, k=k, intercept=alpha), index=df_eval.index)
            score, _ = adjusted_sharpe_score(
                df_eval, alloc, market_col=market_use, rf_col=rf_use, is_scored_col=scored_col
            )
            if pd.notna(score) and score > best_score:
                best_score = score
                best_k = k
                best_alpha = alpha
    return best_k, best_alpha, best_score


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
):
    splits = make_time_splits(df, date_col="date_id", n_splits=n_splits, val_frac=val_frac)
    metrics = []
    use_weights = weight_scored is not None or weight_unscored is not None
    weight_scored = 1.0 if weight_scored is None else weight_scored
    weight_unscored = 1.0 if weight_unscored is None else weight_unscored
    prefix = f"{log_prefix} " if log_prefix else ""
    for i, (mask_tr, mask_val) in enumerate(splits, 1):
        df_tr = df.loc[mask_tr].copy()
        df_val = df.loc[mask_val].copy()
        if train_only_scored and IS_SCORED_COL and IS_SCORED_COL in df_tr.columns:
            df_tr = df_tr.loc[df_tr[IS_SCORED_COL] == 1]
            if df_tr.empty:
                print(f"{prefix}Fold {i}: treino ficou vazio após filtrar is_scored==1; pulando.")
                continue

        df_tr_aligned, df_val_aligned, cols_use = align_feature_frames(df_tr, df_val, feature_cols)
        df_tr_proc, keep_cols = preprocess_basic(df_tr_aligned, cols_use)
        df_val_proc, _ = preprocess_basic(df_val_aligned, cols_use, ref_cols=keep_cols)

        X_tr = df_tr_proc.drop(columns=[target_col], errors="ignore")
        y_tr = df_tr[target_col]
        X_val = df_val_proc.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]

        train_weight = make_sample_weight(df_tr_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored) if use_weights else None
        val_weight = make_sample_weight(df_val_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored) if use_weights else None
        train_ds = lgb.Dataset(X_tr, label=y_tr, weight=train_weight)
        val_ds = lgb.Dataset(X_val, label=y_val, weight=val_weight, reference=train_ds)
        params = dict(BEST_PARAMS)
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
            pred_val, df_val, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
        )
        alloc_val = map_return_to_alloc(pred_val, k=best_k, intercept=best_alpha)
        sharpe_adj, details = adjusted_sharpe_score(
            df_val,
            pd.Series(alloc_val, index=df_val.index),
            market_col=MARKET_COL,
            rf_col=RF_COL,
            is_scored_col=IS_SCORED_COL,
        )

        n_val = len(X_val)
        n_scored = (
            int(df_val[IS_SCORED_COL].sum()) if IS_SCORED_COL and IS_SCORED_COL in df_val.columns else n_val
        )
        if n_scored < min_scored:
            print(
                f"Aviso: fold {i} tem poucas linhas is_scored ({n_scored}); considere ajustar n_splits/val_frac."
            )
        const_sharpe, _ = adjusted_sharpe_score(
            df_val, pd.Series(1.0, index=df_val.index), market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
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


def make_sample_weight(df, weight_scored=1.0, weight_unscored=0.2):
    weight_uns = weight_scored if weight_unscored is None else weight_unscored
    if IS_SCORED_COL and IS_SCORED_COL in df.columns:
        return df[IS_SCORED_COL].map({1: weight_scored, 0: weight_uns}).fillna(weight_uns).to_numpy()
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
):
    """CV temporal com pesos para is_scored (treino ponderado; avaliação segue métrica oficial)."""
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
):
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
        if train_only_scored and IS_SCORED_COL and IS_SCORED_COL in df_tr.columns:
            df_tr = df_tr.loc[df_tr[IS_SCORED_COL] == 1]
            if df_tr.empty:
                continue
        df_tr_aligned, df_val_aligned, cols_use = align_feature_frames(df_tr, df_val, feature_cols)
        df_tr_proc, keep_cols = preprocess_basic(df_tr_aligned, cols_use)
        df_val_proc, _ = preprocess_basic(df_val_aligned, cols_use, ref_cols=keep_cols)
        X_tr = df_tr_proc.drop(columns=[target_col], errors="ignore")
        y_tr = df_tr[target_col]
        X_val = df_val_proc.drop(columns=[target_col], errors="ignore")
        y_val = df_val[target_col]

        train_weight = make_sample_weight(df_tr_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored) if use_weights else None
        val_weight = make_sample_weight(df_val_aligned, weight_scored=weight_scored, weight_unscored=weight_unscored) if use_weights else None
        if model_kind == "lgb":
            params_use = dict(BEST_PARAMS)
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
            pred, df_val, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
        )
        alloc = map_return_to_alloc(pred, k=best_k, intercept=best_alpha)
        sharpe_adj, details = adjusted_sharpe_score(
            df_val,
            pd.Series(alloc, index=df_val.index),
            market_col=MARKET_COL,
            rf_col=RF_COL,
            is_scored_col=IS_SCORED_COL,
        )
        n_scored = (
            int(df_val[IS_SCORED_COL].sum()) if IS_SCORED_COL and IS_SCORED_COL in df_val.columns else len(df_val)
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
                    "forward_returns": df_val.get(MARKET_COL),
                    "risk_free_rate": df_val.get(RF_COL),
                    "is_scored": df_val[IS_SCORED_COL] if IS_SCORED_COL and IS_SCORED_COL in df_val.columns else 1,
                    "fold": i,
                    "seed": seed_use,
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


def expanding_holdout_eval(
    df, feature_cols, target, holdout_frac=0.12, train_only_scored=False, label="holdout", use_weights=False, weight_unscored=0.2
):
    if "date_id" in df.columns:
        df_sorted = df.sort_values("date_id")
    else:
        df_sorted = df.copy()
    n_hold = max(1, int(len(df_sorted) * holdout_frac))
    train_part = df_sorted.iloc[:-n_hold]
    holdout_part = df_sorted.iloc[-n_hold:]
    if train_only_scored and IS_SCORED_COL and IS_SCORED_COL in train_part.columns:
        train_part = train_part.loc[train_part[IS_SCORED_COL] == 1]
    if len(train_part) == 0 or len(holdout_part) == 0:
        return None

    train_aligned, holdout_aligned, cols_use = align_feature_frames(train_part, holdout_part, feature_cols)
    tr_proc, keep_cols = preprocess_basic(train_aligned, cols_use)
    ho_proc, _ = preprocess_basic(holdout_aligned, cols_use, ref_cols=keep_cols)
    X_tr = tr_proc.drop(columns=[target], errors="ignore")
    y_tr = train_part[target]
    X_ho = ho_proc.drop(columns=[target], errors="ignore")

    train_weight = make_sample_weight(train_aligned, weight_scored=1.0, weight_unscored=weight_unscored) if use_weights else None
    params = {"objective": "regression", "metric": "rmse", **BEST_PARAMS}
    model = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, weight=train_weight), num_boost_round=200)
    pred_ho = model.predict(X_ho)
    best_k, best_alpha, _ = optimize_allocation_scale(pred_ho, holdout_part, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL)
    alloc_ho = map_return_to_alloc(pred_ho, k=best_k, intercept=best_alpha)
    sharpe_ho, details = adjusted_sharpe_score(
        holdout_part, alloc_ho, market_col=MARKET_COL, rf_col=RF_COL, is_scored_col=IS_SCORED_COL
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
        "n_scored_holdout": int(holdout_part[IS_SCORED_COL].sum()) if IS_SCORED_COL and IS_SCORED_COL in holdout_part.columns else len(holdout_part),
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
):
    """Aplica pipeline de features compartilhado com a CV, treina modelo especificado e retorna alocação."""
    seed_use = SEED if seed is None else seed
    np.random.seed(seed_use)
    fe_cfg_use = FEATURE_CFG_DEFAULT if fe_cfg is None else fe_cfg
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

    if train_only_scored and IS_SCORED_COL and IS_SCORED_COL in df_train_base.columns:
        mask_scored = df_train_base[IS_SCORED_COL] == 1
        df_train_base = df_train_base.loc[mask_scored]
        df_train_raw = df_train_raw.loc[mask_scored]

    df_train_base, df_test_base, feature_cols_aligned = align_feature_frames(df_train_base, df_test_base, feature_cols_use)

    df_train_proc, keep_cols = preprocess_basic(df_train_base, feature_cols_aligned)
    df_test_proc, _ = preprocess_basic(df_test_base, feature_cols_aligned, ref_cols=keep_cols)
    X_tr = df_train_proc.drop(columns=[target_col], errors="ignore")
    y_tr = df_train_raw.loc[df_train_proc.index, target_col]
    X_te = df_test_proc.drop(columns=[target_col], errors="ignore")
    train_weight = (
        make_sample_weight(df_train_raw.loc[df_train_proc.index], weight_scored=weight_scored, weight_unscored=weight_unscored)
        if use_weights
        else None
    )

    if model_kind == "lgb":
        params_use = dict(BEST_PARAMS)
        if params:
            params_use.update(params)
        params_use["seed"] = seed_use
        train_ds = lgb.Dataset(X_tr, label=y_tr, weight=train_weight)
        model = lgb.train(
            params_use,
            train_ds,
            num_boost_round=int(params_use.get("num_boost_round", 400)),
        )
        pred_test = model.predict(X_te, num_iteration=model.best_iteration or model.current_iteration())
    elif model_kind == "ridge":
        alpha = params.get("alpha", 1.0) if params else 1.0
        model = Ridge(alpha=alpha, random_state=seed_use)
        model.fit(X_tr, y_tr, sample_weight=train_weight)
        pred_test = model.predict(X_te)
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
        if train_weight is not None:
            fit_kwargs["sample_weight"] = train_weight
        model.fit(X_tr, y_tr, **fit_kwargs)
        pred_test = model.predict(X_te)
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
        if train_weight is not None:
            fit_kwargs["sample_weight"] = train_weight
        model.fit(X_tr, y_tr, **fit_kwargs)
        pred_test = model.predict(X_te)
    else:
        raise ValueError(f"Modelo {model_kind} não suportado ou dependência ausente.")

    k_use = alloc_k if alloc_k is not None else ALLOC_K
    alloc_test = map_return_to_alloc(pred_test, k=k_use, intercept=alloc_alpha)
    return pd.Series(alloc_test, index=df_test_proc.index)


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
    "time_cv_lightgbm",
    "sanity_shuffle_test",
    "clipping_sensitivity",
    "summarize_cv_metrics",
    "make_sample_weight",
    "time_cv_lightgbm_weighted",
    "stability_check",
    "run_cv_preds",
    "calibrate_k_from_cv_preds",
    "expanding_holdout_eval",
    "choose_scored_strategy",
    "compute_sharpe_weights",
    "blend_and_eval",
    "train_full_and_predict_model",
    "add_exp_log",
]
