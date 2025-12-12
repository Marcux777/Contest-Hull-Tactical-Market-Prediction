"""Official competition metric helpers (dependency-light).

This module intentionally avoids importing model training libraries. It provides
the evaluation logic used throughout CV/OOF calibration so it can be reused in
restricted Kaggle environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_strategy_returns(
    allocation: pd.Series,
    market_returns: pd.Series,
    rf: pd.Series | None = None,
) -> pd.Series:
    """Computes strategy returns.

    Backward compatible:
    - If `rf` is provided: `rf_t * (1 - alloc_t) + alloc_t * market_t`
    - If `rf` is None: `alloc_t * market_t`
    """
    if rf is None:
        return allocation * market_returns
    return rf * (1.0 - allocation) + allocation * market_returns


def adjusted_sharpe_score(
    df: pd.DataFrame,
    allocation: pd.Series | np.ndarray,
    *,
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str | None = None,
    trading_days_per_yr: int = 252,
    clip_alloc: bool = True,
    min_allocation: float = 0.0,
    max_allocation: float = 2.0,
) -> tuple[float, dict[str, Any]]:
    """Replica a métrica oficial (modified Sharpe), aplicando filtro `is_scored==1` quando disponível."""
    market_use = market_col or "forward_returns"
    rf_use = rf_col or "risk_free_rate"
    scored_col = is_scored_col or "is_scored"

    required = [market_use, rf_use]
    if not all(col in df.columns for col in required):
        return float("nan"), {}

    eval_df = df.copy()
    if scored_col and scored_col in eval_df.columns:
        eval_df = eval_df.loc[eval_df[scored_col] == 1].copy()
    if len(eval_df) == 0:
        return float("nan"), {}

    alloc_series = pd.Series(allocation, index=df.index, dtype=float)
    if clip_alloc:
        alloc_series = alloc_series.clip(float(min_allocation), float(max_allocation))
    alloc_series = alloc_series.reindex(eval_df.index)

    position = alloc_series.astype(float)
    strat_returns = compute_strategy_returns(position, eval_df[market_use].astype(float), eval_df[rf_use].astype(float))
    strategy_excess = strat_returns - eval_df[rf_use].astype(float)

    strategy_excess_cum = (1.0 + strategy_excess).prod()
    strategy_mean_excess = strategy_excess_cum ** (1.0 / len(eval_df)) - 1.0
    strategy_std = float(strat_returns.std(ddof=0))
    if not np.isfinite(strategy_std) or strategy_std <= 0:
        return float("-inf"), {}

    sharpe = float(strategy_mean_excess / (strategy_std + 1e-12) * np.sqrt(float(trading_days_per_yr)))
    strategy_vol = float(strategy_std * np.sqrt(float(trading_days_per_yr)) * 100.0)

    market_excess = eval_df[market_use].astype(float) - eval_df[rf_use].astype(float)
    market_excess_cum = (1.0 + market_excess).prod()
    market_mean_excess = float(market_excess_cum ** (1.0 / len(eval_df)) - 1.0)
    market_std = float(eval_df[market_use].astype(float).std(ddof=0))
    if not np.isfinite(market_std) or market_std <= 0:
        return float("-inf"), {}
    market_vol = float(market_std * np.sqrt(float(trading_days_per_yr)) * 100.0)

    excess_vol = max(0.0, strategy_vol / market_vol - 1.2) if market_vol > 0 else 0.0
    vol_penalty = 1.0 + excess_vol

    return_gap = max(0.0, (market_mean_excess - strategy_mean_excess) * 100.0 * float(trading_days_per_yr))
    return_penalty = 1.0 + (return_gap**2) / 100.0

    adjusted = sharpe / (vol_penalty * return_penalty)
    details: dict[str, Any] = {
        "sharpe_raw": sharpe,
        "strategy_vol": strategy_vol,
        "market_vol": market_vol,
        "strategy_mean_excess": strategy_mean_excess,
        "market_mean_excess": market_mean_excess,
        "vol_penalty": vol_penalty,
        "return_penalty": return_penalty,
    }
    return float(min(adjusted, 1_000_000.0)), details


__all__ = [
    "adjusted_sharpe_score",
    "compute_strategy_returns",
]
