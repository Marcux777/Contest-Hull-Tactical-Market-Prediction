"""Allocation calibration and post-processing utilities.

Goal: treat `pred_return -> allocation` as a first-class module to optimize the
competition metric (adjusted Sharpe), not as a small detail.

This module is dependency-light (numpy/pandas only). It expects the caller to
provide a DataFrame with market/rf/is_scored columns compatible with
`hull_tactical.metric.adjusted_sharpe_score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationConfig:
    """Controls allocation mapping, risk scaling and smoothing."""

    # Search space for global calibration (OOF).
    k_grid: Iterable[float] | None = None
    alpha_grid: Iterable[float] | None = None

    # Hard bounds (Kaggle expects 0..2 where 1 == 100% long market).
    min_allocation: float = 0.0
    max_allocation: float = 2.0

    # Ordering context for rolling/smoothing.
    date_col: str = "date_id"

    # Regime-based aggressiveness reduction.
    regime_col: str = "regime_high_vol"
    high_vol_k_factor: float = 0.6

    # Risk proxy scaling (vol targeting proxy).
    risk_col: str | None = "regime_std_20"
    risk_power: float = 1.0
    risk_clip: Tuple[float, float] = (0.5, 2.0)

    # Optional signal standardization (rolling z-score).
    standardize_window: int | None = None
    standardize_clip: float | None = None

    # Optional volatility targeting based on prediction dispersion (rolling std of `pred_return`).
    pred_vol_window: int | None = None
    pred_vol_power: float = 1.0
    pred_vol_clip: Tuple[float, float] = (0.5, 2.0)

    # Optional allocation smoothing / turnover control.
    smooth_alpha: float | None = 0.2  # EWMA alpha in [0,1]
    smooth_span: int | None = None  # if set and smooth_alpha is None, alpha=2/(span+1)
    delta_cap: float | None = None  # max |Δalloc| per step

    def resolved_k_grid(self) -> np.ndarray:
        if self.k_grid is None:
            return np.linspace(0.0, 3.0, 61)
        return np.asarray(list(self.k_grid), dtype=float)

    def resolved_alpha_grid(self) -> np.ndarray:
        if self.alpha_grid is None:
            return np.asarray([0.8, 1.0, 1.2], dtype=float)
        return np.asarray(list(self.alpha_grid), dtype=float)


@dataclass(frozen=True)
class CalibrationResult:
    best_k: float
    best_alpha: float
    best_score: float
    results: pd.DataFrame


def _sorted_index(df: pd.DataFrame, date_col: str) -> pd.Index:
    if date_col in df.columns:
        return df.sort_values(date_col, kind="mergesort").index
    return df.index


def _standardize_signal(pred_return: pd.Series, df_context: pd.DataFrame, cfg: AllocationConfig) -> pd.Series:
    window = cfg.standardize_window
    if window is None or window <= 1:
        return pred_return
    idx = _sorted_index(df_context, cfg.date_col)
    s = pred_return.reindex(idx)
    roll_mean = s.rolling(window=window, min_periods=max(3, window // 2)).mean()
    roll_std = s.rolling(window=window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
    z = ((s - roll_mean) / roll_std).fillna(0.0)
    if cfg.standardize_clip is not None:
        z = z.clip(-float(cfg.standardize_clip), float(cfg.standardize_clip))
    return z.reindex(df_context.index)


def _pred_vol_factor(pred_return: pd.Series, df_context: pd.DataFrame, cfg: AllocationConfig) -> pd.Series:
    """Vol targeting factor based on rolling std of predictions (causal).

    Uses an expanding-median reference so the factor stays near 1 on average:
      factor_t = median(std_<=t) / (std_t + eps)
    """
    window = cfg.pred_vol_window
    if window is None or window <= 1:
        return pd.Series(1.0, index=df_context.index)
    idx = _sorted_index(df_context, cfg.date_col)
    s = pred_return.reindex(idx).astype(float)
    min_periods = max(3, window // 2)
    roll_std = s.rolling(window=window, min_periods=min_periods).std(ddof=0)
    ref = roll_std.expanding(min_periods=min_periods).median()
    eps = 1e-12
    factor = (ref / (roll_std + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    lo, hi = cfg.pred_vol_clip
    factor = factor.clip(float(lo), float(hi))
    power = float(cfg.pred_vol_power)
    if power != 1.0:
        factor = factor.pow(power)
    return factor.reindex(df_context.index)


def _regime_factor(df_context: pd.DataFrame, cfg: AllocationConfig) -> pd.Series:
    if cfg.regime_col not in df_context.columns:
        return pd.Series(1.0, index=df_context.index)
    flag = df_context[cfg.regime_col].fillna(0).astype(float)
    return (1.0 - flag) + flag * float(cfg.high_vol_k_factor)


def _risk_factor(df_context: pd.DataFrame, cfg: AllocationConfig) -> pd.Series:
    if cfg.risk_col is None or cfg.risk_col not in df_context.columns:
        return pd.Series(1.0, index=df_context.index)
    risk = pd.to_numeric(df_context[cfg.risk_col], errors="coerce").astype(float)
    risk = risk.replace([np.inf, -np.inf], np.nan)
    median = float(risk.median(skipna=True)) if risk.notna().any() else np.nan
    if not np.isfinite(median) or median <= 0:
        return pd.Series(1.0, index=df_context.index)
    raw = (median / (risk + 1e-12)).pow(float(cfg.risk_power)).fillna(1.0)
    lo, hi = cfg.risk_clip
    return raw.clip(float(lo), float(hi))


def smooth_allocation(
    allocation: pd.Series | np.ndarray,
    df_context: pd.DataFrame | None = None,
    *,
    date_col: str = "date_id",
    smooth_alpha: float | None = 0.2,
    smooth_span: int | None = None,
    delta_cap: float | None = None,
) -> pd.Series:
    """EWMA + optional delta cap in chronological order."""
    alloc = pd.Series(allocation, index=df_context.index if df_context is not None else None, dtype=float)
    if smooth_alpha is None and smooth_span is not None and smooth_span > 1:
        smooth_alpha = 2.0 / (float(smooth_span) + 1.0)
    if smooth_alpha is None or smooth_alpha <= 0:
        return alloc

    if df_context is None:
        idx = alloc.index
    else:
        idx = _sorted_index(df_context, date_col)

    a = float(smooth_alpha)
    out = alloc.reindex(idx).copy()
    if out.empty:
        return out.reindex(alloc.index)

    prev = float(out.iloc[0])
    for i in range(1, len(out)):
        raw = float(out.iloc[i])
        curr = (1.0 - a) * prev + a * raw
        if delta_cap is not None:
            cap = float(delta_cap)
            delta = curr - prev
            if abs(delta) > cap:
                curr = prev + np.sign(delta) * cap
        out.iloc[i] = curr
        prev = curr

    return out.reindex(alloc.index)


def apply_allocation_strategy(
    pred_return: pd.Series | np.ndarray,
    df_context: pd.DataFrame,
    *,
    k: float,
    alpha: float = 1.0,
    cfg: AllocationConfig | None = None,
) -> pd.Series:
    """Maps predictions into allocations applying regime/risk scaling and smoothing."""
    cfg = cfg or AllocationConfig()
    pred = pd.Series(pred_return, index=df_context.index, dtype=float)
    pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    signal = _standardize_signal(pred, df_context, cfg)
    k_eff = float(k) * _regime_factor(df_context, cfg) * _risk_factor(df_context, cfg) * _pred_vol_factor(pred, df_context, cfg)
    alloc = float(alpha) + k_eff * signal
    alloc = alloc.clip(cfg.min_allocation, cfg.max_allocation)

    alloc = smooth_allocation(
        alloc,
        df_context,
        date_col=cfg.date_col,
        smooth_alpha=cfg.smooth_alpha,
        smooth_span=cfg.smooth_span,
        delta_cap=cfg.delta_cap,
    )
    return alloc.clip(cfg.min_allocation, cfg.max_allocation)


def calibrate_global_scale(
    pred_df: pd.DataFrame,
    cfg: AllocationConfig | None = None,
    *,
    pred_col: str = "pred_return",
    market_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
    is_scored_col: str = "is_scored",
) -> CalibrationResult:
    """Finds global (k, alpha) on OOF predictions for the full allocation strategy."""
    if pred_df is None or pred_df.empty or pred_col not in pred_df.columns:
        return CalibrationResult(best_k=np.nan, best_alpha=np.nan, best_score=np.nan, results=pd.DataFrame())

    cfg = cfg or AllocationConfig()

    # Local import to keep this module dependency-light.
    from .metric import adjusted_sharpe_score

    k_grid = cfg.resolved_k_grid()
    alpha_grid = cfg.resolved_alpha_grid()
    best_k, best_alpha, best_score = float("nan"), float("nan"), float("-inf")
    rows: list[dict] = []

    for alpha in alpha_grid:
        for k in k_grid:
            alloc = apply_allocation_strategy(
                pred_df[pred_col],
                pred_df,
                k=float(k),
                alpha=float(alpha),
                cfg=cfg,
            )
            score, details = adjusted_sharpe_score(
                pred_df,
                alloc,
                market_col=market_col,
                rf_col=rf_col,
                is_scored_col=is_scored_col,
            )
            if pd.isna(score) or not np.isfinite(score):
                continue
            rows.append(
                {
                    "k": float(k),
                    "alpha": float(alpha),
                    "score": float(score),
                    "strategy_vol": details.get("strategy_vol"),
                    "market_vol": details.get("market_vol"),
                    "vol_penalty": details.get("vol_penalty"),
                    "return_penalty": details.get("return_penalty"),
                }
            )
            if score > best_score:
                best_score = float(score)
                best_k = float(k)
                best_alpha = float(alpha)

    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    if res.empty:
        return CalibrationResult(best_k=np.nan, best_alpha=np.nan, best_score=np.nan, results=res)
    return CalibrationResult(best_k=best_k, best_alpha=best_alpha, best_score=best_score, results=res)


__all__ = [
    "AllocationConfig",
    "CalibrationResult",
    "apply_allocation_strategy",
    "smooth_allocation",
    "calibrate_global_scale",
]
