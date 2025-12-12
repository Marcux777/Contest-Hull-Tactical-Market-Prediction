"""Competition-specific helpers for Hull Tactical - Market Prediction.

Keeps schema/column inference and target normalization in one place so the
notebook and pipeline stay small and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HullColumns:
    """Resolved column names used across the project."""

    target_col: str
    market_col: str
    rf_col: str
    is_scored_col: str | None
    raw_target_col: str


DEFAULT_TARGET_CANDIDATES = ["market_forward_excess_returns", "target", "Target", "forward_returns"]
DEFAULT_MARKET_CANDIDATES = ["forward_returns", "market_forward_excess_returns"]
DEFAULT_RF_CANDIDATES = ["risk_free_rate", "risk_free_returns"]
DEFAULT_IS_SCORED_COL = "is_scored"


def infer_hull_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    target_candidates: list[str] | None = None,
    market_candidates: list[str] | None = None,
    rf_candidates: list[str] | None = None,
    is_scored_col: str = DEFAULT_IS_SCORED_COL,
    normalized_target_col: str = "target",
) -> HullColumns:
    target_candidates = target_candidates or list(DEFAULT_TARGET_CANDIDATES)
    market_candidates = market_candidates or list(DEFAULT_MARKET_CANDIDATES)
    rf_candidates = rf_candidates or list(DEFAULT_RF_CANDIDATES)

    available_targets = [c for c in target_candidates if c in train_df.columns]
    raw_target_col = available_targets[0] if available_targets else None
    if raw_target_col is None:
        raise ValueError("Não encontrei coluna alvo no train (market_forward_excess_returns/target/forward_returns).")

    market_col = next((c for c in market_candidates if c in train_df.columns), None)
    if market_col is None:
        raise ValueError("Não encontrei coluna de retorno de mercado no train (forward_returns).")

    rf_col = next((c for c in rf_candidates if c in train_df.columns), None)
    if rf_col is None:
        raise ValueError("Não encontrei coluna de taxa livre de risco no train (risk_free_rate).")

    has_is_scored = False
    if is_scored_col in train_df.columns:
        has_is_scored = True
    if test_df is not None and is_scored_col in test_df.columns:
        has_is_scored = True
    is_scored_resolved = is_scored_col if has_is_scored else None

    return HullColumns(
        target_col=normalized_target_col,
        market_col=market_col,
        rf_col=rf_col,
        is_scored_col=is_scored_resolved,
        raw_target_col=raw_target_col,
    )


def _ensure_market_excess(
    df: pd.DataFrame,
    *,
    forward_returns_col: str = "forward_returns",
    rf_col: str = "risk_free_rate",
    out_col: str = "market_forward_excess_returns",
) -> pd.DataFrame:
    """Ensures market excess returns exist (forward_returns - risk_free_rate)."""
    if out_col in df.columns:
        return df
    if forward_returns_col not in df.columns or rf_col not in df.columns:
        return df
    out = df.copy()
    out[out_col] = out[forward_returns_col] - out[rf_col]
    return out


def prepare_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    normalized_target_col: str = "target",
    is_scored_col: str = DEFAULT_IS_SCORED_COL,
) -> tuple[pd.DataFrame, pd.DataFrame | None, HullColumns]:
    """Returns copies of train/test with a normalized `target` column.

    - Chooses the raw target column from known candidates.
    - If only `forward_returns` is available, creates `market_forward_excess_returns`.
    - Adds/overwrites `normalized_target_col` on train as the chosen target.
    - Adds `normalized_target_col` on test (NaN) to keep schema consistent.
    """
    cols = infer_hull_columns(
        train_df,
        test_df,
        is_scored_col=is_scored_col,
        normalized_target_col=normalized_target_col,
    )

    train_out = train_df.copy()
    test_out = test_df.copy() if test_df is not None else None

    if cols.raw_target_col == "forward_returns":
        train_out = _ensure_market_excess(train_out, rf_col=cols.rf_col)
        cols = cols.__class__(**{**cols.__dict__, "raw_target_col": "market_forward_excess_returns"})

    if cols.raw_target_col not in train_out.columns:
        raise ValueError(f"Coluna alvo '{cols.raw_target_col}' não encontrada após normalização.")

    train_out[cols.target_col] = train_out[cols.raw_target_col]

    if test_out is not None and cols.target_col not in test_out.columns:
        test_out[cols.target_col] = np.nan

    return train_out, test_out, cols


__all__ = [
    "HullColumns",
    "infer_hull_columns",
    "prepare_train_test",
]

