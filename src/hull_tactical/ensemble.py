"""Ensembling utilities focused on the competition metric.

Key idea: blend model predictions (OOF), then calibrate `pred_return -> allocation`
globally using the full allocation strategy (regime/risk/smoothing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .allocation import AllocationConfig, apply_allocation_strategy
from .metric import adjusted_sharpe_score
from .models import map_return_to_alloc, score_oof_predictions_with_allocation


DEFAULT_CONTEXT_COLS: tuple[str, ...] = (
    "target",
    "date_id",
    "fold",
    "forward_returns",
    "risk_free_rate",
    "is_scored",
    "regime_std_20",
    "regime_high_vol",
)


@dataclass(frozen=True)
class EnsembleScores:
    """Scores for one ensemble candidate."""

    name: str
    oof_sharpe: float
    oof_details: dict
    best_k: float
    best_alpha: float
    metrics: list[dict]
    weights: dict[str, float] | None = None


def merge_oof_predictions(
    pred_frames: Mapping[str, pd.DataFrame],
    *,
    pred_col: str = "pred_return",
    context_cols: tuple[str, ...] = DEFAULT_CONTEXT_COLS,
) -> pd.DataFrame:
    """Merge OOF prediction frames into a single dataframe with columns `pred_<name>`."""
    frames = {k: v for k, v in pred_frames.items() if v is not None and not v.empty and pred_col in v.columns}
    if not frames:
        return pd.DataFrame()

    base_name = next(iter(frames))
    base = frames[base_name]
    keep = [c for c in context_cols if c in base.columns]
    merged = base[keep].copy()
    merged[f"pred_{base_name}"] = pd.to_numeric(base[pred_col], errors="coerce")

    for name, df in frames.items():
        if name == base_name:
            continue
        series = pd.to_numeric(df[pred_col], errors="coerce").rename(f"pred_{name}")
        merged = merged.join(series, how="inner")

    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged = merged.dropna(subset=pred_cols)
    if merged.columns.has_duplicates:
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
    return merged


def compute_oof_sharpe_weights(
    pred_frames: Mapping[str, pd.DataFrame],
    *,
    allocation_cfg: AllocationConfig | None = None,
    pred_col: str = "pred_return",
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str = "is_scored",
    fold_col: str = "fold",
    floor: float = 0.02,
) -> dict[str, float]:
    """Compute non-negative weights proportional to OOF Sharpe (clipped at 0)."""
    cfg = allocation_cfg or AllocationConfig()
    scores: dict[str, float] = {}
    for name, df in pred_frames.items():
        if df is None or df.empty or pred_col not in df.columns:
            continue
        out = score_oof_predictions_with_allocation(
            df,
            allocation_cfg=cfg,
            pred_col=pred_col,
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
            fold_col=fold_col,
        )
        sharpe = out.get("oof_sharpe", np.nan)
        if sharpe is None or not np.isfinite(sharpe):
            continue
        scores[name] = max(float(sharpe), 0.0) + float(floor)
    total = float(sum(scores.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in scores.items()}


def ridge_stack_oof(
    merged: pd.DataFrame,
    pred_cols: list[str],
    *,
    target_col: str = "target",
    fold_col: str = "fold",
    alpha: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Cross-fitted Ridge stacking on OOF predictions using the existing fold ids."""
    if merged.empty or not pred_cols:
        return pd.Series(dtype=float)
    if target_col not in merged.columns:
        raise ValueError(f"target_col '{target_col}' not found in merged frame")
    if fold_col not in merged.columns:
        raise ValueError(f"fold_col '{fold_col}' not found in merged frame")

    out = pd.Series(index=merged.index, dtype=float)
    folds = sorted(pd.Series(merged[fold_col]).dropna().unique().tolist())
    if not folds:
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(merged[pred_cols], merged[target_col])
        out.loc[:] = model.predict(merged[pred_cols])
        return out

    for fold in folds:
        train_mask = merged[fold_col] != fold
        val_mask = merged[fold_col] == fold
        if not bool(val_mask.any()):
            continue
        model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=seed)
        model.fit(merged.loc[train_mask, pred_cols], merged.loc[train_mask, target_col])
        out.loc[val_mask] = model.predict(merged.loc[val_mask, pred_cols])
    return out


def _int_compositions(total: int, k: int) -> list[tuple[int, ...]]:
    """All k-tuples of non-negative ints summing to total (small k only)."""
    if k <= 0:
        return []
    if k == 1:
        return [(int(total),)]
    out: list[tuple[int, ...]] = []
    for head in range(total + 1):
        for tail in _int_compositions(total - head, k - 1):
            out.append((head,) + tail)
    return out


def optimize_blend_weights_grid(
    pred_frames: Mapping[str, pd.DataFrame],
    *,
    allocation_cfg: AllocationConfig | None = None,
    pred_col: str = "pred_return",
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str = "is_scored",
    alloc_k: float = 1.0,
    alloc_alpha: float = 1.0,
    grid_step: float = 0.1,
    max_models: int = 4,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search blend weights on OOF predictions (simplex), scoring with adjusted Sharpe.

    Discipline:
    - uses only OOF predictions (already out-of-fold);
    - keeps allocation mapping fixed during weight search (alloc_k/alloc_alpha + AllocationConfig).

    Returns:
    - best_weights: dict {model_name -> weight} (sums to 1)
    - results: DataFrame with weights/score ranked desc
    """
    merged = merge_oof_predictions(pred_frames, pred_col=pred_col)
    if merged.empty:
        return {}, pd.DataFrame()

    names = [n for n in pred_frames.keys() if f"pred_{n}" in merged.columns]
    if len(names) < 2:
        w = {names[0]: 1.0} if names else {}
        return w, pd.DataFrame([{"score": np.nan, **{f"w_{k}": v for k, v in w.items()}}])
    if len(names) > int(max_models):
        raise ValueError(f"Grid search supports up to {max_models} models; got {len(names)}: {names}")

    step = float(grid_step)
    if step <= 0 or step > 1:
        raise ValueError("grid_step must be in (0, 1]")
    denom = int(round(1.0 / step))
    if not np.isclose(denom * step, 1.0, atol=1e-9):
        raise ValueError("grid_step must evenly divide 1.0 (e.g., 0.5, 0.25, 0.2, 0.1, 0.05)")

    pred_cols = [f"pred_{n}" for n in names]
    mat = merged[pred_cols].to_numpy(dtype=float)

    market_use = market_col or "forward_returns"
    rf_use = rf_col or "risk_free_rate"
    scored_use = is_scored_col or "is_scored"
    cfg = allocation_cfg

    rows: list[dict] = []
    best_score = float("-inf")
    best_weights: dict[str, float] = {}

    for comp in _int_compositions(denom, len(names)):
        w = np.asarray(comp, dtype=float) / float(denom)
        if not np.isclose(w.sum(), 1.0, atol=1e-12):
            continue
        blended = mat @ w
        pred_series = pd.Series(blended, index=merged.index, name=pred_col)
        if cfg is None:
            alloc = pd.Series(map_return_to_alloc(pred_series, k=float(alloc_k), intercept=float(alloc_alpha)), index=merged.index)
        else:
            alloc = apply_allocation_strategy(pred_series, merged, k=float(alloc_k), alpha=float(alloc_alpha), cfg=cfg)
        score, details = adjusted_sharpe_score(merged, alloc, market_col=market_use, rf_col=rf_use, is_scored_col=scored_use)
        if score is None or not np.isfinite(score):
            continue
        row = {"score": float(score), "strategy_vol": details.get("strategy_vol")}
        row.update({f"w_{n}": float(wi) for n, wi in zip(names, w)})
        rows.append(row)
        if float(score) > best_score:
            best_score = float(score)
            best_weights = {n: float(wi) for n, wi in zip(names, w)}

    res = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    return best_weights, res


def evaluate_prediction_ensembles(
    pred_frames: Mapping[str, pd.DataFrame],
    *,
    allocation_cfg: AllocationConfig | None = None,
    pred_col: str = "pred_return",
    market_col: str | None = None,
    rf_col: str | None = None,
    is_scored_col: str = "is_scored",
    fold_col: str = "fold",
    weights: Mapping[str, float] | None = None,
    ridge_alpha: float = 0.1,
    grid_weight_search: bool = False,
    grid_step: float = 0.1,
) -> dict[str, EnsembleScores]:
    """Evaluate simple prediction ensembles (mean/weighted/stacked) with global allocation calibration."""
    merged = merge_oof_predictions(pred_frames, pred_col=pred_col)
    if merged.empty:
        return {}

    cfg = allocation_cfg or AllocationConfig()
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    if not pred_cols:
        return {}

    scores: dict[str, EnsembleScores] = {}

    candidates: dict[str, pd.Series] = {"pred_mean": merged[pred_cols].mean(axis=1)}

    if weights:
        w = {f"pred_{k}": float(v) for k, v in weights.items() if f"pred_{k}" in pred_cols}
        total = float(sum(w.values()))
        if total > 0:
            candidates["pred_weighted"] = sum(w[c] * merged[c] for c in w) / total

    if grid_weight_search and len(pred_cols) >= 2:
        best_w, _res = optimize_blend_weights_grid(
            pred_frames,
            allocation_cfg=cfg,
            pred_col=pred_col,
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
            alloc_k=1.0,
            alloc_alpha=1.0,
            grid_step=grid_step,
            max_models=len(pred_cols),
        )
        if best_w:
            w = {f"pred_{k}": float(v) for k, v in best_w.items() if f"pred_{k}" in pred_cols}
            total = float(sum(w.values()))
            if total > 0:
                candidates["pred_grid_weights"] = sum(w[c] * merged[c] for c in w) / total

    stack = ridge_stack_oof(merged, pred_cols, alpha=ridge_alpha, target_col="target", fold_col=fold_col)
    if not stack.empty and stack.notna().any():
        candidates["pred_stack_ridge"] = stack

    for name, series in candidates.items():
        df_eval = merged.copy()
        df_eval[pred_col] = series
        out = score_oof_predictions_with_allocation(
            df_eval,
            allocation_cfg=cfg,
            pred_col=pred_col,
            market_col=market_col,
            rf_col=rf_col,
            is_scored_col=is_scored_col,
            fold_col=fold_col,
        )
        calib = out.get("calibration")
        scores[name] = EnsembleScores(
            name=name,
            oof_sharpe=float(out.get("oof_sharpe", np.nan)),
            oof_details=out.get("oof_details") or {},
            best_k=float(getattr(calib, "best_k", np.nan)),
            best_alpha=float(getattr(calib, "best_alpha", np.nan)),
            metrics=out.get("metrics") or [],
            weights=dict(weights) if weights else None,
        )
    return scores


__all__ = [
    "DEFAULT_CONTEXT_COLS",
    "EnsembleScores",
    "merge_oof_predictions",
    "compute_oof_sharpe_weights",
    "optimize_blend_weights_grid",
    "ridge_stack_oof",
    "evaluate_prediction_ensembles",
]
