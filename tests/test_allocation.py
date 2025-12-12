import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.models as hm  # noqa: E402
from hull_tactical.allocation import (  # noqa: E402
    AllocationConfig,
    apply_allocation_strategy,
    calibrate_global_scale,
    smooth_allocation,
)


def test_smooth_allocation_ewma_and_cap():
    df = pd.DataFrame({"date_id": [0, 1, 2, 3]})
    alloc = pd.Series([0.0, 2.0, 0.0, 2.0], index=df.index)
    sm = smooth_allocation(alloc, df, smooth_alpha=0.5, delta_cap=None)
    # EWMA: [0.0, 1.0, 0.5, 1.25]
    assert np.allclose(sm.values, [0.0, 1.0, 0.5, 1.25], atol=1e-12)

    sm_cap = smooth_allocation(alloc, df, smooth_alpha=0.5, delta_cap=0.2)
    # After first step, changes capped to 0.2 each step.
    assert np.allclose(sm_cap.values, [0.0, 0.2, 0.1, 0.3], atol=1e-12)


def test_calibrate_global_scale_prefers_nonzero_k():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            "date_id": [0, 1, 2, 3, 4, 5],
            "pred_return": [0.02, 0.01, -0.01, -0.02, 0.015, -0.015],
            "forward_returns": [0.02, 0.01, -0.01, -0.02, 0.015, -0.015],
            "risk_free_rate": [0.0] * 6,
            "is_scored": [1] * 6,
            "regime_high_vol": [0, 0, 1, 1, 0, 1],
            "regime_std_20": [0.01, 0.01, 0.02, 0.02, 0.01, 0.02],
        }
    )
    cfg = AllocationConfig(
        k_grid=[0.0, 0.5, 1.0, 2.0],
        alpha_grid=[1.0],
        smooth_alpha=None,
        delta_cap=None,
        high_vol_k_factor=0.7,
        risk_col="regime_std_20",
    )
    res = calibrate_global_scale(df, cfg, pred_col="pred_return", market_col="forward_returns", rf_col="risk_free_rate")
    assert not res.results.empty
    assert np.isfinite(res.best_score)
    assert res.best_k > 0.0


def test_apply_allocation_strategy_bounds_and_index():
    df = pd.DataFrame({"date_id": [0, 1, 2], "regime_high_vol": [0, 1, 0], "regime_std_20": [0.01, 0.03, 0.01]})
    pred = pd.Series([10.0, -10.0, 10.0], index=df.index)
    cfg = AllocationConfig(min_allocation=0.0, max_allocation=2.0, smooth_alpha=None)
    alloc = apply_allocation_strategy(pred, df, k=1.0, alpha=1.0, cfg=cfg)
    assert list(alloc.index) == list(df.index)
    assert float(alloc.min()) >= 0.0
    assert float(alloc.max()) <= 2.0


def test_apply_allocation_strategy_regime_and_risk_scaling():
    df = pd.DataFrame(
        {
            "date_id": [0, 1],
            "regime_high_vol": [0, 1],
            "regime_std_20": [1.0, 2.0],
        }
    )
    pred = pd.Series([0.1, 0.1], index=df.index)

    # High-vol regime should reduce aggressiveness.
    cfg_regime = AllocationConfig(high_vol_k_factor=0.5, risk_col=None, smooth_alpha=None, delta_cap=None)
    alloc_regime = apply_allocation_strategy(pred, df, k=1.0, alpha=1.0, cfg=cfg_regime)
    assert alloc_regime.iloc[1] < alloc_regime.iloc[0]

    # Higher risk should reduce aggressiveness (median / risk).
    cfg_risk = AllocationConfig(
        high_vol_k_factor=1.0,
        risk_col="regime_std_20",
        risk_power=1.0,
        risk_clip=(0.0, 10.0),
        smooth_alpha=None,
        delta_cap=None,
    )
    alloc_risk = apply_allocation_strategy(pred, df, k=1.0, alpha=1.0, cfg=cfg_risk)
    assert alloc_risk.iloc[0] > alloc_risk.iloc[1]


def test_score_oof_predictions_with_allocation_by_fold():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            "date_id": [0, 1, 2, 3, 4, 5],
            "fold": [1, 1, 1, 2, 2, 2],
            "pred_return": [0.02, 0.01, -0.01, -0.02, 0.015, -0.015],
            "forward_returns": [0.02, 0.01, -0.01, -0.02, 0.015, -0.015],
            "risk_free_rate": [0.0] * 6,
            "is_scored": [1] * 6,
        }
    )
    cfg = AllocationConfig(k_grid=[0.0, 0.5, 1.0, 2.0], alpha_grid=[1.0], smooth_alpha=None, risk_col=None, high_vol_k_factor=1.0)
    out = hm.score_oof_predictions_with_allocation(
        df,
        allocation_cfg=cfg,
        pred_col="pred_return",
        market_col="forward_returns",
        rf_col="risk_free_rate",
        is_scored_col="is_scored",
        fold_col="fold",
    )
    assert out["calibration"] is not None
    assert len(out["metrics"]) == 2
    assert np.isfinite(out["oof_sharpe"])
    assert float(out["calibration"].best_k) > 0.0
