import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.models as hm  # noqa: E402
from hull_tactical.allocation import AllocationConfig  # noqa: E402
from hull_tactical.ensemble import evaluate_prediction_ensembles  # noqa: E402


def test_evaluate_prediction_ensembles_stack_beats_mean_on_synthetic():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    n = 20
    date_id = np.arange(n)
    # Alternating signs with small drift keeps market_std > 0 and gives a learnable signal.
    y = 0.01 * np.sign(np.sin(date_id / 2.0 + 0.1)) + 0.002
    df_base = pd.DataFrame(
        {
            "date_id": date_id,
            "fold": np.where(date_id < n // 2, 1, 2),
            "target": y,
            "forward_returns": y,
            "risk_free_rate": np.zeros(n),
            "is_scored": np.ones(n, dtype=int),
        }
    )
    pred_a = df_base.assign(pred_return=df_base["target"])
    pred_b = df_base.assign(pred_return=-df_base["target"])

    alloc_cfg = AllocationConfig(
        k_grid=[0.0, 0.5, 1.0, 2.0],
        alpha_grid=[1.0],
        risk_col=None,
        high_vol_k_factor=1.0,
        smooth_alpha=None,
        delta_cap=None,
    )
    scores = evaluate_prediction_ensembles({"a": pred_a, "b": pred_b}, allocation_cfg=alloc_cfg, ridge_alpha=0.1)
    assert "pred_mean" in scores
    assert "pred_stack_ridge" in scores
    assert np.isfinite(scores["pred_mean"].oof_sharpe)
    assert np.isfinite(scores["pred_stack_ridge"].oof_sharpe)
    assert scores["pred_stack_ridge"].oof_sharpe > scores["pred_mean"].oof_sharpe

