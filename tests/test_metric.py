import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.metric as ht_metric  # noqa: E402


def test_adjusted_sharpe_score_penalties_and_is_scored_filter():
    df = pd.DataFrame(
        {
            "date_id": list(range(8)),
            "forward_returns": [0.012, 0.008, 0.011, 0.009, 0.013, 0.007, 0.012, 0.008],
            "risk_free_rate": [0.02] * 8,
            "is_scored": [1, 0, 1, 0, 1, 1, 1, 1],
        }
    )
    alloc = pd.Series(2.0, index=df.index)

    score, details = ht_metric.adjusted_sharpe_score(
        df,
        alloc,
        market_col="forward_returns",
        rf_col="risk_free_rate",
        is_scored_col="is_scored",
    )
    assert np.isfinite(score)
    assert details["vol_penalty"] > 1.0
    assert details["return_penalty"] > 1.0
    assert np.isfinite(details["sharpe_raw"])

    scored_mask = df["is_scored"] == 1
    score_scored, _ = ht_metric.adjusted_sharpe_score(
        df.loc[scored_mask],
        alloc.loc[scored_mask],
        market_col="forward_returns",
        rf_col="risk_free_rate",
        is_scored_col=None,
    )
    assert np.isfinite(score_scored)
    assert np.isclose(score, score_scored)
