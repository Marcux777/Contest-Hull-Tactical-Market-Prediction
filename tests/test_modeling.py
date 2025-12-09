import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.models as hm  # noqa: E402


def test_adjusted_sharpe_score_constant():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            "forward_returns": [0.01, -0.005, 0.002, 0.004],
            "risk_free_rate": [0.0002, 0.0002, 0.0002, 0.0002],
            "is_scored": [1, 1, 1, 1],
        }
    )
    alloc = pd.Series(1.0, index=df.index)
    score, _ = hm.adjusted_sharpe_score(df, alloc)
    assert np.isclose(score, 7.515351847771772, atol=1e-9)


def test_make_time_splits_non_empty():
    hm.set_data_columns(is_scored_col="is_scored")
    df = pd.DataFrame({"date_id": list(range(10)), "is_scored": [1] * 10, "forward_returns": np.linspace(0.0, 0.01, 10)})
    splits = hm.make_time_splits(df, n_splits=3, val_frac=0.2)
    assert splits, "Splits should not be empty for sequential dates"
    for train_mask, val_mask in splits:
        assert train_mask.sum() > 0 and val_mask.sum() > 0


def test_prepare_train_test_frames_shapes():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    train = pd.DataFrame(
        {
            "date_id": [0, 1, 2],
            "is_scored": [1, 1, 1],
            "forward_returns": [0.01, 0.0, -0.002],
            "risk_free_rate": [0.0001, 0.0001, 0.0001],
            "target": [0.01, 0.0, -0.002],
            "M1": [1.0, 2.0, 3.0],
        }
    )
    test = pd.DataFrame(
        {
            "date_id": [3],
            "forward_returns": [0.001],
            "risk_free_rate": [0.0001],
            "M1": [4.0],
        }
    )
    prep = hm.prepare_train_test_frames(train, test, ["M1"], target_col="target")
    assert prep["X_tr"].shape[0] == len(train)
    assert prep["X_te"].shape[0] == len(test)
    assert "M1" in prep["feature_cols"]
