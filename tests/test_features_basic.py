import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hull_tactical import features  # noqa: E402


def test_make_features_no_nan_and_shape():
    df_train = pd.DataFrame(
        {
            "date_id": [0, 1, 2],
            "target": [0.1, -0.05, 0.02],
            "forward_returns": [0.1, -0.05, 0.02],
            "risk_free_rate": [0.001, 0.001, 0.001],
            "market_forward_excess_returns": [0.1, -0.05, 0.02],
            "M1": [1.0, 2.0, 3.0],
            "E1": [0.5, 0.6, 0.7],
        }
    )
    df_test = pd.DataFrame(
        {
            "date_id": [3],
            "forward_returns": [0.01],
            "risk_free_rate": [0.001],
            "market_forward_excess_returns": [0.01],
            "M1": [4.0],
            "E1": [0.8],
        }
    )
    train_fe, test_fe, feature_cols, feature_sets, feature_used = features.make_features(
        df_train, test_df=df_test, target_col="target", feature_set="D_intentional"
    )
    assert len(train_fe) == len(df_train)
    assert len(test_fe) == len(df_test)
    # make_features pode produzir NaNs (lags/rollings). O contrato é que o
    # preprocess_basic consiga imputar de forma consistente entre train/test.
    train_proc, proc_cols, medians = features.preprocess_basic(train_fe, feature_cols)
    test_proc, _, _ = features.preprocess_basic(test_fe, feature_cols, ref_cols=proc_cols, ref_medians=medians)
    assert not train_proc.isna().any().any()
    assert not test_proc.isna().any().any()
    assert "M1" in feature_sets[feature_used]


def test_make_features_does_not_use_future_return_columns_as_features():
    # Train has the future returns (for evaluation / target construction).
    # Test does NOT have them, only lagged_* versions -> features must not depend on forward_returns/rf directly.
    df_train = pd.DataFrame(
        {
            "date_id": list(range(10)),
            "target": [0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.01, -0.005, 0.004],
            "forward_returns": [0.011, -0.019, 0.031, 0.001, 0.011, -0.009, 0.021, 0.011, -0.004, 0.005],
            "risk_free_rate": [0.001] * 10,
            "market_forward_excess_returns": [0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, 0.01, -0.005, 0.004],
            "M1": list(range(10)),
            "E1": list(range(10, 20)),
        }
    )
    df_test = pd.DataFrame(
        {
            "date_id": [10, 11],
            "is_scored": [1, 1],
            "lagged_forward_returns": [0.0, 0.01],
            "lagged_risk_free_rate": [0.001, 0.001],
            "lagged_market_forward_excess_returns": [0.0, 0.009],
            "M1": [10.0, 11.0],
            "E1": [20.0, 21.0],
        }
    )

    fe_cfg = {
        "enable_surprise_features": True,  # should only use lagged/aggregate cols
        "enable_cross_sectional_norms": True,  # should be ignored for 1 row/day
        "use_extended_set": True,
    }
    _train_fe, _test_fe, feature_cols, _sets, _used = features.make_features(
        df_train,
        test_df=df_test,
        target_col="target",
        feature_set="D_intentional",
        fe_cfg=fe_cfg,
    )

    assert "forward_returns" not in feature_cols
    assert "risk_free_rate" not in feature_cols
    assert "market_forward_excess_returns" not in feature_cols
    assert "target" not in feature_cols

    for col in feature_cols:
        if "forward_returns" in col and not col.startswith("lagged_"):
            raise AssertionError(f"unexpected feature uses forward_returns: {col}")
        if "risk_free_rate" in col and not col.startswith("lagged_"):
            raise AssertionError(f"unexpected feature uses risk_free_rate: {col}")
        if "market_forward_excess_returns" in col and not col.startswith("lagged_"):
            raise AssertionError(f"unexpected feature uses market_forward_excess_returns: {col}")
        if col.startswith("target"):
            raise AssertionError(f"unexpected feature uses target: {col}")
