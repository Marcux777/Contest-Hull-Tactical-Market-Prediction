import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hull_tactical import features  # noqa: E402


def test_make_features_train_test_have_same_feature_columns_and_no_nan_after_preprocess():
    rng = np.random.default_rng(0)
    n_train = 80
    n_test = 15

    date_train = np.arange(n_train)
    date_test = np.arange(n_train, n_train + n_test)

    market_train = rng.normal(0.001, 0.002, size=n_train)
    market_test = rng.normal(0.001, 0.08, size=n_test)  # much higher vol
    rf_train = np.full(n_train, 0.0001)
    rf_test = np.full(n_test, 0.0001)

    # Base families + a deliberately extreme outlier in test to validate fit_ref winsorization.
    m1_train = rng.normal(0.0, 1.0, size=n_train)
    m1_train[-1] = 20.0
    m1_test = 1_000_000.0 + np.arange(n_test, dtype=float)
    e1_train = rng.normal(0.0, 1.0, size=n_train)
    e1_test = rng.normal(0.0, 1.0, size=n_test)

    df_train = pd.DataFrame(
        {
            "date_id": date_train,
            "target": market_train - rf_train,
            "forward_returns": market_train,
            "risk_free_rate": rf_train,
            "market_forward_excess_returns": market_train - rf_train,
            "M1": m1_train,
            "E1": e1_train,
        }
    )
    df_test = pd.DataFrame(
        {
            "date_id": date_test,
            "forward_returns": market_test,
            "risk_free_rate": rf_test,
            "market_forward_excess_returns": market_test - rf_test,
            "M1": m1_test,
            "E1": e1_test,
        }
    )

    fe_cfg = {
        # Stronger-than-default winsorization so the guardrail is deterministic.
        "winsor_quantile": 0.8,
        "skew_threshold": 0.0,
        "use_extended_set": False,
    }
    train_fe, test_fe, feature_cols, _feature_sets, _used = features.make_features(
        df_train,
        test_df=df_test,
        target_col="target",
        feature_set="D_intentional",
        fe_cfg=fe_cfg,
    )

    missing_train = sorted(set(feature_cols) - set(train_fe.columns))
    missing_test = sorted(set(feature_cols) - set(test_fe.columns))
    assert missing_train == []
    assert missing_test == []

    # After preprocess, both must share the exact same schema and be NaN-safe.
    train_proc, proc_cols, medians = features.preprocess_basic(train_fe, feature_cols)
    test_proc, _, _ = features.preprocess_basic(test_fe, feature_cols, ref_cols=proc_cols, ref_medians=medians)
    assert list(train_proc.columns) == list(test_proc.columns)
    assert not train_proc.isna().any().any()
    assert not test_proc.isna().any().any()

    # fit_ref guardrail: extreme test outlier should be clipped by train quantiles.
    hi = float(train_fe["M1"].quantile(0.8))
    assert float(test_fe["M1"].max()) <= hi + 1e-9

    # fit_ref guardrail for regimes: for a very high-vol test, most rows should be high-vol under train threshold.
    if "regime_high_vol" in test_fe.columns:
        # ignore warm-up NaNs due to rolling std min_periods
        frac_high = float(test_fe.loc[test_fe.index[6:], "regime_high_vol"].mean())
        assert frac_high >= 0.7
