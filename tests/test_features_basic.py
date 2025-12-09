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
    assert not train_fe[feature_cols].isna().any().any()
    assert not test_fe[feature_cols].isna().any().any()
    assert "M1" in feature_sets[feature_used]
