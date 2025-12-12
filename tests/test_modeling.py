import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.models as hm  # noqa: E402
from hull_tactical.allocation import AllocationConfig  # noqa: E402


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


def test_adjusted_sharpe_score_vol_penalty_threshold():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            "forward_returns": [0.01, -0.01, 0.02, -0.02],
            "risk_free_rate": [0.0, 0.0, 0.0, 0.0],
            "is_scored": [1, 1, 1, 1],
        }
    )
    alloc = pd.Series(1.5, index=df.index)
    score, details = hm.adjusted_sharpe_score(df, alloc)
    assert np.isfinite(score)
    assert np.isclose(details["vol_penalty"], 1.3, atol=1e-12)

    alloc_edge = pd.Series(1.2, index=df.index)
    score_edge, details_edge = hm.adjusted_sharpe_score(df, alloc_edge)
    assert np.isfinite(score_edge)
    assert np.isclose(details_edge["vol_penalty"], 1.0, atol=1e-12)


def test_adjusted_sharpe_score_return_penalty_behavior():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            # All positive but not constant -> market_std > 0.
            "forward_returns": [0.01, 0.02, 0.015, 0.005],
            "risk_free_rate": [0.0, 0.0, 0.0, 0.0],
            "is_scored": [1, 1, 1, 1],
        }
    )
    alloc_under = pd.Series(0.5, index=df.index)
    score_under, details_under = hm.adjusted_sharpe_score(df, alloc_under)
    assert np.isfinite(score_under)
    assert details_under["return_penalty"] > 1.0

    alloc_over = pd.Series(1.5, index=df.index)
    score_over, details_over = hm.adjusted_sharpe_score(df, alloc_over)
    assert np.isfinite(score_over)
    assert np.isclose(details_over["return_penalty"], 1.0, atol=1e-12)


def test_adjusted_sharpe_score_respects_is_scored_mask():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    df = pd.DataFrame(
        {
            "forward_returns": [0.01, -0.01, 0.02, -0.02, 0.5, -0.5],
            "risk_free_rate": [0.0] * 6,
            "is_scored": [1, 1, 1, 1, 0, 0],
        }
    )
    alloc = pd.Series(1.0, index=df.index)
    score_all, _ = hm.adjusted_sharpe_score(df, alloc)
    df_scored = df.loc[df["is_scored"] == 1]
    alloc_scored = alloc.reindex(df_scored.index)
    score_scored, _ = hm.adjusted_sharpe_score(df_scored, alloc_scored)
    assert np.isclose(score_all, score_scored, atol=1e-12)

    df_none_scored = df.assign(is_scored=0)
    score_nan, details_nan = hm.adjusted_sharpe_score(df_none_scored, alloc)
    assert np.isnan(score_nan)
    assert details_nan == {}


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


def test_time_cv_lightgbm_fitref_oof_uses_global_k_alpha():
    hm.set_data_columns(market_col="forward_returns", rf_col="risk_free_rate", is_scored_col="is_scored")
    n = 40
    date_id = np.arange(n)
    market_excess = 0.01 + 0.002 * np.sin(date_id / 3.0)
    rf = np.zeros(n)
    df = pd.DataFrame(
        {
            "date_id": date_id,
            "is_scored": np.ones(n, dtype=int),
            "risk_free_rate": rf,
            "market_forward_excess_returns": market_excess,
            "forward_returns": market_excess + rf,
            "target": market_excess,
            "M1": market_excess * 100.0,
            "M2": market_excess * 80.0,
            "E1": np.cos(date_id / 5.0),
        }
    )
    alloc_cfg = AllocationConfig(
        k_grid=[0.0, 0.5, 1.0],
        alpha_grid=[1.0],
        high_vol_k_factor=0.7,
        risk_col="regime_std_20",
        risk_power=1.0,
        risk_clip=(0.5, 2.0),
        smooth_alpha=0.2,
        delta_cap=0.1,
    )
    params_override = {
        "learning_rate": 0.1,
        "num_leaves": 8,
        "min_data_in_leaf": 1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbosity": -1,
    }
    res = hm.time_cv_lightgbm_fitref_oof(
        df,
        "D_intentional",
        "target",
        n_splits=4,
        val_frac=0.2,
        params_override=params_override,
        num_boost_round=20,
        allocation_cfg=alloc_cfg,
    )
    assert res["metrics"]
    assert np.isfinite(res["oof_sharpe"])
    assert np.isfinite(res["best_k"])
    assert np.isfinite(res["best_alpha"])

    for m in res["metrics"]:
        assert np.isclose(m["best_k"], res["best_k"], atol=1e-12)
        assert np.isclose(m["best_alpha"], res["best_alpha"], atol=1e-12)
