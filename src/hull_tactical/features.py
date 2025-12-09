"""
Feature engineering and preprocessing helpers for the Hull Tactical notebook.
Kept dependency-free (pandas/numpy only) so they can run inside Kaggle without extra installs.

Feature sets expostos (chaves usuais do dict retornado por build_feature_sets/make_features):
- A_baseline: colunas numéricas originais (exceto ids/target/forward_returns/rf) sem agregações.
- B_families: baseline + agregações por família (mean/std/median).
- C_regimes: B + regimes (std/mean e flags de vol).
- D_intentional: C + features intencionais (lags, clip, tanh, z).
- E_fe_oriented: D + restante das features engenheiradas (quando use_extended_set=True).
- F_v2_intentional: E + quaisquer colunas adicionais não cobertas em E (fallback).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

INTENTIONAL_DEFAULTS = {
    "clip_bounds": (-0.05, 0.05),  # deterministic clipping on lagged excess return
    "tanh_scale": 1.0,  # scale before tanh
    "zscore_window": 20,  # std window for lagged z-score
    "zscore_clip": 5.0,  # limit for z-score truncation
}
INTENTIONAL_CFG = dict(INTENTIONAL_DEFAULTS)

FEATURE_CFG_DEFAULT = {
    "winsor_quantile": 0.995,  # bilateral clipping for highly skewed features
    "skew_threshold": 3.0,  # apply winsor if |skew| >= threshold
    "add_family_medians": True,
    "add_ratios": True,  # mean/std and mean-median per family
    "add_diffs": True,  # mean-std per family
    "use_extended_set": True,  # expose set E_fe_oriented
}

FEATURE_SET_NAMES = ["A_baseline", "B_families", "C_regimes", "D_intentional", "E_fe_oriented", "F_v2_intentional"]


def prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    sort_key = "date_id" if "date_id" in df.columns else None
    df_sorted = df.sort_values(sort_key) if sort_key else df.copy()
    if df_sorted.columns.has_duplicates:
        df_sorted = df_sorted.loc[:, ~df_sorted.columns.duplicated()]
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {target, "row_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"}
    for col in ["date_id", "is_scored"]:
        if col in df_sorted.columns:
            drop_cols.add(col)
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    feature_cols = [c for c in feature_cols if df_sorted[c].nunique() > 1]
    return df_sorted, feature_cols


def preprocess_basic(
    df: pd.DataFrame,
    feature_cols: list[str],
    ref_cols: list[str] | None = None,
    ref_medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_frame = df.reindex(columns=feature_cols)
    medians = ref_medians if ref_medians is not None else feature_frame.median()
    filled = feature_frame.fillna(medians).fillna(0)

    missing_mask = feature_frame.isna()
    if missing_mask.columns.has_duplicates:
        missing_mask = missing_mask.loc[:, ~missing_mask.columns.duplicated()]
    flag_source_cols = []
    for c in feature_cols:
        mask_c = missing_mask[c]
        has_nan = mask_c.any().any() if isinstance(mask_c, pd.DataFrame) else bool(mask_c.any())
        if has_nan:
            flag_source_cols.append(c)
    flags_df = None
    if flag_source_cols:
        flag_unique = list(dict.fromkeys(flag_source_cols))
        flags_df = missing_mask[flag_unique].astype(int)
        flags_df.columns = [f"{c}_was_nan" for c in flag_unique]

    parts = [filled]
    if flags_df is not None:
        parts.append(flags_df)
    df_proc = pd.concat(parts, axis=1)

    if df_proc.columns.has_duplicates:
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    keep = [col for col in df_proc.columns if df_proc[col].std(ddof=0) > 1e-9]
    out = df_proc[keep] if ref_cols is None else df_proc.reindex(columns=ref_cols, fill_value=0)
    return out, list(out.columns), medians


def time_split(df: pd.DataFrame, cutoff: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * cutoff)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def append_columns(df: pd.DataFrame, cols_dict: dict[str, pd.Series | np.ndarray | float | int]) -> pd.DataFrame:
    """Concatenate derived columns uniquely to avoid fragmentation."""
    if not cols_dict:
        return df
    new_df = pd.DataFrame(cols_dict, index=df.index)
    combined = pd.concat([df, new_df], axis=1)
    if combined.columns.has_duplicates:
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    return combined


def add_missing_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return df
    filler = {c: fill_value for c in missing}
    return append_columns(df, filler)


def align_feature_frames(train_df: pd.DataFrame, other_df: pd.DataFrame | None, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """
    Aligns columns between train/val/test before preprocess_basic:
    - removes duplicates keeping the last occurrence;
    - adds missing columns;
    - drops features that are all-NaN or constant in the train.
    """
    cols_unique = list(dict.fromkeys(feature_cols))
    train_clean = train_df.copy()
    if train_clean.columns.has_duplicates:
        train_clean = train_clean.loc[:, ~train_clean.columns.duplicated(keep="last")]

    other_clean = None
    if other_df is not None:
        other_clean = other_df.copy()
        if other_clean.columns.has_duplicates:
            other_clean = other_clean.loc[:, ~other_clean.columns.duplicated(keep="last")]

    train_clean = add_missing_columns(train_clean, cols_unique)
    if other_clean is not None:
        other_clean = add_missing_columns(other_clean, cols_unique)

    valid_cols = []
    for c in cols_unique:
        series = train_clean[c]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if series.isna().all():
            continue
        if series.std(ddof=0) <= 1e-12:
            continue
        valid_cols.append(c)

    train_clean = add_missing_columns(train_clean, valid_cols)
    if other_clean is not None:
        other_clean = add_missing_columns(other_clean, valid_cols)
    return train_clean, other_clean, valid_cols


def add_lagged_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lagged_* columns to align train/test with lagged test features."""
    if "date_id" not in df.columns:
        return df
    out = df.copy()
    df_sorted = df.sort_values("date_id")
    new_cols = {}

    def add_shift(src_col: str, dest_col: str) -> None:
        if dest_col in out.columns or src_col not in df_sorted.columns:
            return
        shifted = df_sorted[src_col].shift(1).reindex(df.index)
        new_cols[dest_col] = shifted

    add_shift("forward_returns", "lagged_forward_returns")
    add_shift("risk_free_rate", "lagged_risk_free_rate")
    add_shift("market_forward_excess_returns", "lagged_market_forward_excess_returns")
    return append_columns(out, new_cols)


def add_family_aggs(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    fams = {g: [c for c in df_out.columns if c.startswith(g)] for g in ["M", "E", "I", "P", "V"]}
    new_cols = {}
    for fam, cols in fams.items():
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_out[c])]
        if len(cols) >= 2:
            new_cols[f"{fam}_mean"] = df_out[cols].mean(axis=1)
            new_cols[f"{fam}_std"] = df_out[cols].std(axis=1)
            new_cols[f"{fam}_median"] = df_out[cols].median(axis=1)
    return append_columns(df_out, new_cols)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates regime features based on lagged market returns."""
    if "date_id" not in df.columns:
        return df
    df_out = df.copy()
    df_sorted = df.sort_values("date_id")
    if "market_forward_excess_returns" in df_sorted.columns:
        lagged_market = df_sorted["market_forward_excess_returns"].shift(1).reindex(df.index)
        df_out["lagged_market_excess"] = lagged_market
        df_out["regime_std_20"] = lagged_market.rolling(window=20, min_periods=5).std()
        df_out["regime_high_vol"] = (df_out["regime_std_20"] > df_out["regime_std_20"].median()).astype(int)
    return df_out


def add_intentional_features(df: pd.DataFrame, intentional_cfg: dict | None = None) -> pd.DataFrame:
    """Intentional/hand-crafted features (clip/tanh/zscore of lagged excess return)."""
    cfg = dict(INTENTIONAL_DEFAULTS)
    if intentional_cfg:
        cfg.update(intentional_cfg)
    df_out = df.copy()
    if "market_forward_excess_returns" in df_out.columns:
        df_sorted = df_out.sort_values("date_id") if "date_id" in df_out.columns else df_out
        excess = df_sorted["market_forward_excess_returns"]
        lagged_excess = excess.shift(1).reindex(df.index)
        clip_lo, clip_hi = cfg.get("clip_bounds", (-0.05, 0.05))
        clipped = lagged_excess.clip(clip_lo, clip_hi)
        scaled = clipped * cfg.get("tanh_scale", 1.0)
        df_out["lagged_excess_return"] = lagged_excess
        df_out["lagged_excess_clip"] = clipped
        df_out["lagged_excess_tanh"] = np.tanh(scaled)
        window = cfg.get("zscore_window", 20)
        if window and window > 1:
            rolling_std = lagged_excess.rolling(window=window, min_periods=max(3, window // 2)).std()
            z = (lagged_excess - lagged_excess.rolling(window=window, min_periods=max(3, window // 2)).mean()) / rolling_std.replace(0, np.nan)
            z_clip = cfg.get("zscore_clip", None)
            if z_clip:
                z = z.clip(-z_clip, z_clip)
            df_out["lagged_excess_z"] = z.fillna(0)
    return df_out


def winsorize_skewed_features(df: pd.DataFrame, target: str, fe_cfg: dict, fit_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bilateral winsor/clipping on highly skewed features. If fit_ref is provided, uses its quantiles."""
    df_out = df.copy()
    ref = df if fit_ref is None else fit_ref
    q = fe_cfg.get("winsor_quantile")
    skew_thr = fe_cfg.get("skew_threshold", 3.0)
    if q is None or q <= 0.5 or q >= 1:
        return df_out
    drop_cols = {target, "row_id", "date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"}
    num_cols = [c for c in ref.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    new_cols = {}

    def _first_series(obj):
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return pd.Series(obj)

    for col in num_cols:
        if col not in df_out.columns:
            continue
        series_ref = _first_series(ref[col])
        series = _first_series(df_out[col])
        if series.std(ddof=0) <= 0 or series.isna().all():
            continue
        skew_val = series_ref.skew(skipna=True)
        if np.isnan(skew_val) or abs(skew_val) < skew_thr:
            continue
        lo = series_ref.quantile(1 - q)
        hi = series_ref.quantile(q)
        new_cols[col] = series.clip(lo, hi)
    return df_out.assign(**new_cols) if new_cols else df_out


def add_ratio_diff_features(df: pd.DataFrame, fe_cfg: dict) -> pd.DataFrame:
    """Adds simple ratios and differences between family aggregates."""
    df_out = df.copy()
    fams = ["M", "E", "I", "P", "V"]
    new_cols = {}

    def _to_series(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    if fe_cfg.get("add_ratios", True) or fe_cfg.get("add_diffs", True):
        for fam in fams:
            mean_col = f"{fam}_mean"
            std_col = f"{fam}_std"
            median_col = f"{fam}_median"
            if fe_cfg.get("add_ratios", True) and mean_col in df_out and std_col in df_out:
                mean_ser = _to_series(df_out[mean_col])
                std_ser = _to_series(df_out[std_col])
                denom = std_ser.replace(0, np.nan)
                new_cols[f"{fam}_mean_over_std"] = (mean_ser / denom).fillna(0)
            if fe_cfg.get("add_diffs", True) and mean_col in df_out and std_col in df_out:
                mean_ser = _to_series(df_out[mean_col])
                std_ser = _to_series(df_out[std_col])
                new_cols[f"{fam}_mean_minus_std"] = (mean_ser - std_ser).fillna(0)
            if fe_cfg.get("add_ratios", True) and mean_col in df_out and median_col in df_out:
                mean_ser = _to_series(df_out[mean_col])
                med_ser = _to_series(df_out[median_col])
                new_cols[f"{fam}_mean_minus_median"] = (mean_ser - med_ser).fillna(0)
    return append_columns(df_out, new_cols)


def apply_feature_engineering(df: pd.DataFrame, target: str, fe_cfg: dict | None = None, fit_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature engineering pipeline (winsor + ratios/diffs). fit_ref allows applying train quantiles on val/test."""
    cfg = dict(FEATURE_CFG_DEFAULT)
    if fe_cfg:
        cfg.update(fe_cfg)
    df_out = df.copy()
    df_out = winsorize_skewed_features(df_out, target, cfg, fit_ref=fit_ref)
    df_out = add_ratio_diff_features(df_out, cfg)
    return df_out


def add_finance_combos(df: pd.DataFrame) -> pd.DataFrame:
    """Simple factor combinations: spreads and interactions with risk/vol."""
    df_out = df.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    new_cols = {}

    def _to_series(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0]
        return obj

    if "lagged_excess_return" in df_out.columns and "lagged_market_forward_excess_returns" in df_out.columns:
        lhs = _to_series(df_out["lagged_excess_return"])
        rhs = _to_series(df_out["lagged_market_forward_excess_returns"])
        new_cols["lagged_excess_minus_market"] = lhs - rhs
    if "lagged_market_forward_excess_returns" in df_out.columns and "risk_free_rate" in df_out.columns:
        lhs = _to_series(df_out["lagged_market_forward_excess_returns"])
        rhs = _to_series(df_out["risk_free_rate"])
        new_cols["lagged_market_minus_rf"] = lhs - rhs
    fam_pairs = [("M_mean", "V_mean"), ("P_mean", "E_mean")]
    for a, b in fam_pairs:
        if a in df_out.columns and b in df_out.columns:
            lhs = _to_series(df_out[a])
            rhs = _to_series(df_out[b])
            new_cols[f"{a}_minus_{b}"] = lhs - rhs
            denom = rhs.replace(0, np.nan)
            new_cols[f"{a}_over_{b}"] = (lhs / denom).fillna(0)
    return append_columns(df_out, new_cols)


def add_cross_sectional_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional normalization per day for numeric columns (z-score per family)."""
    if "date_id" not in df.columns:
        return df
    df_out = df.copy()
    families = {g: [c for c in df_out.columns if c.startswith(g) and pd.api.types.is_numeric_dtype(df_out[c])] for g in ["M", "E", "I", "P", "V"]}
    new_cols = {}
    for fam, cols in families.items():
        if not cols:
            continue
        grp = df_out.groupby("date_id")[cols]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        new_cols.update({f"{c}_cs_z": ((df_out[c] - mean[c]) / std[c]).fillna(0) for c in cols})
    return append_columns(df_out, new_cols)


def add_surprise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Surprise = deviation from rolling 5/20 mean of the feature (numeric columns)."""
    if "date_id" not in df.columns:
        return df
    df_sorted = df.loc[:, ~df.columns.duplicated()].sort_values("date_id")
    num_cols = [c for c in df_sorted.columns if pd.api.types.is_numeric_dtype(df_sorted[c]) and c not in {"date_id"}]
    new_cols = {}
    for col in num_cols:
        series = df_sorted[col]
        if series.isna().all():
            continue
        roll5 = series.rolling(window=5, min_periods=3).mean()
        roll20 = series.rolling(window=20, min_periods=5).mean()
        new_cols[f"{col}_surprise_5"] = (series - roll5).reindex(df.index)
        new_cols[f"{col}_surprise_20"] = (series - roll20).reindex(df.index)
    return append_columns(df, new_cols)


def build_feature_sets(df: pd.DataFrame, target: str, intentional_cfg: dict | None = None, fe_cfg: dict | None = None, fit_ref: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = df.loc[:, ~df.columns.duplicated()]
    orig_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    df_eng = add_lagged_market_features(df)
    df_eng = add_family_aggs(df_eng)
    df_eng = add_regime_features(df_eng)
    df_eng = add_intentional_features(df_eng, intentional_cfg=intentional_cfg)
    df_eng = apply_feature_engineering(df_eng, target, fe_cfg=fe_cfg, fit_ref=fit_ref)
    df_eng = add_cross_sectional_norms(df_eng)
    df_eng = add_surprise_features(df_eng)
    df_eng = add_finance_combos(df_eng)
    if df_eng.columns.has_duplicates:
        df_eng = df_eng.loc[:, ~df_eng.columns.duplicated(keep="last")]
    df_eng, all_cols = prepare_features(df_eng, target)

    base_cols = [c for c in orig_numeric if c in all_cols]
    if not base_cols:
        base_cols = all_cols

    agg_cols = [c for c in df_eng.columns if c.endswith("_mean") or c.endswith("_std")]
    regime_cols = [c for c in df_eng.columns if c.startswith("regime_")]
    intentional_cols = [c for c in df_eng.columns if any(k in c for k in ["lagged_excess_return", "lagged_market_excess", "_log1p", "_clip", "_tanh", "_z"])]

    feature_sets = {
        "A_baseline": sorted(set(base_cols)),
        "B_families": sorted(set(base_cols + agg_cols)),
        "C_regimes": sorted(set(base_cols + agg_cols + regime_cols)),
        "D_intentional": sorted(set(base_cols + agg_cols + regime_cols + intentional_cols)),
    }
    cfg = FEATURE_CFG_DEFAULT if fe_cfg is None else fe_cfg
    if cfg.get("use_extended_set", True):
        oriented_cols = [c for c in all_cols if c not in feature_sets["D_intentional"]]
        feature_sets["E_fe_oriented"] = sorted(set(feature_sets["D_intentional"] + oriented_cols))
        new_cols = [c for c in all_cols if c not in feature_sets["E_fe_oriented"]]
        feature_sets["F_v2_intentional"] = sorted(set(feature_sets["E_fe_oriented"] + new_cols))
    return df_eng, feature_sets


def make_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_set: str | None = None,
    intentional_cfg: dict | None = None,
    fe_cfg: dict | None = None,
):
    """
    Unified feature pipeline for all flows (CV, diagnostics, final training).
    Inputs:
    - train_df/test_df: precisam conter colunas numéricas e, quando disponível, date_id/forward_returns/risk_free_rate/market_forward_excess_returns.
    - target_col: nome da coluna alvo no train_df (ex.: "target").
    - feature_set: chave do feature set (A_baseline, B_families, C_regimes, D_intentional, E_fe_oriented, F_v2_intentional).
    - intentional_cfg / fe_cfg: dicionários para ajustar clipping/tanh/zscore e FE (winsor, ratios/diffs, etc.).

    Saídas:
    - train_fe/test_fe: dataframes com features engenheiradas (antes do preprocess_basic);
    - feature_cols: lista de colunas do feature_set escolhido;
    - feature_sets: dict {feature_set_name -> lista de colunas};
    - feature_set: nome efetivo usado (cai no default se None).
    """
    if target_col is None:
        raise ValueError("target_col is required in make_features")
    train_fe, feature_sets = build_feature_sets(train_df, target_col, intentional_cfg=intentional_cfg, fe_cfg=fe_cfg, fit_ref=None)
    test_fe = None
    if test_df is not None:
        test_fe, _ = build_feature_sets(test_df, target_col, intentional_cfg=intentional_cfg, fe_cfg=fe_cfg, fit_ref=train_fe)
    if feature_set is None:
        feature_set = "D_intentional" if "D_intentional" in feature_sets else next(iter(feature_sets))
    feature_cols = feature_sets.get(feature_set, next(iter(feature_sets.values())))
    return train_fe, test_fe, feature_cols, feature_sets, feature_set


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    target_col: str | None = None,
    feature_set: str | None = None,
    intentional_cfg: dict | None = None,
    fe_cfg: dict | None = None,
):
    """Alias for backward compatibility; delegates to make_features."""
    return make_features(
        train_df,
        test_df=test_df,
        target_col=target_col,
        feature_set=feature_set,
        intentional_cfg=intentional_cfg,
        fe_cfg=fe_cfg,
    )


__all__ = [
    "INTENTIONAL_DEFAULTS",
    "INTENTIONAL_CFG",
    "FEATURE_CFG_DEFAULT",
    "prepare_features",
    "preprocess_basic",
    "time_split",
    "append_columns",
    "add_missing_columns",
    "align_feature_frames",
    "add_lagged_market_features",
    "add_family_aggs",
    "add_regime_features",
    "add_intentional_features",
    "winsorize_skewed_features",
    "add_ratio_diff_features",
    "apply_feature_engineering",
    "add_finance_combos",
    "add_cross_sectional_norms",
    "add_surprise_features",
    "build_feature_sets",
    "make_features",
    "build_features",
]
