import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.config as ht_config  # noqa: E402
import hull_tactical.data as ht_data  # noqa: E402
import hull_tactical.models as hm  # noqa: E402
from hull_tactical import artifacts, competition, features, pipeline  # noqa: E402
from hull_tactical.allocation import apply_allocation_strategy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inferência rápida (sem treinar): carrega artefatos e gera alocações.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path, default=None, help="Opcional: para validar/ler AllocationConfig do YAML.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifact", type=str, default="artifact.json")
    parser.add_argument("--out", type=Path, default=Path("data/submissions/submission.csv"))
    args = parser.parse_args()

    artifact_path = args.artifacts_dir / args.artifact
    art = artifacts.load_artifact(artifact_path)
    if not art.model_files:
        raise FileNotFoundError(f"artifact sem model_files: {artifact_path}")

    # Load data (Kaggle-safe path resolution happens inside ht_data/io).
    df_train, df_test = ht_data.load_raw_data(args.data_dir)
    df_train, df_test, cols = competition.prepare_train_test(df_train, df_test, normalized_target_col=art.target_col)

    # Apply the same fit_ref policy used in training when generating features.
    df_fit = df_train
    if art.train_only_scored and cols.is_scored_col and cols.is_scored_col in df_fit.columns:
        df_fit = df_fit.loc[df_fit[cols.is_scored_col] == 1].copy()

    train_fe, test_fe, _cols, _sets, used_set = features.make_features(
        df_fit,
        test_df=df_test,
        target_col=art.target_col,
        feature_set=art.feature_set,
        intentional_cfg=art.intentional_cfg,
        fe_cfg=art.feature_cfg,
    )
    if used_set != art.feature_set:
        raise ValueError(f"feature_set mismatch: artifact={art.feature_set} computed={used_set}")

    # Preprocess test into the exact model schema.
    medians = pd.Series(art.medians, dtype=float)
    missing = sorted(set(art.feature_cols) - set(test_fe.columns))
    if missing:
        raise ValueError(f"Test is missing {len(missing)} feature columns: {missing[:20]}")

    X_te, _, _ = features.preprocess_basic(
        test_fe,
        art.feature_cols,
        ref_cols=art.model_feature_cols,
        ref_medians=medians,
    )
    X_te = X_te.reindex(columns=art.model_feature_cols, fill_value=0.0)

    # Predict (supports simple bagging: average over model_files).
    preds: list[np.ndarray] = []
    for rel in art.model_files:
        model_path = args.artifacts_dir / rel
        booster = lgb.Booster(model_file=str(model_path))
        preds.append(booster.predict(X_te))
    pred_mean = np.mean(np.vstack(preds), axis=0)
    pred_series = pd.Series(pred_mean, index=X_te.index, name="pred_return")

    # Allocation mapping (frozen in artifact).
    k_use = float(art.alloc_k) if art.alloc_k is not None else float(hm.ALLOC_K)
    alpha_use = float(art.alloc_alpha) if art.alloc_alpha is not None else 1.0

    alloc_cfg = ht_config.allocation_config_from_dict(art.allocation_cfg)
    if alloc_cfg is None:
        allocations = pd.Series(hm.map_return_to_alloc(pred_series, k=k_use, intercept=alpha_use), index=pred_series.index)
    else:
        allocations = apply_allocation_strategy(pred_series, test_fe, k=k_use, alpha=alpha_use, cfg=alloc_cfg)

    row_id_col = "row_id" if "row_id" in df_test.columns else df_test.columns[0]
    pipeline.make_submission_csv(args.out, allocations, df_test[row_id_col])
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

