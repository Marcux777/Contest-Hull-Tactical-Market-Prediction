import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.config as ht_config  # noqa: E402
import hull_tactical.data as ht_data  # noqa: E402
import hull_tactical.models as hm  # noqa: E402
from hull_tactical import artifacts, competition  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Treino offline: gera artefatos para inferência (sem Kaggle API key).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--artifact-name", type=str, default="artifact.json")
    args = parser.parse_args()

    loaded = ht_config.load_all_configs(args.config_dir)
    run_cfg = loaded.run_cfg if isinstance(loaded.run_cfg, dict) else {}
    model_cfg = run_cfg.get("model") if isinstance(run_cfg.get("model"), dict) else {}
    feature_set = args.feature_set or run_cfg.get("feature_set") or "D_intentional"
    target_col = args.target_col or run_cfg.get("target_col") or "target"
    num_boost_round = model_cfg.get("num_boost_round")
    bagging_seeds = model_cfg.get("bagging_seeds")
    if isinstance(bagging_seeds, list) and bagging_seeds:
        bagging_seeds = [int(x) for x in bagging_seeds]
    else:
        bagging_seeds = [int(run_cfg.get("seed", 42))]

    train_only_scored = bool(run_cfg.get("train_only_scored", False))
    weight_scored = run_cfg.get("weight_scored")
    weight_unscored = run_cfg.get("weight_unscored")

    alloc_k = run_cfg.get("alloc_k")
    alloc_alpha = float(run_cfg.get("alloc_alpha", 1.0))

    df_train, df_test = ht_data.load_raw_data(args.data_dir)
    df_train, _df_test_unused, cols = competition.prepare_train_test(df_train, df_test, normalized_target_col=target_col)

    cfg = hm.default_config(args.config_dir)
    cfg.market_col = cols.market_col
    cfg.rf_col = cols.rf_col
    cfg.is_scored_col = cols.is_scored_col
    hm.set_data_columns(cols.market_col, cols.rf_col, cols.is_scored_col)

    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prep = hm.prepare_train_test_frames(
        df_train,
        df_test=None,
        feature_cols=None,
        target_col=target_col,
        feature_set=feature_set,
        intentional_cfg=cfg.intentional_cfg,
        fe_cfg=cfg.feature_cfg,
        train_only_scored=train_only_scored,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        cfg=cfg,
    )
    X_tr = prep["X_tr"]
    y_tr = prep["y_tr"]
    train_weight = prep["train_weight"]

    params_use = dict(cfg.best_params or hm.BEST_PARAMS)
    params_use["objective"] = "regression"
    params_use["metric"] = "rmse"

    num_round = int(num_boost_round) if num_boost_round is not None else int(params_use.pop("num_boost_round", 200))
    model_files: list[str] = []
    for seed in bagging_seeds:
        params_use_seed = dict(params_use)
        params_use_seed["seed"] = int(seed)
        model = lgb.LGBMRegressor(**params_use_seed, n_estimators=num_round, random_state=int(seed))
        model.fit(X_tr, y_tr, sample_weight=train_weight)
        model_path = output_dir / ("model.txt" if len(bagging_seeds) == 1 else f"model_seed{seed}.txt")
        model.booster_.save_model(str(model_path))
        model_files.append(model_path.name)

    medians = prep["medians"]
    medians_dict = medians.to_dict() if isinstance(medians, pd.Series) else {}
    allocation_cfg_dict = loaded.allocation_cfg.__dict__ if loaded.allocation_cfg else None
    artifact = artifacts.HullModelArtifact(
        model_kind="lgb",
        feature_set=feature_set,
        target_col=target_col,
        model_files=model_files,
        feature_cols=list(prep["feature_cols"]),
        model_feature_cols=list(prep["model_feature_cols"]),
        medians={str(k): float(v) for k, v in medians_dict.items()},
        train_only_scored=train_only_scored,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        feature_cfg=cfg.feature_cfg,
        intentional_cfg=cfg.intentional_cfg,
        alloc_k=float(alloc_k) if alloc_k is not None else None,
        alloc_alpha=float(alloc_alpha),
        allocation_cfg=allocation_cfg_dict,
    )
    artifact_path = output_dir / args.artifact_name
    artifacts.save_artifact(artifact_path, artifact)

    summary = {
        "feature_set": feature_set,
        "target_col": target_col,
        "feature_cfg": cfg.feature_cfg,
        "intentional_cfg": cfg.intentional_cfg,
        "best_params": cfg.best_params,
        "allocation_cfg": loaded.allocation_cfg.__dict__ if loaded.allocation_cfg else None,
        "model_files": model_files,
        "artifact_path": str(artifact_path),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"OK: artefatos salvos em {output_dir} (models={len(model_files)})")


if __name__ == "__main__":
    main()
