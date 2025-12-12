import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.config as ht_config  # noqa: E402
import hull_tactical.data as ht_data  # noqa: E402
import hull_tactical.models as hm  # noqa: E402
from hull_tactical import competition, pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Treina e gera submission CSV (local).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/submissions/submission.csv"))
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
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
        bagging_seeds = None

    train_only_scored = bool(run_cfg.get("train_only_scored", False))
    weight_scored = run_cfg.get("weight_scored")
    weight_unscored = run_cfg.get("weight_unscored")
    alloc_k = run_cfg.get("alloc_k")
    alloc_alpha = float(run_cfg.get("alloc_alpha", 1.0))

    df_train, df_test = ht_data.load_raw_data(args.data_dir)
    df_train, df_test, cols = competition.prepare_train_test(df_train, df_test, normalized_target_col=target_col)

    cfg = hm.default_config(args.config_dir)
    cfg.market_col = cols.market_col
    cfg.rf_col = cols.rf_col
    cfg.is_scored_col = cols.is_scored_col
    hm.set_data_columns(cols.market_col, cols.rf_col, cols.is_scored_col)

    allocations = pipeline.train_pipeline(
        df_train=df_train,
        df_test=df_test,
        cfg=cfg,
        target_col=target_col,
        feature_set=feature_set,
        allocation_cfg=loaded.allocation_cfg,
        train_only_scored=train_only_scored,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        num_boost_round=int(num_boost_round) if num_boost_round is not None else None,
        alloc_k=float(alloc_k) if alloc_k is not None else None,
        alloc_alpha=alloc_alpha,
        bagging_seeds=bagging_seeds,
    )
    row_id_col = "row_id" if "row_id" in df_test.columns else df_test.columns[0]
    pipeline.make_submission_csv(args.out, allocations, df_test[row_id_col])
    print(f"Submissão gerada em {args.out}")


if __name__ == "__main__":
    main()
