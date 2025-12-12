import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hull_tactical.config as ht_config  # noqa: E402
import hull_tactical.data as ht_data  # noqa: E402
import hull_tactical.models as hm  # noqa: E402
from hull_tactical import competition, pipeline  # noqa: E402


def _append_experiment_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run time-aware CV (fit_ref) using YAML configs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--feature-set", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--num-boost-round", type=int, default=None)
    parser.add_argument("--weight-scored", type=float, default=None)
    parser.add_argument("--weight-unscored", type=float, default=None)
    parser.add_argument("--train-only-scored", action="store_true")
    parser.add_argument("--log-csv", type=Path, default=Path("reports/experiments.csv"))
    args = parser.parse_args()

    loaded = ht_config.load_all_configs(args.config_dir)
    run_cfg = loaded.run_cfg if isinstance(loaded.run_cfg, dict) else {}
    cv_cfg = run_cfg.get("cv") if isinstance(run_cfg.get("cv"), dict) else {}
    model_cfg = run_cfg.get("model") if isinstance(run_cfg.get("model"), dict) else {}

    feature_set = args.feature_set or run_cfg.get("feature_set") or "D_intentional"
    target_col = args.target_col or run_cfg.get("target_col") or "target"
    n_splits = int(args.n_splits or cv_cfg.get("n_splits") or 4)
    val_frac = float(args.val_frac or cv_cfg.get("val_frac") or 0.12)
    num_boost_round = int(args.num_boost_round or model_cfg.get("num_boost_round") or cv_cfg.get("num_boost_round") or 200)

    weight_scored_cfg = run_cfg.get("weight_scored", cv_cfg.get("weight_scored"))
    weight_unscored_cfg = run_cfg.get("weight_unscored", cv_cfg.get("weight_unscored"))
    train_only_scored_cfg = run_cfg.get("train_only_scored", cv_cfg.get("train_only_scored", False))

    weight_scored = args.weight_scored if args.weight_scored is not None else weight_scored_cfg
    weight_unscored = args.weight_unscored if args.weight_unscored is not None else weight_unscored_cfg
    train_only_scored = bool(args.train_only_scored or train_only_scored_cfg or False)

    df_train, _ = ht_data.load_raw_data(args.data_dir)
    df_train, _, cols = competition.prepare_train_test(df_train, None, normalized_target_col=target_col)

    cfg = hm.default_config(args.config_dir)
    cfg.market_col = cols.market_col
    cfg.rf_col = cols.rf_col
    cfg.is_scored_col = cols.is_scored_col
    hm.set_data_columns(cols.market_col, cols.rf_col, cols.is_scored_col)

    out = pipeline.run_time_cv_fitref_oof(
        df_train,
        feature_set=feature_set,
        target_col=target_col,
        cfg=cfg,
        n_splits=n_splits,
        val_frac=val_frac,
        num_boost_round=num_boost_round,
        weight_scored=weight_scored,
        weight_unscored=weight_unscored,
        train_only_scored=train_only_scored,
        allocation_cfg=loaded.allocation_cfg,
        return_oof_df=False,
    )

    summary = out.get("summary") or {}
    payload = {
        "feature_set": feature_set,
        "target_col": target_col,
        "n_splits": n_splits,
        "val_frac": val_frac,
        "num_boost_round": num_boost_round,
        "weight_scored": weight_scored,
        "weight_unscored": weight_unscored,
        "train_only_scored": train_only_scored,
        "oof_sharpe": out.get("oof_sharpe"),
        "best_k": out.get("best_k"),
        "best_alpha": out.get("best_alpha"),
        **{f"cv_{k}": v for k, v in summary.items()},
    }
    print(json.dumps(payload, indent=2, default=str))

    _append_experiment_row(
        args.log_csv,
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "feature_set": feature_set,
            "target_col": target_col,
            "n_splits": n_splits,
            "val_frac": val_frac,
            "num_boost_round": num_boost_round,
            "train_only_scored": int(train_only_scored),
            "weight_scored": weight_scored if weight_scored is not None else "",
            "weight_unscored": weight_unscored if weight_unscored is not None else "",
            "oof_sharpe": out.get("oof_sharpe"),
            "best_k": out.get("best_k"),
            "best_alpha": out.get("best_alpha"),
            "sharpe_mean": summary.get("sharpe_mean", ""),
            "sharpe_std": summary.get("sharpe_std", ""),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
