# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02_submission — Hull Tactical – Market Prediction
# Notebook enxuto de **submissão** (treino full + inferência).
#
# - Configs em `configs/` (`features.yaml`, `lgb.yaml`, `run.yaml`)
# - Lógica em `src/hull_tactical/` (features/model/pipeline)
# - Para pesquisa/EDA/CV/tuning, use `01_research`

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_src_dir(package_name: str = "hull_tactical") -> Path | None:
    def _check_root(root: Path) -> Path | None:
        src_init = root / "src" / package_name / "__init__.py"
        if src_init.exists():
            return src_init.parent.parent
        pkg_init = root / package_name / "__init__.py"
        if pkg_init.exists():
            return pkg_init.parent
        return None

    cwd = Path.cwd()
    for base in [cwd] + list(cwd.parents):
        found = _check_root(base)
        if found is not None:
            return found

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        try:
            for ds_root in sorted([p for p in kaggle_input.iterdir() if p.is_dir()]):
                for base in [ds_root] + [p for p in ds_root.iterdir() if p.is_dir()]:
                    found = _check_root(base)
                    if found is not None:
                        return found
        except Exception:
            pass
    return None


_src_dir = _find_src_dir("hull_tactical")
if _src_dir is not None and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Se existir `data/` ao lado de `src/`, usa como default local.
if _src_dir is not None:
    try:
        root_guess = _src_dir.resolve().parent
        data_guess = root_guess / "data"
        if data_guess.exists():
            os.environ.setdefault("HT_DATA_DIR", str(data_guess))
    except Exception:
        pass

# %%
import pandas as pd  # noqa: E402

import hull_tactical.config as ht_config  # noqa: E402
import hull_tactical.data as ht_data  # noqa: E402
import hull_tactical.models as hm  # noqa: E402
from hull_tactical import competition, io, pipeline  # noqa: E402

# %%
# 1) Configs (edite YAML em `configs/`)
loaded = ht_config.load_all_configs()
run_cfg = loaded.run_cfg if isinstance(loaded.run_cfg, dict) else {}
cv_cfg = run_cfg.get("cv") if isinstance(run_cfg.get("cv"), dict) else {}
model_cfg = run_cfg.get("model") if isinstance(run_cfg.get("model"), dict) else {}

feature_set = run_cfg.get("feature_set") or "D_intentional"
target_col = run_cfg.get("target_col") or "target"
train_only_scored = bool(run_cfg.get("train_only_scored", False))
weight_scored = run_cfg.get("weight_scored")
weight_unscored = run_cfg.get("weight_unscored")
num_boost_round = model_cfg.get("num_boost_round")
bagging_seeds = model_cfg.get("bagging_seeds")
if isinstance(bagging_seeds, list) and bagging_seeds:
    bagging_seeds = [int(x) for x in bagging_seeds]
else:
    bagging_seeds = None

alloc_k = run_cfg.get("alloc_k")
alloc_alpha = float(run_cfg.get("alloc_alpha", 1.0))

print("configs_dir:", loaded.config_dir)
print("feature_set:", feature_set)
print("target_col:", target_col)
print("train_policy:", {"train_only_scored": train_only_scored, "weight_scored": weight_scored, "weight_unscored": weight_unscored})
print("model:", {"num_boost_round": num_boost_round, "bagging_seeds": bagging_seeds})
print("alloc:", {"alloc_k": alloc_k, "alloc_alpha": alloc_alpha, "allocation_cfg": bool(loaded.allocation_cfg)})
print("cv:", cv_cfg)

# %%
# 2) Dados (não baixa via Kaggle API/CLI)
paths = io.get_data_paths()
io.ensure_local_data(paths, download_if_missing=False)
df_train, df_test = ht_data.load_raw_data(paths.data_dir)

print(df_train.shape, df_test.shape)

# %%
# 3) Normaliza colunas e prepara config do pipeline
df_train, df_test, cols = competition.prepare_train_test(df_train, df_test, normalized_target_col=target_col)

cfg = hm.default_config()
cfg.market_col = cols.market_col
cfg.rf_col = cols.rf_col
cfg.is_scored_col = cols.is_scored_col
hm.set_data_columns(cols.market_col, cols.rf_col, cols.is_scored_col)

# %%
# 4) Treina full e prediz allocation no test
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

allocations.describe()

# %%
# 5) Export local (CSV). No Kaggle rerun, adapte para o gateway oficial se necessário.
row_id_col = "row_id" if "row_id" in df_test.columns else df_test.columns[0]
sub_path = paths.submissions_dir / "submission.csv"
pipeline.make_submission_csv(sub_path, allocations, df_test[row_id_col])
sub_path
