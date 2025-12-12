"""Data loading and simple splits for Hull Tactical."""
from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd

from . import io

DEFAULT_DATA_DIR = pathlib.Path("data")


def load_raw_data(data_dir: pathlib.Path | str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train/test as DataFrames.

    - Local: reads from `data_dir` (or `HT_DATA_DIR`/`./data`) and supports `raw/`.
    - Kaggle: reads directly from `/kaggle/input/<competition>/train.csv|test.csv`.

    Never triggers an automatic download (Kaggle API keys should stay out of the repo).
    """
    base = pathlib.Path(data_dir) if data_dir else None
    paths = io.get_data_paths(base)
    io.ensure_local_data(paths, download_if_missing=False)
    return pd.read_csv(paths.train_path), pd.read_csv(paths.test_path)


def train_valid_split(df: pd.DataFrame, valid_frac: float = 0.2, seed: int | None = 42):
    if not 0 < valid_frac < 1:
        raise ValueError("valid_frac deve estar entre 0 e 1")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * valid_frac))
    val = shuffled.iloc[-n_val:]
    train = shuffled.iloc[:-n_val]
    return train, val


__all__ = ["load_raw_data", "train_valid_split", "DEFAULT_DATA_DIR"]
