"""I/O helpers (paths, local data sync) for the Hull Tactical project."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


COMPETITION_SLUG = "hull-tactical-market-prediction"


@dataclass(frozen=True)
class DataPaths:
    data_dir: Path
    raw_dir: Path
    submissions_dir: Path
    train_path: Path
    test_path: Path
    raw_train_path: Path
    raw_test_path: Path
    zip_path: Path


def pick_data_dir(env_var: str = "HT_DATA_DIR") -> Path:
    env_val = os.environ.get(env_var)
    env_dir = Path(env_val) if env_val else None
    if env_dir:
        return env_dir
    candidates = [
        Path("/kaggle/input/hull-tactical-market-prediction"),
        (Path("..").resolve() / "data"),
        (Path.cwd() / "data"),
        (Path.cwd().parent / "data"),
        Path("/kaggle/working/data"),
        Path("/content/data"),
    ]
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    return Path("data")


def get_data_paths(data_dir: Path | None = None, competition_slug: str = COMPETITION_SLUG) -> DataPaths:
    base = data_dir or pick_data_dir()
    raw_dir = base / "raw"
    submissions_dir = base / "submissions"
    train_path = base / "train.csv"
    test_path = base / "test.csv"
    raw_train_path = raw_dir / "train.csv"
    raw_test_path = raw_dir / "test.csv"
    zip_path = raw_dir / f"{competition_slug}.zip"
    return DataPaths(
        data_dir=base,
        raw_dir=raw_dir,
        submissions_dir=submissions_dir,
        train_path=train_path,
        test_path=test_path,
        raw_train_path=raw_train_path,
        raw_test_path=raw_test_path,
        zip_path=zip_path,
    )


def download_competition_zip(
    paths: DataPaths,
    *,
    competition_slug: str = COMPETITION_SLUG,
    kaggle_bin: str = "kaggle",
) -> Path:
    """Downloads the competition zip to paths.raw_dir using Kaggle CLI."""
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle_bin, "competitions", "download", "-c", competition_slug, "-p", str(paths.raw_dir)]
    subprocess.run(cmd, check=True)
    if not paths.zip_path.exists():
        raise FileNotFoundError(f"Esperado {paths.zip_path} após download do Kaggle CLI.")
    return paths.zip_path


def ensure_local_data(
    paths: DataPaths,
    *,
    competition_slug: str = COMPETITION_SLUG,
    download_if_missing: bool = False,
) -> DataPaths:
    """Ensures train/test exist in paths.data_dir.

    Order:
    1) If running on Kaggle: copy from `/kaggle/input/<competition_slug>/`.
    2) If already present in data_dir: no-op.
    3) If present in raw_dir: copy to data_dir.
    4) If zip exists (or download_if_missing=True): extract and copy.
    """
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)

    kaggle_input_dir = Path(f"/kaggle/input/{competition_slug}")
    if kaggle_input_dir.exists():
        for src, dst in [
            (kaggle_input_dir / "train.csv", paths.train_path),
            (kaggle_input_dir / "test.csv", paths.test_path),
        ]:
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

    if paths.train_path.exists() and paths.test_path.exists():
        return paths

    if paths.raw_train_path.exists() and paths.raw_test_path.exists():
        for src, dst in [(paths.raw_train_path, paths.train_path), (paths.raw_test_path, paths.test_path)]:
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        if paths.train_path.exists() and paths.test_path.exists():
            return paths

    if not paths.zip_path.exists() and download_if_missing:
        download_competition_zip(paths, competition_slug=competition_slug)

    if paths.zip_path.exists():
        with zipfile.ZipFile(paths.zip_path, "r") as zf:
            zf.extractall(paths.raw_dir)
        for src, dst in [(paths.raw_train_path, paths.train_path), (paths.raw_test_path, paths.test_path)]:
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

    if not (paths.train_path.exists() and paths.test_path.exists()):
        raise FileNotFoundError(
            f"train.csv/test.csv não encontrados em {paths.data_dir.resolve()}. "
            "Coloque os arquivos manualmente (ex.: ajuste `HT_DATA_DIR`) "
            "ou baixe via Kaggle CLI (download_if_missing=True)."
        )
    return paths


__all__ = [
    "COMPETITION_SLUG",
    "DataPaths",
    "pick_data_dir",
    "get_data_paths",
    "download_competition_zip",
    "ensure_local_data",
]
