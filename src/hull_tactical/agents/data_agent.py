"""Agent stub for data preparation tasks."""
from __future__ import annotations

from pathlib import Path
from .. import data


def run(data_dir: Path | None = None):
    return data.load_raw_data(data_dir)
