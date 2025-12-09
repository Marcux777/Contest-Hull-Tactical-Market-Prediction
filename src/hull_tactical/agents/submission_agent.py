"""Agent stub to generate submission allocations."""
from __future__ import annotations

from pathlib import Path

from .. import pipeline


def run(output_path: Path, allocations, row_ids):
    return pipeline.make_submission_csv(output_path, allocations, row_ids)
