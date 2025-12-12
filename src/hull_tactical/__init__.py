"""Hull Tactical package (dependency-light utilities for Kaggle notebooks).

Note: avoid importing heavy modules at package import-time; prefer explicit
imports like `from hull_tactical import pipeline` or `import hull_tactical.models`.
"""

__all__ = [
    "artifacts",
    "allocation",
    "competition",
    "config",
    "data",
    "ensemble",
    "features",
    "io",
    "metric",
    "models",
    "pipeline",
]
