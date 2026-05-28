"""Metrics used by the synthetic examples."""

from __future__ import annotations

import numpy as np


def mae(values: np.ndarray, target: float | np.ndarray) -> float:
    """Mean absolute error, ignoring NaN values."""
    return float(np.nanmean(np.abs(np.asarray(values, dtype=float) - target)))


def coverage(values: np.ndarray) -> float:
    """Fraction of non-NaN estimates."""
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.isfinite(values)))

