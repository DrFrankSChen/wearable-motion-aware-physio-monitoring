"""Simple signal-quality helpers for public synthetic examples."""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    """Return root mean square amplitude."""
    return float(np.sqrt(np.mean(np.square(x))))


def motion_flags(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Flag windows whose motion score exceeds a threshold."""
    return np.asarray(scores) > threshold

