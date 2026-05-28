"""Basic signal preprocessing helpers for synthetic demos."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 3) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def heart_band(x: np.ndarray, fs: float) -> np.ndarray:
    """Filter a PPG-like signal to a broad heart-rate band."""
    return bandpass(x, 0.5, 5.0, fs)


def respiratory_band(x: np.ndarray, fs: float) -> np.ndarray:
    """Filter an IMU-like signal to a broad breathing-rate band."""
    return bandpass(x, 0.08, 0.7, fs)

