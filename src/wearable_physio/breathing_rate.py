"""Breathing-rate estimation helpers for synthetic IMU demos."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .preprocessing import respiratory_band
from .signal_quality import rms


def dominant_breathing_rate_bpm(x: np.ndarray, fs: float) -> float:
    """Estimate breathing rate from a respiratory-band spectral peak."""
    freqs, power = welch(
        x - np.mean(x),
        fs=fs,
        nperseg=min(len(x), int(24 * fs)),
        noverlap=min(len(x) // 2, int(12 * fs)),
        nfft=4096,
    )
    mask = (freqs >= 0.08) & (freqs <= 0.7)
    if not np.any(mask):
        return float("nan")
    peak_hz = freqs[mask][np.argmax(power[mask])]
    return float(60.0 * peak_hz)


def estimate_br_windows(
    t: np.ndarray,
    imu: np.ndarray,
    fs: float,
    window_sec: float,
    hop_sec: float,
    motion_reference: np.ndarray | None = None,
    motion_threshold: float | None = None,
    true_br_bpm: np.ndarray | None = None,
) -> list[dict[str, float | bool]]:
    """Estimate breathing rate from filtered IMU windows."""
    filtered_imu = respiratory_band(imu, fs)
    motion_source = imu if motion_reference is None else motion_reference

    rows: list[dict[str, float | bool]] = []
    win = int(window_sec * fs)
    hop = int(hop_sec * fs)

    for start in range(0, len(t) - win + 1, hop):
        end = start + win
        filtered_win = filtered_imu[start:end]
        motion_score = rms(motion_source[start:end])
        motion_flag = False if motion_threshold is None else motion_score > motion_threshold
        raw_br = dominant_breathing_rate_bpm(filtered_win, fs)
        true_window_br = float("nan")
        if true_br_bpm is not None:
            true_window_br = float(np.mean(true_br_bpm[start:end]))

        rows.append(
            {
                "time_sec": float(t[start + win // 2]),
                "raw_br_bpm": raw_br,
                "accepted_br_bpm": float("nan") if motion_flag else raw_br,
                "true_br_bpm": true_window_br,
                "motion_score": motion_score,
                "motion_flag": motion_flag,
            }
        )

    return rows


def toy_sequence_model_br_bpm(x: np.ndarray, fs: float) -> float:
    """Toy sequence-model surrogate for the public demo.

    This is not a trained CNN-LSTM and does not reproduce the private model.
    It uses fixed convolution-like smoothing filters followed by a spectral peak
    estimate to mimic the shape of a sequence-model demo without model weights.
    """
    filtered = respiratory_band(x, fs)
    smooth_kernel = np.ones(max(3, int(0.8 * fs))) / max(3, int(0.8 * fs))
    local_average = np.convolve(filtered, smooth_kernel, mode="same")
    residual = filtered - 0.35 * local_average
    return dominant_breathing_rate_bpm(residual, fs)
