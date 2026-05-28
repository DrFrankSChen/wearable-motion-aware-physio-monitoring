"""Heart-rate estimation helpers for the synthetic PPG demo."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .preprocessing import heart_band
from .signal_quality import rms


def dominant_hr_bpm(x: np.ndarray, fs: float) -> float:
    """Estimate heart rate from the dominant spectral peak."""
    freqs, power = welch(
        x - np.mean(x),
        fs=fs,
        nperseg=min(len(x), int(8 * fs)),
        noverlap=min(len(x) // 2, int(4 * fs)),
        nfft=2048,
    )
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return float("nan")
    peak_hz = freqs[mask][np.argmax(power[mask])]
    return float(60.0 * peak_hz)


def estimate_hr_windows(
    t: np.ndarray,
    ppg: np.ndarray,
    imu: np.ndarray,
    fs: float,
    window_sec: float,
    hop_sec: float,
    motion_threshold: float,
    true_hr_bpm: np.ndarray | None = None,
) -> list[dict[str, float | bool]]:
    """Estimate HR in windows and suppress motion-dominated windows."""
    filtered_ppg = heart_band(ppg, fs)
    filtered_imu = heart_band(imu, fs)

    rows: list[dict[str, float | bool]] = []
    win = int(window_sec * fs)
    hop = int(hop_sec * fs)

    for start in range(0, len(t) - win + 1, hop):
        end = start + win
        ppg_win = filtered_ppg[start:end]
        imu_win = filtered_imu[start:end]
        motion_score = rms(imu_win)
        motion_flag = motion_score > motion_threshold
        raw_hr = dominant_hr_bpm(ppg_win, fs)
        true_window_hr = float("nan")
        if true_hr_bpm is not None:
            true_window_hr = float(np.mean(true_hr_bpm[start:end]))

        rows.append(
            {
                "time_sec": float(t[start + win // 2]),
                "raw_hr_bpm": raw_hr,
                "accepted_hr_bpm": float("nan") if motion_flag else raw_hr,
                "true_hr_bpm": true_window_hr,
                "motion_score": motion_score,
                "motion_flag": motion_flag,
            }
        )

    return rows
