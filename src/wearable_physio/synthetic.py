"""Synthetic signal generators for public demos.

These generators are deliberately simple and are not private research data.
"""

from __future__ import annotations

import numpy as np


def synthetic_ppg_with_motion(
    fs: float,
    duration_sec: float,
    heart_rate_bpm: float,
    motion_hz: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create clean PPG, motion-contaminated PPG, IMU motion, and true HR."""
    t = np.arange(0, duration_sec, 1.0 / fs)
    true_hr_bpm = heart_rate_bpm + 4.0 * np.sin(2 * np.pi * t / 155.0)
    true_hr_bpm += 1.5 * np.sin(2 * np.pi * t / 54.0 + 0.8)
    heart_hz = true_hr_bpm / 60.0
    heart_phase = 2 * np.pi * np.cumsum(heart_hz) / fs

    clean_ppg = (
        np.sin(heart_phase)
        + 0.25 * np.sin(2 * heart_phase)
        + 0.05 * rng.normal(size=t.size)
    )

    motion_envelope = np.zeros_like(t)
    for start, end in [(42, 70), (112, 138)]:
        motion_envelope[(t >= start) & (t <= end)] = 1.0

    imu_motion = motion_envelope * (
        np.sin(2 * np.pi * motion_hz * t)
        + 0.35 * np.sin(2 * np.pi * 2 * motion_hz * t)
    )
    imu_motion += 0.03 * rng.normal(size=t.size)

    contaminated_ppg = clean_ppg + 1.4 * motion_envelope * np.sin(2 * np.pi * motion_hz * t)
    contaminated_ppg += 0.08 * rng.normal(size=t.size)

    return t, clean_ppg, contaminated_ppg, imu_motion, true_hr_bpm


def synthetic_respiratory_imu(
    fs: float,
    duration_sec: float,
    breathing_rate_bpm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create respiratory motion, contaminated IMU, gross motion, and true BR."""
    t = np.arange(0, duration_sec, 1.0 / fs)
    true_br_bpm = breathing_rate_bpm + 1.2 * np.sin(2 * np.pi * t / 130.0 + 0.3)
    true_br_bpm += 0.5 * np.sin(2 * np.pi * t / 48.0)
    breath_hz = true_br_bpm / 60.0
    breath_phase = 2 * np.pi * np.cumsum(breath_hz) / fs

    respiratory_motion = (
        np.sin(breath_phase)
        + 0.20 * np.sin(2 * breath_phase + 0.4)
    )
    respiratory_motion += 0.04 * rng.normal(size=t.size)

    gross_motion = np.zeros_like(t)
    for start, end, freq in [(55, 76, 0.95), (126, 146, 1.15)]:
        idx = (t >= start) & (t <= end)
        gross_motion[idx] = 1.6 * np.sin(2 * np.pi * freq * t[idx])
        gross_motion[idx] += 0.6 * np.sin(2 * np.pi * 2 * freq * t[idx])

    imu_signal = respiratory_motion + gross_motion + 0.08 * rng.normal(size=t.size)
    return t, respiratory_motion, imu_signal, gross_motion, true_br_bpm
