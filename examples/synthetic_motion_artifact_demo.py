#!/usr/bin/env python3
"""Synthetic demonstration of motion-aware PPG window handling.

This script uses generated signals only. It is not the private research
implementation and should not be used for physiological inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch


FS = 25.0
DURATION_SEC = 180
WINDOW_SEC = 12
HOP_SEC = 4
TRUE_HR_BPM = 72.0
MOTION_HZ = 1.8
RNG = np.random.default_rng(7)


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(3, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def dominant_hr_bpm(x: np.ndarray, fs: float) -> float:
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


def build_synthetic_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0, DURATION_SEC, 1.0 / FS)
    heart_hz = TRUE_HR_BPM / 60.0

    clean_ppg = (
        np.sin(2 * np.pi * heart_hz * t)
        + 0.25 * np.sin(2 * np.pi * 2 * heart_hz * t)
        + 0.05 * RNG.normal(size=t.size)
    )

    motion_envelope = np.zeros_like(t)
    for start, end in [(42, 70), (112, 138)]:
        motion_envelope[(t >= start) & (t <= end)] = 1.0

    imu_motion = motion_envelope * (
        np.sin(2 * np.pi * MOTION_HZ * t)
        + 0.35 * np.sin(2 * np.pi * 2 * MOTION_HZ * t)
    )
    imu_motion += 0.03 * RNG.normal(size=t.size)

    contaminated_ppg = clean_ppg + 1.4 * motion_envelope * np.sin(2 * np.pi * MOTION_HZ * t)
    contaminated_ppg += 0.08 * RNG.normal(size=t.size)

    return t, clean_ppg, contaminated_ppg, imu_motion


def estimate_windows(
    t: np.ndarray,
    ppg: np.ndarray,
    imu: np.ndarray,
    motion_threshold: float,
) -> list[dict[str, float | bool]]:
    filtered_ppg = bandpass(ppg, 0.5, 5.0, FS)
    filtered_imu = bandpass(imu, 0.5, 5.0, FS)

    rows: list[dict[str, float | bool]] = []
    win = int(WINDOW_SEC * FS)
    hop = int(HOP_SEC * FS)

    for start in range(0, len(t) - win + 1, hop):
        end = start + win
        ppg_win = filtered_ppg[start:end]
        imu_win = filtered_imu[start:end]
        motion_score = float(np.sqrt(np.mean(imu_win**2)))
        motion_flag = motion_score > motion_threshold

        raw_hr = dominant_hr_bpm(ppg_win, FS)
        accepted_hr = float("nan") if motion_flag else raw_hr

        rows.append(
            {
                "time_sec": float(t[start + win // 2]),
                "raw_hr_bpm": raw_hr,
                "accepted_hr_bpm": accepted_hr,
                "motion_score": motion_score,
                "motion_flag": motion_flag,
            }
        )

    return rows


def summarize(rows: list[dict[str, float | bool]]) -> None:
    raw = np.array([float(r["raw_hr_bpm"]) for r in rows])
    accepted = np.array([float(r["accepted_hr_bpm"]) for r in rows])
    motion_flags = np.array([bool(r["motion_flag"]) for r in rows])

    raw_mae = np.nanmean(np.abs(raw - TRUE_HR_BPM))
    accepted_mae = np.nanmean(np.abs(accepted - TRUE_HR_BPM))

    print("Synthetic motion-aware PPG demo")
    print(f"True heart rate: {TRUE_HR_BPM:.1f} bpm")
    print(f"Windows analyzed: {len(rows)}")
    print(f"Motion-dominated windows flagged: {int(np.sum(motion_flags))}")
    print(f"Raw window MAE: {raw_mae:.2f} bpm")
    print(f"Accepted-window MAE after motion gating: {accepted_mae:.2f} bpm")
    print("Note: rejected windows are excluded instead of forced into an estimate.")


def save_plot(
    t: np.ndarray,
    clean_ppg: np.ndarray,
    contaminated_ppg: np.ndarray,
    imu: np.ndarray,
    rows: list[dict[str, float | bool]],
) -> Path:
    out_dir = Path("demo_outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "synthetic_motion_artifact_demo.png"

    row_t = np.array([float(r["time_sec"]) for r in rows])
    raw_hr = np.array([float(r["raw_hr_bpm"]) for r in rows])
    accepted_hr = np.array([float(r["accepted_hr_bpm"]) for r in rows])
    motion_score = np.array([float(r["motion_score"]) for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(t, clean_ppg, label="clean synthetic PPG", linewidth=1.0)
    axes[0].plot(t, contaminated_ppg, label="motion-contaminated PPG", linewidth=0.9, alpha=0.75)
    axes[0].set_ylabel("PPG")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, imu, color="tab:orange", linewidth=1.0)
    axes[1].set_ylabel("IMU motion")
    axes[1].grid(alpha=0.25)

    axes[2].plot(row_t, raw_hr, "o-", label="raw estimate", markersize=4)
    axes[2].plot(row_t, accepted_hr, "o-", label="accepted estimate", markersize=4)
    axes[2].axhline(TRUE_HR_BPM, color="black", linestyle="--", linewidth=1.0, label="true HR")
    axes[2].fill_between(
        row_t,
        TRUE_HR_BPM - 20 * (motion_score > np.percentile(motion_score, 65)),
        TRUE_HR_BPM + 20 * (motion_score > np.percentile(motion_score, 65)),
        color="tab:red",
        alpha=0.08,
        label="higher motion score",
    )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("HR estimate (bpm)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    t, clean_ppg, contaminated_ppg, imu = build_synthetic_signals()
    baseline_motion = bandpass(imu[: int(30 * FS)], 0.5, 5.0, FS)
    motion_threshold = float(np.sqrt(np.mean(baseline_motion**2)) + 0.15)

    rows = estimate_windows(t, contaminated_ppg, imu, motion_threshold)
    summarize(rows)
    out_path = save_plot(t, clean_ppg, contaminated_ppg, imu, rows)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

