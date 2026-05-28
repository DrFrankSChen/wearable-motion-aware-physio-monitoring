#!/usr/bin/env python3
"""Synthetic demonstration of motion-aware PPG window handling.

This script uses generated signals only. It is not the private research
implementation and should not be used for physiological inference.

The motion handling here is a public-safe simplification of the research
direction: IMU-dominated windows are gated out rather than processed by the
full private heart-rate pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearable_physio.heart_rate import estimate_hr_windows  # noqa: E402
from wearable_physio.metrics import mae  # noqa: E402
from wearable_physio.preprocessing import heart_band  # noqa: E402
from wearable_physio.synthetic import synthetic_ppg_with_motion  # noqa: E402
from wearable_physio.visualization import save_heart_rate_demo_plot  # noqa: E402


FS = 25.0
DURATION_SEC = 180
WINDOW_SEC = 12
HOP_SEC = 4
TRUE_HR_BPM = 72.0
MOTION_HZ = 1.8
RNG = np.random.default_rng(7)


def summarize(rows: list[dict[str, float | bool]]) -> None:
    raw = np.array([float(r["raw_hr_bpm"]) for r in rows])
    accepted = np.array([float(r["accepted_hr_bpm"]) for r in rows])
    true_hr = np.array([float(r["true_hr_bpm"]) for r in rows])
    motion_flags = np.array([bool(r["motion_flag"]) for r in rows])

    print("Synthetic motion-aware PPG heart-rate demo")
    print(f"Synthetic heart-rate range: {np.nanmin(true_hr):.1f}-{np.nanmax(true_hr):.1f} bpm")
    print(f"Windows analyzed: {len(rows)}")
    print(f"Motion-dominated windows flagged: {int(np.sum(motion_flags))}")
    print(f"Raw window MAE: {mae(raw, true_hr):.2f} bpm")
    print(f"Accepted-window MAE after motion gating: {mae(accepted, true_hr):.2f} bpm")
    print("Note: rejected windows are excluded instead of forced into an estimate.")


def main() -> None:
    t, _clean_ppg, contaminated_ppg, imu, true_hr_bpm = synthetic_ppg_with_motion(
        fs=FS,
        duration_sec=DURATION_SEC,
        heart_rate_bpm=TRUE_HR_BPM,
        motion_hz=MOTION_HZ,
        rng=RNG,
    )
    baseline_motion = heart_band(imu[: int(30 * FS)], FS)
    motion_threshold = float(np.sqrt(np.mean(baseline_motion**2)) + 0.15)

    rows = estimate_hr_windows(
        t=t,
        ppg=contaminated_ppg,
        imu=imu,
        fs=FS,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        motion_threshold=motion_threshold,
        true_hr_bpm=true_hr_bpm,
    )
    summarize(rows)
    out_path = save_heart_rate_demo_plot(t, contaminated_ppg, imu, rows, true_hr_bpm, FS)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
