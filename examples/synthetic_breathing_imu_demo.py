#!/usr/bin/env python3
"""Synthetic breathing-rate demo using IMU-like respiratory motion.

This example is intentionally public-safe. The spectral estimator and toy
sequence-model path are illustrative only and do not reproduce the private
breathing-rate model or its architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearable_physio.breathing_rate import estimate_br_windows, toy_sequence_model_br_bpm  # noqa: E402
from wearable_physio.metrics import coverage, mae  # noqa: E402
from wearable_physio.signal_quality import rms  # noqa: E402
from wearable_physio.synthetic import synthetic_respiratory_imu  # noqa: E402
from wearable_physio.visualization import save_breathing_rate_demo_plot  # noqa: E402


FS = 25.0
DURATION_SEC = 180
WINDOW_SEC = 32
HOP_SEC = 8
TRUE_BR_BPM = 14.0
RNG = np.random.default_rng(11)


def toy_sequence_model_windows(t: np.ndarray, imu: np.ndarray) -> np.ndarray:
    """Run the toy model path on the same windows as the PSD estimator."""
    win = int(WINDOW_SEC * FS)
    hop = int(HOP_SEC * FS)
    estimates = []
    for start in range(0, len(t) - win + 1, hop):
        end = start + win
        estimates.append(toy_sequence_model_br_bpm(imu[start:end], FS))
    return np.array(estimates, dtype=float)


def summarize(rows: list[dict[str, float | bool]], model_estimates: np.ndarray) -> None:
    raw = np.array([float(r["raw_br_bpm"]) for r in rows])
    accepted = np.array([float(r["accepted_br_bpm"]) for r in rows])
    true_br = np.array([float(r["true_br_bpm"]) for r in rows])
    motion_flags = np.array([bool(r["motion_flag"]) for r in rows])

    print("Synthetic respiratory-IMU breathing-rate demo")
    print(f"Synthetic breathing-rate range: {np.nanmin(true_br):.1f}-{np.nanmax(true_br):.1f} breaths/min")
    print(f"Windows analyzed: {len(rows)}")
    print(f"Motion-dominated windows flagged: {int(np.sum(motion_flags))}")
    print(f"Respiratory-band PSD MAE: {mae(raw, true_br):.2f} breaths/min")
    print(f"Accepted-window PSD MAE: {mae(accepted, true_br):.2f} breaths/min")
    print(f"Toy sequence-model demo MAE: {mae(model_estimates, true_br):.2f} breaths/min")
    print(f"Accepted-window coverage: {100.0 * coverage(accepted):.1f}%")
    print("Note: the toy sequence model is a fixed-filter demo, not the private trained model.")


def main() -> None:
    t, _clean_respiration, imu_signal, gross_motion, true_br_bpm = synthetic_respiratory_imu(
        fs=FS,
        duration_sec=DURATION_SEC,
        breathing_rate_bpm=TRUE_BR_BPM,
        rng=RNG,
    )
    baseline_motion = rms(gross_motion[: int(30 * FS)])
    motion_threshold = baseline_motion + 0.45

    rows = estimate_br_windows(
        t=t,
        imu=imu_signal,
        fs=FS,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        motion_reference=gross_motion,
        motion_threshold=motion_threshold,
        true_br_bpm=true_br_bpm,
    )
    model_estimates = toy_sequence_model_windows(t, imu_signal)
    summarize(rows, model_estimates)

    out_path = save_breathing_rate_demo_plot(
        t=t,
        imu_signal=imu_signal,
        gross_motion=gross_motion,
        rows=rows,
        model_estimates=model_estimates,
        true_br_bpm=true_br_bpm,
        fs=FS,
    )
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
