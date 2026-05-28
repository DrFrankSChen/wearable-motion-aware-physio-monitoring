"""Plotting helpers for synthetic demo outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import heart_band, respiratory_band


def ensure_output_dir(path: str | Path = "demo_outputs") -> Path:
    """Create and return the demo output directory."""
    out_dir = Path(path)
    out_dir.mkdir(exist_ok=True)
    return out_dir


def save_heart_rate_demo_plot(
    t: np.ndarray,
    contaminated_ppg: np.ndarray,
    imu: np.ndarray,
    rows: list[dict[str, float | bool]],
    true_hr_bpm: np.ndarray,
    fs: float,
) -> Path:
    """Save the synthetic motion-artifact HR demo plot."""
    out_path = ensure_output_dir() / "synthetic_motion_artifact_demo.png"
    row_t = np.array([float(r["time_sec"]) for r in rows])
    raw_hr = np.array([float(r["raw_hr_bpm"]) for r in rows])
    accepted_hr = np.array([float(r["accepted_hr_bpm"]) for r in rows])
    true_window_hr = np.array([float(r["true_hr_bpm"]) for r in rows])
    motion_score = np.array([float(r["motion_score"]) for r in rows])
    motion_flags = np.array([bool(r["motion_flag"]) for r in rows])

    filtered_ppg = heart_band(contaminated_ppg, fs)
    handled_ppg = filtered_ppg.copy()
    for center, flagged in zip(row_t, motion_flags):
        if flagged:
            handled_ppg[np.abs(t - center) <= 4.0] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, contaminated_ppg, color="tab:blue", linewidth=0.9, label="synthetic PPG with motion artifacts")
    axes[0].set_ylabel("PPG")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, imu, color="tab:orange", linewidth=1.0, label="corresponding IMU motion")
    axes[1].set_ylabel("IMU")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, filtered_ppg, color="0.65", linewidth=0.8, label="bandpassed PPG before gating")
    axes[2].plot(t, handled_ppg, color="tab:green", linewidth=1.0, label="public demo: accepted signal after gating")
    axes[2].set_ylabel("processed PPG")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    axes[3].plot(t, true_hr_bpm, color="black", linestyle="--", linewidth=1.0, label="true synthetic HR")
    axes[3].plot(row_t, raw_hr, "o-", color="tab:orange", label="raw PSD estimate", markersize=4)
    axes[3].plot(row_t, accepted_hr, "o-", color="tab:green", label="accepted estimate", markersize=4)
    axes[3].plot(row_t, true_window_hr, ".", color="black", alpha=0.7, label="window mean HR")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("HR estimate (bpm)")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.25)

    for ax in axes:
        for center, score, flagged in zip(row_t, motion_score, motion_flags):
            if flagged:
                ax.axvspan(center - 4.0, center + 4.0, color="tab:red", alpha=0.06)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_breathing_rate_demo_plot(
    t: np.ndarray,
    imu_signal: np.ndarray,
    gross_motion: np.ndarray,
    rows: list[dict[str, float | bool]],
    model_estimates: np.ndarray,
    true_br_bpm: np.ndarray,
    fs: float,
) -> Path:
    """Save the synthetic breathing-rate demo plot."""
    out_path = ensure_output_dir() / "synthetic_breathing_imu_demo.png"
    row_t = np.array([float(r["time_sec"]) for r in rows])
    raw_br = np.array([float(r["raw_br_bpm"]) for r in rows])
    accepted_br = np.array([float(r["accepted_br_bpm"]) for r in rows])
    true_window_br = np.array([float(r["true_br_bpm"]) for r in rows])
    motion_flags = np.array([bool(r["motion_flag"]) for r in rows])

    filtered_resp = respiratory_band(imu_signal, fs)
    handled_resp = filtered_resp.copy()
    for center, flagged in zip(row_t, motion_flags):
        if flagged:
            handled_resp[np.abs(t - center) <= 10.0] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, imu_signal, color="tab:blue", label="synthetic IMU with respiratory motion and artifacts", linewidth=0.9)
    axes[0].set_ylabel("IMU")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, gross_motion, color="tab:red", linewidth=1.0)
    axes[1].set_ylabel("gross motion")
    axes[1].legend(["corresponding motion artifact"], loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, filtered_resp, color="0.65", linewidth=0.9, label="respiratory-band filtered IMU")
    axes[2].plot(t, handled_resp, color="tab:green", linewidth=1.0, label="public demo: accepted respiratory signal")
    axes[2].set_ylabel("processed IMU")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    axes[3].plot(t, true_br_bpm, color="black", linestyle="--", linewidth=1.0, label="true synthetic BR")
    axes[3].plot(row_t, raw_br, "o-", color="tab:orange", label="respiratory-band PSD", markersize=4)
    axes[3].plot(row_t, accepted_br, "o-", color="tab:green", label="accepted PSD estimate", markersize=4)
    axes[3].plot(row_t, model_estimates, "s-", color="tab:purple", label="toy sequence-model demo", markersize=3)
    axes[3].plot(row_t, true_window_br, ".", color="black", alpha=0.7, label="window mean BR")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("BR estimate (breaths/min)")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.25)

    for ax in axes:
        for center, flagged in zip(row_t, motion_flags):
            if flagged:
                ax.axvspan(center - 10.0, center + 10.0, color="tab:red", alpha=0.06)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
