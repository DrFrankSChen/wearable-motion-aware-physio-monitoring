# Wearable Motion-Aware Physio Monitoring

Research prototypes for motion-aware heart-rate and breathing-rate estimation from wearable and head-worn PPG and IMU signals.

This repository is a public research summary of active work in wearable health sensing. It highlights the problem framing, modeling approaches, validation strategy, deployment direction, and a synthetic demonstration of motion artifact handling without publishing private data, ongoing research models, or the full unpublished implementation.

## Motivation

Wearable physiological signals are noisy in real settings. Photoplethysmography (PPG) can carry heart-rate and respiration-related information, but motion artifacts often create peaks, amplitude shifts, dropouts, and misleading frequency content. This project explores ways to estimate physiology from wearable signals while explicitly accounting for motion.

The work focuses on two research tracks:

- **Breathing-rate estimation:** Prototypes using head-worn PPG plus accelerometer features, wrist PPG plus accelerometer deep learning, and head accelerometer-only signals.
- **Heart-rate estimation:** PPG-derived heart-rate estimation with IMU-assisted motion suppression, signal-quality checks, temporal tracking, and comparison against external reference devices.

## High-Level Approach

The private research implementation explores a pipeline with these stages:

1. Synchronize wearable PPG and IMU streams.
2. Detect low-quality or motion-dominated windows.
3. Apply signal filtering and motion-aware gating.
4. Estimate heart rate or breathing rate from window-level features or learned models.
5. Compare estimates with reference devices and summarize error, correlation, and bias.

See [docs/method_overview.md](docs/method_overview.md) for a fuller overview.

The public disclosure boundary is described in [docs/repo_boundary.md](docs/repo_boundary.md).

## Synthetic Demo

The example script uses only generated signals. It creates a simulated PPG waveform, injects motion artifacts, generates an IMU-like motion trace, and shows how motion-aware window handling can reduce misleading estimates.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/synthetic_motion_artifact_demo.py
```

The demo saves a plot under `demo_outputs/`, which is ignored by git.

## Preliminary Results

The research prototypes have shown promising preliminary performance in controlled evaluations, but the work is not clinically validated and should not be used for medical decision-making. See [docs/results_summary.md](docs/results_summary.md).

## Deployment Prototype

The private implementation also includes iOS deployment prototypes for real-time estimation from wearable streams. See [docs/deployment.md](docs/deployment.md).

## Status

This is an active research repository. The public materials are designed to describe the research direction and engineering scope while preserving unpublished methods and private data.
