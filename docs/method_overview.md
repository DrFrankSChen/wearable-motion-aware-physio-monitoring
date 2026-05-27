# Method Overview

This project investigates motion-aware physiological monitoring from wearable and head-worn sensors. The core idea is to treat motion as a first-class signal rather than as background noise.

## Signal Sources

The research prototypes use combinations of:

- PPG channels from wearable or head-worn devices.
- IMU accelerometer streams for motion context.
- Reference heart-rate or breathing-rate signals for offline evaluation.

## General Pipeline

The public version summarizes the method at a high level:

1. **Input alignment:** Bring PPG and IMU streams onto compatible time windows.
2. **Signal-quality screening:** Identify windows with dropouts, near-zero PPG, low amplitude, or unstable spectra.
3. **Motion-aware processing:** Use IMU information to flag high-motion windows, suppress motion-dominated spectral content, or avoid unreliable estimates.
4. **Physiology estimation:** Estimate heart rate or breathing rate from window-level signal features or learned models.
5. **Reference comparison:** Compare estimates with reference devices using error, correlation, bias, and threshold-based summaries.

## Breathing-Rate Estimation

The private research code explores three prototype paths:

- **Head PPG plus accelerometer feature model:** Extracts pulse-interval variability, respiration-band PPG features, spectral features, peak-valley features, and motion-quality signals before regression.
- **Wrist PPG plus accelerometer deep model:** Uses multi-channel PPG and accelerometer windows as model input for breathing-rate regression.
- **Head accelerometer-only model:** Estimates breathing-related motion from head-worn IMU signals and prepares a compact deployment path.

These approaches are intended for research comparison across sensing locations and signal types.

## Heart-Rate Estimation

The private heart-rate prototype estimates heart rate from head-worn PPG while accounting for motion artifacts:

- Reconstructs a usable IMU time base for motion context.
- Applies bandpass filtering and windowed spectral estimation to PPG.
- Uses IMU spectra to identify motion-dominated frequency regions.
- Skips or holds estimates in windows with excessive motion or poor signal quality.
- Applies temporal tracking to reduce implausible jumps.
- Compares derived estimates with external reference devices.

## Why Motion Awareness Matters

Motion can create strong frequency components that overlap physiological bands. Without motion handling, a model may confuse movement with heart rhythm or respiration. The research goal is to make the estimator aware of when a signal is trustworthy, when it should be corrected, and when it should be excluded from evaluation.

