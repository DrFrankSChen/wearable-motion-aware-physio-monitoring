# Preliminary Results Summary

The results below are intentionally summarized. Full experiment logs, exact implementation details, raw data, and trained models are not included because this is active research.

## Breathing Rate

Preliminary breathing-rate prototypes reached roughly **1 to 3 breaths per minute mean absolute error** depending on dataset, sensor placement, signal type, and evaluation setup.

Observed trends:

- Head-worn PPG plus accelerometer features showed promising performance on selected datasets.
- Wrist PPG plus accelerometer deep learning required more careful validation because performance depended strongly on subject, dataset, and motion condition.
- Head accelerometer-only signals were useful for studying respiration-related motion and deployment feasibility.
- A trained breathing-rate model was converted into an Apple Core ML format and embedded in an iOS prototype for real-time testing with wearable streams.

## Heart Rate

Preliminary heart-rate evaluation compared head-worn PPG-derived estimates with external reference devices.

Observed trends:

- Raw PPG-derived heart-rate estimates were sensitive to motion and timing alignment.
- IMU-assisted motion handling improved robustness in motion-contaminated windows.
- Lag correction improved agreement in evaluated sessions.
- In real-time experiments with lying-still and large-movement conditions, the derived heart-rate trace remained stable and aligned with the finger-worn reference recording.
- Post-experiment comparison against an O2Ring-style reference reached **Pearson correlation up to 0.809** and **mean absolute error as low as 0.99 beats per minute** in evaluated sessions.

## Limitations

These findings are preliminary and dataset-dependent.

This repository does not claim clinical validation, regulatory readiness, or production reliability. The work is best understood as a research prototype for wearable physiological monitoring.
