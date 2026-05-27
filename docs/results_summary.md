# Preliminary Results Summary

The results below are intentionally summarized. Full experiment logs, exact implementation details, raw data, and trained models are not included because this is active research.

## Breathing Rate

Preliminary breathing-rate prototypes reached roughly **1 to 3 breaths per minute mean absolute error** depending on dataset, sensor placement, signal type, and evaluation setup.

Observed trends:

- Head-worn PPG plus accelerometer features showed promising performance on selected datasets.
- Wrist PPG plus accelerometer deep learning required more careful validation because performance depended strongly on subject, dataset, and motion condition.
- Head accelerometer-only signals were useful for studying respiration-related motion and deployment feasibility.

## Heart Rate

Preliminary heart-rate evaluation compared head-worn PPG-derived estimates with external reference devices.

Observed trends:

- Raw PPG-derived heart-rate estimates were sensitive to motion and timing alignment.
- IMU-assisted motion handling improved robustness in motion-contaminated windows.
- Lag correction improved agreement in evaluated sessions.
- FRENZ-derived heart rate compared against an O2Ring-style reference showed roughly **1.5 to 2.2 beats per minute mean absolute error after lag correction** across evaluated sessions.

## Limitations

These findings are preliminary and dataset-dependent.

This repository does not claim clinical validation, regulatory readiness, or production reliability. The work is best understood as a research prototype and engineering portfolio artifact for wearable physiological monitoring.

