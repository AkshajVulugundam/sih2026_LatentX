# LatentX

**AI-Driven Component Burn-In & Reliability Screening**

> LatentX detects hidden component anomalies during burn-in and predicts future degradation before conventional threshold-based testing can identify a failure.

## Overview

Burn-in testing applies controlled stress to electronic components and monitors electrical parameters over time. Conventional screening can rely on fixed absolute limits, which may miss a component that is still inside the limit but is degrading abnormally compared with its batch peers.

**LatentX** is a modular classical-ML screening prototype designed to address this gap.

It combines:

1. **Batch-relative anomaly detection** — identifies unusual readings and drift relative to the component's batch.
2. **Risk classification** — classifies a component as `NORMAL`, `MONITOR`, or `HIGH_RISK`.
3. **168h drift prediction** — uses earlier burn-in readings to estimate future parameter values.
4. **Risk assessment** — compares predicted behaviour with screening limits.
5. **Explainable output** — reports the parameter/trajectory that contributed most to the flag.
6. **Local dashboard** — provides an interactive screening interface without requiring a server.

## Architecture

```text
Burn-In Measurements
      |
      v
Data Preprocessing
      |
      v
Feature Engineering
      |
      +-------------------+
      |                   |
      v                   v
Batch-Relative       Time-Series /
Anomaly Detection    Drift Features
      |                   |
      v                   v
Risk Classification  168h Prediction
      |                   |
      +---------+---------+
                |
                v
         Risk Assessment
                |
                v
       Explainable Result
                |
                v
          LatentX Dashboard
```

## Screening Workflow

The prototype works with burn-in measurements at:

- **0h**
- **24h**
- **96h**

and predicts behaviour at:

- **168h**

The monitored parameters in the prototype are:

| Parameter | Unit | Prototype limit |
|---|---:|---:|
| Iddq | mA | 5.0 |
| Leakage | uA | 200.0 |
| Propagation delay | ns | 15.0 |

The limits above are **prototype assumptions used for the synthetic SIH-style dataset**, not claimed ISRO specifications.

## ML Pipeline

### Stage 1 — Batch-relative anomaly detection

LatentX computes robust batch-relative statistics using the median and MAD-derived scale.

For each parameter it evaluates:

- absolute level at each available time point
- change between burn-in time points
- robust z-scores
- maximum absolute robust z-score
- dominant anomaly driver

A robust-z threshold of `|z| > 3.5` is used for the prototype anomaly flag.

This is intended to detect cases such as:

```text
Component:
10.2 -> 15.1 -> 28.7 -> 44.8

Absolute limit:
50
```

Even though the component has not crossed the absolute limit, its trajectory can be abnormal relative to its batch.

### Stage 2 — Risk classification

A classical decision-tree classifier produces:

- `NORMAL`
- `MONITOR`
- `HIGH_RISK`

The deployed feature set differs by screening point based on validation/ablation results:

- **24h:** deployment without chained `anomaly_*` inputs
- **96h:** deployment with chained anomaly outputs

The model is exported into `reliefos_live_models.js` so the dashboard can perform local inference.

### Stage 3 — 168h drift prediction

Classical regression models are used to predict the 168h value of each monitored parameter.

The selected prototype models include:

- Random Forest
- Gradient Boosting

depending on the parameter and available cutoff.

The dashboard reports the predicted value, percentage of the prototype limit, and status.

## Prototype Validation

Validation was performed on the SIH-style synthetic dataset created for development.

| Model / task | Result |
|---|---|
| 24h risk classifier | Accuracy 86.8%, macro-F1 0.716 |
| 96h risk classifier | Accuracy 90.8%, macro-F1 0.805 |
| 96h Iddq predictor | MAE 0.072, R2 0.945 |
| 96h Leakage predictor | MAE 3.301, R2 0.886 |

These are **prototype validation results**, not results on confidential ISRO data.

## Data

The public SIH26170 dataset was not available to the prototype during development.

Therefore, LatentX uses a **synthetic SIH-style burn-in dataset** for model development and demonstration.

The synthetic data was designed to represent:

- normal batch variation
- batch-relative anomalies
- gradual drift
- sudden anomalies
- latent defects

The dataset is deliberately kept separate from the NASA degradation data used during research.

### NASA research reference

NASA's public capacitor degradation data was used as a **research/prototype reference for degradation-pattern thinking**. It is **not ISRO data** and is not presented as the SIH dataset.

## Repository Structure

```text
LatentX/
|
+-- LatentX_Dashboard.html
+-- reliefos_live_models.js
|
+-- anomaly_detector.py
+-- classifier.py
+-- drift_predictor.py
+-- stage3_drift_predictor.py
+-- risk_engine.py
|
+-- generate_sih_synthetic_burnin_v2.py
+-- build_console_data.py
+-- export_reliefos_live_models.py
```

### Main files

| File | Purpose |
|---|---|
| `LatentX_Dashboard.html` | Interactive local screening dashboard |
| `reliefos_live_models.js` | Exported trained model bundle used by the dashboard |
| `anomaly_detector.py` | Batch-relative anomaly detection |
| `classifier.py` | Risk-classification experiments/training |
| `stage3_drift_predictor.py` | 168h drift-prediction pipeline |
| `drift_predictor.py` | Regression/drift prediction utilities |
| `risk_engine.py` | Risk assessment logic |
| `generate_sih_synthetic_burnin_v2.py` | Synthetic SIH-style data generation |
| `build_console_data.py` | Dashboard/console data preparation |
| `export_reliefos_live_models.py` | Exports trained models for browser inference |

## Running the Dashboard

Keep these files in the **same directory**:

```text
LatentX_Dashboard.html
reliefos_live_models.js
```

Then open `LatentX_Dashboard.html` in a modern web browser.

No Python server is required for the exported model inference used by the dashboard.

## Important Notes

- The current prototype uses synthetic SIH-style data.
- NASA data is a research reference, not ISRO data.
- Prototype limits are assumptions and must be replaced/recalibrated with authorized component-specific limits.
- Model performance can change substantially when evaluated on real burn-in data.
- Different component families may require different parameters, limits, batch profiles, and model configurations.
- High-risk results are intended to support engineering screening and should not be treated as an autonomous replacement for qualification or QA decisions.

## Future Work

- Validate and recalibrate the pipeline on authorized real burn-in data.
- Add component-specific configuration profiles.
- Improve false-negative performance, especially for latent and late-onset defects.
- Add uncertainty estimates to future predictions.
- Expand the parameter set beyond the current prototype.
- Add model monitoring and periodic recalibration.
- Support additional component families through the modular architecture.
- Integrate engineer feedback into validation and retraining workflows.

## Technology Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **HTML / CSS / JavaScript**
- Classical machine learning
- Local browser-based inference

## Project Status

**Prototype / Hackathon Development**

LatentX is currently a research and hackathon prototype. Its purpose is to demonstrate a practical architecture for moving burn-in screening from purely threshold-based pass/fail analysis toward **batch-aware, predictive, and explainable screening**.
