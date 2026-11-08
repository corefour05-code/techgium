# Self-Calibrating & Self-Healing Medical Devices: Infusion Pump ML System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Phase%202%20Completed-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()
[![MedTech](https://img.shields.io/badge/domain-MedTech%20%7C%20Digital%20Twin-red.svg)]()

Research prototype developed for the MedTech Challenge: **"Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins"**, targeting smart intravenous infusion pumps.

---

## 1. System Vision & Architecture

The ML subsystem provides continuous sensor validation, fault isolation, statistical drift detection, measurement error estimation, and sensor health scoring across heterogeneous infusion pump architectures.

```
                      ┌─────────────────────────────────────────┐
                      │        Infusion Pump Telemetry          │
                      │  (Pressure, Force, Flow, Temp, Encoder) │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │     Modular Preprocessing Pipeline      │
                      │ (Validation, Unit Range, Outliers, NaN) │
                      └────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  Statistical     │             │  Cross-Sensor    │             │  Unsupervised    │
│  Drift Engine    │             │  Consistency     │             │  Anomaly Engine  │
│  (EWMA / CUSUM)  │             │  (Hydraulic/Load)│             │ (Isolation Forest)
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │       Fault Classification Engine       │
                      │    (XGBoost Multi-Class Classifier)     │
                      │     [Distinguishes Drift vs Occlusion]  │
                      └────────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────────────┐             ┌─────────────────────────────────────────┐
│     Sensor Error Estimation Module      │             │       Interpretable Sensor Health       │
│     (XGBoost Regressor + Residuals)     │             │       Score & Decision Engine           │
│  [Only when paired reference is valid]  │             │       (Confidence & Recommendations)    │
└────────────────────┬────────────────────┘             └────────────────────┬────────────────────┘
                     │                                                       │
                     └─────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                      ┌─────────────────────────────────────────┐
                      │            FastAPI REST API             │
                      │   (Digital Twin & Dashboard Gateway)    │
                      └─────────────────────────────────────────┘
```

---

## 2. Phase 2 Accomplishments & Pipeline Implementation

- **A. Data Validation Pipeline (`src/data/validation.py`):**
  - Audits missing value rates and longest contiguous gaps.
  - Analyzes timestamp regularity, detects jitter, and flags duplicate timestamps.
  - Implements non-destructive physical range checking (never silently deletes potential clinical faults).
  - Detects sensor freeze (ADC lockup) and sensor dropout.
- **B. Sensor-Specific Preprocessing (`src/data/preprocessing.py`):**
  - Configurable sensor channel preprocessing (supports pumps with any subset of sensors).
  - Sentinel value translation for DosePI (`-99.0` recalibration events and `-1.0` warm-up steps).
  - Normalizes headers and purges empty trailing columns for Zenodo Typical Sensor Defects.
  - Completely preserves raw data files intact.
- **C. Configurable Time-Series Feature Engineering (`src/features/`):**
  - Instantaneous features: numerical derivatives ($dx/dt$), acceleration ($d^2x/dt^2$), and SNR.
  - Rolling features: configurable multi-window sliding statistics (mean, std, median, min, max, IQR, rolling variance, and OLS trend slope $\beta_1$).
  - Cross-sensor features: hydraulic pressure-to-force coupling ($F = P \cdot A + F_f$), delivery flow error, and Digital Twin residuals.
- **D. Research-Oriented Physics-Informed Simulator (`src/data/synthetic_generator.py`):**
  - Disclaims ISO 60601-2-24 compliance (explicitly research-oriented).
  - Fixed seed reproducibility (`seed=42`).
  - Strict field separation: `sensor_pressure_mmhg`, `reference_pressure_mmhg`, `dt_predicted_pressure_mmhg`.
  - Provenance tagging: `SYNTHETIC_DATA` on every generated record.
- **E. Statistical Drift Detection Engines (`src/drift/ewma.py`, `src/drift/cusum.py`):**
  - Mathematically grounded EWMA with time-varying variance limits.
  - Tabular two-sided CUSUM with slack $K$ and decision threshold $H$.
  - In-control baseline calibration bounding empirical false alarm rates ($\le 0.5\% - 1.0\%$). No arbitrary thresholds.
- **F. Verification & Testing:**
  - 18 comprehensive unit tests passing with zero failures (`pytest tests/`).
  - Quantitative benchmark report generated: `reports/DRIFT_EVALUATION_REPORT.md` and `reports/drift_metrics.json`.
  - Publication-quality diagnostic figures generated in `reports/figures/`.

---

## 3. Quickstart & Verification

### Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Run Dataset Acquisition & Inspection
```powershell
# Run acquisition across all 5 research datasets
python scripts/download_datasets.py --dataset all

# Run automated EDA and inventory compilation
python scripts/run_eda.py

# Generate EDA figures
python scripts/generate_eda_figures.py
```

---

## 4. Safety & Medical Device Constraints

1. **No Automatic In-Line Calibration:** In safety-critical medical devices (IEC 60601 / IEC 62304), ML models must never alter pump actuation parameters automatically. All calibration recommendations require validated engineering review.
2. **True Accuracy Requires Reference Ground Truth:** An ML model or Digital Twin simulation cannot assert true sensor accuracy in the absence of an independent physical reference measurement.
3. **Occlusion vs. Sensor Drift Distinction:** An occlusion represents a hydraulic blockage (requiring immediate clinical response), whereas sensor drift represents transducer wear. Our multi-sensor consistency rules ensure these two critical phenomena are never conflated.

