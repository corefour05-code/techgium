# Phase 4 Engineering Report: Supervised Machine Learning, Transducer Error Estimation, SHAP Explainability & Production REST API

**Project:** Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins  
**Target Device:** Infusion Pump (Syringe / Volumetric Line Architecture)  
**Module:** Machine Learning Intelligence Engine (Phase 4 Deliverable)  
**Status:** VALIDATED & COMPLETE (55/55 Unit & Integration Tests Passing)  
**Date:** November 2026  

---

## 1. Executive Summary

Phase 4 delivers the complete supervised machine learning, regression estimation, model explainability, and REST API integration layer for the Infusion Pump Sensor Intelligence ML Module. Building strictly on the validated foundations of Phase 1 (EDA and data provenance), Phase 2 (statistical drift detection via EWMA and CUSUM), and Phase 3 (multivariate anomaly detection and cross-sensor consistency validation), Phase 4 introduces:

1. **Decoupled Sensor Health vs. Process Risk Scoring:** Explicit separation of physical/electrical sensor transducer health (0–100) from fluidic infusion process risk (0–100). Crucially, a genuine downstream catheter occlusion triggers a **CRITICAL process risk (95/100)** while maintaining an unpenalized **sensor health score ($\ge 90/100$)**, resolving a classic biomedical false-degradation failure mode.
2. **XGBoost 9-Class Fault Classifier:** A production gradient-boosted decision tree ensemble trained with run-level experimental isolation (zero temporal leakage), balanced sample weighting, and configurable channel inputs. Achieved **97.48% Balanced Accuracy**, **0.9778 Macro F1-score**, and a low **1.73% False Alarm Rate** on nominal infusion runs.
3. **Paired Sensor Error Regressor:** A dedicated XGBoost regressor predicting physical transducer bias ($\Delta P = P_{\text{sensor}} - P_{\text{ref}}$) when verified independent reference standards are connected. Achieved **MAE = 0.5895 mmHg**, **RMSE = 1.9369 mmHg**, and **$R^2 = 0.9945$**, equipped with a mandatory `INSUFFICIENT_REFERENCE_DATA` safety safeguard when physical reference data is missing.
4. **SHAP TreeExplainer Attribution:** Model interpretability engine producing per-prediction local feature rankings, attribution directions (`PUSHED_TOWARD` / `PUSHED_AWAY`), and human-readable narrative summaries accompanied by medical AI non-causality disclaimers.
5. **FastAPI Production REST API:** Asynchronous REST microservice exposing 6 endpoints (`/health`, `/models/status`, `/predict/fault`, `/predict/sensor-error`, `/predict/sensor-health`, `/predict/batch`) with Pydantic v2 data validation, comprehensive error handling, and advisory safety banners.
6. **Zero-Defect Test Suite:** 55 passing unit and integration tests across all four project phases and the REST API.

---

## 2. Sensor Health Scoring vs. Device/Process Risk Decoupling

### 2.1 The Clinical & Physical Motivation
In previous naive architectures, any signal anomaly flagged by the system penalized a single aggregate "Sensor Health Score". In an infusion pump, this represents a severe engineering flaw:
- When a patient's venous line clots or a stopcock is closed, fluid cannot exit the tubing.
- In accordance with fluid mechanics, line pressure surges exponentially ($\Delta P \ge 250\text{ mmHg}$) and plunger mechanical force increases in direct proportion ($F = P \cdot A + F_{\text{fric}}$).
- The pressure transducer and force sensor are **functioning with perfect fidelity** by reporting this dangerous pressure surge.
- Penalizing the sensor transducer health score during an occlusion falsely flags healthy hardware as defective, misleading clinical technicians into replacing functioning transducers while ignoring the acute line obstruction.

### 2.2 Decoupled Scoring Architecture
In `src/health/sensor_health.py`, the scoring engine maintains two mathematically distinct score vectors:

$$\text{Sensor Health Score} = 100 - \sum \text{Transducer Degradation Deductions}$$

$$\text{Process Risk Score} = \min\left(100, \; \text{Base Risk} + \sum \text{Fluidic Pathway Risk Factors}\right)$$

| Dimension | Measured Entity | Deductions / Escalations | Occlusion Behavior |
| :--- | :--- | :--- | :--- |
| **Sensor Health Score (0–100)** | Physical and electrical integrity of transducer hardware | Drift, sudden calibration jump, electrical freeze (ADC lockup), ground dropout, excessive variance | **Score remains $\ge 90/100$ (HEALTHY)** — Transducer operates correctly |
| **Process Risk Score (0–100)** | Patient safety and infusion fluid delivery integrity | Downstream occlusion, delivery interruption, hydraulic divergence | **Surges to $\ge 90/100$ (CRITICAL)** — Immediate clinical alarm |

### 2.3 Clarification on Isolation Forest Severity Score
Isolation Forest decision function values $s(\mathbf{x})$ are converted to a normalized anomaly severity metric:

$$\text{severity}(\mathbf{x}) = \text{clip}\left(\frac{s_{\text{nominal}} - s(\mathbf{x})}{s_{\text{nominal}} - s_{\text{extreme}}}, \; 0.0, \; 1.0\right)$$

> [!IMPORTANT]
> This metric is strictly documented and logged as an **Anomaly Severity Score in $[0, 1]$** reflecting geometric isolation depth in feature space. It is **NOT** a calibrated probability of hardware failure.

---

## 3. Configurable Time-Series Feature Engineering

The feature engineering layer (`src/features/feature_pipeline.py`) transforms multi-sensor windows into 23 fixed numerical features without assuming every infusion pump contains all sensor channels:

```
Multi-Sensor Input Window (Pressure, Force, Flow, Temp, Commanded Rate, DT Predict)
  │
  ├── 1. Pressure Transducer Dynamics
  │     ├── p_raw: Current instantaneous pressure (mmHg)
  │     ├── p_roll_mean_60: 60-second rolling moving average
  │     ├── p_roll_std_60: 60-second rolling signal variance
  │     ├── p_roll_slope_60: Ordinary least-squares linear trend (mmHg/s)
  │     └── p_dt_residual: Deviation from Digital Twin theoretical expectation
  │
  ├── 2. Force Transducer Dynamics & Channel Availability
  │     ├── has_force_channel: Binary indicator {0.0, 1.0}
  │     ├── force_raw: Plunger force (N) or nominal hydraulic imputation
  │     ├── force_expected_from_p: Theoretical force: (P * Area) + Friction
  │     └── force_hydraulic_residual: Physical divergence between force & pressure
  │
  ├── 3. Flow Transducer Dynamics & Channel Availability
  │     ├── has_flow_channel: Binary indicator {0.0, 1.0}
  │     ├── flow_raw: Delivered volumetric flow (mL/h) or nominal imputation
  │     ├── flow_residual_from_cmd: Discrepancy from commanded rate
  │     └── flow_roll_std_60: 60-second flow delivery variability
  │
  ├── 4. Environmental Thermal Coupling
  │     ├── has_temp_channel: Binary indicator {0.0, 1.0}
  │     ├── temp_raw: Ambient enclosure temperature (°C)
  │     └── temp_roll_delta_60: 60-second temperature drift rate (°C)
  │
  └── 5. Electrical & ADC Integrity Counters
        ├── sensor_freeze_counter: Consecutive identical ADC samples (Δ = 0)
        └── sensor_dropout_counter: Consecutive zero/ground readings (≤ 0.01)
```

If a pump lacks force, flow, or temperature sensors, the pipeline sets `has_<channel>_channel = 0.0` and imputes nominal physics-informed baselines, ensuring deterministic feature shapes across diverse hardware configurations.

---

## 4. XGBoost Multiclass Classification

### 4.1 Architecture and Training Rigor
- **Algorithm:** Hist-based Gradient Boosted Decision Tree Classifier (`src/classification/xgboost_classifier.py`).
- **Classes (9):**
  1. `NORMAL`
  2. `POSITIVE_DRIFT`
  3. `NEGATIVE_DRIFT`
  4. `SUDDEN_OFFSET`
  5. `EXCESSIVE_NOISE`
  6. `SENSOR_FREEZE`
  7. `SENSOR_DROPOUT`
  8. `GENUINE_OCCLUSION`
  9. `TEMPERATURE_INFLUENCE`
- **Data Provenance:** Strict `SYNTHETIC_DATA` tagging across all records. No incompatible public datasets (e.g., Zenodo DosePI or industrial bearing vibration) were co-mingled into the pump fault classifier.
- **Run-Level Isolation (Zero Temporal Leakage):** Complete 30-minute experimental runs were partitioned at the run level:
  - **Training:** 140,940 samples (Seeds 42, 101, 202) across commanded rates of 20, 50, and 100 mL/h with randomized channel presence and fault magnitudes.
  - **Validation:** 15,660 samples (Seed 303).
  - **Testing:** 93,960 samples (Seeds 404, 505) — completely unseen runs, seeds, and disturbance realizations.
- **Class Imbalance:** Handled via balanced inverse-frequency sample weights.

### 4.2 Test Set Performance Evaluation (93,960 Test Samples)

| Fault Class | Precision | Recall | F1-Score | Support (Samples) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NORMAL** | **0.9687** | **0.9827** | **0.9756** | 36,360 | Validated |
| **POSITIVE_DRIFT** | **0.9656** | **0.9440** | **0.9547** | 7,200 | Validated |
| **NEGATIVE_DRIFT** | **0.9180** | **0.9543** | **0.9358** | 7,200 | Validated |
| **SUDDEN_OFFSET** | **1.0000** | **1.0000** | **1.0000** | 7,200 | Validated |
| **EXCESSIVE_NOISE** | **0.9994** | **1.0000** | **0.9997** | 7,200 | Validated |
| **SENSOR_FREEZE** | **1.0000** | **0.9992** | **0.9996** | 7,200 | Validated |
| **SENSOR_DROPOUT** | **1.0000** | **1.0000** | **1.0000** | 7,200 | Validated |
| **GENUINE_OCCLUSION** | **1.0000** | **0.9994** | **0.9997** | 7,200 | Validated |
| **TEMPERATURE_INFLUENCE** | **0.9814** | **0.8936** | **0.9354** | 7,200 | Validated |
| **Macro Average** | **0.9815** | **0.9748** | **0.9778** | 93,960 | **Target Exceeded** |
| **Overall Accuracy** | — | — | **0.9772** | 93,960 | **Target Exceeded** |

- **Balanced Accuracy:** **97.48%**
- **False Alarm Rate on Normal Operation:** **1.73%** (630 false positives out of 36,360 normal test samples)

### 4.3 Confusion Matrix Analysis
As illustrated in `reports/figures/classification_confusion_matrix.png`:
- `SUDDEN_OFFSET`, `SENSOR_DROPOUT`, and `EXCESSIVE_NOISE` achieve near-perfect separation (F1 $\ge 0.999$).
- `GENUINE_OCCLUSION` achieved 99.94% recall and 100% precision, demonstrating flawless cross-sensor disambiguation against sensor drift.
- The slight confusion between `TEMPERATURE_INFLUENCE` and gradual drift occurs during the early initial thermal ramp ($\Delta T < 0.5^\circ\text{C}$), which is physically expected when temperature swings are subtle.

---

## 5. Supervised Regression: Transducer Error Estimation

### 5.1 Problem Formulation & Physics Constraints
$$\text{Sensor Error} = P_{\text{sensor}} - P_{\text{reference}}$$

When a biomedical engineer connects an independent, NIST-traceable calibrated manometer during maintenance, the ML module estimates the systematic transducer error and computes a 95% confidence interval.

> [!WARNING]
> **Strict Clinical Safety Rule:**
> An ML model cannot establish true sensor accuracy without independent reference measurements. Digital Twin predictions are theoretical approximations and must **never** be substituted for reference ground truth. When independent reference data is omitted, the regressor automatically halts and returns status `INSUFFICIENT_REFERENCE_DATA`.

### 5.2 Regressor Performance Metrics
- **Model:** Hist-based Gradient Boosted Regressor (`src/regression/xgboost_regressor.py`).
- **Mean Absolute Error (MAE):** **0.5895 mmHg**
- **Root Mean Squared Error (RMSE):** **1.9369 mmHg**
- **Coefficient of Determination ($R^2$):** **0.9945**
- **Residual Uncertainty:** 95% Confidence Interval width = $\pm 1.96 \times 1.9369 = \pm 3.796\text{ mmHg}$.

The parity plot (`reports/figures/regression_error_parity.png`) shows tight clustering along the identity line across offset magnitudes from $-40\text{ mmHg}$ to $+50\text{ mmHg}$.

---

## 6. SHAP Model Explainability

### 6.1 Explainability Integration
To ensure clinical engineering trust, every classification prediction is accompanied by a local feature attribution report generated via `shap.TreeExplainer` (`src/explainability/shap_explainer.py`).

For each prediction, the explainer returns:
1. **Predicted Class & Model Confidence** (e.g., `TEMPERATURE_INFLUENCE`, 99.8%).
2. **Top Contributing Features:** Ranked by absolute SHAP value $|\phi_i|$.
3. **Attribution Direction:**
   - `PUSHED_TOWARD`: Feature value positively increased the log-odds of the predicted class.
   - `PUSHED_AWAY`: Feature value suppressed alternative classes or moderated the score.
4. **Human-Readable Clinical Summary:** Synthesizes the quantitative SHAP values into an intuitive sentence.

### 6.2 Sample SHAP Output (Thermal Drift Scenario)
```json
{
  "predicted_class": "TEMPERATURE_INFLUENCE",
  "confidence": 0.9984,
  "top_contributing_features": [
    {
      "feature": "temp_raw",
      "value": 27.60,
      "shap_attribution": 5.1386,
      "influence_direction": "PUSHED_TOWARD"
    },
    {
      "feature": "has_flow_channel",
      "value": 0.0,
      "shap_attribution": -1.4305,
      "influence_direction": "PUSHED_AWAY"
    },
    {
      "feature": "p_dt_residual",
      "value": 6.16,
      "shap_attribution": -0.3975,
      "influence_direction": "PUSHED_AWAY"
    }
  ],
  "explanation_summary": "Prediction: TEMPERATURE_INFLUENCE (confidence: 99.8%). Primary factors: temp_raw (27.60) increased likelihood of TEMPERATURE_INFLUENCE (attribution: +5.139); has_flow_channel (0.00) reduced likelihood of TEMPERATURE_INFLUENCE (attribution: -1.431).",
  "disclaimer": "Interpretability Notice: SHAP feature attributions reflect statistical model associations within the decision trees and must NOT be interpreted as causal physiological proof."
}
```

---

## 7. Production FastAPI REST Service

### 7.1 Architecture & Endpoints
The REST API is implemented in `api/app.py` with Pydantic v2 validation models in `api/schemas/payloads.py`:

| Method | Endpoint | Description | Key Request / Response Elements |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service and model availability status | Returns `status: "OK"`, loaded models dictionary, and advisory notice |
| `GET` | `/models/status` | Model registry inventory & parameters | Lists classes, feature counts, and residual RMSEs |
| `POST` | `/predict/fault` | Real-time 9-class fault classification | Input: sensor window; Output: class, probabilities, SHAP attribution |
| `POST` | `/predict/sensor-error` | Transducer error estimation | Input: paired reference; Output: $\Delta P$, calibrated estimate, 95% CI |
| `POST` | `/predict/sensor-health`| Integrated health and process scoring | Output: decoupled `sensor_health_score`, `process_risk_score`, risk level |
| `POST` | `/predict/batch` | Batch inference processing | Array of sensor windows processed concurrently |

### 7.2 Safety & Advisory Disclaimers
Every endpoint response includes the mandatory regulatory advisory banner:
> *"Advisory Medical AI Notice: Recommendations are advisory only. No automated device calibration is permitted without validated biomedical engineering sign-off. True sensor accuracy cannot be confirmed without independent reference measurements."*

---

## 8. Verification and Test Suite Execution

A comprehensive Pytest suite covers all 4 development phases and the API layer:

```
============================= test session starts =============================
platform win32 -- Python 3.12.7, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\techgium
collected 55 items

tests/test_api.py::test_api_health_endpoint PASSED                       [  1%]
tests/test_api.py::test_api_models_status_endpoint PASSED                [  3%]
tests/test_api.py::test_api_predict_fault_normal PASSED                  [  5%]
tests/test_api.py::test_api_predict_fault_dropout PASSED                 [  7%]
tests/test_api.py::test_api_predict_fault_occlusion PASSED               [  9%]
tests/test_api.py::test_api_predict_sensor_error_with_reference PASSED   [ 10%]
tests/test_api.py::test_api_predict_sensor_error_missing_reference PASSED [ 12%]
tests/test_api.py::test_api_predict_sensor_error_unknown_sensor_type PASSED [ 14%]
tests/test_api.py::test_api_predict_sensor_health_normal PASSED          [ 16%]
tests/test_api.py::test_api_predict_sensor_health_occlusion_decoupling PASSED [ 18%]
tests/test_api.py::test_api_predict_batch PASSED                         [ 20%]
tests/test_api.py::test_api_invalid_input_validation_error PASSED        [ 21%]
tests/test_phase1.py::test_core_dependencies PASSED                      [ 23%]
tests/test_phase1.py::test_config_enums_and_paths PASSED                 [ 25%]
tests/test_phase1.py::test_inspect_csv_encoding_fallback PASSED          [ 27%]
tests/test_phase1.py::test_dosepi_loader PASSED                          [ 29%]
tests/test_phase1.py::test_sensor_defects_loader PASSED                  [ 30%]
tests/test_phase2.py::test_missing_values_audit PASSED                   [ 32%]
tests/test_phase2.py::test_timestamp_regularity_and_duplicates PASSED    [ 34%]
tests/test_phase2.py::test_non_destructive_range_validation PASSED       [ 36%]
tests/test_phase2.py::test_frozen_sensor_detection PASSED                [ 38%]
tests/test_phase2.py::test_sensor_dropout_detection PASSED               [ 40%]
tests/test_phase2.py::test_synthetic_data_provenance_and_field_separation PASSED [ 41%]
tests/test_phase2.py::test_configurable_sensor_channels PASSED           [ 43%]
tests/test_phase2.py::test_dosepi_sentinel_cleaning PASSED               [ 45%]
tests/test_phase2.py::test_zenodo_cleaning_trailing_column PASSED        [ 47%]
tests/test_phase2.py::test_ewma_drift_detection PASSED                   [ 49%]
tests/test_phase2.py::test_cusum_drift_detection PASSED                  [ 50%]
tests/test_phase2.py::test_rolling_trend_slope PASSED                    [ 52%]
tests/test_phase2.py::test_cross_sensor_hydraulic_consistency PASSED     [ 54%]
tests/test_phase3.py::test_isolation_forest_fit_predict_and_persistence PASSED [ 56%]
tests/test_phase3.py::test_scenario_normal_operation PASSED              [ 58%]
tests/test_phase3.py::test_scenario_gradual_positive_drift PASSED        [ 60%]
tests/test_phase3.py::test_scenario_gradual_negative_drift PASSED        [ 61%]
tests/test_phase3.py::test_scenario_sudden_offset PASSED                 [ 63%]
tests/test_phase3.py::test_scenario_genuine_downstream_occlusion PASSED  [ 65%]
tests/test_phase3.py::test_scenario_sensor_freeze PASSED                 [ 67%]
tests/test_phase3.py::test_scenario_sensor_dropout PASSED                [ 69%]
tests/test_phase3.py::test_scenario_environmental_temperature_influence PASSED [ 70%]
tests/test_phase3.py::test_scenario_missing_sensor_channels_safeguard PASSED [ 72%]
tests/test_phase3.py::test_scenario_conflicting_sensor_evidence PASSED   [ 74%]
tests/test_phase3.py::test_independent_reference_ground_truth_and_disclaimer PASSED [ 76%]
tests/test_phase4.py::test_classifier_predict_normal PASSED              [ 78%]
tests/test_phase4.py::test_classifier_predict_positive_drift PASSED      [ 80%]
tests/test_phase4.py::test_classifier_predict_negative_drift PASSED      [ 81%]
tests/test_phase4.py::test_classifier_predict_sudden_offset PASSED       [ 83%]
tests/test_phase4.py::test_classifier_predict_excessive_noise PASSED     [ 85%]
tests/test_phase4.py::test_classifier_predict_sensor_freeze PASSED       [ 87%]
tests/test_phase4.py::test_classifier_predict_sensor_dropout PASSED      [ 89%]
tests/test_phase4.py::test_classifier_predict_genuine_occlusion PASSED   [ 90%]
tests/test_phase4.py::test_classifier_predict_temperature_influence PASSED [ 92%]
tests/test_phase4.py::test_regressor_with_paired_reference_data PASSED   [ 94%]
tests/test_phase4.py::test_regressor_missing_reference_safeguard PASSED  [ 96%]
tests/test_phase4.py::test_shap_explanation_contents PASSED              [ 98%]
tests/test_phase4.py::test_sensor_health_decoupling_in_occlusion PASSED  [100%]

============================= 55 passed in 7.84s ==============================
```

---

## 9. Artifacts Created & Modified

### New Phase 4 Modules:
- `src/features/feature_pipeline.py`: 23-feature temporal, physical residual, and channel availability extractor.
- `src/classification/xgboost_classifier.py`: 9-class XGBoost fault classification engine with balanced weighting.
- `src/regression/xgboost_regressor.py`: Paired transducer error estimator with 95% confidence intervals and safety safeguards.
- `src/explainability/shap_explainer.py`: Model explainability wrapper using TreeExplainer for local feature ranking.
- `scripts/train_and_validate_models.py`: Complete training, validation, and serialization pipeline with run-level separation.
- `api/schemas/payloads.py`: Pydantic request and response schemas for all 6 endpoints.
- `api/app.py`: Production FastAPI REST API microservice with async handlers and CORS middleware.
- `tests/test_phase4.py`: 13 scenario and unit tests verifying Phase 4 ML models.
- `tests/test_api.py`: 12 integration tests verifying API routes, error codes, and Pydantic validation.

### Updated Previous Modules:
- `src/health/sensor_health.py`: Refactored to completely decouple Sensor Health Score from Process Risk Score and document Isolation Forest severity scores.

### Serialized Model Bundles & Deliverables:
- `models/classification/xgboost_classifier.joblib`
- `models/regression/xgboost_regressor_pressure.joblib`
- `models/anomaly_detection/isolation_forest.joblib`
- `reports/model_evaluation_metrics.json`
- `reports/figures/classification_confusion_matrix.png`
- `reports/figures/regression_error_parity.png`
- `reports/figures/xgboost_feature_importance.png`
- `reports/PHASE4_REPORT.md`

---

## 10. Known Limitations & Next Steps

### Known Limitations
1. **Research Simulator Foundation:** The models were trained on physics-informed synthetic simulations. While rigorously modeled with realistic fluid dynamics, electrical noise, and environmental coupling, real infusion pumps exhibit proprietary mechanical elastomeric compliance curves that require bench-test calibration.
2. **Computational Footprint of SHAP:** Real-time TreeExplainer attribution adds $\approx 15\text{ ms}$ of latency per instance. For high-frequency edge microcontrollers (e.g., STM32 / ARM Cortex-M), explainability should be computed asynchronously on the dashboard host.
3. **No Automated Device Calibration:** In accordance with medical device software risk management principles (IEC 62304 / ISO 14971), the ML system outputs **recommendations only**; closed-loop recalibration must never be executed autonomously without validated biomedical engineer authorization.

### Recommended Next Steps (Phase 5: Remaining Useful Life & Digital Twin Integration)
1. **Predictive Maintenance & RUL Estimation:** Implement degradation trajectory tracking (Weibull / Cox proportional hazards or linear drift extrapolation) to forecast Remaining Useful Life before sensor tolerance exceedance.
2. **Digital Twin Co-Simulation Loop:** Connect the REST API to the team's real-time 3D Digital Twin visualization dashboard via WebSockets / REST webhooks.
3. **Hardware-in-the-Loop (HIL) Bench Validation:** Deploy the REST service against physical syringe pump test benches streaming live serial sensor telemetry.

