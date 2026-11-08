# Machine Learning Backend API Integration Contract

**Document Version:** 1.0.0-phase6  
**System Target:** Infusion Pump AI & Digital Twin Co-Simulation  
**Subsystem Responsibility:** Machine Learning (ML) Sensor Intelligence Module  
**Data Provenance Status:** `SYNTHETIC_DATA` (Simulation Benchmarking; Physical Hardware Validation Pending)  
**Safety Classification:** Advisory Medical Device AI (Non-Controlling, Audit-Gated)  

---

## 1. Executive Summary & Purpose

This document establishes the formal, binding Application Programming Interface (API) contract between the **Infusion Pump Machine Learning Backend** and the **Digital Twin (DT)** developed by the simulation team.

The ML Backend provides:
1. **Real-Time Sequential Ingestion:** Stateful 1 Hz telemetry streaming with sliding-window dynamics.
2. **Multi-Class Fault Attribution:** 9-class supervised classification distinguishing transducer degradation from true physiological/process blockage.
3. **Calibrated Confidence:** Post-hoc probability calibration (Temperature Scaling) providing empirical frequency alignment.
4. **Finite-Sample Measurement Error Intervals:** Distribution-free Split Conformal Prediction intervals (95% marginal coverage) when physical references exist.
5. **Decoupled Scoring:** Strictly independent **Sensor Health Score (0–100)** and **Process Delivery Risk Score (0–100)**.
6. **Stateless Batch Prediction:** High-throughput window evaluation for retrospective audit trails and simulation batch runs.

---

## 2. Communication Protocol & Global Conventions

### 2.1 Transport & Networking
- **Protocol:** HTTP/1.1 (JSON payloads over REST)
- **Default Base URL:** `http://127.0.0.1:8000`
- **Default Port:** `8000` (Configurable via `UVICORN_PORT` environment variable)
- **Content-Type:** `application/json`
- **Accept:** `application/json`

### 2.2 Expected Telemetry Sampling Rate
- **Nominal Sampling Frequency:** **1.0 Hz** ($T_s = 1.0\text{ s}$).
- **Minimum Sampling Frequency:** 0.2 Hz (5.0s between telemetry frames). Telemetry gaps exceeding $5.0\text{ s}$ trigger `STALE_TELEMETRY_GAP` flags.
- **Maximum Sampling Frequency:** Up to 50 Hz. (Inference latency is ~16.9 ms per step, leaving ample headroom).

### 2.3 Sensor Channel Specifications & Valid Engineering Ranges

| Channel Name in JSON | Sensor Type | Physical Meaning | Engineering Unit | Valid Operational Range | Channel Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sensor_pressure_mmhg` | Piezoresistive Transducer | Downstream catheter/line pressure | $\text{mmHg}$ | $[-50.0, 2000.0]$ | **MANDATORY** |
| `sensor_force_n` | Strain Gauge Load Cell | Syringe plunger driving force | $\text{N}$ | $[0.0, 200.0]$ | *OPTIONAL* |
| `sensor_flow_ml_h` | Ultrasonic/Coriolis Flow | Delivered effluent volume rate | $\text{mL/h}$ | $[0.0, 2000.0]$ | *OPTIONAL* |
| `sensor_temperature_c` | NTC Thermistor / RTD | Internal/ambient manifold temp | $^\circ\text{C}$ | $[-20.0, 100.0]$ | *OPTIONAL* |
| `commanded_rate_ml_h` | Stepper Controller Setpoint | Commanded pump delivery rate | $\text{mL/h}$ | $[0.0, 1500.0]$ | *OPTIONAL* (Default: 50.0) |
| `dt_predicted_pressure_mmhg` | Hydrodynamic Digital Twin | DT expected fluid backpressure | $\text{mmHg}$ | $[0.0, 2000.0]$ | *OPTIONAL* |
| `independent_reference_value`| NIST/Calibrated Manometer | External physical reference gauge | $\text{mmHg}$ | $[0.0, 2000.0]$ | *OPTIONAL* (Advisory Only) |
| `baseline_pressure_mmhg` | Clinical Tare / Pre-infusion | Established catheter pre-tare | $\text{mmHg}$ | $[0.0, 300.0]$ | *OPTIONAL* |

> [!IMPORTANT]
> The only strictly mandatory physical sensor channel is `sensor_pressure_mmhg`. If secondary channels (`force`, `flow`, `temperature`) are not installed or offline, the ML pipeline automatically activates nominal imputation without crashing.

---

## 3. Real-Time Sequential Telemetry API (`/stream/observe`)

Used for continuous, sample-by-sample 1 Hz streaming during live simulation. The ML backend internally buffers telemetry in a FIFO deque (capacity: 120 samples) to compute multi-scale rolling feature dynamics (15s mean/slope, 60s slope/variance, rate-of-change, hydraulic coupling).

### 3.1 Endpoint Definition
- **Method:** `POST`
- **Path:** `/stream/observe`

### 3.2 Request Payload (`SequentialObservationRequest`)

```json
{
  "session_id": "pump_station_alpha_01",
  "sensor_id": "PRESS_LINE_01",
  "timestamp": 124.0,
  "sensor_pressure_mmhg": 84.2,
  "sensor_force_n": 4.92,
  "sensor_flow_ml_h": 49.8,
  "sensor_temperature_c": 22.4,
  "commanded_rate_ml_h": 50.0,
  "dt_predicted_pressure_mmhg": 80.0,
  "independent_reference_value": null,
  "baseline_pressure_mmhg": null
}
```

#### Field Specifications:
- `session_id` (`string`, required): Unique session/pump identifier. Used to isolate telemetry queues between multiple concurrent pumps.
- `sensor_id` (`string`, required): Identifier of the physical sensor transducer.
- `timestamp` (`float`, required): Relative or UNIX epoch time in seconds. Monotonically increasing.
- `sensor_pressure_mmhg` (`float`, required): Transducer pressure measurement. Must be between -50.0 and 2000.0 mmHg.
- `sensor_force_n` (`float | null`, optional): Plunger load cell reading.
- `sensor_flow_ml_h` (`float | null`, optional): Flowmeter reading.
- `sensor_temperature_c` (`float | null`, optional): Thermal sensor reading.
- `commanded_rate_ml_h` (`float`, optional, default: 50.0): Commanded pump infusion setpoint.
- `dt_predicted_pressure_mmhg` (`float | null`, optional): Hydrodynamic line pressure expected by the Digital Twin.
- `independent_reference_value` (`float | null`, optional): Physical reference measurement (used only for empirical measurement error estimation).
- `baseline_pressure_mmhg` (`float | null`, optional): Clinically validated catheter pre-tare. If omitted, the adaptive pre-infusion tare engine estimates it automatically.

---

### 3.3 Response Payload (`SequentialPredictionResponse`)

```json
{
  "session_id": "pump_station_alpha_01",
  "sensor_id": "PRESS_LINE_01",
  "timestamp": 124.0,
  "provenance": "SYNTHETIC_DATA",
  "stream_quality": "OK",
  "warmup_status": "READY",
  "buffered_samples": 60,
  "predicted_fault_class": "NORMAL",
  "class_index": 0,
  "model_confidence": 0.9854,
  "calibrated_confidence": 0.9412,
  "calibrated_probabilities": {
    "NORMAL": 0.9412,
    "POSITIVE_DRIFT": 0.0074,
    "NEGATIVE_DRIFT": 0.0074,
    "SUDDEN_OFFSET": 0.0074,
    "EXCESSIVE_NOISE": 0.0074,
    "SENSOR_FREEZE": 0.0074,
    "SENSOR_DROPOUT": 0.0074,
    "GENUINE_OCCLUSION": 0.0074,
    "TEMPERATURE_INFLUENCE": 0.0074
  },
  "raw_class_probabilities": {
    "NORMAL": 0.9854,
    "POSITIVE_DRIFT": 0.0018,
    "NEGATIVE_DRIFT": 0.0018,
    "SUDDEN_OFFSET": 0.0018,
    "EXCESSIVE_NOISE": 0.0018,
    "SENSOR_FREEZE": 0.0018,
    "SENSOR_DROPOUT": 0.0018,
    "GENUINE_OCCLUSION": 0.0018,
    "TEMPERATURE_INFLUENCE": 0.0018
  },
  "is_sensor_fault": false,
  "is_process_anomaly": false,
  "sensor_health_score": 100.0,
  "process_risk_score": 5.0,
  "process_risk_level": "LOW",
  "anomaly_detected": false,
  "anomaly_severity_score": 0.0,
  "estimated_sensor_error": null,
  "conformal_error_interval_95": null,
  "error_status": "INSUFFICIENT_REFERENCE_DATA",
  "advisory_recommendation": "Normal infusion telemetry. Transducer operating nominally.",
  "disclaimer": "Advisory Medical AI Notice: Recommendations are advisory only. No automated device calibration is permitted without validated biomedical engineering sign-off."
}
```

#### Key Output Fields Explained:
- `stream_quality`: Telemetry stream integrity indicator:
  - `"OK"`: Monotonic timestamp with normal gap.
  - `"DUPLICATE_TIMESTAMP"`: Telemetry frame has identical timestamp to preceding frame ($\Delta t \approx 0$).
  - `"OUT_OF_ORDER_TIMESTAMP"`: Telemetry arrived reversed in time ($\Delta t < 0$).
  - `"STALE_TELEMETRY_GAP"`: Telemetry gap exceeds 5.0 seconds ($\Delta t > 5.0\text{ s}$).
- `warmup_status`:
  - `"WARMUP"`: Buffer contains $< 15$ samples. Multi-scale feature slopes (15s, 60s) are not yet mature. Emits default safe `"NORMAL"` attribution.
  - `"READY"`: Buffer contains $\ge 15$ samples. Full ML inference active.
- `predicted_fault_class`: Target fault category (see Section 5 for enumeration).
- `model_confidence` vs `calibrated_confidence`:
  - `model_confidence`: Raw XGBoost tree leaf softmax probability.
  - `calibrated_confidence`: Post-hoc Temperature-Scaled probability, aligning output scores with empirical true-positive frequencies (ECE reduced from 0.1506 to 0.0247).
- `sensor_health_score` vs `process_risk_score` (Strictly Decoupled):
  - `sensor_health_score` ($0.0 \text{ to } 100.0$): Transducer physical integrity. Penalized by drift, ADC freeze, rail dropout, and excessive stochastic noise. **NOT penalized by genuine line occlusion.**
  - `process_risk_score` ($0.0 \text{ to } 100.0$): Patient delivery safety hazard. Escalated by occlusion (95.0), severe drift (50.0), or loss of pressure signal (90.0).
- `conformal_error_interval_95`: Split Conformal Prediction interval `[lower, upper]` guaranteeing 95% marginal coverage without assuming Gaussian errors. Populated *only* when `independent_reference_value` is supplied.

---

## 4. Session Lifecycle Management

### 4.1 Reset Streaming Buffer (`POST /session/reset`)
Resets internal rolling queues, timestamps, and quality error counters for a given session. Should be called by the Digital Twin when starting a new infusion simulation run or priming a new catheter line.

- **Request Payload:**
```json
{
  "session_id": "pump_station_alpha_01"
}
```
- **Response Payload:**
```json
{
  "session_id": "pump_station_alpha_01",
  "reset_success": true,
  "timestamp": "2026-11-08T10:59:09.123456Z",
  "message": "Session 'pump_station_alpha_01' buffer reset successfully."
}
```

### 4.2 Query Session Buffer Status (`GET /session/status/{session_id}`)
Queries buffer health, sample count, and stream quality issue counts.

- **URL Example:** `GET /session/status/pump_station_alpha_01`
- **Response Payload:**
```json
{
  "session_id": "pump_station_alpha_01",
  "buffered_samples": 60,
  "max_buffer_size": 120,
  "warmup_threshold": 15,
  "warmup_status": "READY",
  "total_observations_received": 145,
  "last_timestamp": 144.0,
  "telemetry_quality_issues": {
    "DUPLICATE_TIMESTAMP": 0,
    "OUT_OF_ORDER_TIMESTAMP": 0,
    "STALE_TELEMETRY_GAP": 0
  }
}
```

---

## 5. Classification Schema & Fault Enumeration

The ML Backend classifies operating conditions into exactly 9 mutually exclusive categories:

| Class Index | Class Identifier | Fault Category | Physical Mechanism / Clinical Interpretation | `is_sensor_fault` | `is_process_anomaly` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `NORMAL` | In-Control | Nominal infusion; sensor and line operate within normal hydraulic tolerances. | `false` | `false` |
| `1` | `POSITIVE_DRIFT` | Sensor Defect | Piezoresistive diaphragm oxidation / slow positive transducer offset. | `true` | `false` |
| `2` | `NEGATIVE_DRIFT` | Sensor Defect | Piezoresistive diaphragm leakage / slow negative transducer loss. | `true` | `false` |
| `3` | `SUDDEN_OFFSET` | Sensor Defect | Hardware step shift / mechanical impact on transducer body. | `true` | `false` |
| `4` | `EXCESSIVE_NOISE` | Sensor Defect | EMI interference / degraded electrical shielding / loose ground. | `true` | `false` |
| `5` | `SENSOR_FREEZE` | Sensor Defect | ADC conversion lock / telemetry bus freeze at invariant constant. | `true` | `false` |
| `6` | `SENSOR_DROPOUT` | Sensor Defect | Lead disconnection / rail loss to 0.0 mmHg. | `true` | `false` |
| `7` | `GENUINE_OCCLUSION`| Process Blockage| Physical downstream fluid pathway occlusion (clot, closed stopcock, kinked tubing). Pressure and force surge; flow drops. | `false` | `true` |
| `8` | `TEMPERATURE_INFLUENCE`| Environmental | Ambient thermal excursion affecting uncompensated piezo element. | `true` | `false` |

---

## 6. Stateless Window & Batch Prediction Endpoints

For non-streaming batch simulation, offline replay, and model validation.

### 6.1 `POST /predict/fault`
Evaluates a single window array of telemetry and returns classification with local SHAP feature attributions.

- **Request:**
```json
{
  "sensor_id": "PRESS_01",
  "sensor_pressure_mmhg": [80.0, 80.1, 80.2, 80.1, 80.3, ...],
  "sensor_force_n": [4.85, 4.86, ...],
  "sensor_flow_ml_h": [50.0, 49.9, ...],
  "sensor_temperature_c": [22.0, 22.0, ...],
  "commanded_rate_ml_h": 50.0,
  "dt_predicted_pressure_mmhg": 80.0
}
```

### 6.2 `POST /predict/sensor-error`
Estimates transducer measurement error against independent physical reference gauges.
- **Status Codes:**
  - If `independent_reference_value` is `null`: Returns `status: "INSUFFICIENT_REFERENCE_DATA"`.
  - If `independent_reference_value` is provided: Returns `status: "ESTIMATION_AVAILABLE"`, estimated error, and 95% conformal interval.

### 6.3 `POST /predict/sensor-health`
Evaluates multi-modal evidence across consistency checks, Isolation Forest anomaly detection, and drift detectors to produce decoupled Sensor Health and Process Risk scores.

### 6.4 `POST /predict/batch`
Batch wrapper accepting `{"items": [FaultPredictionRequest, ...]}` returning high-throughput evaluations (throughput: ~45 samples/sec).

---

## 7. Error Codes and Failure Modes

The API uses standard HTTP response codes:

| HTTP Status | Error Type | Cause | Recommended Digital Twin Action |
| :--- | :--- | :--- | :--- |
| `400 Bad Request` | `Feature Extraction Error` | Empty array or non-numeric entries in telemetry window. | Inspect JSON format and ensure valid numbers are transmitted. |
| `404 Not Found` | `Session Missing` | Querying `/session/status/{id}` for an uninitialized session. | Ingest an observation via `/stream/observe` or initialize session first. |
| `422 Unprocessable Entity` | `Pydantic Validation Error` | Value out of physical range (e.g. pressure < -50 mmHg, negative flow/force). | Clamp digital twin physical variables to valid sensor operating bounds. |
| `503 Service Unavailable` | `Model Loading Failure` | Persisted model artifact missing on server or initialization failure. | Check server logs; verify `models/` directory contains joblib files. |

---

## 8. Regulatory, Safety & Attribution Notice

> [!CAUTION]
> **Advisory Medical AI Notice:**
> All outputs produced by this Machine Learning subsystem (including classifications, sensor health scores, error estimates, and conformal intervals) are strictly **ADVISORY**.
> Under medical device safety principles:
> 1. The ML subsystem shall **NEVER** autonomously recalibrate transducer offsets or modify infusion pump motor velocity without independent biomedical engineering validation and clinician sign-off.
> 2. True sensor accuracy cannot be confirmed without external physical reference standards (e.g. calibrated fluid columns or NIST-traceable pressure manometers).
> 3. All simulation evaluations are designated `SYNTHETIC_DATA`. Physical hardware validation remains pending.

