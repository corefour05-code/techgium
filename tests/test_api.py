"""Comprehensive Unit and Integration Tests for FastAPI Endpoints and Pydantic Schemas.

Covers:
- GET /health
- GET /models/status
- POST /predict/fault (Normal, Dropout, Occlusion, SHAP explanations)
- POST /predict/sensor-error (Valid reference, Missing reference, Unknown sensor type)
- POST /predict/sensor-health (Decoupled health & risk, occlusion handling)
- POST /predict/batch
- Error handling (Invalid input, empty payload, 422 validation errors, 400 unknown sensor)
"""

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig
from src.utils.config import FaultClass
from api.app import (
    load_production_models,
    get_health,
    get_models_status,
    predict_fault,
    predict_sensor_error,
    predict_sensor_health,
    predict_batch,
)
from api.schemas.payloads import (
    FaultPredictionRequest,
    SensorErrorRequest,
    SensorHealthRequest,
    BatchPredictionRequest,
)


def run_coro(coro):
    """Execute coroutine synchronously without external event loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


@pytest.fixture(scope="module", autouse=True)
def init_models():
    """Ensure production models and engines are loaded before running tests."""
    load_production_models()


# ---------------------------------------------------------------------------
# 1. Health & Status Endpoints
# ---------------------------------------------------------------------------

def test_api_health_endpoint():
    """Verify /health endpoint returns valid HealthResponse."""
    res = run_coro(get_health())
    assert res.status in ("OK", "DEGRADED")
    assert res.service == "Infusion Pump ML Intelligence Engine"
    assert "isolation_forest" in res.models_loaded
    assert "xgboost_classifier" in res.models_loaded
    assert "xgboost_regressor_pressure" in res.models_loaded
    assert "safety_disclaimer" in res.safety_disclaimer.lower() or "medical" in res.safety_disclaimer.lower()


def test_api_models_status_endpoint():
    """Verify /models/status returns model metadata and feature information."""
    res = run_coro(get_models_status())
    assert "xgboost_classifier" in res.models
    assert "xgboost_regressor_pressure" in res.models
    clf_info = res.models["xgboost_classifier"]
    assert clf_info["num_features"] > 0
    assert len(clf_info["supported_classes"]) == 9


# ---------------------------------------------------------------------------
# 2. Fault Classification Endpoint
# ---------------------------------------------------------------------------

def test_api_predict_fault_normal():
    """Verify normal window returns NORMAL class with confidence and probabilities."""
    req = FaultPredictionRequest(
        sensor_id="P_LINE_01",
        sensor_pressure_mmhg=[80.0 + (i % 2) * 0.2 for i in range(30)],
        sensor_force_n=[4.85 for _ in range(30)],
        sensor_flow_ml_h=[50.0 for _ in range(30)],
        commanded_rate_ml_h=50.0,
    )
    res = run_coro(predict_fault(req))
    assert res.predicted_fault_class == "NORMAL"
    assert res.model_confidence >= 0.70
    assert not res.is_sensor_fault
    assert not res.is_process_anomaly
    assert "NORMAL" in res.class_probabilities
    assert res.shap_explanation is not None


def test_api_predict_fault_dropout():
    """Verify disconnected sensor (0.0 reading) returns SENSOR_DROPOUT with SHAP explanation."""
    req = FaultPredictionRequest(
        sensor_id="P_LINE_01",
        sensor_pressure_mmhg=[0.0] * 30,
        sensor_force_n=[4.85] * 30,
        sensor_flow_ml_h=[50.0] * 30,
    )
    res = run_coro(predict_fault(req))
    assert res.predicted_fault_class == "SENSOR_DROPOUT"
    assert res.is_sensor_fault
    assert res.shap_explanation is not None
    assert len(res.shap_explanation["top_contributing_features"]) > 0


def test_api_predict_fault_occlusion():
    """Verify occlusion pattern returns GENUINE_OCCLUSION process anomaly."""
    # To avoid triggering freeze (flat line), give slight realistic variance
    req = FaultPredictionRequest(
        sensor_id="P_LINE_01",
        sensor_pressure_mmhg=[320.0 + (i % 3) * 0.4 for i in range(30)],
        sensor_force_n=[14.8 + (i % 2) * 0.05 for i in range(30)],
        sensor_flow_ml_h=[0.0] * 30,
    )
    res = run_coro(predict_fault(req))
    assert res.predicted_fault_class == "GENUINE_OCCLUSION"
    assert res.is_process_anomaly
    assert not res.is_sensor_fault


# ---------------------------------------------------------------------------
# 3. Sensor Error Regression Endpoint
# ---------------------------------------------------------------------------

def test_api_predict_sensor_error_with_reference():
    """Verify regression estimation when independent reference is provided."""
    cfg = SimulatorConfig(duration_seconds=300, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.SUDDEN_OFFSET,
        config=cfg,
        fault_start_sec=100,
        offset_magnitude_mmhg=20.0,
    )
    last_p = float(df_run["sensor_pressure_mmhg"].iloc[-1])
    ref_p = float(df_run["reference_pressure_mmhg"].iloc[-1])
    expected_err = last_p - ref_p

    req = SensorErrorRequest(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        sensor_pressure_mmhg=df_run["sensor_pressure_mmhg"].iloc[-60:].tolist(),
        sensor_force_n=df_run["sensor_force_n"].iloc[-60:].tolist() if "sensor_force_n" in df_run else None,
        sensor_flow_ml_h=df_run["sensor_flow_ml_h"].iloc[-60:].tolist() if "sensor_flow_ml_h" in df_run else None,
        commanded_rate_ml_h=float(df_run["commanded_rate_ml_h"].iloc[-1]),
        dt_predicted_pressure_mmhg=float(df_run["dt_predicted_pressure_mmhg"].iloc[-1]),
        observed_sensor_value=last_p,
        independent_reference_value=ref_p,
    )
    res = run_coro(predict_sensor_error(req))
    assert res.status == "ESTIMATION_AVAILABLE"
    assert res.units == "mmHg"
    assert res.estimated_sensor_error == pytest.approx(expected_err, abs=4.0)
    assert res.estimated_calibrated_value == pytest.approx(ref_p, abs=4.0)
    assert len(res.confidence_interval_95) == 2


def test_api_predict_sensor_error_missing_reference():
    """Verify missing reference produces INSUFFICIENT_REFERENCE_DATA."""
    req = SensorErrorRequest(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        sensor_pressure_mmhg=[80.0] * 15,
        independent_reference_value=None,
        observed_sensor_value=80.0,
    )
    res = run_coro(predict_sensor_error(req))
    assert res.status == "INSUFFICIENT_REFERENCE_DATA"
    assert res.estimated_sensor_error is None


def test_api_predict_sensor_error_unknown_sensor_type():
    """Verify unknown sensor type raises HTTP 400 Bad Request."""
    req = SensorErrorRequest(
        sensor_id="RAD_01",
        sensor_type="radiation",
        sensor_pressure_mmhg=[80.0] * 10,
    )
    with pytest.raises(HTTPException) as exc_info:
        run_coro(predict_sensor_error(req))
    assert exc_info.value.status_code == 400
    assert "unsupported sensor type" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# 4. Integrated Sensor Health Endpoint
# ---------------------------------------------------------------------------

def test_api_predict_sensor_health_normal():
    """Verify normal window produces HEALTHY status, high health score, low process risk."""
    req = SensorHealthRequest(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        sensor_pressure_mmhg=[80.0 + (i % 2) * 0.1 for i in range(20)],
        sensor_force_n=[4.85 for _ in range(20)],
        sensor_flow_ml_h=[50.0 for _ in range(20)],
        commanded_rate_ml_h=50.0,
    )
    res = run_coro(predict_sensor_health(req))
    assert res.status == "HEALTHY"
    assert res.sensor_health_score >= 90.0
    assert res.process_risk_score <= 20.0
    assert res.process_risk_level == "LOW"


def test_api_predict_sensor_health_occlusion_decoupling():
    """Verify genuine occlusion maintains high sensor health while process risk surges to CRITICAL."""
    # Use slight variance to avoid constant flatline sensor freeze trigger
    f_val = (310.0 * 0.0133322 * 3.14) + 1.5
    req = SensorHealthRequest(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        sensor_pressure_mmhg=[310.0 + (i % 3) * 0.4 for i in range(20)],
        sensor_force_n=[f_val + (i % 2) * 0.05 for i in range(20)],
        sensor_flow_ml_h=[0.0] * 20,
        commanded_rate_ml_h=50.0,
    )
    res = run_coro(predict_sensor_health(req))
    assert res.sensor_health_score >= 90.0  # Sensor is healthy!
    assert res.process_risk_score >= 90.0   # Process is critical!
    assert res.process_risk_level == "CRITICAL"
    assert "CLINICAL ALERT" in res.recommended_action


# ---------------------------------------------------------------------------
# 5. Batch Prediction & Validation Failure Handling
# ---------------------------------------------------------------------------

def test_api_predict_batch():
    """Verify predict_batch processes multiple window requests."""
    batch_req = BatchPredictionRequest(
        items=[
            FaultPredictionRequest(
                sensor_id="P_01",
                sensor_pressure_mmhg=[80.0 + (i % 2) * 0.2 for i in range(15)],
                sensor_force_n=[4.85 for _ in range(15)],
                sensor_flow_ml_h=[50.0 for _ in range(15)],
            ),
            FaultPredictionRequest(
                sensor_id="P_02",
                sensor_pressure_mmhg=[0.0] * 15,
                sensor_force_n=[4.85] * 15,
                sensor_flow_ml_h=[50.0] * 15,
            ),
        ]
    )
    res = run_coro(predict_batch(batch_req))
    assert res.total_processed == 2
    assert len(res.results) == 2
    assert res.results[0].predicted_fault_class == "NORMAL"
    assert res.results[1].predicted_fault_class == "SENSOR_DROPOUT"


def test_api_invalid_input_validation_error():
    """Verify empty input fails Pydantic schema validation with ValidationError."""
    with pytest.raises(ValidationError):
        FaultPredictionRequest(
            sensor_id="P_01",
            sensor_pressure_mmhg=[],  # Violates min_length=1
        )
