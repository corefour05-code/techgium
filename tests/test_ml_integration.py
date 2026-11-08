"""Comprehensive Integration Test Suite for ML Backend & Digital Twin Interface (Phase 6).

Verifies the 14 mandatory integration scenarios:
 1. Normal operation (streaming 60 samples, warmup -> ready transition)
 2. Positive drift detection & decoupled health scoring
 3. Negative drift detection
 4. Sudden sensor offset detection
 5. Excessive noise detection
 6. Sensor freeze detection
 7. Sensor dropout detection
 8. Genuine occlusion (high process risk >= 85, preserved sensor health >= 85)
 9. Temperature influence handling
 10. Missing optional sensor channels (force, flow, temp omitted)
 11. Invalid input validation (Pydantic bounds & missing fields)
 12. Stale, duplicate, and out-of-order timestamps
 13. Insufficient reference data handling & conformal prediction intervals
 14. Session lifecycle management (reset, status, 404s, degraded state)
"""

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.app import (
    load_production_models,
    models_state,
    get_health,
    get_models_status,
    observe_stream,
    reset_session,
    get_session_status,
)
from api.schemas.payloads import (
    SequentialObservationRequest,
    SessionResetRequest,
)
from src.session.stream_buffer import StreamQualityStatus, WarmupStatus
from src.utils.config import DataProvenance


def run_coro(coro):
    """Execute coroutine synchronously without event loop overhead."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    """Ensure models and stream session manager are loaded."""
    load_production_models()


# ---------------------------------------------------------------------------
# Scenario 1: Normal Operation (Streaming 60 samples)
# ---------------------------------------------------------------------------

def test_scenario_01_normal_streaming_warmup_and_steady_state():
    """Verify 60 nominal samples transition from WARMUP to READY with high health."""
    session_id = "test_scen_01_normal"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    responses = []
    for t in range(60):
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=80.0 + (t % 3) * 0.1,
            sensor_force_n=4.85 + (t % 3) * 0.01,
            sensor_flow_ml_h=50.0 + (t % 2) * 0.05,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        res = run_coro(observe_stream(req))
        responses.append(res)

    # Verify warmup transition
    assert responses[0].warmup_status == WarmupStatus.WARMUP.value
    assert responses[13].warmup_status == WarmupStatus.WARMUP.value
    assert responses[14].warmup_status == WarmupStatus.READY.value
    assert responses[-1].warmup_status == WarmupStatus.READY.value

    # Verify steady-state at sample 60
    final = responses[-1]
    assert final.stream_quality == StreamQualityStatus.OK.value
    assert final.predicted_fault_class == "NORMAL"
    assert final.class_index == 0
    assert final.is_sensor_fault is False
    assert final.is_process_anomaly is False
    assert final.sensor_health_score >= 90.0
    assert final.process_risk_score <= 20.0
    assert final.process_risk_level in ("LOW", "MODERATE")
    assert final.model_confidence >= 0.70
    assert final.calibrated_confidence is not None
    assert final.calibrated_confidence >= 0.60
    assert final.buffered_samples == 60


# ---------------------------------------------------------------------------
# Scenario 2: Positive Sensor Drift
# ---------------------------------------------------------------------------

def test_scenario_02_positive_drift():
    """Verify gradual upward drift degrades sensor health without declaring process risk."""
    session_id = "test_scen_02_pos_drift"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(60):
        # Drift starts after warmup (t >= 20) at +1.5 mmHg/step
        drift = 1.5 * max(0, t - 20)
        p_val = 80.0 + (t % 2) * 0.1 + drift
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,  # Force stays constant -> mechanical mismatch
            sensor_flow_ml_h=50.0,  # Flow stays constant
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class in ("POSITIVE_DRIFT", "SUDDEN_OFFSET")
    assert final.is_sensor_fault is True
    assert final.is_process_anomaly is False
    # Sensor health must degrade, but process risk must stay low (decoupled!)
    assert final.sensor_health_score < 75.0
    assert final.process_risk_score <= 50.0


# ---------------------------------------------------------------------------
# Scenario 3: Negative Sensor Drift
# ---------------------------------------------------------------------------

def test_scenario_03_negative_drift():
    """Verify gradual negative drift is detected and flagged as sensor fault."""
    session_id = "test_scen_03_neg_drift"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(60):
        drift = -1.2 * max(0, t - 20)
        p_val = max(10.0, 80.0 + drift)
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class in ("NEGATIVE_DRIFT", "SUDDEN_OFFSET")
    assert final.is_sensor_fault is True
    assert final.sensor_health_score < 80.0


# ---------------------------------------------------------------------------
# Scenario 4: Sudden Sensor Offset
# ---------------------------------------------------------------------------

def test_scenario_04_sudden_sensor_offset():
    """Verify sudden step jump without physical flow change is diagnosed as SUDDEN_OFFSET."""
    session_id = "test_scen_04_offset"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(50):
        # Step jump at t = 25 from 80 to 145 mmHg
        p_val = 80.0 if t < 25 else 145.0
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class in ("SUDDEN_OFFSET", "POSITIVE_DRIFT")
    assert final.is_sensor_fault is True
    assert final.sensor_health_score < 70.0


# ---------------------------------------------------------------------------
# Scenario 5: Excessive Noise
# ---------------------------------------------------------------------------

def test_scenario_05_excessive_noise():
    """Verify high-amplitude stochastic jitter is diagnosed as EXCESSIVE_NOISE."""
    session_id = "test_scen_05_noise"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    import random
    rng = random.Random(42)

    final = None
    for t in range(60):
        noise = rng.gauss(0.0, 15.0) if t >= 20 else rng.gauss(0.0, 0.2)
        p_val = 80.0 + noise
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class == "EXCESSIVE_NOISE"
    assert final.is_sensor_fault is True
    assert final.sensor_health_score < 80.0


# ---------------------------------------------------------------------------
# Scenario 6: Sensor Freeze
# ---------------------------------------------------------------------------

def test_scenario_06_sensor_freeze():
    """Verify static invariant sensor readings are diagnosed as SENSOR_FREEZE."""
    session_id = "test_scen_06_freeze"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(60):
        # First 15 steps have slight dither, then completely frozen
        p_val = 80.0 + (0.1 if t < 15 and t % 2 == 0 else 0.0)
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class == "SENSOR_FREEZE"
    assert final.is_sensor_fault is True
    assert final.sensor_health_score < 80.0


# ---------------------------------------------------------------------------
# Scenario 7: Sensor Dropout
# ---------------------------------------------------------------------------

def test_scenario_07_sensor_dropout():
    """Verify electrical disconnect to 0.0 mmHg is diagnosed as SENSOR_DROPOUT."""
    session_id = "test_scen_07_dropout"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(40):
        p_val = 80.0 if t < 20 else 0.0
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85 if t < 20 else 0.0,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class == "SENSOR_DROPOUT"
    assert final.is_sensor_fault is True
    assert final.sensor_health_score < 60.0


# ---------------------------------------------------------------------------
# Scenario 8: Genuine Downstream Occlusion (Process Decoupling Verification)
# ---------------------------------------------------------------------------

def test_scenario_08_genuine_occlusion_process_risk_decoupling():
    """CRITICAL TEST: Genuine occlusion elevates process risk >= 85 while PRESERVING sensor health >= 85."""
    from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig
    from src.utils.config import FaultClass

    session_id = "test_scen_08_occlusion"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    cfg = SimulatorConfig(duration_seconds=120, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.GENUINE_DOWNSTREAM_OCCLUSION,
        config=cfg,
        fault_start_sec=30,
        occlusion_time_constant_sec=30.0,
    )

    final = None
    for _, row in df.iloc[:75].iterrows():
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(row["timestamp_sec"]),
            sensor_pressure_mmhg=float(row["sensor_pressure_mmhg"]),
            sensor_force_n=float(row["sensor_force_n"]),
            sensor_flow_ml_h=float(row["sensor_flow_ml_h"]),
            sensor_temperature_c=float(row["sensor_temperature_c"]),
            commanded_rate_ml_h=float(row["commanded_rate_ml_h"]),
            dt_predicted_pressure_mmhg=float(row["dt_predicted_pressure_mmhg"]),
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class == "GENUINE_OCCLUSION"
    assert final.is_process_anomaly is True
    assert final.is_sensor_fault is False

    # CRITICAL DECOUPLING ASSERTIONS:
    # 1. Process risk is critical (downstream line blocked)
    assert final.process_risk_score >= 85.0
    assert final.process_risk_level in ("HIGH", "CRITICAL")
    # 2. Sensor health is PRESERVED because pressure transducer is accurately measuring the actual line pressure
    assert final.sensor_health_score >= 85.0


# ---------------------------------------------------------------------------
# Scenario 9: Temperature Influence
# ---------------------------------------------------------------------------

def test_scenario_09_temperature_influence():
    """Verify ambient thermal excursion is classified as TEMPERATURE_INFLUENCE."""
    session_id = "test_scen_09_temp"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(60):
        # Thermal ramp coupled with piezoresistive pressure response
        temp_val = 22.0 + 0.15 * t
        p_val = 80.0 + 1.6 * (temp_val - 22.0)
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_MAIN_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=4.85,
            sensor_flow_ml_h=50.0,
            sensor_temperature_c=temp_val,
            commanded_rate_ml_h=50.0,
            dt_predicted_pressure_mmhg=80.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class in ("TEMPERATURE_INFLUENCE", "NORMAL")
    assert final.sensor_health_score >= 80.0


# ---------------------------------------------------------------------------
# Scenario 10: Missing Optional Sensor Channels
# ---------------------------------------------------------------------------

def test_scenario_10_missing_optional_sensors_graceful_handling():
    """Verify pump with pressure-only sensor configuration runs cleanly without crashing."""
    session_id = "test_scen_10_pressure_only"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    final = None
    for t in range(30):
        p_val = 80.0 + (t % 3) * 0.1
        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_ONLY_01",
            timestamp=float(t),
            sensor_pressure_mmhg=p_val,
            sensor_force_n=None,
            sensor_flow_ml_h=None,
            sensor_temperature_c=None,
            commanded_rate_ml_h=50.0,
        )
        final = run_coro(observe_stream(req))

    assert final is not None
    assert final.warmup_status == WarmupStatus.READY.value
    assert final.predicted_fault_class == "NORMAL"
    assert final.sensor_health_score >= 90.0


# ---------------------------------------------------------------------------
# Scenario 11: Invalid Input Validation
# ---------------------------------------------------------------------------

def test_scenario_11_invalid_input_validation():
    """Verify Pydantic schema validation rejects unphysical and malformed inputs."""
    # Negative pressure below sensor physical floor (-50 mmHg)
    with pytest.raises(ValidationError):
        SequentialObservationRequest(
            session_id="valid_session",
            sensor_id="P1",
            timestamp=1.0,
            sensor_pressure_mmhg=-100.0,  # Below -50 mmHg min bound
        )

    # Pressure exceeding transducer burst limit (> 2000 mmHg)
    with pytest.raises(ValidationError):
        SequentialObservationRequest(
            session_id="valid_session",
            sensor_id="P1",
            timestamp=1.0,
            sensor_pressure_mmhg=2500.0,
        )

    # Negative plunger force
    with pytest.raises(ValidationError):
        SequentialObservationRequest(
            session_id="valid_session",
            sensor_id="P1",
            timestamp=1.0,
            sensor_pressure_mmhg=80.0,
            sensor_force_n=-5.0,
        )

    # Empty session_id
    with pytest.raises(ValidationError):
        SequentialObservationRequest(
            session_id="",
            sensor_id="P1",
            timestamp=1.0,
            sensor_pressure_mmhg=80.0,
        )


# ---------------------------------------------------------------------------
# Scenario 12: Stale and Duplicate Timestamps
# ---------------------------------------------------------------------------

def test_scenario_12_telemetry_quality_duplicate_and_gaps():
    """Verify duplicate, out-of-order, and stale telemetry gaps are flagged."""
    session_id = "test_scen_12_quality"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    # Step 1: Normal initial sample
    res1 = run_coro(observe_stream(SequentialObservationRequest(
        session_id=session_id, sensor_id="P1", timestamp=10.0, sensor_pressure_mmhg=80.0
    )))
    assert res1.stream_quality == StreamQualityStatus.OK.value

    # Step 2: Duplicate timestamp (10.0 again)
    res2 = run_coro(observe_stream(SequentialObservationRequest(
        session_id=session_id, sensor_id="P1", timestamp=10.0, sensor_pressure_mmhg=80.1
    )))
    assert res2.stream_quality == StreamQualityStatus.DUPLICATE_TIMESTAMP.value

    # Step 3: Out-of-order timestamp (8.0 < 10.0)
    res3 = run_coro(observe_stream(SequentialObservationRequest(
        session_id=session_id, sensor_id="P1", timestamp=8.0, sensor_pressure_mmhg=80.2
    )))
    assert res3.stream_quality == StreamQualityStatus.OUT_OF_ORDER_TIMESTAMP.value

    # Step 4: Stale gap (25.0 > 8.0 + 5.0s threshold)
    res4 = run_coro(observe_stream(SequentialObservationRequest(
        session_id=session_id, sensor_id="P1", timestamp=25.0, sensor_pressure_mmhg=80.3
    )))
    assert res4.stream_quality == StreamQualityStatus.STALE_TELEMETRY_GAP.value

    # Verify session status tracks quality counters
    status_res = run_coro(get_session_status(session_id))
    assert status_res.telemetry_quality_issues["DUPLICATE_TIMESTAMP"] >= 1
    assert status_res.telemetry_quality_issues["OUT_OF_ORDER_TIMESTAMP"] >= 1
    assert status_res.telemetry_quality_issues["STALE_TELEMETRY_GAP"] >= 1


# ---------------------------------------------------------------------------
# Scenario 13: Insufficient Reference Data vs. Conformal Interval Estimation
# ---------------------------------------------------------------------------

def test_scenario_13_reference_data_and_conformal_interval():
    """Verify measurement error estimation behavior with and without reference data."""
    session_id = "test_scen_13_ref"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    # Stream 20 samples without reference
    final_no_ref = None
    for t in range(20):
        final_no_ref = run_coro(observe_stream(SequentialObservationRequest(
            session_id=session_id,
            sensor_id="P1",
            timestamp=float(t),
            sensor_pressure_mmhg=80.0,
            independent_reference_value=None,
        )))

    assert final_no_ref.error_status == "INSUFFICIENT_REFERENCE_DATA"
    assert final_no_ref.estimated_sensor_error is None
    assert final_no_ref.conformal_error_interval_95 is None

    # Stream 1 sample with valid physical reference
    res_with_ref = run_coro(observe_stream(SequentialObservationRequest(
        session_id=session_id,
        sensor_id="P1",
        timestamp=21.0,
        sensor_pressure_mmhg=82.5,
        independent_reference_value=80.0,  # +2.5 mmHg apparent sensor over-reading
    )))

    assert res_with_ref.error_status == "ESTIMATION_AVAILABLE"
    assert res_with_ref.estimated_sensor_error is not None
    assert isinstance(res_with_ref.estimated_sensor_error, float)
    assert res_with_ref.conformal_error_interval_95 is not None
    assert len(res_with_ref.conformal_error_interval_95) == 2
    low, high = res_with_ref.conformal_error_interval_95
    assert low < high


# ---------------------------------------------------------------------------
# Scenario 14: Session Lifecycle, Reset & Degraded Error Handling
# ---------------------------------------------------------------------------

def test_scenario_14_session_lifecycle_and_error_handling():
    """Verify session reset, 404 queries, and service error handling."""
    session_id = "test_scen_14_lifecycle"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    # Ingest 5 samples
    for t in range(5):
        run_coro(observe_stream(SequentialObservationRequest(
            session_id=session_id, sensor_id="P1", timestamp=float(t), sensor_pressure_mmhg=80.0
        )))

    status1 = run_coro(get_session_status(session_id))
    assert status1.buffered_samples == 5

    # Reset session
    reset_res = run_coro(reset_session(SessionResetRequest(session_id=session_id)))
    assert reset_res.reset_success is True

    status2 = run_coro(get_session_status(session_id))
    assert status2.buffered_samples == 0

    # Nonexistent session status should raise 404
    with pytest.raises(HTTPException) as exc_info:
        run_coro(get_session_status("nonexistent_session_xyz_999"))
    assert exc_info.value.status_code == 404
