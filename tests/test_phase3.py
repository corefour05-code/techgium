"""Comprehensive Unit and Scenario Integration Tests for Phase 3.

Covers:
- Multivariate Anomaly Detection using Isolation Forest (fit, predict, save, load)
- Multi-channel Cross-Sensor Consistency Engine across all key physical scenarios:
  1. Normal operation
  2. Gradual positive sensor drift
  3. Gradual negative sensor drift
  4. Sudden sensor offset
  5. Sensor freeze (ADC lock)
  6. Sensor dropout (disconnection/rail)
  7. Genuine downstream occlusion
  8. Environmental temperature influence
  9. Missing sensor channels (single channel -> INSUFFICIENT_EVIDENCE)
  10. Conflicting sensor evidence (ambiguous multi-sensor state)
- Sensor Health Assessment Engine (evidence deductions, status, recommendations, safety disclaimer)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import SensorType, DataProvenance, FaultClass, SensorHealthStatus
from src.anomaly.isolation_forest import MultivariateAnomalyDetector
from src.sensor_fusion.consistency_engine import (
    CrossSensorConsistencyEngine,
    ConsistencyVerdict,
    ConsistencyAssessment,
)
from src.health.sensor_health import SensorHealthEngine, HealthAssessmentReport
from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig


# ---------------------------------------------------------------------------
# 1. Isolation Forest Detector Tests
# ---------------------------------------------------------------------------

def test_isolation_forest_fit_predict_and_persistence(tmp_path):
    """Verify Isolation Forest training, scoring bounds, and joblib serialization."""
    np.random.seed(42)
    # Generate in-control baseline feature vectors
    n_samples = 200
    baseline_features = pd.DataFrame({
        "rolling_mean": np.random.normal(50.0, 1.0, n_samples),
        "rolling_std": np.random.normal(1.2, 0.2, n_samples),
        "rate_of_change": np.random.normal(0.0, 0.05, n_samples),
    })

    detector = MultivariateAnomalyDetector(contamination=0.02, n_estimators=50, random_state=42)
    detector.fit(baseline_features)
    assert detector.is_fitted_
    assert len(detector.feature_names_) == 3

    # Predict on normal test data
    normal_test = pd.DataFrame({
        "rolling_mean": np.random.normal(50.0, 1.0, 50),
        "rolling_std": np.random.normal(1.2, 0.2, 50),
        "rate_of_change": np.random.normal(0.0, 0.05, 50),
    })
    preds_normal = detector.predict(normal_test)
    assert "anomaly_detected" in preds_normal.columns
    assert "anomaly_severity" in preds_normal.columns
    assert "confidence" in preds_normal.columns
    assert preds_normal["anomaly_severity"].between(0.0, 1.0).all()
    assert preds_normal["confidence"].between(0.0, 1.0).all()

    # Predict on extreme outlier data
    outlier_test = pd.DataFrame({
        "rolling_mean": [150.0, 200.0, 300.0],
        "rolling_std": [25.0, 30.0, 40.0],
        "rate_of_change": [10.0, 15.0, 20.0],
    })
    preds_outlier = detector.predict(outlier_test)
    assert preds_outlier["anomaly_detected"].all()
    assert (preds_outlier["anomaly_severity"] > 0.6).all()

    # Save and reload model
    model_file = tmp_path / "test_iso_forest.joblib"
    saved_path = detector.save(model_file)
    assert saved_path.exists()

    loaded_detector = MultivariateAnomalyDetector.load(saved_path)
    assert loaded_detector.is_fitted_
    loaded_preds = loaded_detector.predict(outlier_test)
    np.testing.assert_allclose(
        preds_outlier["anomaly_severity"].values,
        loaded_preds["anomaly_severity"].values,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# 2. Scenario: Normal Operation
# ---------------------------------------------------------------------------

def test_scenario_normal_operation():
    """Verify clean in-control infusion run produces NORMAL verdict and HEALTHY status."""
    cfg = SimulatorConfig(duration_seconds=300, random_seed=42)
    df = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)

    engine = CrossSensorConsistencyEngine()
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.NORMAL
    assert not assessment.is_sensor_fault
    assert not assessment.is_process_anomaly
    assert assessment.confidence >= 0.85

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    assert health.status == SensorHealthStatus.HEALTHY
    assert health.health_score >= 90.0
    assert not health.drift_detected
    assert not health.anomaly_detected
    assert "Continue normal operation" in health.recommended_action


# ---------------------------------------------------------------------------
# 3. Scenario: Gradual Positive Drift
# ---------------------------------------------------------------------------

def test_scenario_gradual_positive_drift():
    """Verify that pressure drift with normal force and flow is flagged as SENSOR_DRIFT_SUSPECTED."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.POSITIVE_DRIFT,
        config=cfg,
        fault_start_sec=100,
        drift_rate_mmhg_per_hour=150.0,
    )

    engine = CrossSensorConsistencyEngine(pressure_drift_threshold_mmhg=15.0)
    assessment = engine.assess_run(df, ewma_alarm_active=True)

    assert assessment.verdict == ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED
    assert assessment.is_sensor_fault
    assert not assessment.is_process_anomaly

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        ewma_drift_detected=True,
        ewma_drift_direction="POSITIVE",
    )

    assert health.status == SensorHealthStatus.DRIFT_SUSPECTED
    assert health.health_score <= 60.0
    assert "statistical_drift" in health.deductions
    assert health.drift_direction == "POSITIVE"
    assert "calibration check" in health.recommended_action.lower()


# ---------------------------------------------------------------------------
# 4. Scenario: Gradual Negative Drift
# ---------------------------------------------------------------------------

def test_scenario_gradual_negative_drift():
    """Verify negative sensor drift triggers drift verdict and appropriate direction."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.NEGATIVE_DRIFT,
        config=cfg,
        fault_start_sec=100,
        drift_rate_mmhg_per_hour=150.0,
    )

    engine = CrossSensorConsistencyEngine(pressure_drift_threshold_mmhg=15.0)
    assessment = engine.assess_run(df, cusum_alarm_active=True)

    assert assessment.verdict == ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED
    assert assessment.is_sensor_fault

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        cusum_drift_detected=True,
        cusum_drift_direction="NEGATIVE",
    )

    assert health.status == SensorHealthStatus.DRIFT_SUSPECTED
    assert health.drift_direction == "NEGATIVE"


# ---------------------------------------------------------------------------
# 5. Scenario: Sudden Offset
# ---------------------------------------------------------------------------

def test_scenario_sudden_offset():
    """Verify sudden sensor offset (step change) is caught by cross-sensor validation."""
    cfg = SimulatorConfig(duration_seconds=400, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.SUDDEN_OFFSET,
        config=cfg,
        fault_start_sec=100,
        offset_magnitude_mmhg=40.0,
    )

    engine = CrossSensorConsistencyEngine()
    assessment = engine.assess_run(df, ewma_alarm_active=True)

    # Because force and flow remain nominal while pressure jumped by 40 mmHg,
    # the engine detects sensor divergence
    assert assessment.verdict == ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED
    assert assessment.is_sensor_fault
    assert not assessment.is_process_anomaly


# ---------------------------------------------------------------------------
# 6. Scenario: Genuine Downstream Occlusion
# ---------------------------------------------------------------------------

def test_scenario_genuine_downstream_occlusion():
    """Verify true occlusion (pressure rise + force surge + flow collapse) confirms GENUINE_OCCLUSION_CONFIRMED."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.GENUINE_DOWNSTREAM_OCCLUSION,
        config=cfg,
        fault_start_sec=150,
        occlusion_time_constant_sec=80.0,
    )

    engine = CrossSensorConsistencyEngine(pressure_occlusion_threshold_mmhg=250.0)
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED
    assert not assessment.is_sensor_fault
    assert assessment.is_process_anomaly
    assert assessment.confidence >= 0.90

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    # The sensor is functioning correctly (it correctly measured the occlusion)
    # but the process state is alarming
    assert health.status == SensorHealthStatus.WARNING
    assert "CLINICAL ALERT: Downstream line occlusion confirmed" in health.recommended_action
    assert "process_occlusion" in health.deductions


# ---------------------------------------------------------------------------
# 7. Scenario: Sensor Freeze
# ---------------------------------------------------------------------------

def test_scenario_sensor_freeze():
    """Verify locked ADC readings trigger SENSOR_FREEZE_SUSPECTED."""
    cfg = SimulatorConfig(duration_seconds=200, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.SENSOR_FREEZE,
        config=cfg,
        fault_start_sec=100,
    )

    engine = CrossSensorConsistencyEngine(min_consecutive_freeze_steps=15)
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.SENSOR_FREEZE_SUSPECTED
    assert assessment.is_sensor_fault
    assert not assessment.is_process_anomaly

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    assert health.status == SensorHealthStatus.FAULT_SUSPECTED
    assert health.health_score <= 30.0
    assert "sensor_freeze" in health.deductions
    assert "transducer locked" in health.recommended_action.lower()


# ---------------------------------------------------------------------------
# 8. Scenario: Sensor Dropout
# ---------------------------------------------------------------------------

def test_scenario_sensor_dropout():
    """Verify sensor rail/zero dropout triggers SENSOR_DROPOUT_SUSPECTED."""
    cfg = SimulatorConfig(duration_seconds=200, random_seed=42)
    df = generate_infusion_run(
        scenario=FaultClass.SENSOR_DROPOUT,
        config=cfg,
        fault_start_sec=100,
    )

    engine = CrossSensorConsistencyEngine(min_consecutive_dropout_steps=5)
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.SENSOR_DROPOUT_SUSPECTED
    assert assessment.is_sensor_fault

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    assert health.status == SensorHealthStatus.FAULT_SUSPECTED
    assert health.health_score <= 25.0
    assert "sensor_dropout" in health.deductions
    assert "disconnected" in health.recommended_action.lower()


# ---------------------------------------------------------------------------
# 9. Scenario: Environmental Temperature Influence
# ---------------------------------------------------------------------------

def test_scenario_environmental_temperature_influence():
    """Verify ambient temperature correlation triggers TEMPERATURE_INFLUENCE_SUSPECTED."""
    n = 60
    # Simulate strong thermal coupling: pressure sensor reading tracks ambient temperature rise
    temp_profile = 22.0 + 0.1 * np.arange(n)
    thermal_p_drift = 50.0 + 3.0 * (temp_profile - 22.0)  # r ~ 1.0

    df = pd.DataFrame({
        "sensor_pressure_mmhg": thermal_p_drift,
        "dt_predicted_pressure_mmhg": np.full(n, 50.0),
        "sensor_temperature_c": temp_profile,
        "sensor_force_n": np.full(n, 2.8),     # Ambiguous force
        "sensor_flow_ml_h": np.full(n, 30.0),  # Ambiguous flow
        "commanded_rate_ml_h": np.full(n, 50.0),
    })

    engine = CrossSensorConsistencyEngine(pressure_drift_threshold_mmhg=10.0)
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.TEMPERATURE_INFLUENCE_SUSPECTED
    assert assessment.is_sensor_fault
    assert any("thermal correlation" in e.lower() for e in assessment.evidence)

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    assert health.status == SensorHealthStatus.WARNING
    assert "thermal_influence" in health.deductions
    assert "temperature" in health.recommended_action.lower()


# ---------------------------------------------------------------------------
# 10. Scenario: Missing Sensor Channels (Single Channel Safeguard)
# ---------------------------------------------------------------------------

def test_scenario_missing_sensor_channels_safeguard():
    """Verify that single-channel architectures return INSUFFICIENT_EVIDENCE rather than guessing."""
    cfg = SimulatorConfig(
        duration_seconds=300,
        enable_pressure=True,
        enable_force=False,
        enable_flow=False,
        enable_motor_encoder=False,
        enable_temperature=False,
        random_seed=42,
    )
    df = generate_infusion_run(
        scenario=FaultClass.POSITIVE_DRIFT,
        config=cfg,
        fault_start_sec=50,
        drift_rate_mmhg_per_hour=150.0,
    )

    assert "sensor_force_n" not in df.columns
    assert "sensor_flow_ml_h" not in df.columns

    engine = CrossSensorConsistencyEngine(pressure_drift_threshold_mmhg=15.0)
    assessment = engine.assess_run(
        df,
        pressure_col="sensor_pressure_mmhg",
        force_col="sensor_force_n",  # Not in df
        flow_col="sensor_flow_ml_h",  # Not in df
        ewma_alarm_active=True,
    )

    # Inviolable Rule: Without redundant secondary channels, cannot distinguish occlusion from drift
    assert assessment.verdict == ConsistencyVerdict.INSUFFICIENT_EVIDENCE
    assert not assessment.is_sensor_fault
    assert not assessment.is_process_anomaly
    assert any("cannot distinguish" in ev.lower() for ev in assessment.evidence)

    health_engine = SensorHealthEngine()
    health = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
    )

    assert health.status == SensorHealthStatus.INSUFFICIENT_DATA
    assert "insufficient" in health.recommended_action.lower()


# ---------------------------------------------------------------------------
# 11. Scenario: Conflicting Sensor Evidence
# ---------------------------------------------------------------------------

def test_scenario_conflicting_sensor_evidence():
    """Verify ambiguous or conflicting sensor patterns yield CONFLICTING_EVIDENCE."""
    n = 60
    np.random.seed(42)
    # Add small realistic sensor jitter so readings are not frozen
    p_jitter = 120.0 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        "sensor_pressure_mmhg": p_jitter,
        "dt_predicted_pressure_mmhg": np.full(n, 50.0),
        "sensor_force_n": np.full(n, 2.8),
        "sensor_flow_ml_h": np.full(n, 35.0),
        "commanded_rate_ml_h": np.full(n, 50.0),
        "sensor_temperature_c": np.full(n, 22.0),
    })

    engine = CrossSensorConsistencyEngine(
        pressure_occlusion_threshold_mmhg=250.0,
        pressure_drift_threshold_mmhg=20.0,
    )
    assessment = engine.assess_run(df)

    assert assessment.verdict == ConsistencyVerdict.CONFLICTING_EVIDENCE
    assert not assessment.is_sensor_fault
    assert not assessment.is_process_anomaly


# ---------------------------------------------------------------------------
# 12. Independent Reference Ground Truth & AI Safety Notice
# ---------------------------------------------------------------------------

def test_independent_reference_ground_truth_and_disclaimer():
    """Verify independent reference calculation, error deduction, and mandatory safety disclaimer."""
    engine = CrossSensorConsistencyEngine()
    cfg = SimulatorConfig(duration_seconds=100, random_seed=42)
    df = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)
    assessment = engine.assess_run(df)

    health_engine = SensorHealthEngine()

    # Case 1: Independent reference provided (e.g. gravimetric scale balance or calibrated manometer)
    report_with_ref = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        independent_reference_value=50.0,
        observed_sensor_value=72.0,  # +22 mmHg true error
    )
    assert report_with_ref.reference_available
    assert report_with_ref.estimated_measurement_error == pytest.approx(22.0)
    assert "reference_error_penalty" in report_with_ref.deductions
    assert "Medical Device AI Safety Notice" in report_with_ref.disclaimer

    # Case 2: No independent reference provided
    report_no_ref = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        independent_reference_value=None,
    )
    assert not report_no_ref.reference_available
    assert report_no_ref.estimated_measurement_error is None
    assert any("no independent reference" in ev.lower() for ev in report_no_ref.evidence_breakdown)
