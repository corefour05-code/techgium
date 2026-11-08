"""Unit and scenario tests for Phase 4 Machine Learning models.

Covers:
- XGBoost Multiclass Classifier predictions across all 9 classes:
  1. NORMAL
  2. POSITIVE_DRIFT
  3. NEGATIVE_DRIFT
  4. SUDDEN_OFFSET
  5. EXCESSIVE_NOISE
  6: SENSOR_FREEZE
  7: SENSOR_DROPOUT
  8: GENUINE_OCCLUSION
  9: TEMPERATURE_INFLUENCE
- XGBoost Regressor measurement error estimation (with paired reference data & missing reference data)
- SHAP Explainability (local top-k feature attribution, influence direction, summary)
- Decoupled Sensor Health Score vs Process Risk Score (occlusion does not penalize sensor health)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import MODELS_DIR, FaultClass
from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig
from src.features.feature_pipeline import SensorFeaturePipeline
from src.classification.xgboost_classifier import (
    InfusionPumpFaultClassifier,
    CLASS_NAMES,
    CLASS_NAME_TO_INDEX,
)
from src.regression.xgboost_regressor import SensorErrorEstimator
from src.explainability.shap_explainer import InfusionSHAPExplainer
from src.sensor_fusion.consistency_engine import CrossSensorConsistencyEngine, ConsistencyVerdict
from src.health.sensor_health import SensorHealthEngine, HealthAssessmentReport


@pytest.fixture(scope="module")
def pipeline():
    return SensorFeaturePipeline()


@pytest.fixture(scope="module")
def classifier():
    clf_path = MODELS_DIR / "classification" / "xgboost_classifier.joblib"
    assert clf_path.exists(), "Classifier artifact must exist before testing."
    return InfusionPumpFaultClassifier.load(clf_path)


@pytest.fixture(scope="module")
def regressor():
    reg_path = MODELS_DIR / "regression" / "xgboost_regressor_pressure.joblib"
    assert reg_path.exists(), "Regressor artifact must exist before testing."
    return SensorErrorEstimator.load(reg_path)


@pytest.fixture(scope="module")
def explainer(classifier):
    return InfusionSHAPExplainer(classifier)


# ---------------------------------------------------------------------------
# 1. Classification Tests Across Fault Types
# ---------------------------------------------------------------------------

def test_classifier_predict_normal(pipeline, classifier):
    """Verify clean normal window predicts NORMAL class."""
    n = 60
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": np.random.normal(80.0, 0.5, n),
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": np.full(n, 22.0),
        "commanded_rate_ml_h": np.full(n, 50.0),
        "dt_predicted_pressure_mmhg": np.full(n, 80.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "NORMAL"
    assert detail["confidence"] >= 0.70
    assert not detail["is_fault"]
    assert not detail["is_process_anomaly"]


def test_classifier_predict_positive_drift(pipeline, classifier):
    """Verify upward drifting pressure predicts POSITIVE_DRIFT."""
    cfg = SimulatorConfig(duration_seconds=900, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.POSITIVE_DRIFT,
        config=cfg,
        fault_start_sec=300,
        drift_rate_mmhg_per_hour=45.0,
    )
    feats = pipeline.transform_dataframe(df_run)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "POSITIVE_DRIFT"
    assert detail["is_fault"]


def test_classifier_predict_negative_drift(pipeline, classifier):
    """Verify downward drifting pressure predicts NEGATIVE_DRIFT."""
    cfg = SimulatorConfig(duration_seconds=900, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.NEGATIVE_DRIFT,
        config=cfg,
        fault_start_sec=300,
        drift_rate_mmhg_per_hour=45.0,
    )
    feats = pipeline.transform_dataframe(df_run)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "NEGATIVE_DRIFT"
    assert detail["is_fault"]


def test_classifier_predict_sudden_offset(pipeline, classifier):
    """Verify sudden step bias predicts SUDDEN_OFFSET."""
    n = 60
    p_step = np.concatenate([np.full(30, 80.0), np.full(30, 125.0)])
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": p_step,
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": np.full(n, 22.0),
        "commanded_rate_ml_h": np.full(n, 50.0),
        "dt_predicted_pressure_mmhg": np.full(n, 80.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "SUDDEN_OFFSET"
    assert detail["is_fault"]


def test_classifier_predict_excessive_noise(pipeline, classifier):
    """Verify high-variance noisy signal predicts EXCESSIVE_NOISE."""
    cfg = SimulatorConfig(duration_seconds=900, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.EXCESSIVE_NOISE,
        config=cfg,
        fault_start_sec=300,
    )
    feats = pipeline.transform_dataframe(df_run)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "EXCESSIVE_NOISE"
    assert detail["is_fault"]


def test_classifier_predict_sensor_freeze(pipeline, classifier):
    """Verify completely frozen constant reading predicts SENSOR_FREEZE."""
    n = 60
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": np.full(n, 78.4321),
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": np.full(n, 22.0),
        "commanded_rate_ml_h": np.full(n, 50.0),
        "dt_predicted_pressure_mmhg": np.full(n, 80.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "SENSOR_FREEZE"
    assert detail["is_fault"]


def test_classifier_predict_sensor_dropout(pipeline, classifier):
    """Verify rail/zero disconnection predicts SENSOR_DROPOUT."""
    n = 60
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": np.full(n, 0.0),
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": np.full(n, 22.0),
        "commanded_rate_ml_h": np.full(n, 50.0),
        "dt_predicted_pressure_mmhg": np.full(n, 80.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "SENSOR_DROPOUT"
    assert detail["is_fault"]


def test_classifier_predict_genuine_occlusion(pipeline, classifier):
    """Verify high pressure + surging force + collapsed flow predicts GENUINE_OCCLUSION."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.GENUINE_DOWNSTREAM_OCCLUSION,
        config=cfg,
        fault_start_sec=150,
        occlusion_time_constant_sec=80.0,
    )
    feats = pipeline.transform_dataframe(df_run)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "GENUINE_OCCLUSION"
    assert detail["is_process_anomaly"]
    assert not detail["is_fault"]  # Process anomaly, not sensor fault!


def test_classifier_predict_temperature_influence(pipeline, classifier):
    """Verify strong thermal ramp coupled with temperature predicts TEMPERATURE_INFLUENCE."""
    n = 60
    temp_profile = 22.0 + 0.15 * np.arange(n)
    p_thermal = 80.0 + 1.6 * (temp_profile - 22.0)
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": p_thermal,
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": temp_profile,
        "commanded_rate_ml_h": np.full(n, 50.0),
        "dt_predicted_pressure_mmhg": np.full(n, 80.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    detail = classifier.predict_detailed(feats.iloc[[-1]])[0]

    assert detail["predicted_class"] == "TEMPERATURE_INFLUENCE"


# ---------------------------------------------------------------------------
# 2. Regression Tests (Sensor Error Estimation)
# ---------------------------------------------------------------------------

def test_regressor_with_paired_reference_data(pipeline, regressor):
    """Verify pressure error regressor computes error and 95% confidence bounds."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.SUDDEN_OFFSET,
        config=cfg,
        fault_start_sec=200,
        offset_magnitude_mmhg=25.0,
    )
    feats = pipeline.transform_dataframe(df_run)
    observed_val = float(df_run["sensor_pressure_mmhg"].iloc[-1])
    ref_val = float(df_run["reference_pressure_mmhg"].iloc[-1])
    true_error = observed_val - ref_val

    res = regressor.estimate_sensor_error(
        feats.iloc[[-1]],
        observed_sensor_value=observed_val,
        reference_measurement_available=True,
    )

    assert res["status"] == "ESTIMATION_AVAILABLE"
    assert res["estimated_sensor_error"] == pytest.approx(true_error, abs=4.0)
    assert res["estimated_calibrated_value"] == pytest.approx(ref_val, abs=4.0)
    assert len(res["confidence_interval_95"]) == 2
    assert res["confidence_interval_95"][0] < res["confidence_interval_95"][1]
    assert "Advisory recommendation only" in res["advisory_recommendation"]


def test_regressor_missing_reference_safeguard(pipeline, regressor):
    """Verify that omitting reference measurements yields INSUFFICIENT_REFERENCE_DATA."""
    feats = pd.DataFrame(0.0, index=[0], columns=pipeline.transform_dataframe(
        pd.DataFrame({"sensor_pressure_mmhg": [80.0] * 15})
    ).columns)

    res = regressor.estimate_sensor_error(
        feats,
        observed_sensor_value=None,
        reference_measurement_available=False,
    )

    assert res["status"] == "INSUFFICIENT_REFERENCE_DATA"
    assert res["estimated_sensor_error"] is None
    assert "cannot establish true sensor accuracy" in res["message"].lower()


# ---------------------------------------------------------------------------
# 3. SHAP Explainability Tests
# ---------------------------------------------------------------------------

def test_shap_explanation_contents(pipeline, explainer):
    """Verify SHAP explanation structure, feature ranking, and attribution direction."""
    n = 60
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": np.full(n, 0.0),  # Dropout
        "sensor_force_n": np.random.normal(4.85, 0.05, n),
        "sensor_flow_ml_h": np.random.normal(50.0, 0.5, n),
        "sensor_temperature_c": np.full(n, 22.0),
    })
    feats = pipeline.transform_dataframe(df_win)
    explanation = explainer.explain_instance(feats, top_k=5)

    assert explanation["predicted_class"] == "SENSOR_DROPOUT"
    assert len(explanation["top_contributing_features"]) == 5
    assert "explanation_summary" in explanation
    assert "SHAP feature attributions reflect statistical model associations" in explanation["disclaimer"]

    # Top feature for dropout should be related to pressure or dropout counter
    top_feat = explanation["top_contributing_features"][0]
    assert "feature" in top_feat
    assert "shap_attribution" in top_feat
    assert top_feat["influence_direction"] in ("PUSHED_TOWARD", "PUSHED_AWAY")


# ---------------------------------------------------------------------------
# 4. Decoupled Sensor Health vs Process Risk Scoring
# ---------------------------------------------------------------------------

def test_sensor_health_decoupling_in_occlusion():
    """Verify genuine downstream occlusion preserves sensor health score while escalating process risk to CRITICAL."""
    cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
    df_run = generate_infusion_run(
        scenario=FaultClass.GENUINE_DOWNSTREAM_OCCLUSION,
        config=cfg,
        fault_start_sec=150,
        occlusion_time_constant_sec=80.0,
    )

    engine = CrossSensorConsistencyEngine(pressure_occlusion_threshold_mmhg=250.0)
    assessment = engine.assess_run(df_run)
    assert assessment.verdict == ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED

    health_engine = SensorHealthEngine()
    report = health_engine.evaluate_health(
        sensor_id="PRESS_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        anomaly_detected=True,
        anomaly_severity=0.85,
    )

    # Core requirement: Sensor health MUST NOT be penalized for a fluid line occlusion!
    assert report.sensor_health_score >= 90.0
    assert report.health_score >= 90.0

    # Core requirement: Process risk MUST escalate to CRITICAL!
    assert report.process_risk_score >= 90.0
    assert report.process_risk_level == "CRITICAL"
    assert "fluid_pathway_occlusion" in report.process_risk_factors
    assert report.anomaly_severity_score == pytest.approx(0.85)
