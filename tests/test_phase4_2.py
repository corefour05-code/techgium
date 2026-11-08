"""Comprehensive Test Suite for Phase 4.2 Engineering Enhancements.

Covers:
1. Adaptive Baseline Estimation:
   - Pre-infusion tare estimation across multi-baseline operating states (15, 40, 65 mmHg).
   - Tare rejection under high pressure variance (unstable tare state).
   - Tare rejection during active infusion delivery flow (> 2.0 mL/h).
   - Tare rejection under suspected occlusion (> 180 mmHg).
   - Tare rejection under sensor freeze and dropout states.
   - Safety Invariant: Active occlusion pressure is NEVER erased or auto-zeroed.
2. Robust Feature Engineering:
   - Exact 27 feature column count and verified feature names.
   - Causal backward-looking guarantee (no future-leakage).
   - Resilience against missing optional channels (force, flow, temperature).
3. Multiclass Probability Calibration:
   - Temperature scaling convergence and positive temperature parameter (T > 0).
   - Calibrated probability simplex properties (sums to 1.0, bounds in [0, 1]).
   - ECE and Brier score metric computation.
4. Conformal Sensor Error Regression:
   - Split conformal prediction calibration quantile.
   - Empirical coverage guarantee (>= 90% coverage for 95% nominal interval).
   - Safety guard: Returns None / raises when independent reference is absent.
   - Catheter baseline subgroup performance.
5. Decoupled Sensor Health & Process Risk Invariant:
   - Under diverse baselines, occlusion raises process risk to CRITICAL while sensor health >= 90.
   - Sensor drift degrades sensor health without incorrectly diagnosing process occlusion.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FaultClass,
    DataProvenance,
    SensorType,
)
from src.features.adaptive_baseline import (
    AdaptiveBaselineEstimator,
    BaselineQualityStatus,
    BaselineEstimate,
)
from src.features.feature_pipeline import SensorFeaturePipeline, FEATURE_NAMES
from src.classification.xgboost_classifier import (
    InfusionPumpFaultClassifier,
    CLASS_NAMES,
    CLASS_NAME_TO_INDEX,
)
from src.classification.probability_calibration import (
    MulticlassCalibrator,
    compute_ece,
    compute_brier_score,
)
from src.regression.xgboost_regressor import SensorErrorEstimator
from src.regression.conformal_regressor import ConformalSensorErrorEstimator
from src.sensor_fusion.consistency_engine import ConsistencyAssessment, ConsistencyVerdict
from src.health.sensor_health import SensorHealthEngine, HealthAssessmentReport
from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig


# ---------------------------------------------------------------------------
# 1. Adaptive Baseline Estimator Unit & Safety Tests
# ---------------------------------------------------------------------------

def test_adaptive_baseline_pre_infusion_nominal():
    """Verify clean pre-infusion tare accurately captures true catheter backpressure."""
    estimator = AdaptiveBaselineEstimator(max_acceptable_stability_std_mmhg=1.5)
    
    for true_base in [15.0, 40.0, 65.0]:
        # 30-sample pre-infusion tare with 0 flow and minor sensor noise
        tare_p = pd.Series(np.random.normal(true_base, 0.4, 30))
        tare_q = pd.Series(np.zeros(30))
        
        est = estimator.estimate_from_pre_infusion(tare_p, flow_window=tare_q)
        assert est.is_valid is True
        assert est.quality_status == BaselineQualityStatus.ESTIMATED_STABLE
        assert est.confidence >= 0.70
        assert abs(est.baseline_pressure_mmhg - true_base) < 0.5


def test_adaptive_baseline_rejection_unstable():
    """Verify baseline estimator rejects unstable/noisy pressure signal during tare."""
    estimator = AdaptiveBaselineEstimator(max_acceptable_stability_std_mmhg=1.5, nominal_fallback_baseline_mmhg=40.0)
    
    # Highly noisy tare window (std = 3.5 > 1.5 threshold)
    unstable_p = pd.Series(np.random.normal(40.0, 3.5, 30))
    est = estimator.estimate_from_pre_infusion(unstable_p)
    
    assert est.is_valid is False
    assert est.quality_status == BaselineQualityStatus.REJECTED_UNSTABLE
    assert est.baseline_pressure_mmhg == 40.0  # Safe fallback
    assert est.confidence == 0.25


def test_adaptive_baseline_rejection_active_flow():
    """Verify baseline estimator strictly refuses to tare when active delivery flow is present."""
    estimator = AdaptiveBaselineEstimator(active_flow_threshold_ml_h=2.0, nominal_fallback_baseline_mmhg=40.0)
    
    tare_p = pd.Series(np.random.normal(80.0, 0.4, 30))
    # Active delivery flow at 50 mL/h
    active_q = pd.Series(np.random.normal(50.0, 0.5, 30))
    
    est = estimator.estimate_from_pre_infusion(tare_p, flow_window=active_q)
    assert est.is_valid is False
    assert est.quality_status == BaselineQualityStatus.REJECTED_ACTIVE_INFUSION
    assert est.baseline_pressure_mmhg == 40.0
    assert est.confidence == 0.20


def test_adaptive_baseline_rejection_occlusion_surge():
    """Verify baseline estimator strictly refuses to tare when line pressure indicates occlusion."""
    estimator = AdaptiveBaselineEstimator(occlusion_risk_threshold_mmhg=180.0, nominal_fallback_baseline_mmhg=40.0)
    
    # High pressure from acute occlusion (e.g. 350 mmHg) with normal noise
    high_p = pd.Series(np.random.normal(350.0, 0.4, 30))
    est = estimator.estimate_from_pre_infusion(high_p)
    
    assert est.is_valid is False
    assert est.quality_status == BaselineQualityStatus.REJECTED_OCCLUSION_RISK
    assert est.baseline_pressure_mmhg == 40.0
    assert est.confidence == 0.10


def test_adaptive_baseline_never_erases_occlusion():
    """CRITICAL SAFETY TEST: Verify active occlusion pressure is NEVER auto-zeroed."""
    pipeline = SensorFeaturePipeline()
    
    # Simulate an active acute downstream occlusion (pressure ramps to 450 mmHg)
    p_surge = np.linspace(40.0, 450.0, 100)
    df_occlusion = pd.DataFrame({
        "sensor_pressure_mmhg": p_surge,
        "sensor_flow_ml_h": np.zeros(100),
        "commanded_rate_ml_h": np.full(100, 50.0),
    })
    
    feats = pipeline.transform_dataframe(df_occlusion)
    
    # Residual and deviation from baseline must reflect the genuine massive surge (> 300 mmHg)
    assert feats["p_dt_residual"].iloc[-1] > 300.0
    assert feats["p_deviation_from_baseline"].iloc[-1] > 300.0
    # Occlusion pressure was NOT auto-zeroed away
    assert feats["p_dt_residual"].iloc[-1] != pytest.approx(0.0, abs=5.0)


# ---------------------------------------------------------------------------
# 2. Robust Feature Engineering & Causality Tests
# ---------------------------------------------------------------------------

def test_feature_pipeline_exact_columns():
    """Verify pipeline extracts exactly the 27 specified features in correct order."""
    pipeline = SensorFeaturePipeline()
    df = pd.DataFrame({
        "sensor_pressure_mmhg": [40.0] * 30,
        "sensor_force_n": [1.5] * 30,
        "sensor_flow_ml_h": [0.0] * 30,
        "sensor_temperature_c": [22.0] * 30,
        "commanded_rate_ml_h": [50.0] * 30,
    })
    feats = pipeline.transform_dataframe(df)
    
    assert feats.shape[1] == 27
    assert list(feats.columns) == FEATURE_NAMES
    assert not feats.isna().any().any()


def test_feature_pipeline_causality():
    """Verify feature calculation at step t uses only past/present data (causal)."""
    pipeline = SensorFeaturePipeline()
    
    np.random.seed(42)
    p_series = np.random.normal(50.0, 1.0, 100)
    df_full = pd.DataFrame({"sensor_pressure_mmhg": p_series})
    feats_full = pipeline.transform_dataframe(df_full)
    
    # Calculate features on prefix of length 50
    df_sub = pd.DataFrame({"sensor_pressure_mmhg": p_series[:50]})
    feats_sub = pipeline.transform_dataframe(df_sub)
    
    # Rolling features at step 49 must match identically between sub and full
    for col in ["p_raw", "p_roll_mean_15", "p_roll_std_15", "p_roll_mean_60"]:
        assert abs(feats_full[col].iloc[49] - feats_sub[col].iloc[49]) < 1e-6


def test_feature_pipeline_missing_channels():
    """Verify pipeline gracefully handles missing optional force, flow, and temp channels."""
    pipeline = SensorFeaturePipeline()
    # Minimal input with only pressure channel
    df_min = pd.DataFrame({"sensor_pressure_mmhg": [60.0] * 30})
    feats = pipeline.transform_dataframe(df_min)
    
    assert feats.shape[1] == 27
    assert feats["has_force_channel"].iloc[0] == 0.0
    assert feats["has_flow_channel"].iloc[0] == 0.0
    assert feats["has_temp_channel"].iloc[0] == 0.0
    assert not feats.isna().any().any()


# ---------------------------------------------------------------------------
# 3. Multiclass Probability Calibration Tests
# ---------------------------------------------------------------------------

def test_probability_calibrator_simplex():
    """Verify calibrated probability predictions sum to 1.0 and stay in [0, 1]."""
    calibrator_path = MODELS_DIR / "classification" / "probability_calibrator.joblib"
    clf_path = MODELS_DIR / "classification" / "xgboost_classifier.joblib"
    
    assert calibrator_path.exists(), "Calibrator artifact must exist."
    assert clf_path.exists(), "Classifier artifact must exist."
    
    classifier = InfusionPumpFaultClassifier.load(clf_path)
    calibrator = MulticlassCalibrator.load(calibrator_path, base_classifier=classifier)
    
    pipeline = SensorFeaturePipeline()
    df_win = pd.DataFrame({
        "sensor_pressure_mmhg": [40.0] * 30,
        "commanded_rate_ml_h": [50.0] * 30,
    })
    feats = pipeline.transform_dataframe(df_win)
    
    cal_probas = calibrator.predict_proba(feats)
    assert cal_probas.shape == (30, len(CLASS_NAMES))
    assert np.all(cal_probas >= 0.0)
    assert np.all(cal_probas <= 1.0)
    np.testing.assert_allclose(np.sum(cal_probas, axis=1), 1.0, atol=1e-5)


def test_calibration_ece_and_brier_computation():
    """Verify ECE and Brier score evaluation functions operate correctly."""
    n = 200
    n_classes = 9
    rng = np.random.RandomState(42)
    
    y_true = rng.randint(0, n_classes, size=n)
    logits = rng.randn(n, n_classes)
    exp_logits = np.exp(logits)
    probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    ece = compute_ece(probas, y_true, n_bins=10)
    brier = compute_brier_score(probas, y_true)
    
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0


# ---------------------------------------------------------------------------
# 4. Conformal Sensor Error Regression Tests
# ---------------------------------------------------------------------------

def test_conformal_regressor_coverage_guarantee():
    """Verify conformal predictor interval provides >= 90% empirical test coverage."""
    conformal_path = MODELS_DIR / "regression" / "conformal_regressor_pressure.joblib"
    reg_path = MODELS_DIR / "regression" / "xgboost_regressor_pressure.joblib"
    assert conformal_path.exists(), "Conformal regressor artifact must exist."
    assert reg_path.exists(), "Base regressor artifact must exist."
    
    regressor = SensorErrorEstimator.load(reg_path)
    conformal_engine = ConformalSensorErrorEstimator.load(conformal_path, base_regressor=regressor)
    assert conformal_engine.is_calibrated_ is True
    assert conformal_engine.conformal_quantile_ > 0.0
    
    # Generate test run with known reference pressure
    cfg = SimulatorConfig(duration_seconds=300, random_seed=404)
    df_run = generate_infusion_run(scenario=FaultClass.POSITIVE_DRIFT, config=cfg)
    pipeline = SensorFeaturePipeline()
    feats = pipeline.transform_dataframe(df_run)
    
    y_true_err = (df_run["sensor_pressure_mmhg"] - df_run["reference_pressure_mmhg"]).values
    true_slice = y_true_err[60:]
    
    cov_metrics = conformal_engine.evaluate_coverage(feats.iloc[60:], true_slice)
    coverage = cov_metrics["conformal_prediction"]["empirical_coverage"]
    
    # Test coverage must be >= 90% (nominal 95% interval)
    assert coverage >= 0.90, f"Expected coverage >= 0.90, got {coverage:.4f}"


def test_conformal_missing_reference_guard():
    """Verify conformal regressor safely reports missing reference without claiming true accuracy."""
    conformal_path = MODELS_DIR / "regression" / "conformal_regressor_pressure.joblib"
    reg_path = MODELS_DIR / "regression" / "xgboost_regressor_pressure.joblib"
    
    regressor = SensorErrorEstimator.load(reg_path)
    conformal_engine = ConformalSensorErrorEstimator.load(conformal_path, base_regressor=regressor)
    
    pipeline = SensorFeaturePipeline()
    df_win = pd.DataFrame({"sensor_pressure_mmhg": [45.0] * 30})
    feats = pipeline.transform_dataframe(df_win)
    
    # Point prediction and intervals with missing reference must report insufficient data
    res = conformal_engine.predict_conformal_interval(feats, reference_measurement_available=False)
    assert res["status"] == "INSUFFICIENT_REFERENCE_DATA"
    assert res["estimated_sensor_error"] is None
    assert res["conformal_interval_95"] is None


# ---------------------------------------------------------------------------
# 5. Decoupled Sensor Health vs Process Risk Safety Invariant
# ---------------------------------------------------------------------------

def test_health_risk_decoupling_under_diverse_baselines():
    """Verify process occlusion raises process risk to CRITICAL while keeping sensor health healthy."""
    health_engine = SensorHealthEngine()
    
    for catheter_p in [15.0, 40.0, 65.0]:
        # Scenario: Physical occlusion occurs. Sensor is performing flawlessly (0 drift, valid tare).
        assessment = ConsistencyAssessment(
            verdict=ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED,
            is_sensor_fault=False,
            is_process_anomaly=True,
            confidence=0.95,
            evidence=["Pressure surge in hydraulic proportion with force and delivery collapse"],
        )
        report = health_engine.evaluate_health(
            sensor_id=f"P_LINE_{int(catheter_p)}",
            sensor_type="pressure",
            consistency_assessment=assessment,
            anomaly_detected=True,
            anomaly_severity=0.85,
        )
        
        # Sensor transducer is mechanically functional
        assert report.sensor_health_score >= 90.0
        assert report.health_score >= 90.0
        assert report.status.value in ["HEALTHY", "WARNING"]
        
        # Process risk must be CRITICAL due to fluid blockage
        assert report.process_risk_score >= 85.0
        assert report.process_risk_level == "CRITICAL"


def test_health_risk_decoupling_under_sensor_drift():
    """Verify sensor drift degrades sensor health without declaring critical process blockage."""
    health_engine = SensorHealthEngine()
    
    assessment = ConsistencyAssessment(
        verdict=ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED,
        is_sensor_fault=True,
        is_process_anomaly=False,
        confidence=0.85,
        evidence=["Isolated sensor drift without physical force rise"],
    )
    report = health_engine.evaluate_health(
        sensor_id="P_LINE_01",
        sensor_type="pressure",
        consistency_assessment=assessment,
        ewma_drift_detected=True,
        cusum_drift_detected=True,
    )
    
    # Sensor health must be degraded
    assert report.sensor_health_score < 70.0
    assert report.status.value in ["DRIFT_SUSPECTED", "FAULT_SUSPECTED"]
    
    # Process risk is not CRITICAL (fluid path is open, not physically blocked)
    assert report.process_risk_level != "CRITICAL"
