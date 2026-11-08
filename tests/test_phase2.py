"""Comprehensive Unit and Integration Tests for Phase 2.

Covers:
- Missing value audit & gap lengths
- Timestamp regularity, monotonicity, and duplicates
- Non-destructive range validation
- Sensor freeze & dropout detection
- Provenance tracking & data field separation
- Configurable sensor channels
- Sentinel value transformation in derived datasets
- EWMA & CUSUM baseline calibration and drift detection
- Rolling feature extraction & OLS slope
- Cross-sensor hydraulic consistency
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import SensorType, DataProvenance, FaultClass
from src.data.validation import (
    check_missing_values,
    validate_timestamps,
    detect_frozen_sensor,
    detect_sensor_dropout,
    validate_sensor_dataframe,
)
from src.data.preprocessing import (
    clean_dosepi_dataframe,
    clean_sensor_defects_dataframe,
    SensorPreprocessor,
)
from src.data.synthetic_generator import (
    generate_infusion_run,
    SimulatorConfig,
)
from src.features.rolling_features import (
    compute_rolling_trend_slope,
    RollingFeatureExtractor,
)
from src.features.cross_sensor_features import CrossSensorFeatureExtractor
from src.drift.ewma import EWMADetector
from src.drift.cusum import CUSUMDetector


# 1. Missing Values Tests
def test_missing_values_audit():
    df = pd.DataFrame({
        "timestamp": np.arange(10),
        "pressure": [10.0, 10.2, np.nan, np.nan, np.nan, 10.5, np.nan, 10.6, 10.7, 10.8],
    })
    audit = check_missing_values(df, channels=["pressure"])
    assert audit["pressure"]["total_missing"] == 4
    assert audit["pressure"]["missing_pct"] == 40.0
    assert audit["pressure"]["max_contiguous_gap"] == 3


# 2. Timestamp Validation Tests
def test_timestamp_regularity_and_duplicates():
    # Irregular intervals and duplicates
    df = pd.DataFrame({
        "timestamp": [0.0, 1.0, 2.0, 2.0, 3.5, 4.5, 5.5],
        "reading": np.ones(7),
    })
    res = validate_timestamps(df, timestamp_col="timestamp", expected_interval_sec=1.0)
    assert res["duplicate_count"] == 1
    assert not res["is_strictly_monotonic"]
    assert res["irregular_intervals_count"] >= 1


# 3. Non-Destructive Range Validation Tests
def test_non_destructive_range_validation():
    # Value 2500 mmHg is out-of-range, but must NOT be deleted
    df = pd.DataFrame({
        "timestamp": np.arange(5),
        "sensor_p": [40.0, 42.0, 2500.0, 43.0, 41.0],
    })
    report = validate_sensor_dataframe(
        df,
        sensor_type_mapping={"sensor_p": SensorType.PRESSURE},
        timestamp_col="timestamp",
        provenance=DataProvenance.REAL_CUSTOM_DATA,
    )
    # Check that out of range flag was registered
    assert report.out_of_range_flags["sensor_p"] == 1
    # Check that record count was preserved (5 rows intact!)
    assert len(df) == 5
    assert df.loc[2, "sensor_p"] == 2500.0


# 4. Sensor Freeze Detection
def test_frozen_sensor_detection():
    # 20 consecutive identical values
    series = pd.Series([12.5] * 20 + [12.6, 12.7, 12.8])
    is_frozen, count, mask = detect_frozen_sensor(series, max_consecutive_frozen=15)
    assert is_frozen
    assert count > 0
    assert mask[18]  # Step 18 should be flagged as frozen


# 5. Sensor Dropout Detection
def test_sensor_dropout_detection():
    # Signal drops to 0.0 for 8 steps
    series = pd.Series([40.0, 41.0] + [0.0] * 8 + [42.0, 43.0])
    is_drop, count, mask = detect_sensor_dropout(series, min_consecutive_dropouts=5)
    assert is_drop
    assert count >= 4


# 6. Provenance & Field Separation in Synthetic Generator
def test_synthetic_data_provenance_and_field_separation():
    cfg = SimulatorConfig(duration_seconds=60, random_seed=42)
    df = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)

    # 1. Provenance check
    assert (df["provenance"] == DataProvenance.SYNTHETIC_DATA.value).all()
    assert (df["is_synthetic"] == True).all()
    assert "Research-oriented" in df["simulation_disclaimer"].iloc[0]

    # 2. Strict field separation
    assert "sensor_pressure_mmhg" in df.columns
    assert "reference_pressure_mmhg" in df.columns
    assert "dt_predicted_pressure_mmhg" in df.columns

    # 3. Reproducibility check with same seed
    df2 = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)
    pd.testing.assert_frame_equal(df, df2)


# 7. Configurable Sensor Channels Test
def test_configurable_sensor_channels():
    # Pump with only pressure and temperature, no force or flow sensors
    cfg = SimulatorConfig(
        duration_seconds=30,
        enable_pressure=True,
        enable_temperature=True,
        enable_force=False,
        enable_flow=False,
    )
    df = generate_infusion_run(scenario=FaultClass.NORMAL, config=cfg)
    assert "sensor_pressure_mmhg" in df.columns
    assert "sensor_temperature_c" in df.columns
    assert "sensor_force_n" not in df.columns
    assert "sensor_flow_ml_h" not in df.columns


# 8. DosePI Sentinel Cleaning
def test_dosepi_sentinel_cleaning():
    raw_df = pd.DataFrame({"p [g]": [0.201, -99.0, 0.198, 0.202]})
    cleaned = clean_dosepi_dataframe(raw_df, source_filename="0.2 ml.csv")

    # Original value preserved
    assert cleaned.loc[1, "raw_sensor_reading"] == -99.0
    # Cleaned value is NaN
    assert pd.isna(cleaned.loc[1, "measured_mass_g"])
    # Sentinel flag is True
    assert cleaned.loc[1, "is_recalibration_event"] == True
    assert cleaned["provenance"].iloc[0] == DataProvenance.REAL_PUBLIC_DATA.value


# 9. Zenodo Trailing Column Stripping
def test_zenodo_cleaning_trailing_column():
    raw_df = pd.DataFrame({
        "Time [sec]": [0, 1, 2],
        "Temp_Inside_0 [°C]": [20.1, 20.2, 20.1],
        "Temp_Inside_2 [°C]": [20.0, 20.0, 20.1],
        "Unnamed: 23": [np.nan, np.nan, np.nan],
    })
    cleaned = clean_sensor_defects_dataframe(raw_df, source_filename="test.csv")
    assert "Unnamed: 23" not in cleaned.columns
    assert "Temp_Inside_0" in cleaned.columns
    assert "sensor_residual" in cleaned.columns


# 10. EWMA Baseline Calibration and Drift Detection
def test_ewma_drift_detection():
    rng = np.random.default_rng(42)
    # 500 in-control samples (mean=0, std=1)
    baseline = rng.normal(0.0, 1.0, 500)
    # Test data: 200 normal + 300 linear positive drift
    t_post = np.arange(300)
    test_data = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(0.0, 1.0, 300) + (0.05 * t_post)])

    detector = EWMADetector(lambda_param=0.15)
    detector.calibrate_baseline(baseline, target_false_alarm_rate=0.01)

    assert detector.is_calibrated_
    assert abs(detector.mu_0_) < 0.2
    assert 0.8 < detector.sigma_0_ < 1.2

    res = detector.monitor(test_data, fault_start_idx=200)
    assert res.is_drift_detected
    assert res.drift_direction == "POSITIVE"
    assert res.false_alarm_rate < 0.05
    assert res.detection_delay_steps is not None
    assert res.detection_delay_steps < 80


# 11. CUSUM Baseline Calibration and Drift Detection
def test_cusum_drift_detection():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.0, 1.0, 500)
    t_post = np.arange(300)
    test_data = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(0.0, 1.0, 300) + (0.05 * t_post)])

    detector = CUSUMDetector(k_factor=0.5)
    detector.calibrate_baseline(baseline, target_false_alarm_rate=0.01)

    assert detector.is_calibrated_
    res = detector.monitor(test_data, fault_start_idx=200)
    assert res.is_drift_detected
    assert res.drift_direction == "POSITIVE"
    assert res.detection_delay_steps is not None
    assert res.detection_delay_steps < 80


# 12. Rolling Feature Extraction & Trend Slope
def test_rolling_trend_slope():
    # Strictly linear ramp y = 2 * t
    t = np.arange(50, dtype=np.float64)
    y = pd.Series(2.0 * t)
    slope_series = compute_rolling_trend_slope(y, window=10)
    # Once window is full (index >= 9), slope must be exactly 2.0
    valid_slopes = slope_series.dropna()
    np.testing.assert_allclose(valid_slopes.values, 2.0, atol=1e-5)


# 13. Cross-Sensor Hydraulic Consistency
def test_cross_sensor_hydraulic_consistency():
    extractor = CrossSensorFeatureExtractor()

    # Case A: Sensor Drift (Pressure rises, Force stays normal)
    df_drift = pd.DataFrame({
        "sensor_pressure": [40.0, 100.0, 200.0],
        "sensor_force": [1.8, 1.81, 1.82],  # Plunger force stays normal!
    })
    feat_drift = extractor.extract(df_drift, pressure_col="sensor_pressure", force_col="sensor_force")
    # Discrepancy between actual force and expected hydraulic force widens dramatically
    assert feat_drift["force_hydraulic_residual_n"].iloc[-1] < -5.0

    # Case B: True Occlusion (Pressure and Force surge together in hydraulic coupling)
    # Expected force from 200 mmHg is ~ 200 * 0.0133322 * 3.14 ~ 8.37 N
    df_occlusion = pd.DataFrame({
        "sensor_pressure": [40.0, 100.0, 200.0],
        "sensor_force": [3.17, 5.68, 9.87],  # F = P * A + F_friction
    })
    feat_occ = extractor.extract(df_occlusion, pressure_col="sensor_pressure", force_col="sensor_force")
    # Residual remains near nominal mechanical friction (~1.5 N)
    np.testing.assert_allclose(feat_occ["force_hydraulic_residual_n"].values, 1.5, atol=0.2)

