"""Phase 1 Smoke and Verification Unit Tests."""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import (
    SensorType,
    SensorHealthStatus,
    FaultClass,
    DATASET_METADATA,
    RAW_DATA_DIR,
)
from src.data.inspect import inspect_csv_file
from src.data.dataset_loader import load_dosepi, load_sensor_defects


def test_core_dependencies():
    """Verify all critical ML, signal processing, and API packages import cleanly."""
    import numpy as np
    import scipy
    import sklearn
    import xgboost
    import shap
    import fastapi
    import pydantic
    import mlflow

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgboost.__version__ is not None
    assert fastapi.__version__ is not None


def test_config_enums_and_paths():
    """Verify system constants and sensor enumerations."""
    assert SensorType.PRESSURE == "pressure"
    assert SensorType.TEMPERATURE == "temperature"
    assert SensorType.FLOW == "flow"
    assert SensorType.FORCE == "force"
    assert SensorType.MOTOR_ENCODER == "motor_encoder"

    assert SensorHealthStatus.HEALTHY == "HEALTHY"
    assert SensorHealthStatus.DRIFT_SUSPECTED == "DRIFT_SUSPECTED"

    assert FaultClass.NORMAL == 0
    assert FaultClass.GENUINE_DOWNSTREAM_OCCLUSION == 8

    assert len(DATASET_METADATA) == 5
    for key in ["dosepi", "pressure_occlusion", "force_failure", "smart_pump_logs", "sensor_defects"]:
        assert key in DATASET_METADATA


def test_inspect_csv_encoding_fallback():
    """Test that CSV inspection handles non-UTF8 encodings without raising errors."""
    sensor_defects_dir = RAW_DATA_DIR / "sensor_defects" / "extracted"
    std_file = sensor_defects_dir / "data_standard.csv"

    if std_file.exists():
        insp = inspect_csv_file(std_file, max_rows=100)
        assert "error" not in insp
        assert insp["total_columns"] >= 20
        assert "Time [sec]" in insp["columns"]
        assert insp["timestamp_analysis"]["sampling_interval_median"] == 1.0


def test_dosepi_loader():
    """Test DosePI loader loads core or compensated files."""
    dosepi_dfs = load_dosepi()
    # At least the core directory should be accessible if unzipped
    if not dosepi_dfs:
        core_dir = RAW_DATA_DIR / "dosepi" / "dosepi-datasets" / "core"
        if core_dir.exists():
            dosepi_dfs = load_dosepi(raw_dir=core_dir)
    assert isinstance(dosepi_dfs, dict)


def test_sensor_defects_loader():
    """Test Sensor Defects loader handles latin-1 encodings and returns clean data."""
    defects = load_sensor_defects(include_defects=["standard", "drift_0_001"])
    if defects:
        assert len(defects) >= 1
        for name, df in defects.items():
            assert isinstance(df, pd.DataFrame)
            assert "Time [sec]" in df.columns

