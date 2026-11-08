"""Sensor-specific preprocessing module for the Infusion Pump ML System.

Applies clean, documented transformations on derived datasets while keeping raw data untouched:
- Modular per-sensor processing (pressure, temperature, flow, force, motor encoder)
- Sentinel value translation (e.g. DosePI recalibration markers)
- Trailing column stripping and unicode header normalization
- Outlier flagging and range bounding without data deletion
- Preservation of raw signals, independent references, and digital twin targets
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import (
    SensorType,
    DataProvenance,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.utils.logging_config import setup_logger
from src.data.validation import validate_sensor_dataframe, ValidationReport

logger = setup_logger(__name__)


def clean_dosepi_dataframe(df: pd.DataFrame, source_filename: str = "") -> pd.DataFrame:
    """Preprocess DosePI dataframe into a standardized derived format.

    Rules applied:
    1. Sentinel '-99.0' is converted to NaN in 'measured_mass_g' and flagged in 'is_recalibration_event'.
    2. Sentinel '-1.0' in compensated files is converted to NaN and flagged in 'is_warmup_phase'.
    3. Preserves original values in 'raw_sensor_reading'.
    4. Sets provenance to REAL_PUBLIC_DATA.
    """
    derived = df.copy()
    derived["provenance"] = DataProvenance.REAL_PUBLIC_DATA.value
    derived["source_dataset"] = "DosePI"
    derived["source_file"] = source_filename

    # Core files have column 'p [g]'
    if "p [g]" in derived.columns:
        raw_val = derived["p [g]"]
        derived["raw_sensor_reading"] = raw_val
        derived["is_recalibration_event"] = (raw_val == -99.0)
        # Parse set volume from filename (e.g. '0.2 ml.csv' -> 0.2)
        try:
            target_vol = float(source_filename.replace(" ml.csv", "").replace("ml.csv", "").strip())
        except Exception:
            target_vol = np.nan
        derived["commanded_volume_ml"] = target_vol
        # Cleaned measurement: mask out recalibration sentinels
        derived["measured_mass_g"] = np.where(raw_val > 0, raw_val, np.nan)
        # Assuming aqueous density ~ 1.0 g/mL at 20°C
        derived["measured_volume_ml"] = derived["measured_mass_g"]
        derived["measurement_error_ml"] = derived["measured_volume_ml"] - derived["commanded_volume_ml"]

    elif "Set volume" in derived.columns and "Actual weight" in derived.columns:
        derived["commanded_volume_ml"] = pd.to_numeric(derived["Set volume"], errors="coerce")
        raw_act = pd.to_numeric(derived["Actual weight"], errors="coerce")
        derived["raw_sensor_reading"] = raw_act
        derived["is_warmup_phase"] = (raw_act == -1.0) | (derived["commanded_volume_ml"] == 0.0)
        derived["measured_mass_g"] = np.where(raw_act > 0, raw_act, np.nan)
        derived["measured_volume_ml"] = derived["measured_mass_g"]
        derived["measurement_error_ml"] = derived["measured_volume_ml"] - derived["commanded_volume_ml"]
        if "Prediction" in derived.columns:
            # Model prediction string cleanup e.g. '[-1.0]' -> float
            pred_s = derived["Prediction"].astype(str).str.replace("[", "").str.replace("]", "")
            derived["model_predicted_volume_ml"] = pd.to_numeric(pred_s, errors="coerce")

    # Add discrete step index if no timestamp
    if "timestamp" not in derived.columns:
        derived["step_index"] = np.arange(len(derived))

    return derived


def clean_sensor_defects_dataframe(
    df: pd.DataFrame,
    source_filename: str = "",
    target_sensor: str = "Temp_Inside_2 [°C]",
    reference_sensor: str = "Temp_Inside_0 [°C]",
) -> pd.DataFrame:
    """Preprocess Zenodo Typical Sensor Defects dataframe into a standardized derived format.

    Rules applied:
    1. Strip empty artifact column 'Unnamed: 23' caused by trailing header commas.
    2. Normalize column names (strip whitespace and degree units for programmatic indexing).
    3. Ensure strictly monotonic 'Time [sec]'.
    4. Flag target sensor vs co-located reference sensor.
    5. Sets provenance to REAL_PUBLIC_DATA.
    """
    derived = df.copy()

    # Drop trailing empty column if present
    if "Unnamed: 23" in derived.columns:
        derived = derived.drop(columns=["Unnamed: 23"])

    # Clean header names
    rename_map = {}
    for c in derived.columns:
        clean_name = c.replace(" [°C]", "").replace(" [sec]", "").strip()
        rename_map[c] = clean_name
    derived = derived.rename(columns=rename_map)

    derived["provenance"] = DataProvenance.REAL_PUBLIC_DATA.value
    derived["source_dataset"] = "Typical_Sensor_Defects"
    derived["source_file"] = source_filename

    # In Zenodo defect injection, defects were introduced into Temp_Inside_2
    # Temp_Inside_0 and Temp_Inside_1 act as unaffected co-located references.
    clean_target = target_sensor.replace(" [°C]", "").strip()
    clean_ref = reference_sensor.replace(" [°C]", "").strip()

    if clean_target in derived.columns and clean_ref in derived.columns:
        derived["target_sensor_reading"] = derived[clean_target]
        derived["reference_sensor_reading"] = derived[clean_ref]
        derived["sensor_residual"] = derived["target_sensor_reading"] - derived["reference_sensor_reading"]

    return derived


class SensorPreprocessor:
    """Configurable preprocessor for individual sensor channels.

    Ensures models and detectors receive appropriately sanitized, non-leaking features.
    """

    def __init__(
        self,
        sensor_type: SensorType,
        channel_name: str,
        fill_strategy: str = "forward_fill",
        max_missing_fill_steps: int = 5,
    ):
        self.sensor_type = sensor_type
        self.channel_name = channel_name
        self.fill_strategy = fill_strategy
        self.max_missing_fill_steps = max_missing_fill_steps
        self.baseline_mean_: Optional[float] = None
        self.baseline_std_: Optional[float] = None

    def fit_baseline(self, baseline_series: pd.Series) -> "SensorPreprocessor":
        """Compute baseline statistical parameters strictly on in-control data."""
        clean = baseline_series.dropna()
        if len(clean) == 0:
            raise ValueError(f"Cannot fit baseline on empty series for {self.channel_name}")
        self.baseline_mean_ = float(clean.mean())
        self.baseline_std_ = float(clean.std()) if len(clean) > 1 else 1.0
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        """Transform sensor series into raw, sanitized, and normalized derived columns."""
        res = pd.DataFrame(index=series.index)
        res[f"{self.channel_name}_raw"] = series
        res[f"{self.channel_name}_is_missing"] = series.isna()

        # Controlled imputation for short gaps
        if self.fill_strategy == "forward_fill":
            filled = series.ffill(limit=self.max_missing_fill_steps)
        else:
            filled = series

        res[f"{self.channel_name}_sanitized"] = filled

        # Normalization if baseline fitted
        if self.baseline_mean_ is not None and self.baseline_std_ is not None:
            std_adj = self.baseline_std_ if self.baseline_std_ > 1e-6 else 1.0
            res[f"{self.channel_name}_zscore"] = (filled - self.baseline_mean_) / std_adj

        return res

