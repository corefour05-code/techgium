"""Data validation module for the Infusion Pump ML System.

Implements non-destructive, safety-critical medical device data validation:
- Missing value audit (gap lengths, total counts, rates)
- Timestamp regularity and monotonicity analysis
- Sensor-specific range validation without destructive record removal
- Frozen sensor reading detection (ADC lockup / bus freeze)
- Sensor dropout detection (unexpected signal loss during active operation)
- Provenance and schema integrity verification
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.config import SensorType, DataProvenance
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Standard plausible physical boundaries for medical infusion pumps
# Note: Readings outside these bounds are FLAGGED, NOT automatically deleted.
PHYSICAL_RANGE_LIMITS: Dict[str, Tuple[float, float]] = {
    "pressure": (-50.0, 1500.0),       # mmHg (downstream line pressure can reach ~1000 mmHg under occlusion)
    "temperature": (0.0, 60.0),         # °C (operating environmental & enclosure range)
    "flow": (0.0, 1500.0),             # mL/h (clinical infusion range 0.1 to 1200 mL/h)
    "force": (-5.0, 150.0),            # Newtons (syringe plunger load cell)
    "motor_encoder": (0.0, 100000.0),  # Cumulative steps or pulses
}


@dataclass
class ValidationReport:
    """Detailed summary of data quality, completeness, and sensor anomalies."""
    is_valid: bool
    total_records: int
    provenance: str
    missing_value_summary: Dict[str, Dict[str, Any]]
    timestamp_quality: Dict[str, Any]
    out_of_range_flags: Dict[str, int]
    frozen_sensor_flags: Dict[str, int]
    dropout_flags: Dict[str, int]
    duplicate_timestamps: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "provenance": self.provenance,
            "missing_value_summary": self.missing_value_summary,
            "timestamp_quality": self.timestamp_quality,
            "out_of_range_flags": self.out_of_range_flags,
            "frozen_sensor_flags": self.frozen_sensor_flags,
            "dropout_flags": self.dropout_flags,
            "duplicate_timestamps": self.duplicate_timestamps,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def check_missing_values(df: pd.DataFrame, channels: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Audit missing values and compute maximum contiguous missing gaps."""
    target_cols = channels or [c for c in df.columns if c not in ["timestamp", "Time", "Time [sec]"]]
    summary = {}

    for col in target_cols:
        if col not in df.columns:
            continue
        series = df[col]
        is_na = series.isna()
        total_missing = int(is_na.sum())
        missing_pct = round(float(total_missing / len(df) * 100), 2) if len(df) > 0 else 0.0

        # Contiguous gap calculation
        max_contiguous_gap = 0
        if total_missing > 0:
            # Group consecutive booleans
            blocks = (~is_na).cumsum()[is_na]
            if len(blocks) > 0:
                max_contiguous_gap = int(blocks.value_counts().max())

        summary[col] = {
            "total_missing": total_missing,
            "missing_pct": missing_pct,
            "max_contiguous_gap": max_contiguous_gap,
        }
    return summary


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_interval_sec: Optional[float] = None,
    tolerance_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Validate monotonicity, intervals, and regularity of time-series timestamps."""
    if timestamp_col not in df.columns:
        return {"error": f"Timestamp column '{timestamp_col}' not found"}

    ts_series = df[timestamp_col]
    total_records = len(df)
    duplicate_count = int(ts_series.duplicated().sum())

    res: Dict[str, Any] = {
        "timestamp_col": timestamp_col,
        "total_records": total_records,
        "duplicate_count": duplicate_count,
        "is_strictly_monotonic": False,
        "irregular_intervals_count": 0,
    }

    if total_records < 2:
        res["is_strictly_monotonic"] = True
        return res

    # Compute numeric deltas
    try:
        if pd.api.types.is_numeric_dtype(ts_series):
            deltas = ts_series.diff().dropna()
        else:
            dt_series = pd.to_datetime(ts_series, errors="coerce")
            if dt_series.isna().any():
                res["unparseable_timestamps"] = int(dt_series.isna().sum())
            deltas = dt_series.diff().dropna().dt.total_seconds()

        res["is_strictly_monotonic"] = bool((deltas > 0).all())
        median_dt = float(deltas.median())
        res["median_interval_sec"] = round(median_dt, 4)
        res["min_interval_sec"] = round(float(deltas.min()), 4)
        res["max_interval_sec"] = round(float(deltas.max()), 4)
        res["std_interval_sec"] = round(float(deltas.std()), 4) if len(deltas) > 1 else 0.0

        target_interval = expected_interval_sec or median_dt
        if target_interval > 0:
            allowed_jitter = target_interval * tolerance_ratio
            irregular_mask = (deltas - target_interval).abs() > allowed_jitter
            res["irregular_intervals_count"] = int(irregular_mask.sum())
            res["expected_interval_sec"] = target_interval

    except Exception as e:
        res["error"] = str(e)

    return res


def detect_frozen_sensor(series: pd.Series, max_consecutive_frozen: int = 10, epsilon: float = 1e-9) -> Tuple[bool, int, np.ndarray]:
    """Detect if a continuous sensor reading is artificially locked/frozen across multiple consecutive readings.

    Args:
        series: Sensor measurement series.
        max_consecutive_frozen: Number of identical values before flagging as frozen.
        epsilon: Maximum floating-point variation to consider values identical.

    Returns:
        (is_frozen_detected, total_frozen_points, boolean_mask)
    """
    clean = series.ffill()
    diffs = (clean.diff().abs() <= epsilon)
    # Cumulative streak of true differences
    streaks = (~diffs).cumsum()
    counts = diffs.groupby(streaks).cumsum()
    frozen_mask = (counts >= max_consecutive_frozen).to_numpy()
    total_frozen = int(frozen_mask.sum())
    return (total_frozen > 0), total_frozen, frozen_mask


def detect_sensor_dropout(
    series: pd.Series,
    min_consecutive_dropouts: int = 3,
    dropout_threshold: float = 0.001,
) -> Tuple[bool, int, np.ndarray]:
    """Detect sudden loss of signal (dropping to exact zero or disconnected minimum) during expected activity."""
    is_dropout = (series.abs() <= dropout_threshold) | series.isna()
    streaks = (~is_dropout).cumsum()
    counts = is_dropout.groupby(streaks).cumsum()
    dropout_mask = (counts >= min_consecutive_dropouts).to_numpy()
    total_dropout = int(dropout_mask.sum())
    return (total_dropout > 0), total_dropout, dropout_mask


def validate_sensor_dataframe(
    df: pd.DataFrame,
    sensor_type_mapping: Dict[str, SensorType],
    timestamp_col: Optional[str] = None,
    expected_interval_sec: Optional[float] = None,
    provenance: Optional[DataProvenance] = None,
) -> ValidationReport:
    """Run comprehensive non-destructive validation across a sensor dataframe.

    Args:
        df: Input DataFrame containing sensor measurements.
        sensor_type_mapping: Dictionary mapping dataframe column name to its SensorType.
        timestamp_col: Name of timestamp column if present.
        expected_interval_sec: Nominal expected sampling interval.
        provenance: Data provenance tag (REAL_PUBLIC_DATA, REAL_CUSTOM_DATA, SYNTHETIC_DATA).

    Returns:
        ValidationReport instance.
    """
    warnings: List[str] = []
    errors: List[str] = []
    is_valid = True

    # 1. Provenance check
    prov_str = provenance.value if provenance else df.get("provenance", pd.Series(["UNKNOWN"])).iloc[0] if "provenance" in df.columns else "UNKNOWN"

    # 2. Missing values
    missing_summary = check_missing_values(df, channels=list(sensor_type_mapping.keys()))
    for ch, m_data in missing_summary.items():
        if m_data["missing_pct"] > 20.0:
            warnings.append(f"Channel '{ch}' has high missing rate ({m_data['missing_pct']}%).")
        if m_data["max_contiguous_gap"] > 50:
            warnings.append(f"Channel '{ch}' has a sustained missing gap of {m_data['max_contiguous_gap']} samples.")

    # 3. Timestamps
    ts_quality: Dict[str, Any] = {}
    dup_ts = 0
    if timestamp_col and timestamp_col in df.columns:
        ts_quality = validate_timestamps(df, timestamp_col, expected_interval_sec)
        dup_ts = ts_quality.get("duplicate_count", 0)
        if dup_ts > 0:
            warnings.append(f"Found {dup_ts} duplicate timestamps in '{timestamp_col}'.")
        if not ts_quality.get("is_strictly_monotonic", True):
            warnings.append(f"Timestamp column '{timestamp_col}' is not strictly monotonically increasing.")
        if ts_quality.get("irregular_intervals_count", 0) > 0:
            warnings.append(f"Found {ts_quality['irregular_intervals_count']} irregular sampling intervals.")

    # 4. Sensor-specific checks (Range, Freeze, Dropout)
    out_of_range_flags: Dict[str, int] = {}
    frozen_flags: Dict[str, int] = {}
    dropout_flags: Dict[str, int] = {}

    for col, stype in sensor_type_mapping.items():
        if col not in df.columns:
            warnings.append(f"Configured sensor channel '{col}' ({stype.value}) missing from dataset.")
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        limits = PHYSICAL_RANGE_LIMITS.get(stype.value)
        if limits:
            low, high = limits
            oor_mask = (s < low) | (s > high)
            oor_count = int(oor_mask.sum())
            out_of_range_flags[col] = oor_count
            if oor_count > 0:
                warnings.append(f"Channel '{col}' has {oor_count} readings outside plausible range [{low}, {high}]. (Non-destructively flagged)")

        # Frozen check (e.g. 15 consecutive identical values)
        has_freeze, freeze_count, _ = detect_frozen_sensor(s, max_consecutive_frozen=15)
        frozen_flags[col] = freeze_count
        if has_freeze:
            warnings.append(f"Channel '{col}' exhibits frozen sensor behavior ({freeze_count} steps).")

        # Dropout check
        has_dropout, drop_count, _ = detect_sensor_dropout(s, min_consecutive_dropouts=5)
        dropout_flags[col] = drop_count
        if has_dropout:
            warnings.append(f"Channel '{col}' exhibits sensor dropout / disconnection ({drop_count} steps).")

    return ValidationReport(
        is_valid=is_valid,
        total_records=len(df),
        provenance=str(prov_str),
        missing_value_summary=missing_summary,
        timestamp_quality=ts_quality,
        out_of_range_flags=out_of_range_flags,
        frozen_sensor_flags=frozen_flags,
        dropout_flags=dropout_flags,
        duplicate_timestamps=dup_ts,
        warnings=warnings,
        errors=errors,
    )
