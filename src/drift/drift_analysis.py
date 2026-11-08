"""Comprehensive benchmark and comparative analysis for EWMA and CUSUM drift detection."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.drift.ewma import EWMADetector, EWMAResult
from src.drift.cusum import CUSUMDetector, CUSUMResult
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class DriftBenchmarkSummary:
    """Consolidated metrics across multiple drift experiments."""
    detector_name: str
    scenario_name: str
    magnitude: float
    magnitude_unit: str
    detection_rate_pct: float
    sample_false_alarm_rate_pct: float
    false_alarm_events_count: int
    detection_delay_steps: Optional[int]
    detection_delay_sec: Optional[float]
    drift_direction_detected: Optional[str]
    in_control_mean: float
    in_control_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "scenario_name": self.scenario_name,
            "magnitude": self.magnitude,
            "magnitude_unit": self.magnitude_unit,
            "detection_rate_pct": round(self.detection_rate_pct, 2),
            "sample_false_alarm_rate_pct": round(self.sample_false_alarm_rate_pct, 4),
            "false_alarm_events_count": self.false_alarm_events_count,
            "detection_delay_steps": self.detection_delay_steps,
            "detection_delay_sec": round(self.detection_delay_sec, 1) if self.detection_delay_sec is not None else None,
            "drift_direction_detected": self.drift_direction_detected,
        }


def benchmark_detectors_on_series(
    baseline_series: pd.Series,
    test_series: pd.Series,
    scenario_name: str,
    magnitude: float,
    magnitude_unit: str,
    fault_start_idx: int,
    sampling_rate_hz: float = 1.0,
    target_far: float = 0.005,
) -> Tuple[DriftBenchmarkSummary, DriftBenchmarkSummary, EWMAResult, CUSUMResult]:
    """Calibrate EWMA and CUSUM on baseline and evaluate on a test series."""
    # 1. EWMA
    ewma = EWMADetector(lambda_param=0.15, consecutive_alarms_to_confirm=3)
    ewma.calibrate_baseline(baseline_series, target_false_alarm_rate=target_far)
    ewma_res = ewma.monitor(test_series, fault_start_idx=fault_start_idx)

    ewma_delay_sec = (
        float(ewma_res.detection_delay_steps / sampling_rate_hz)
        if ewma_res.detection_delay_steps is not None
        else None
    )

    ewma_summary = DriftBenchmarkSummary(
        detector_name="EWMA",
        scenario_name=scenario_name,
        magnitude=magnitude,
        magnitude_unit=magnitude_unit,
        detection_rate_pct=ewma_res.detection_rate * 100.0,
        sample_false_alarm_rate_pct=ewma_res.false_alarm_rate * 100.0,
        false_alarm_events_count=ewma_res.false_alarm_events_count,
        detection_delay_steps=ewma_res.detection_delay_steps,
        detection_delay_sec=ewma_delay_sec,
        drift_direction_detected=ewma_res.drift_direction,
        in_control_mean=ewma_res.in_control_mean,
        in_control_std=ewma_res.in_control_std,
    )

    # 2. CUSUM
    cusum = CUSUMDetector(k_factor=0.5, consecutive_alarms_to_confirm=3, reset_on_alarm=True)
    cusum.calibrate_baseline(baseline_series, target_false_alarm_rate=target_far)
    cusum_res = cusum.monitor(test_series, fault_start_idx=fault_start_idx)

    cusum_delay_sec = (
        float(cusum_res.detection_delay_steps / sampling_rate_hz)
        if cusum_res.detection_delay_steps is not None
        else None
    )

    cusum_summary = DriftBenchmarkSummary(
        detector_name="CUSUM",
        scenario_name=scenario_name,
        magnitude=magnitude,
        magnitude_unit=magnitude_unit,
        detection_rate_pct=cusum_res.detection_rate * 100.0,
        sample_false_alarm_rate_pct=cusum_res.false_alarm_rate * 100.0,
        false_alarm_events_count=cusum_res.false_alarm_events_count,
        detection_delay_steps=cusum_res.detection_delay_steps,
        detection_delay_sec=cusum_delay_sec,
        drift_direction_detected=cusum_res.drift_direction,
        in_control_mean=cusum_res.in_control_mean,
        in_control_std=cusum_res.in_control_std,
    )

    return ewma_summary, cusum_summary, ewma_res, cusum_res
