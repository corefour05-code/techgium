"""Rule-based Cross-Sensor Consistency & Fault Disambiguation Engine.

Differentiates between:
- Process Anomalies (e.g. Genuine Downstream Occlusion)
- Sensor Transducer Faults (e.g. Gradual Sensor Drift, Offset, Freeze, Dropout)
- Environmental Influences (e.g. Ambient Temperature Coupling)
- Insufficient Evidence (e.g. Single-channel architectures lacking cross-validation signals)

Inviolable Rule:
If physical evidence is insufficient (e.g., only a single pressure transducer is present
without force, flow, or independent reference), the engine returns INSUFFICIENT_EVIDENCE
rather than inventing an unverified medical diagnosis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ConsistencyVerdict(str, Enum):
    """Physically grounded diagnostic verdicts."""
    NORMAL = "NORMAL"
    SENSOR_DRIFT_SUSPECTED = "SENSOR_DRIFT_SUSPECTED"
    SENSOR_FREEZE_SUSPECTED = "SENSOR_FREEZE_SUSPECTED"
    SENSOR_DROPOUT_SUSPECTED = "SENSOR_DROPOUT_SUSPECTED"
    GENUINE_OCCLUSION_CONFIRMED = "GENUINE_OCCLUSION_CONFIRMED"
    TEMPERATURE_INFLUENCE_SUSPECTED = "TEMPERATURE_INFLUENCE_SUSPECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


@dataclass
class ConsistencyAssessment:
    """Detailed result of cross-sensor physical validation."""
    verdict: ConsistencyVerdict
    is_sensor_fault: bool
    is_process_anomaly: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "is_sensor_fault": self.is_sensor_fault,
            "is_process_anomaly": self.is_process_anomaly,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "metrics": self.metrics,
        }


class CrossSensorConsistencyEngine:
    """Multi-channel physical validation and occlusion disambiguation engine."""

    def __init__(
        self,
        nominal_plunger_area_cm2: float = 3.14,
        nominal_friction_n: float = 1.5,
        pressure_occlusion_threshold_mmhg: float = 250.0,
        pressure_drift_threshold_mmhg: float = 20.0,
        min_consecutive_freeze_steps: int = 15,
        min_consecutive_dropout_steps: int = 5,
    ):
        self.plunger_area_cm2 = nominal_plunger_area_cm2
        self.nominal_friction_n = nominal_friction_n
        self.pressure_occlusion_threshold_mmhg = pressure_occlusion_threshold_mmhg
        self.pressure_drift_threshold_mmhg = pressure_drift_threshold_mmhg
        self.min_consecutive_freeze_steps = min_consecutive_freeze_steps
        self.min_consecutive_dropout_steps = min_consecutive_dropout_steps

    def assess_run(
        self,
        df: pd.DataFrame,
        pressure_col: Optional[str] = "sensor_pressure_mmhg",
        force_col: Optional[str] = "sensor_force_n",
        flow_col: Optional[str] = "sensor_flow_ml_h",
        temperature_col: Optional[str] = "sensor_temperature_c",
        commanded_rate_col: Optional[str] = "commanded_rate_ml_h",
        dt_pressure_col: Optional[str] = "dt_predicted_pressure_mmhg",
        ewma_alarm_active: bool = False,
        cusum_alarm_active: bool = False,
        recent_window_steps: int = 60,
    ) -> ConsistencyAssessment:
        """Evaluate cross-sensor consistency over recent time window.

        Args:
            df: DataFrame containing available sensor channels.
            pressure_col: Column name for line pressure.
            force_col: Column name for plunger drive force.
            flow_col: Column name for delivered flow.
            temperature_col: Column name for ambient/sensor temperature.
            commanded_rate_col: Column name for commanded infusion rate.
            dt_pressure_col: Column name for Digital Twin expected pressure.
            ewma_alarm_active: Flag from EWMA detector.
            cusum_alarm_active: Flag from CUSUM detector.
            recent_window_steps: Number of recent samples to evaluate.

        Returns:
            ConsistencyAssessment instance.
        """
        evidence = []
        metrics = {}

        # Extract available channels
        has_pressure = pressure_col is not None and pressure_col in df.columns
        has_force = force_col is not None and force_col in df.columns
        has_flow = flow_col is not None and flow_col in df.columns
        has_temp = temperature_col is not None and temperature_col in df.columns
        has_dt = dt_pressure_col is not None and dt_pressure_col in df.columns
        has_cmd = commanded_rate_col is not None and commanded_rate_col in df.columns

        metrics["channels_available"] = {
            "pressure": has_pressure,
            "force": has_force,
            "flow": has_flow,
            "temperature": has_temp,
            "digital_twin": has_dt,
        }

        # Check data sufficiency
        if not has_pressure:
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.INSUFFICIENT_EVIDENCE,
                is_sensor_fault=False,
                is_process_anomaly=False,
                confidence=0.0,
                evidence=["Primary line pressure channel is missing."],
                metrics=metrics,
            )

        # Slice recent evaluation window
        win_df = df.iloc[-recent_window_steps:].copy() if len(df) >= recent_window_steps else df.copy()
        p_series = win_df[pressure_col]

        # 1. Check Sensor Dropout (at least min_consecutive_dropout_steps at tail of window)
        is_zero_p = ((p_series.abs() < 0.01) | p_series.isna()).values
        tail_dropout = 0
        for val in reversed(is_zero_p):
            if val:
                tail_dropout += 1
            else:
                break
        if tail_dropout >= self.min_consecutive_dropout_steps:
            evidence.append(f"Pressure sensor dropped to rail/zero for {tail_dropout} consecutive steps.")
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.SENSOR_DROPOUT_SUSPECTED,
                is_sensor_fault=True,
                is_process_anomaly=False,
                confidence=0.95,
                evidence=evidence,
                metrics=metrics,
            )

        # 2. Check Sensor Freeze (at least min_consecutive_freeze_steps consecutive zero-diff at tail)
        p_diffs = (p_series.diff().abs() <= 1e-6).values
        tail_freeze = 0
        for val in reversed(p_diffs[1:] if len(p_diffs) > 1 else []):
            if val:
                tail_freeze += 1
            else:
                break
        if tail_freeze >= self.min_consecutive_freeze_steps:
            evidence.append(f"Pressure sensor locked at constant value for {tail_freeze} consecutive steps.")
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.SENSOR_FREEZE_SUSPECTED,
                is_sensor_fault=True,
                is_process_anomaly=False,
                confidence=0.95,
                evidence=evidence,
                metrics=metrics,
            )

        # Compute pressure residual against Digital Twin if available
        current_p = float(p_series.iloc[-1])
        metrics["current_pressure_mmhg"] = round(current_p, 2)

        p_residual = 0.0
        if has_dt:
            dt_p = float(win_df[dt_pressure_col].iloc[-1])
            p_residual = current_p - dt_p
            metrics["dt_pressure_residual_mmhg"] = round(p_residual, 2)

        # Check if an anomaly or statistical drift alarm is active
        statistical_drift_indicated = ewma_alarm_active or cusum_alarm_active or (abs(p_residual) > self.pressure_drift_threshold_mmhg)
        metrics["statistical_drift_indicated"] = statistical_drift_indicated

        # If pressure is normal and no drift alarm, verdict is NORMAL
        if not statistical_drift_indicated and current_p < self.pressure_occlusion_threshold_mmhg:
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.NORMAL,
                is_sensor_fault=False,
                is_process_anomaly=False,
                confidence=0.92,
                evidence=["Pressure readings within normal bounds; no statistical drift detected."],
                metrics=metrics,
            )

        # 3. Pressure is elevated or drifting: MULTI-CHANNEL CROSS-VALIDATION
        # CRITICAL SAFEGUARD: If no secondary sensor exists (no force, no flow, no reference),
        # we CANNOT physically confirm whether it is occlusion or transducer drift!
        if not has_force and not has_flow:
            evidence.append("Pressure anomaly detected, but neither force nor flow sensor is available for cross-validation.")
            evidence.append("Single-channel evidence cannot distinguish sensor drift from downstream occlusion.")
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.INSUFFICIENT_EVIDENCE,
                is_sensor_fault=False,
                is_process_anomaly=False,
                confidence=0.40,
                evidence=evidence,
                metrics=metrics,
            )

        # 4. Scenario B: True Downstream Occlusion Check
        # Physics: Under occlusion, pressure surges >> 250 mmHg, Plunger Force surges (F = P*A + F_f), Flow -> 0
        hydraulic_force_surging = False
        flow_collapsed = False

        if has_force:
            f_series = win_df[force_col]
            current_f = float(f_series.iloc[-1])
            expected_f_from_p = (current_p * 0.0133322 * self.plunger_area_cm2) + self.nominal_friction_n
            force_discrepancy = abs(current_f - expected_f_from_p)

            metrics["current_force_n"] = round(current_f, 3)
            metrics["expected_hydraulic_force_n"] = round(expected_f_from_p, 3)
            metrics["force_discrepancy_n"] = round(force_discrepancy, 3)

            # In true occlusion, actual force closely tracks expected hydraulic force (discrepancy < 2.0 N)
            # and current force is significantly elevated (> 3.5 N)
            if current_f > 3.0 and force_discrepancy < 2.5:
                hydraulic_force_surging = True
                evidence.append(f"Plunger force ({current_f:.2f} N) matches hydraulic rise predicted by pressure ({expected_f_from_p:.2f} N).")

        if has_flow:
            q_series = win_df[flow_col]
            current_q = float(q_series.iloc[-1])
            cmd_q = float(win_df[commanded_rate_col].iloc[-1]) if has_cmd else 50.0
            metrics["current_flow_ml_h"] = round(current_q, 2)
            metrics["commanded_flow_ml_h"] = round(cmd_q, 2)

            if current_q < (0.2 * cmd_q):  # Flow collapsed below 20% of setpoint
                flow_collapsed = True
                evidence.append(f"Measured flow ({current_q:.2f} mL/h) collapsed well below commanded setpoint ({cmd_q:.2f} mL/h).")

        # Confirm Occlusion if both pressure surges AND cross-sensor confirms hydraulic state
        if current_p >= self.pressure_occlusion_threshold_mmhg and (hydraulic_force_surging or flow_collapsed):
            evidence.append("Cross-sensor hydraulic coupling confirms genuine downstream fluid pathway occlusion.")
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED,
                is_sensor_fault=False,
                is_process_anomaly=True,
                confidence=0.96,
                evidence=evidence,
                metrics=metrics,
            )

        # 5. Scenario D: Environmental Temperature Influence Check
        # Check if divergence is strongly driven by ambient temperature changes
        if has_temp and abs(p_residual) > 5.0:
            # Use longer thermal window (up to 300 steps / 5 min) to match thermodynamic time constants
            thermal_win_len = min(len(df), max(recent_window_steps, 300))
            thermal_win = df.iloc[-thermal_win_len:]
            temp_series = thermal_win[temperature_col]
            p_res_series = thermal_win[pressure_col] - (thermal_win[dt_pressure_col] if has_dt else current_p)

            # Apply rolling smoothing to filter out high-frequency sensor noise and detect true physical thermal trend
            smooth_w = max(5, min(30, thermal_win_len // 6))
            t_smooth = temp_series.rolling(smooth_w).mean().dropna()
            p_smooth = p_res_series.rolling(smooth_w).mean().dropna()
            delta_t_span = float(t_smooth.max() - t_smooth.min()) if len(t_smooth) > 0 else 0.0
            if len(t_smooth) >= 15 and delta_t_span > 0.8 and p_smooth.std() > 1e-4:
                corr = float(np.corrcoef(t_smooth, p_smooth)[0, 1])
            else:
                corr = 0.0
            metrics["temp_pressure_correlation"] = round(corr, 3)
            metrics["temp_window_swing_c"] = round(delta_t_span, 2)
            if abs(corr) > 0.70 and delta_t_span > 0.8:
                evidence.append(f"Strong thermal correlation (r={corr:.2f}) between ambient temperature and sensor divergence.")
                return ConsistencyAssessment(
                    verdict=ConsistencyVerdict.TEMPERATURE_INFLUENCE_SUSPECTED,
                    is_sensor_fault=True,
                    is_process_anomaly=False,
                    confidence=0.85,
                    evidence=evidence,
                    metrics=metrics,
                )

        # 6. Scenario A: Sensor Drift Check
        # Pressure is drifting/elevated, BUT force is NORMAL and flow is NORMAL
        force_is_normal = False
        flow_is_normal = False

        if has_force:
            current_f = float(win_df[force_col].iloc[-1])
            # Normal syringe friction + nominal baseline force is typically 1.5 - 2.5 N
            if current_f < 2.5:
                force_is_normal = True
                evidence.append(f"Plunger force is normal ({current_f:.2f} N) despite elevated pressure reading ({current_p:.1f} mmHg).")

        if has_flow:
            current_q = float(win_df[flow_col].iloc[-1])
            cmd_q = float(win_df[commanded_rate_col].iloc[-1]) if has_cmd else 50.0
            if abs(current_q - cmd_q) < (0.25 * cmd_q):
                flow_is_normal = True
                evidence.append(f"Measured flow delivery ({current_q:.1f} mL/h) matches commanded setpoint ({cmd_q:.1f} mL/h).")

        if force_is_normal or flow_is_normal:
            evidence.append("Cross-sensor disagreement: Pressure sensor has diverged from true hydraulic state.")
            return ConsistencyAssessment(
                verdict=ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED,
                is_sensor_fault=True,
                is_process_anomaly=False,
                confidence=0.93,
                evidence=evidence,
                metrics=metrics,
            )

        # Fallback if sensors disagree ambiguously
        evidence.append("Ambiguous sensor evidence: Cross-sensor relationships do not cleanly match known fault signatures.")
        return ConsistencyAssessment(
            verdict=ConsistencyVerdict.CONFLICTING_EVIDENCE,
            is_sensor_fault=False,
            is_process_anomaly=False,
            confidence=0.50,
            evidence=evidence,
            metrics=metrics,
        )
