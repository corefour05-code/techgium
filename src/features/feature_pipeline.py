"""Unified, configurable Feature Extraction Pipeline for Infusion Pump ML.

Extracts signal dynamics, multi-scale rolling statistics, cross-sensor physical coupling,
adaptive catheter baseline relationships, and hardware integrity indicators.
Robust to missing sensor channels.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.features.rolling_features import compute_rolling_trend_slope
from src.features.signal_features import compute_rate_of_change, compute_acceleration
from src.features.adaptive_baseline import AdaptiveBaselineEstimator, BaselineEstimate


FEATURE_NAMES = [
    "p_raw",
    "p_roll_mean_15",
    "p_roll_std_15",
    "p_roll_slope_15",
    "p_roll_mean_60",
    "p_roll_std_60",
    "p_roll_slope_60",
    "p_roll_variance_60",
    "p_rate_of_change",
    "p_acceleration",
    "p_dt_residual",
    "p_deviation_from_baseline",
    "baseline_confidence",
    "signal_quality_index",
    "p_flow_hydraulic_consistency",
    "force_raw",
    "force_expected_from_p",
    "force_hydraulic_residual",
    "flow_raw",
    "flow_delivery_error",
    "temp_raw",
    "temp_roll_delta_60",
    "sensor_freeze_counter",
    "sensor_dropout_counter",
    "has_force_channel",
    "has_flow_channel",
    "has_temp_channel",
]


class SensorFeaturePipeline:
    """Extracts standardized, leakage-free feature vectors for classification and regression."""

    def __init__(
        self,
        plunger_area_cm2: float = 3.14,
        friction_n: float = 1.5,
        nominal_baseline_pressure_mmhg: float = 40.0,
        line_hydraulic_resistance: float = 0.8,
    ):
        self.plunger_area_cm2 = plunger_area_cm2
        self.friction_n = friction_n
        self.nominal_baseline_pressure_mmhg = nominal_baseline_pressure_mmhg
        self.line_hydraulic_resistance = line_hydraulic_resistance
        self.baseline_estimator = AdaptiveBaselineEstimator(
            nominal_fallback_baseline_mmhg=nominal_baseline_pressure_mmhg
        )

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        pressure_col: str = "sensor_pressure_mmhg",
        force_col: str = "sensor_force_n",
        flow_col: str = "sensor_flow_ml_h",
        temp_col: str = "sensor_temperature_c",
        cmd_rate_col: str = "commanded_rate_ml_h",
        dt_pressure_col: str = "dt_predicted_pressure_mmhg",
        baseline_mmhg: Optional[float] = None,
    ) -> pd.DataFrame:
        """Transform a time-series run into aligned feature vectors with adaptive baseline tracking."""
        n = len(df)
        feats = pd.DataFrame(index=df.index)

        # 1. Pressure Channel Features
        has_pressure = pressure_col in df.columns
        if not has_pressure:
            raise ValueError(f"Required pressure channel '{pressure_col}' missing from DataFrame.")

        p_series = pd.to_numeric(df[pressure_col], errors="coerce").ffill().bfill()
        feats["p_raw"] = p_series

        # Rolling 15-step window
        feats["p_roll_mean_15"] = p_series.rolling(15, min_periods=1).mean()
        feats["p_roll_std_15"] = p_series.rolling(15, min_periods=1).std().fillna(0.0)
        feats["p_roll_slope_15"] = compute_rolling_trend_slope(p_series, 15).bfill().fillna(0.0)

        # Rolling 60-step window
        feats["p_roll_mean_60"] = p_series.rolling(60, min_periods=1).mean()
        feats["p_roll_std_60"] = p_series.rolling(60, min_periods=1).std().fillna(0.0)
        feats["p_roll_slope_60"] = compute_rolling_trend_slope(p_series, 60).bfill().fillna(0.0)
        feats["p_roll_variance_60"] = p_series.rolling(60, min_periods=1).var().fillna(0.0)

        # Signal derivatives
        feats["p_rate_of_change"] = compute_rate_of_change(p_series).fillna(0.0)
        feats["p_acceleration"] = compute_acceleration(p_series).fillna(0.0)

        # Commanded setpoint
        if cmd_rate_col in df.columns:
            cmd_rate = pd.to_numeric(df[cmd_rate_col], errors="coerce").fillna(50.0)
        else:
            cmd_rate = pd.Series(50.0, index=df.index)

        # 2. Adaptive Baseline & Digital Twin Hydrodynamic Expectation
        # Determine active baseline
        if baseline_mmhg is not None:
            active_baseline = float(baseline_mmhg)
            baseline_conf = 1.0
        elif dt_pressure_col in df.columns:
            # If explicit dt_predicted_pressure exists, derive catheter baseline from it: P_base = P_dt - R*Q
            p_dt_raw = pd.to_numeric(df[dt_pressure_col], errors="coerce")
            inferred_base = p_dt_raw - (self.line_hydraulic_resistance * cmd_rate)
            active_baseline = float(inferred_base.iloc[0]) if not inferred_base.isna().all() else self.nominal_baseline_pressure_mmhg
            baseline_conf = 0.95
        else:
            # Estimate adaptively from initial pre-infusion window (first 25 samples)
            init_n = min(25, len(p_series))
            tare_p = p_series.iloc[:init_n]
            tare_q = pd.to_numeric(df[flow_col].iloc[:init_n], errors="coerce") if flow_col in df.columns else None
            est = self.baseline_estimator.estimate_from_pre_infusion(tare_p, flow_window=tare_q)
            if est.is_valid:
                active_baseline = est.baseline_pressure_mmhg
                baseline_conf = est.confidence
            else:
                # If during active flow without pre-infusion tare or explicit DT prediction,
                # infer catheter baseline hydrodynamically: P_base = P_initial - R_line * Q_init
                q_init = float(tare_q.iloc[0]) if (tare_q is not None and len(tare_q) > 0) else float(cmd_rate.iloc[0])
                p_init = float(tare_p.iloc[0])
                if q_init > 2.0:
                    inferred_p0 = p_init - (self.line_hydraulic_resistance * q_init)
                    active_baseline = max(0.0, inferred_p0)
                    baseline_conf = 0.70
                else:
                    active_baseline = est.baseline_pressure_mmhg
                    baseline_conf = est.confidence

        # Hydrodynamic Digital Twin expectation: P_dt = P_baseline + (R_line * Q_cmd)
        p_dt = active_baseline + (self.line_hydraulic_resistance * cmd_rate)
        feats["p_dt_residual"] = p_series - p_dt
        feats["p_deviation_from_baseline"] = p_series - active_baseline
        feats["baseline_confidence"] = baseline_conf
        feats["signal_quality_index"] = 1.0 / (1.0 + feats["p_roll_std_60"])

        # 3. Force Channel Features
        has_force = force_col in df.columns
        feats["has_force_channel"] = 1.0 if has_force else 0.0
        expected_f = (p_series * 0.0133322 * self.plunger_area_cm2) + self.friction_n
        feats["force_expected_from_p"] = expected_f

        if has_force:
            f_series = pd.to_numeric(df[force_col], errors="coerce").ffill().bfill()
            feats["force_raw"] = f_series
            feats["force_hydraulic_residual"] = f_series - expected_f
        else:
            # Nominal imputation for missing channel
            feats["force_raw"] = expected_f
            feats["force_hydraulic_residual"] = 0.0

        # 4. Flow Channel Features & Hydraulic Consistency
        has_flow = flow_col in df.columns
        feats["has_flow_channel"] = 1.0 if has_flow else 0.0
        if has_flow:
            q_series = pd.to_numeric(df[flow_col], errors="coerce").ffill().bfill()
            feats["flow_raw"] = q_series
            feats["flow_delivery_error"] = q_series - cmd_rate
            # Hydraulic consistency: Delta P / Q - R_line
            measured_res = feats["p_deviation_from_baseline"] / np.maximum(1e-2, q_series)
            feats["p_flow_hydraulic_consistency"] = measured_res - self.line_hydraulic_resistance
        else:
            feats["flow_raw"] = cmd_rate
            feats["flow_delivery_error"] = 0.0
            feats["p_flow_hydraulic_consistency"] = 0.0

        # 5. Temperature Channel Features
        has_temp = temp_col in df.columns
        feats["has_temp_channel"] = 1.0 if has_temp else 0.0
        if has_temp:
            t_series = pd.to_numeric(df[temp_col], errors="coerce").ffill().bfill()
            feats["temp_raw"] = t_series
            t_base_60 = t_series.shift(60).fillna(t_series.iloc[0])
            feats["temp_roll_delta_60"] = t_series - t_base_60
        else:
            feats["temp_raw"] = 22.0
            feats["temp_roll_delta_60"] = 0.0

        # 6. Electrical & ADC Integrity Counters
        # Freeze counter: consecutive diff == 0, excluding regions undergoing sudden offset jumps (> 3.0 mmHg)
        has_recent_step = (p_series.diff().abs() > 3.0).rolling(60, min_periods=1).max().fillna(0).astype(bool)
        diff_zero = ((p_series.diff().abs() <= 1e-6) & (~has_recent_step)).astype(int)
        freeze_cumsum = diff_zero.cumsum()
        freeze_reset = freeze_cumsum.where(diff_zero == 0).ffill().fillna(0)
        feats["sensor_freeze_counter"] = freeze_cumsum - freeze_reset

        # Dropout counter: consecutive readings <= 0.01
        is_zero = ((p_series.abs() <= 0.01) | p_series.isna()).astype(int)
        zero_cumsum = is_zero.cumsum()
        zero_reset = zero_cumsum.where(is_zero == 0).ffill().fillna(0)
        feats["sensor_dropout_counter"] = zero_cumsum - zero_reset

        # Ensure exact column order and no remaining NaNs
        clean_feats = feats[FEATURE_NAMES].copy()
        return clean_feats.ffill().bfill().fillna(0.0)

    def transform_window_dict(self, window_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform a real-time payload containing array/list readings into a feature row."""
        length = len(window_data.get("sensor_pressure_mmhg", [50.0]))
        df_win = pd.DataFrame({
            "sensor_pressure_mmhg": window_data.get("sensor_pressure_mmhg", [50.0] * length),
        })
        if "sensor_force_n" in window_data and window_data["sensor_force_n"] is not None:
            df_win["sensor_force_n"] = window_data["sensor_force_n"]
        if "sensor_flow_ml_h" in window_data and window_data["sensor_flow_ml_h"] is not None:
            df_win["sensor_flow_ml_h"] = window_data["sensor_flow_ml_h"]
        if "sensor_temperature_c" in window_data and window_data["sensor_temperature_c"] is not None:
            df_win["sensor_temperature_c"] = window_data["sensor_temperature_c"]
        if "commanded_rate_ml_h" in window_data and window_data["commanded_rate_ml_h"] is not None:
            cmd_val = window_data["commanded_rate_ml_h"]
            df_win["commanded_rate_ml_h"] = cmd_val if isinstance(cmd_val, list) else [cmd_val] * length
        if "dt_predicted_pressure_mmhg" in window_data and window_data["dt_predicted_pressure_mmhg"] is not None:
            dt_val = window_data["dt_predicted_pressure_mmhg"]
            df_win["dt_predicted_pressure_mmhg"] = dt_val if isinstance(dt_val, list) else [dt_val] * length

        base_val = window_data.get("baseline_pressure_mmhg") or window_data.get("pre_infusion_tare_mmhg")
        base_num = float(base_val) if base_val is not None else None

        full_feats = self.transform_dataframe(df_win, baseline_mmhg=base_num)
        return full_feats.iloc[[-1]]
