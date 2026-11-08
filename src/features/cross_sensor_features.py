"""Cross-sensor consistency and residual feature extraction.

Constructs physically meaningful relational features between available sensor channels
without requiring or assuming that every pump contains all sensor types.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd


class CrossSensorFeatureExtractor:
    """Extracts relational and residual features across available paired channels."""

    def __init__(
        self,
        nominal_plunger_area_cm2: float = 3.14,  # Standard 20 mL syringe cross-sectional area
        baseline_temperature_c: float = 22.0,    # Nominal room temperature
    ):
        self.nominal_plunger_area_cm2 = nominal_plunger_area_cm2
        self.baseline_temperature_c = baseline_temperature_c

    def extract(
        self,
        df: pd.DataFrame,
        pressure_col: Optional[str] = None,
        force_col: Optional[str] = None,
        flow_col: Optional[str] = None,
        commanded_rate_col: Optional[str] = None,
        temperature_col: Optional[str] = None,
        reference_col: Optional[str] = None,
        digital_twin_col: Optional[str] = None,
        target_sensor_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extract all available cross-sensor relationships.

        Returns:
            DataFrame containing extracted cross-sensor columns.
        """
        out_df = pd.DataFrame(index=df.index)

        # 1. Hydraulic Coupling: Pressure (mmHg) vs Plunger Force (N)
        # In a closed syringe/tubing system, fluid pressure translates to drive force:
        # F = P * A + F_friction. (1 mmHg = 133.322 Pa = 0.0133322 N/cm2)
        if pressure_col and force_col and pressure_col in df.columns and force_col in df.columns:
            p_val = pd.to_numeric(df[pressure_col], errors="coerce")
            f_val = pd.to_numeric(df[force_col], errors="coerce")

            # Hydraulic force component in Newtons
            f_hydraulic_expected = (p_val * 0.0133322) * self.nominal_plunger_area_cm2
            out_df["expected_force_from_pressure_n"] = f_hydraulic_expected
            # Force residual (discrepancy from pure hydraulic coupling)
            out_df["force_hydraulic_residual_n"] = f_val - f_hydraulic_expected
            # Ratio of force to pressure
            safe_p = np.where(p_val.abs() > 1.0, p_val, np.nan)
            out_df["force_to_pressure_ratio"] = f_val / safe_p

        # 2. Delivery Accuracy: Commanded Rate vs Measured Flow / Delivery
        if flow_col and commanded_rate_col and flow_col in df.columns and commanded_rate_col in df.columns:
            flow_val = pd.to_numeric(df[flow_col], errors="coerce")
            cmd_val = pd.to_numeric(df[commanded_rate_col], errors="coerce")
            delivery_err = flow_val - cmd_val
            out_df["delivery_flow_error"] = delivery_err
            safe_cmd = np.where(cmd_val > 0.01, cmd_val, np.nan)
            out_df["delivery_relative_error_pct"] = (delivery_err / safe_cmd) * 100.0

        # 3. Reference Residual (Ground Truth paired error)
        # Only valid when an independent reference measurement exists
        if reference_col and target_sensor_col and reference_col in df.columns and target_sensor_col in df.columns:
            ref_val = pd.to_numeric(df[reference_col], errors="coerce")
            tgt_val = pd.to_numeric(df[target_sensor_col], errors="coerce")
            out_df["true_sensor_error_residual"] = tgt_val - ref_val

        # 4. Digital Twin Expected Value Residual
        # Note: A Digital Twin prediction is NOT an independent measurement
        if digital_twin_col and target_sensor_col and digital_twin_col in df.columns and target_sensor_col in df.columns:
            dt_val = pd.to_numeric(df[digital_twin_col], errors="coerce")
            tgt_val = pd.to_numeric(df[target_sensor_col], errors="coerce")
            out_df["digital_twin_residual"] = tgt_val - dt_val

        # 5. Environmental Temperature Interaction
        if temperature_col and temperature_col in df.columns:
            temp_val = pd.to_numeric(df[temperature_col], errors="coerce")
            out_df["temperature_deviation_from_nominal"] = temp_val - self.baseline_temperature_c

        return out_df

