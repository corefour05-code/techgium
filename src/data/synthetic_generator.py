"""Research-oriented, physics-informed infusion pump simulator.

DISCLAIMER:
This module is a research-oriented, physics-informed mathematical simulator
designed solely for algorithmic benchmarking, testing fault detection pipelines,
and validating cross-sensor consistency algorithms. It is NOT a certified or
standards-compliant medical device simulator (e.g., it does NOT claim ISO 60601-2-24
compliance).

Features:
- Configurable sensor channel inventory (pressure, force, flow, motor encoder, temperature)
- Strict separation between:
    * Actual sensor readings (subject to drift, noise, faults)
    * Independent reference measurements (ground-truth physical state)
    * Digital Twin model predictions (ideal theoretical physics predictions)
- Explicit provenance tagging ('SYNTHETIC_DATA') on every generated record
- Reproducible random generation using fixed seeds
- Controlled fault scenarios:
    0: Normal
    1: Gradual positive sensor drift
    2: Gradual negative sensor drift
    3: Sudden offset bias
    4: Excessive noise
    5: Sensor dropout / disconnection
    6: Sensor freeze / ADC lockup
    7: Temperature-induced sensor variation
    8: Genuine downstream line occlusion (hydraulic exponential pressure rise)
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.config import DataProvenance, FaultClass, RANDOM_SEED, SYNTHETIC_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for physics-informed simulation runs."""
    sampling_rate_hz: float = 1.0           # 1 Hz sampling
    duration_seconds: int = 7200            # 2 hours default
    commanded_rate_ml_h: float = 50.0       # 50 mL/h nominal infusion
    nominal_baseline_pressure_mmhg: float = 40.0  # Normal venous catheter backpressure
    line_hydraulic_resistance: float = 0.8  # mmHg / (mL/h)
    line_compliance: float = 0.05           # mL / mmHg
    syringe_plunger_area_cm2: float = 3.14  # Standard 20 mL syringe
    syringe_friction_n: float = 1.5         # Mechanical friction
    ambient_temp_mean_c: float = 22.0       # Ambient room temperature
    ambient_temp_swing_c: float = 2.0       # Diurnal fluctuation
    sensor_pressure_noise_std: float = 0.8  # Normal sensor noise (mmHg)
    sensor_force_noise_std: float = 0.05    # Normal force noise (N)
    random_seed: int = RANDOM_SEED

    # Configurable channel presence
    enable_pressure: bool = True
    enable_force: bool = True
    enable_flow: bool = True
    enable_motor_encoder: bool = True
    enable_temperature: bool = True


def generate_infusion_run(
    scenario: FaultClass = FaultClass.NORMAL,
    config: Optional[SimulatorConfig] = None,
    fault_start_sec: int = 1800,            # Fault initiates at 30 min
    drift_rate_mmhg_per_hour: float = 20.0, # Rate of linear sensor drift
    offset_magnitude_mmhg: float = 35.0,    # Sudden step bias
    occlusion_time_constant_sec: float = 300.0, # Exponential occlusion ramp
) -> pd.DataFrame:
    """Generate a single physics-informed continuous infusion run.

    Args:
        scenario: Fault scenario enumeration.
        config: Simulator parameters and channel flags.
        fault_start_sec: Time in seconds when the fault scenario activates.
        drift_rate_mmhg_per_hour: Magnitude of drift rate if scenario is drift.
        offset_magnitude_mmhg: Magnitude of step offset if scenario is offset.
        occlusion_time_constant_sec: Characteristic hydraulic filling time for occlusion.

    Returns:
        DataFrame with clearly separated sensor, reference, digital twin, and ground-truth fields.
    """
    cfg = config or SimulatorConfig()
    rng = np.random.default_rng(cfg.random_seed)

    n_steps = int(cfg.duration_seconds * cfg.sampling_rate_hz)
    t_sec = np.linspace(0, cfg.duration_seconds, n_steps, endpoint=False)

    # 1. Operating setpoints & Digital Twin ideal expectations
    cmd_rate = np.full(n_steps, cfg.commanded_rate_ml_h)
    # Flow in mL/sec
    q_ideal_ml_s = cmd_rate / 3600.0

    # Digital Twin assumes normal operating physics:
    # P_dt = P_baseline + R_line * Q
    p_dt = cfg.nominal_baseline_pressure_mmhg + (cfg.line_hydraulic_resistance * cmd_rate)
    # F_dt = P_dt * area_conversion + friction (1 mmHg = 0.0133322 N/cm2)
    f_dt = (p_dt * 0.0133322 * cfg.syringe_plunger_area_cm2) + cfg.syringe_friction_n

    # 2. Environmental physics: Temperature model
    # Sinusoidal thermal cycle with minor random walk
    diurnal_cycle = np.sin(2.0 * np.pi * t_sec / 86400.0)
    temp_true = cfg.ambient_temp_mean_c + (cfg.ambient_temp_swing_c * diurnal_cycle) + rng.normal(0, 0.05, n_steps)

    # 3. Ground Truth Physical State (What actually happens in the fluid pathway)
    p_true = p_dt.copy()
    f_true = f_dt.copy()
    q_true = cmd_rate.copy()

    # Ground-truth labels array
    labels = np.full(n_steps, FaultClass.NORMAL.value, dtype=int)
    operating_state = np.full(n_steps, "NORMAL", dtype=object)

    # Handle Genuine Downstream Occlusion (Scenario 8)
    if scenario == FaultClass.GENUINE_DOWNSTREAM_OCCLUSION:
        fault_mask = t_sec >= fault_start_sec
        labels[fault_mask] = FaultClass.GENUINE_DOWNSTREAM_OCCLUSION.value
        operating_state[fault_mask] = "DOWNSTREAM_OCCLUSION"

        # Under physical occlusion: fluid cannot exit, pressure rises exponentially toward pump max
        # delta_P(t) = (P_max - P_baseline) * (1 - exp(-(t - t0) / tau))
        p_occlusion_max = 850.0  # Max occlusion pressure threshold in mmHg
        delta_t = np.maximum(0.0, t_sec - fault_start_sec)
        occlusion_rise = (p_occlusion_max - cfg.nominal_baseline_pressure_mmhg) * (
            1.0 - np.exp(-delta_t / occlusion_time_constant_sec)
        )
        p_true[fault_mask] = cfg.nominal_baseline_pressure_mmhg + occlusion_rise[fault_mask]
        # True plunger force rises in strict hydraulic proportion!
        f_true[fault_mask] = (p_true[fault_mask] * 0.0133322 * cfg.syringe_plunger_area_cm2) + cfg.syringe_friction_n
        # True delivered flow drops exponentially to zero
        q_true[fault_mask] = cmd_rate[fault_mask] * np.exp(-delta_t[fault_mask] / (occlusion_time_constant_sec / 2.0))

    # 4. Sensor Transducer Realization (What the physical sensors measure)
    # Baseline sensor readings = true physical state + normal transducer noise
    p_sensor = p_true + rng.normal(0, cfg.sensor_pressure_noise_std, n_steps)
    f_sensor = f_true + rng.normal(0, cfg.sensor_force_noise_std, n_steps)
    q_sensor = q_true + rng.normal(0, 0.5, n_steps)
    temp_sensor = temp_true + rng.normal(0, 0.1, n_steps)

    # Apply Sensor Transducer Faults (Scenarios 1-7)
    # NOTE: In sensor faults, true fluid physics remains normal, but the sensor output degrades!
    if scenario in [
        FaultClass.POSITIVE_DRIFT,
        FaultClass.NEGATIVE_DRIFT,
        FaultClass.SUDDEN_OFFSET,
        FaultClass.EXCESSIVE_NOISE,
        FaultClass.SENSOR_DROPOUT,
        FaultClass.SENSOR_FREEZE,
        FaultClass.TEMPERATURE_VARIATION,
    ]:
        fault_mask = t_sec >= fault_start_sec
        labels[fault_mask] = scenario.value

        if scenario == FaultClass.POSITIVE_DRIFT:
            operating_state[fault_mask] = "SENSOR_POSITIVE_DRIFT"
            # Linear positive ramp in hours: slope = drift_rate_mmhg_per_hour / 3600
            drift_val = (drift_rate_mmhg_per_hour / 3600.0) * np.maximum(0.0, t_sec - fault_start_sec)
            p_sensor[fault_mask] += drift_val[fault_mask]

        elif scenario == FaultClass.NEGATIVE_DRIFT:
            operating_state[fault_mask] = "SENSOR_NEGATIVE_DRIFT"
            drift_val = (drift_rate_mmhg_per_hour / 3600.0) * np.maximum(0.0, t_sec - fault_start_sec)
            p_sensor[fault_mask] -= drift_val[fault_mask]

        elif scenario == FaultClass.SUDDEN_OFFSET:
            operating_state[fault_mask] = "SENSOR_OFFSET"
            p_sensor[fault_mask] += offset_magnitude_mmhg

        elif scenario == FaultClass.EXCESSIVE_NOISE:
            operating_state[fault_mask] = "SENSOR_EXCESSIVE_NOISE"
            # 5x higher variance
            p_sensor[fault_mask] += rng.normal(0, cfg.sensor_pressure_noise_std * 5.0, np.sum(fault_mask))

        elif scenario == FaultClass.SENSOR_DROPOUT:
            operating_state[fault_mask] = "SENSOR_DROPOUT"
            p_sensor[fault_mask] = 0.0

        elif scenario == FaultClass.SENSOR_FREEZE:
            operating_state[fault_mask] = "SENSOR_FREEZE"
            # Freeze at the exact value immediately prior to fault
            freeze_idx = np.searchsorted(t_sec, fault_start_sec)
            frozen_val = p_sensor[freeze_idx]
            p_sensor[fault_mask] = frozen_val

        elif scenario == FaultClass.TEMPERATURE_VARIATION:
            operating_state[fault_mask] = "TEMPERATURE_EFFECT"
            # Environmental thermal ramp (e.g. ambient warming or incubator transition)
            thermal_ramp = 0.0035 * np.maximum(0.0, t_sec - fault_start_sec)
            temp_sensor[fault_mask] += thermal_ramp[fault_mask]
            # Sensor uncompensated thermal sensitivity coefficient (1.6 mmHg / °C drift)
            temp_effect = 1.6 * thermal_ramp
            p_sensor[fault_mask] += temp_effect[fault_mask]

    # Cumulative motor position (steps)
    steps_per_ml = 2000.0
    motor_pos = np.cumsum(q_ideal_ml_s / cfg.sampling_rate_hz) * steps_per_ml

    # Build Output DataFrame
    data: Dict[str, Any] = {
        "timestamp_sec": t_sec,
        "provenance": DataProvenance.SYNTHETIC_DATA.value,
        "is_synthetic": True,
        "simulation_disclaimer": "Research-oriented physics-informed simulator (not ISO standards compliant)",
        "operating_state": operating_state,
        "ground_truth_fault_label": labels,
        "commanded_rate_ml_h": cmd_rate,
        # Digital Twin Predictions (theoretical expectations)
        "dt_predicted_pressure_mmhg": p_dt,
        "dt_predicted_force_n": f_dt,
        # Independent Reference Measurements (simulated true fluid physics)
        "reference_pressure_mmhg": p_true,
        "reference_force_n": f_true,
        "reference_flow_ml_h": q_true,
        "reference_temperature_c": temp_true,
    }

    # Sensor measurements (only included if channel is enabled)
    if cfg.enable_pressure:
        data["sensor_pressure_mmhg"] = p_sensor
    if cfg.enable_force:
        data["sensor_force_n"] = f_sensor
    if cfg.enable_flow:
        data["sensor_flow_ml_h"] = q_sensor
    if cfg.enable_motor_encoder:
        data["motor_encoder_steps"] = motor_pos
    if cfg.enable_temperature:
        data["sensor_temperature_c"] = temp_sensor

    df_out = pd.DataFrame(data)
    return df_out

