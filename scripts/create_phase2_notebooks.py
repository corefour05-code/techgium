"""Generate 02_data_preprocessing.ipynb and 03_drift_analysis.ipynb."""

from pathlib import Path
import nbformat as nbf

# ----------------- Notebook 2: Preprocessing -----------------
nb2 = nbf.v4.new_notebook()
nb2.cells.append(nbf.v4.new_markdown_cell("""# 02. Sensor Validation, Data Preprocessing & Feature Engineering

**Project:** Self-Calibrating and Self-Healing Medical Devices (Infusion Pump Target)  
**Phase:** Phase 2 - Foundational Validation & Preprocessing Pipeline  

---

## Overview
This notebook demonstrates:
1. Non-destructive range validation and sensor anomaly detection (freeze, dropout, missing gaps).
2. Documented cleaning of sentinel values in DosePI (`-99.0` and `-1.0`) while preserving raw readings.
3. Zenodo trailing empty column removal and unicode header normalization.
4. Configurable multi-window rolling feature extraction and OLS trend slopes.
5. Physics-informed infusion pump simulation with strict field separation (`sensor`, `reference`, `digital_twin`).
"""))

nb2.cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.config import SensorType, DataProvenance, FaultClass, RAW_DATA_DIR
from src.data.validation import validate_sensor_dataframe, detect_frozen_sensor, detect_sensor_dropout
from src.data.preprocessing import clean_dosepi_dataframe, clean_sensor_defects_dataframe
from src.data.synthetic_generator import generate_infusion_run, SimulatorConfig
from src.features.rolling_features import RollingFeatureExtractor
from src.features.cross_sensor_features import CrossSensorFeatureExtractor

print("Imports ready.")
"""))

nb2.cells.append(nbf.v4.new_code_cell("""# Inspect Physics-Informed Simulator Output
cfg = SimulatorConfig(duration_seconds=600, random_seed=42)
df_sim = generate_infusion_run(scenario=FaultClass.POSITIVE_DRIFT, config=cfg, fault_start_sec=300, drift_rate_mmhg_per_hour=30.0)

print(f"Generated {len(df_sim)} rows.")
print("Provenance:", df_sim['provenance'].iloc[0])
print("Separated fields available:")
for col in df_sim.columns:
    print(f" - {col}")
display(df_sim.head())
"""))

nb2.cells.append(nbf.v4.new_code_cell("""# Feature Engineering Demo
extractor = RollingFeatureExtractor(windows=[15, 60], features=['mean', 'std', 'slope', 'snr'])
roll_feats = extractor.extract_features(df_sim['sensor_pressure_mmhg'], prefix="p_sensor")

cross_ext = CrossSensorFeatureExtractor()
cross_feats = cross_ext.extract(
    df_sim,
    pressure_col="sensor_pressure_mmhg",
    force_col="sensor_force_n",
    flow_col="sensor_flow_ml_h",
    commanded_rate_col="commanded_rate_ml_h",
    digital_twin_col="dt_predicted_pressure_mmhg",
    target_sensor_col="sensor_pressure_mmhg"
)

combined_features = pd.concat([df_sim[['timestamp_sec', 'sensor_pressure_mmhg']], roll_feats, cross_feats], axis=1)
display(combined_features.head())
"""))

p2 = Path(r"D:\techgium\notebooks\02_data_preprocessing.ipynb")
with open(p2, "w", encoding="utf-8") as f:
    nbf.write(nb2, f)
print(f"Generated {p2}")

# ----------------- Notebook 3: Drift Analysis -----------------
nb3 = nbf.v4.new_notebook()
nb3.cells.append(nbf.v4.new_markdown_cell("""# 03. Statistical Drift Detection: EWMA & CUSUM

**Project:** Self-Calibrating and Self-Healing Medical Devices (Infusion Pump Target)  
**Phase:** Phase 2 - Foundational Statistical Drift Engines  

---

## Objectives
1. Calibrate EWMA and CUSUM control charts strictly on in-control baseline data (no arbitrary thresholds).
2. Benchmark detection delay, false alarm rate, and detection rate across drift magnitudes.
3. Distinguish single-sensor transducer drift from hydraulic downstream line occlusion via cross-sensor consistency.
"""))

nb3.cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import matplotlib.pyplot as plt

# Display drift evaluation report summary table
report_json = PROJECT_ROOT / "reports" / "drift_metrics.json"
if report_json.exists():
    with open(report_json, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    df_metrics = pd.DataFrame(data)
    display(df_metrics[['scenario_name', 'detector_name', 'magnitude', 'magnitude_unit', 'detection_rate_pct', 'false_alarm_rate_pct', 'detection_delay_sec', 'drift_direction_detected']])
"""))

p3 = Path(r"D:\techgium\notebooks\03_drift_analysis.ipynb")
with open(p3, "w", encoding="utf-8") as f:
    nbf.write(nb3, f)
print(f"Generated {p3}")

