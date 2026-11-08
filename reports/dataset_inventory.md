# Research Dataset Inventory & Availability Report

Generated for the MedTech Challenge: Self-Calibrating and Self-Healing Infusion Pump ML System.

| Dataset ID | Title | Status | Files Count | Size (MB) | Supported Sensors |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **dosepi** | DosePI: Extensive Dataset for Peristaltic Pum... | `ACQUIRED` | 19 | 4.567 MB | flow, weight_dosing, motor_steps, operating_parameters |
| **pressure_occlusion** | Multi-infusion with integrated multiple press... | `NOT_ACQUIRED` | 0 | 0.0 MB | pressure |
| **force_failure** | Development of a Real-time Force-based Algori... | `NOT_ACQUIRED` | 0 | 0.0 MB | force, pressure |
| **smart_pump_logs** | Hospitals Infusion Data Set: Spain (Determina... | `NOT_ACQUIRED` | 0 | 0.0 MB | event_logs, operating_conditions, alarm_records |
| **sensor_defects** | Typical Sensor Defects Dataset... | `ACQUIRED` | 15 | 86.949 MB | temperature, smoke_defect_channels |


---

## Dataset: DosePI: Extensive Dataset for Peristaltic Pump Accuracy Enhancement

- **ID:** `dosepi`
- **Publication / URL:** [10.1038/s41597-025-05902-z](https://www.nature.com/articles/s41597-025-05902-z)
- **Status:** `ACQUIRED`
- **Dataset Type:** Pharmaceutical peristaltic filling pump dataset (not clinical intravenous continuous infusion).
- **Analytical Purpose:** Pump delivery accuracy, dosing variation, compensation modelling, pump output behaviour.

### Inspected Files:

#### File: `dosepi-datasets.zip` (0.926 MB)


#### File: `0.2 ml.csv` (0.032 MB)

- Total rows inspected: 2,652
- Total columns: 1
- Columns found: `['p [g]']`

#### File: `0.3 ml.csv` (0.089 MB)

- Total rows inspected: 16,221
- Total columns: 1
- Columns found: `['p [g]']`

#### File: `1.2 ml.csv` (0.09 MB)

- Total rows inspected: 16,249
- Total columns: 1
- Columns found: `['p [g]']`

#### File: `2.0 ml.csv` (0.043 MB)

- Total rows inspected: 7,824
- Total columns: 1
- Columns found: `['p [g]']`

#### File: `p10-0.1ml.csv` (0.381 MB)

- Total rows inspected: 9,628
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `p10-0.2ml.csv` (0.416 MB)

- Total rows inspected: 11,438
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `p10-1.2ml.csv` (0.242 MB)

- Total rows inspected: 8,375
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `p10-2.0ml.csv` (0.275 MB)

- Total rows inspected: 9,519
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `6neur-0.1ml.csv` (0.092 MB)

- Total rows inspected: 2,372
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `6neur-0.3ml.csv` (0.205 MB)

- Total rows inspected: 5,719
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `6neur-1.2ml.csv` (0.134 MB)

- Total rows inspected: 4,756
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `10neur-Vc+V+IncrAv-0.3ml.csv` (0.323 MB)

- Total rows inspected: 8,990
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `10neur-Vc+V+IncrAv-1.2ml.csv` (0.257 MB)

- Total rows inspected: 9,120
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `8neur-V+IncrAv-1.2ml.csv` (0.204 MB)

- Total rows inspected: 7,224
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `6neur-Vc+V+IncrAv-1.2ml.csv` (0.255 MB)

- Total rows inspected: 9,022
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `8neur-V+IncrAv-0.3ml.csv` (0.28 MB)

- Total rows inspected: 7,826
- Total columns: 3
- Columns found: `['Set volume', 'Actual weight', 'Prediction']`

#### File: `0.3 ml-AR10-ADCS-active.csv` (0.223 MB)

- Total rows inspected: 6,228
- Total columns: 3
- Columns found: `['V_c', 'p [g]', 'Prediction']`

#### File: `1.2 ml-AR10-ADCS-active.csv` (0.099 MB)

- Total rows inspected: 5,547
- Total columns: 2
- Columns found: `['V_c', 'p [g]']`

## Dataset: Multi-infusion with integrated multiple pressure sensing allows earlier detection of line occlusions

- **ID:** `pressure_occlusion`
- **Publication / URL:** [10.3390/s21206934](https://pmc.ncbi.nlm.nih.gov/articles/PMC8555317/)
- **Status:** `NOT_ACQUIRED`
- **Dataset Type:** In vitro multi-infusion clinical pressure-event research study.
- **Analytical Purpose:** Pressure behaviour, normal infusion patterns, genuine occlusion events, pressure anomaly analysis.

> **Note on Acquisition:** This dataset has not yet been populated or requires author/institutional data governance approval.

## Dataset: Development of a Real-time Force-based Algorithm for Infusion Failure Detection

- **ID:** `force_failure`
- **Publication / URL:** [10.1177/19322968241247530](https://pmc.ncbi.nlm.nih.gov/articles/PMC11535327/)
- **Status:** `NOT_ACQUIRED`
- **Dataset Type:** In vitro force and line-pressure experimental failure study.
- **Analytical Purpose:** Pressure and force relationships, infusion failure classification, cross-sensor consistency analysis.

> **Note on Acquisition:** This dataset has not yet been populated or requires author/institutional data governance approval.

## Dataset: Hospitals Infusion Data Set: Spain (Determinants of Intravenous Infusion Longevity and Infusion Failure)

- **ID:** `smart_pump_logs`
- **Publication / URL:** [10.2196/48628](https://ai.jmir.org/2023/1/e48628/)
- **Status:** `NOT_ACQUIRED`
- **Dataset Type:** Retrospective hospital smart pump event logs (operational discrete events, not continuous raw waveforms).
- **Analytical Purpose:** Infusion operating conditions, pump event information, occlusion and infusion termination context.

> **Note on Acquisition:** This dataset has not yet been populated or requires author/institutional data governance approval.

## Dataset: Typical Sensor Defects Dataset

- **ID:** `sensor_defects`
- **Publication / URL:** [10.5281/zenodo.56358](https://zenodo.org/records/56358)
- **Status:** `ACQUIRED`
- **Dataset Type:** Generic multi-sensor defect benchmark dataset (hazardous material storage container, not an infusion pump).
- **Analytical Purpose:** Sensor drift, offset, noise, peaks and abnormal sensor behaviour benchmark.

### Inspected Files:

#### File: `Configuration.pdf` (0.058 MB)


#### File: `data.zip` (3.859 MB)


#### File: `data_drift_0_001.csv` (6.488 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_drift_0_0025.csv` (6.487 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_drift_0_005.csv` (6.487 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_noise_0dB.csv` (6.488 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_noise_10dB.csv` (6.487 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_offset_1.csv` (6.322 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_offset_2.csv` (6.322 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_offset_5.csv` (6.322 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_peak_1.csv` (6.323 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_peak_10.csv` (6.333 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_peak_2.csv` (6.324 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_peak_5.csv` (6.327 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

#### File: `data_standard.csv` (6.322 MB)

- Total rows inspected: 50,000
- Total columns: 24
- Columns found: `['Time [sec]', 'Smoke_Inside_0', 'Smoke_Ventilation_In', 'Leakage_0', 'Leakage_1', 'Pressure_Ventilation_Out', 'Smoke_Ventilation_Out', 'Smoke_Inside_1', 'Gas_Ventilation_Out', 'Temp_Outside_0 [°C]', 'Temp_Outside_1 [°C]', 'Temp_Ventilation_Out [°C]', 'Temp_Inside_3 [°C]', 'Temp_Inside_0 [°C]', 'Temp_Inside_1 [°C]', 'Temp_Inside_2 [°C]', 'Temp_Inside_4 [°C]', 'Temp_Ventilation_In [°C]', 'Temp_Inside_6 [°C]', 'Temp_Inside_7 [°C]', 'Temp_Inside_9 [°C]', 'Temp_Inside_8 [°C]', 'Temp_Inside_5 [°C]', 'Unnamed: 23']`
- Timestamp Analysis: {"column_identified": "Time [sec]", "sampling_interval_median": 1.0, "sampling_interval_min": 1.0, "sampling_interval_max": 1.0, "is_strictly_monotonic": true}

