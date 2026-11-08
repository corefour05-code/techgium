# Exploratory Data Analysis (EDA) & Data Quality Report

**Project:** Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins (Infusion Pump Target)
**Author:** Senior ML & Biomedical AI Team
**Phase:** Phase 1 - Environment, Acquisition, Inspection & Baseline Analysis

---

## 1. Executive Summary

- **Total Target Datasets:** 5
- **Datasets Acquired / Local Files Inspected:** 2 (`dosepi, sensor_defects`)
- **Datasets Requiring Manual / Author Request:** 3 (`pressure_occlusion, force_failure, smart_pump_logs`)

### Critical ML Engineering Distinctions:

1. **No Continuous Raw Intravenous Pump Waveform in Open Domain:** None of the public datasets provide an open-access, continuous multi-sensor intravenous infusion stream with simultaneous pressure, force, flow, and motor encoder signals over months of aging.
2. **DosePI Domain Specifics:** DosePI is an exceptional high-resolution dataset, but it represents pharmaceutical peristaltic dosing/filling cycles (dispensing target volumes into vials with high-precision balance readouts), not continuous clinical IV delivery in an ambulatory patient setting.
3. **Typical Sensor Defects Role:** Zenodo 56358 provides clean, controlled synthetic injections (drift, offset, peaks, noise) atop real multi-channel sensor baseline in a container. It serves as a gold standard for validating our statistical drift and offset detection engines (EWMA/CUSUM), but is not an infusion pump dataset.
4. **Clinical Data Governance Restrictions:** Datasets B (PMC8555317), C (PMC11535327), and D (JMIR AI e48628) represent vital infusion domain literature, but their continuous raw sensor archives are protected by hospital clinical governance and author-access restrictions.

---

## 2. In-Depth Inspection by Dataset

### Dataset: DosePI: Extensive Dataset for Peristaltic Pump Accuracy Enhancement (`dosepi`)

- **DOI / Link:** [10.1038/s41597-025-05902-z](https://www.nature.com/articles/s41597-025-05902-z)
- **Acquisition Status:** `ACQUIRED`
- **Domain Context:** Pharmaceutical peristaltic filling pump dataset (not clinical intravenous continuous infusion).
- **Analytical Purpose in ML System:** Pump delivery accuracy, dosing variation, compensation modelling, pump output behaviour.
- **Supported Sensors:** `flow, weight_dosing, motor_steps, operating_parameters`

#### Inspected Files (19 total):

##### File: `dosepi-datasets.zip`
- **Path:** `raw\dosepi\dosepi-datasets.zip`
- **Size:** 0.93 MB (970,545 bytes)
##### File: `0.2 ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\core\0.2 ml.csv`
- **Size:** 0.03 MB (33,494 bytes)
- **Rows Inspected:** 2,652
- **Columns (1):** `p [g]`
- **Duplicate Rows:** 2584

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p [g]` | `float64` | 0 (0.0%) | 68 | -99.0000 | 0.2130 | -1.7453 ± 13.7565 |

##### File: `0.3 ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\core\0.3 ml.csv`
- **Size:** 0.09 MB (93,090 bytes)
- **Rows Inspected:** 16,221
- **Columns (1):** `p [g]`
- **Duplicate Rows:** 16187

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p [g]` | `float64` | 0 (0.0%) | 34 | -99.0000 | 0.3150 | -0.0304 ± 5.7200 |

##### File: `1.2 ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\core\1.2 ml.csv`
- **Size:** 0.09 MB (94,727 bytes)
- **Rows Inspected:** 16,249
- **Columns (1):** `p [g]`
- **Duplicate Rows:** 16209

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p [g]` | `float64` | 0 (0.0%) | 40 | -99.0000 | 1.2200 | 0.8670 ± 5.7669 |

##### File: `2.0 ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\core\2.0 ml.csv`
- **Size:** 0.04 MB (45,148 bytes)
- **Rows Inspected:** 7,824
- **Columns (1):** `p [g]`
- **Duplicate Rows:** 7765

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `p [g]` | `float64` | 0 (0.0%) | 59 | -99.0000 | 2.0290 | 1.6618 ± 5.8128 |

##### File: `p10-0.1ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\AR-onlineTraining\p10-0.1ml.csv`
- **Size:** 0.38 MB (399,854 bytes)
- **Rows Inspected:** 9,628
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 3110

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 6,398 | 0.0000 | 0.1085 | 0.1012 ± 0.0020 |
| `Actual weight` | `float64` | 0 (0.0%) | 61 | -1.0000 | 0.1090 | 0.0962 ± 0.0634 |
| `Prediction` | `str` | 0 (0.0%) | 6,388 | N/A | N/A | N/A |

##### File: `p10-0.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\AR-onlineTraining\p10-0.2ml.csv`
- **Size:** 0.42 MB (436,337 bytes)
- **Rows Inspected:** 11,438
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 3683

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 7,602 | 0.0000 | 0.2177 | 0.2015 ± 0.0031 |
| `Actual weight` | `float64` | 0 (0.0%) | 85 | -1.0000 | 0.2130 | 0.1965 ± 0.0692 |
| `Prediction` | `str` | 0 (0.0%) | 7,615 | N/A | N/A | N/A |

##### File: `p10-1.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\AR-onlineTraining\p10-1.2ml.csv`
- **Size:** 0.24 MB (254,063 bytes)
- **Rows Inspected:** 8,375
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 2725

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 5,549 | 0.0000 | 1.2228 | 1.2041 ± 0.0235 |
| `Actual weight` | `float64` | 0 (0.0%) | 51 | -1.0000 | 1.2290 | 1.1921 ± 0.1271 |
| `Prediction` | `str` | 0 (0.0%) | 5,565 | N/A | N/A | N/A |

##### File: `p10-2.0ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\AR-onlineTraining\p10-2.0ml.csv`
- **Size:** 0.28 MB (288,362 bytes)
- **Rows Inspected:** 9,519
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 3121

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 6,291 | 0.0000 | 2.0395 | 2.0055 ± 0.0214 |
| `Actual weight` | `float64` | 0 (0.0%) | 47 | -1.0000 | 2.0180 | 1.9875 ± 0.1736 |
| `Prediction` | `str` | 0 (0.0%) | 6,313 | N/A | N/A | N/A |

##### File: `6neur-0.1ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-onlineTraining\6neur-0.1ml.csv`
- **Size:** 0.09 MB (96,019 bytes)
- **Rows Inspected:** 2,372
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 742

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 1,565 | 0.0000 | 0.1209 | 0.1010 ± 0.0031 |
| `Actual weight` | `float64` | 0 (0.0%) | 56 | -1.0000 | 0.1090 | 0.0963 ± 0.0639 |
| `Prediction` | `str` | 0 (0.0%) | 1,571 | N/A | N/A | N/A |

##### File: `6neur-0.3ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-onlineTraining\6neur-0.3ml.csv`
- **Size:** 0.20 MB (214,598 bytes)
- **Rows Inspected:** 5,719
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 1806

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 3,785 | 0.0000 | 0.3158 | 0.3003 ± 0.0046 |
| `Actual weight` | `float64` | 0 (0.0%) | 88 | -1.0000 | 0.3210 | 0.2970 ± 0.0750 |
| `Prediction` | `str` | 0 (0.0%) | 3,803 | N/A | N/A | N/A |

##### File: `6neur-1.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-onlineTraining\6neur-1.2ml.csv`
- **Size:** 0.13 MB (140,665 bytes)
- **Rows Inspected:** 4,756
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 1548

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 3,099 | 0.0000 | 1.2273 | 1.2051 ± 0.0253 |
| `Actual weight` | `float64` | 0 (0.0%) | 43 | -1.0000 | 1.2210 | 1.1907 ± 0.1274 |
| `Prediction` | `str` | 0 (0.0%) | 3,114 | N/A | N/A | N/A |

##### File: `10neur-Vc+V+IncrAv-0.3ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-preTrained\10neur-Vc+V+IncrAv-0.3ml.csv`
- **Size:** 0.32 MB (338,932 bytes)
- **Rows Inspected:** 8,990
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 2882

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 5,915 | 0.2937 | 0.3093 | 0.3008 ± 0.0020 |
| `Actual weight` | `float64` | 0 (0.0%) | 88 | -1.0000 | 0.3140 | 0.2953 ± 0.0738 |
| `Prediction` | `str` | 0 (0.0%) | 5,942 | N/A | N/A | N/A |

##### File: `10neur-Vc+V+IncrAv-1.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-preTrained\10neur-Vc+V+IncrAv-1.2ml.csv`
- **Size:** 0.26 MB (269,845 bytes)
- **Rows Inspected:** 9,120
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 3014

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 5,898 | 0.0000 | 1.2206 | 1.2045 ± 0.0137 |
| `Actual weight` | `float64` | 0 (0.0%) | 39 | -1.0000 | 1.2170 | 1.1916 ± 0.1260 |
| `Prediction` | `str` | 0 (0.0%) | 5,928 | N/A | N/A | N/A |

##### File: `8neur-V+IncrAv-1.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\GRU-preTrained\8neur-V+IncrAv-1.2ml.csv`
- **Size:** 0.20 MB (214,174 bytes)
- **Rows Inspected:** 7,224
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 2335

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 4,733 | 0.0000 | 1.2213 | 1.2036 ± 0.0251 |
| `Actual weight` | `float64` | 0 (0.0%) | 49 | -1.0000 | 1.2350 | 1.1923 ± 0.1267 |
| `Prediction` | `str` | 0 (0.0%) | 4,754 | N/A | N/A | N/A |

##### File: `6neur-Vc+V+IncrAv-1.2ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\LSTM-preTrained\6neur-Vc+V+IncrAv-1.2ml.csv`
- **Size:** 0.26 MB (267,832 bytes)
- **Rows Inspected:** 9,022
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 2922

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 5,885 | 0.0000 | 1.2205 | 1.2049 ± 0.0139 |
| `Actual weight` | `float64` | 0 (0.0%) | 48 | -1.0000 | 1.2280 | 1.1913 ± 0.1267 |
| `Prediction` | `str` | 0 (0.0%) | 5,914 | N/A | N/A | N/A |

##### File: `8neur-V+IncrAv-0.3ml.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\LSTM-preTrained\8neur-V+IncrAv-0.3ml.csv`
- **Size:** 0.28 MB (293,519 bytes)
- **Rows Inspected:** 7,826
- **Columns (3):** `Set volume, Actual weight, Prediction`
- **Duplicate Rows:** 2522

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Set volume` | `float64` | 0 (0.0%) | 5,159 | 0.0000 | 0.3093 | 0.3008 ± 0.0040 |
| `Actual weight` | `float64` | 0 (0.0%) | 66 | -1.0000 | 0.3140 | 0.2957 ± 0.0749 |
| `Prediction` | `str` | 0 (0.0%) | 5,183 | N/A | N/A | N/A |

##### File: `0.3 ml-AR10-ADCS-active.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\training-data\0.3 ml-AR10-ADCS-active.csv`
- **Size:** 0.22 MB (233,895 bytes)
- **Rows Inspected:** 6,228
- **Columns (3):** `V_c, p [g], Prediction`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `V_c` | `float64` | 0 (0.0%) | 6,197 | 0.2843 | 0.3156 | 0.3045 ± 0.0034 |
| `p [g]` | `float64` | 0 (0.0%) | 6,208 | 0.2866 | 0.3135 | 0.3004 ± 0.0036 |
| `Prediction` | `str` | 0 (0.0%) | 6,206 | N/A | N/A | N/A |

##### File: `1.2 ml-AR10-ADCS-active.csv`
- **Path:** `raw\dosepi\dosepi-datasets\compensated\training-data\1.2 ml-AR10-ADCS-active.csv`
- **Size:** 0.10 MB (103,915 bytes)
- **Rows Inspected:** 5,547
- **Columns (2):** `V_c, p [g]`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `V_c` | `float64` | 0 (0.0%) | 5,546 | 1.1799 | 1.2228 | 1.2068 ± 0.0057 |
| `p [g]` | `float64` | 0 (0.0%) | 38 | 1.1640 | 1.2180 | 1.1995 ± 0.0050 |


### Dataset: Multi-infusion with integrated multiple pressure sensing allows earlier detection of line occlusions (`pressure_occlusion`)

- **DOI / Link:** [10.3390/s21206934](https://pmc.ncbi.nlm.nih.gov/articles/PMC8555317/)
- **Acquisition Status:** `NOT_ACQUIRED`
- **Domain Context:** In vitro multi-infusion clinical pressure-event research study.
- **Analytical Purpose in ML System:** Pressure behaviour, normal infusion patterns, genuine occlusion events, pressure anomaly analysis.
- **Supported Sensors:** `pressure`

> **Status:** No local files found. Refer to `DATASET_GUIDE.md` for manual acquisition or author request procedures.

### Dataset: Development of a Real-time Force-based Algorithm for Infusion Failure Detection (`force_failure`)

- **DOI / Link:** [10.1177/19322968241247530](https://pmc.ncbi.nlm.nih.gov/articles/PMC11535327/)
- **Acquisition Status:** `NOT_ACQUIRED`
- **Domain Context:** In vitro force and line-pressure experimental failure study.
- **Analytical Purpose in ML System:** Pressure and force relationships, infusion failure classification, cross-sensor consistency analysis.
- **Supported Sensors:** `force, pressure`

> **Status:** No local files found. Refer to `DATASET_GUIDE.md` for manual acquisition or author request procedures.

### Dataset: Hospitals Infusion Data Set: Spain (Determinants of Intravenous Infusion Longevity and Infusion Failure) (`smart_pump_logs`)

- **DOI / Link:** [10.2196/48628](https://ai.jmir.org/2023/1/e48628/)
- **Acquisition Status:** `NOT_ACQUIRED`
- **Domain Context:** Retrospective hospital smart pump event logs (operational discrete events, not continuous raw waveforms).
- **Analytical Purpose in ML System:** Infusion operating conditions, pump event information, occlusion and infusion termination context.
- **Supported Sensors:** `event_logs, operating_conditions, alarm_records`

> **Status:** No local files found. Refer to `DATASET_GUIDE.md` for manual acquisition or author request procedures.

### Dataset: Typical Sensor Defects Dataset (`sensor_defects`)

- **DOI / Link:** [10.5281/zenodo.56358](https://zenodo.org/records/56358)
- **Acquisition Status:** `ACQUIRED`
- **Domain Context:** Generic multi-sensor defect benchmark dataset (hazardous material storage container, not an infusion pump).
- **Analytical Purpose in ML System:** Sensor drift, offset, noise, peaks and abnormal sensor behaviour benchmark.
- **Supported Sensors:** `temperature, smoke_defect_channels`

#### Inspected Files (15 total):

##### File: `Configuration.pdf`
- **Path:** `raw\sensor_defects\Configuration.pdf`
- **Size:** 0.06 MB (60,828 bytes)
##### File: `data.zip`
- **Path:** `raw\sensor_defects\data.zip`
- **Size:** 3.86 MB (4,046,559 bytes)
##### File: `data_drift_0_001.csv`
- **Path:** `raw\sensor_defects\extracted\data_drift_0_001.csv`
- **Size:** 6.49 MB (6,802,773 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 354 | 19.5000 | 20.2710 | 19.8974 ± 0.2594 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_drift_0_0025.csv`
- **Path:** `raw\sensor_defects\extracted\data_drift_0_0025.csv`
- **Size:** 6.49 MB (6,802,339 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 822 | 19.5000 | 20.6770 | 20.1005 ± 0.3744 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_drift_0_005.csv`
- **Path:** `raw\sensor_defects\extracted\data_drift_0_005.csv`
- **Size:** 6.49 MB (6,802,497 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 1,530 | 19.5000 | 21.3540 | 20.4390 ± 0.5682 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_noise_0dB.csv`
- **Path:** `raw\sensor_defects\extracted\data_noise_0dB.csv`
- **Size:** 6.49 MB (6,802,765 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 1,998 | 18.5020 | 21.1420 | 19.7615 ± 0.3386 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_noise_10dB.csv`
- **Path:** `raw\sensor_defects\extracted\data_noise_10dB.csv`
- **Size:** 6.49 MB (6,802,517 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 1,012 | 19.1230 | 20.3370 | 19.7617 ± 0.2045 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_offset_1.csv`
- **Path:** `raw\sensor_defects\extracted\data_offset_1.csv`
- **Size:** 6.32 MB (6,628,806 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 6 | 20.5000 | 21.0000 | 20.7620 ± 0.1843 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_offset_2.csv`
- **Path:** `raw\sensor_defects\extracted\data_offset_2.csv`
- **Size:** 6.32 MB (6,628,806 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 6 | 21.5000 | 22.0000 | 21.7620 ± 0.1843 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_offset_5.csv`
- **Path:** `raw\sensor_defects\extracted\data_offset_5.csv`
- **Size:** 6.32 MB (6,628,806 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 6 | 24.5000 | 25.0000 | 24.7620 ± 0.1843 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_peak_1.csv`
- **Path:** `raw\sensor_defects\extracted\data_peak_1.csv`
- **Size:** 6.32 MB (6,630,014 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 12 | 19.5000 | 120.0000 | 21.4280 ± 12.8008 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_peak_10.csv`
- **Path:** `raw\sensor_defects\extracted\data_peak_10.csv`
- **Size:** 6.33 MB (6,640,891 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 12 | 19.5000 | 120.0000 | 36.4300 ± 37.2699 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_peak_2.csv`
- **Path:** `raw\sensor_defects\extracted\data_peak_2.csv`
- **Size:** 6.32 MB (6,631,222 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 12 | 19.5000 | 120.0000 | 23.0960 ± 17.9533 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_peak_5.csv`
- **Path:** `raw\sensor_defects\extracted\data_peak_5.csv`
- **Size:** 6.33 MB (6,634,847 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 12 | 19.5000 | 120.0000 | 28.1000 ± 27.6464 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`

##### File: `data_standard.csv`
- **Path:** `raw\sensor_defects\extracted\data_standard.csv`
- **Size:** 6.32 MB (6,628,806 bytes)
- **Rows Inspected:** 50,000
- **Columns (24):** `Time [sec], Smoke_Inside_0, Smoke_Ventilation_In, Leakage_0, Leakage_1, Pressure_Ventilation_Out, Smoke_Ventilation_Out, Smoke_Inside_1, Gas_Ventilation_Out, Temp_Outside_0 [°C], Temp_Outside_1 [°C], Temp_Ventilation_Out [°C], Temp_Inside_3 [°C], Temp_Inside_0 [°C], Temp_Inside_1 [°C], Temp_Inside_2 [°C], Temp_Inside_4 [°C], Temp_Ventilation_In [°C], Temp_Inside_6 [°C], Temp_Inside_7 [°C], Temp_Inside_9 [°C], Temp_Inside_8 [°C], Temp_Inside_5 [°C], Unnamed: 23`
- **Duplicate Rows:** 0

| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Time [sec]` | `int64` | 0 (0.0%) | 50,000 | 0.0000 | 49999.0000 | 24999.5000 ± 14433.9011 |
| `Smoke_Inside_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_In` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_0` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Leakage_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Pressure_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Ventilation_Out` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Smoke_Inside_1` | `int64` | 0 (0.0%) | 1 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| `Gas_Ventilation_Out` | `float64` | 0 (0.0%) | 1 | 43.6000 | 43.6000 | 43.6000 ± 0.0000 |
| `Temp_Outside_0 [°C]` | `float64` | 0 (0.0%) | 12 | 20.5000 | 21.6000 | 21.1114 ± 0.2666 |
| `Temp_Outside_1 [°C]` | `float64` | 0 (0.0%) | 12 | 20.8000 | 21.9000 | 21.4150 ± 0.2688 |
| `Temp_Ventilation_Out [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.8019 ± 0.2139 |
| `Temp_Inside_3 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9159 ± 0.2254 |
| `Temp_Inside_0 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8768 ± 0.2162 |
| `Temp_Inside_1 [°C]` | `float64` | 0 (0.0%) | 8 | 19.4000 | 20.1000 | 19.7897 ± 0.2173 |
| `Temp_Inside_2 [°C]` | `float64` | 0 (0.0%) | 6 | 19.5000 | 20.0000 | 19.7620 ± 0.1843 |
| `Temp_Inside_4 [°C]` | `float64` | 0 (0.0%) | 7 | 19.5000 | 20.1000 | 19.7783 ± 0.2156 |
| `Temp_Ventilation_In [°C]` | `float64` | 0 (0.0%) | 8 | 19.6000 | 20.3000 | 19.9467 ± 0.2321 |
| `Temp_Inside_6 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8867 ± 0.2224 |
| `Temp_Inside_7 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.8597 ± 0.2056 |
| `Temp_Inside_9 [°C]` | `float64` | 0 (0.0%) | 7 | 19.3000 | 19.9000 | 19.6015 ± 0.2075 |
| `Temp_Inside_8 [°C]` | `float64` | 0 (0.0%) | 7 | 19.6000 | 20.2000 | 19.9034 ± 0.2184 |
| `Temp_Inside_5 [°C]` | `float64` | 0 (0.0%) | 8 | 19.3000 | 20.0000 | 19.6109 ± 0.2270 |
| `Unnamed: 23` | `float64` | 50000 (100.0%) | 0 | N/A | N/A | N/A |

**Sampling & Temporal Dynamics:**
- Primary Timestamp Column: `Time [sec]`
- Median Sampling Interval: 1.0
- Min/Max Interval: 1.0 / 1.0
- Strictly Monotonic Increasing: `True`


