# Research Dataset Guide & Clinical Governance Inventory

**Project:** Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins  
**Target Device:** Infusion Pump  
**Engineering Document:** Dataset Provenance, Field Mapping, Data Quality, and Scientific Limitations  

---

## 1. Dataset Acquisition Overview & Inventory

| Dataset Identifier | Full Publication Title | Source / Repository | Acquisition Status | Total Size | Primary Channels | Valid ML Application |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dosepi`** | DosePI: A comprehensive dataset for peristaltic pump accuracy enhancement | *Nature Scientific Data* / Figshare `10.6084/m9.figshare.28391645` | **ACQUIRED** (19 CSV files) | 4.57 MB | Gravimetric mass balance ($p [g]$), commanded volume ($V_{set} [mL]$), ML compensations | Pump delivery accuracy, tubing fatigue non-linearities, reference error estimation |
| **`pressure_occlusion`** | Multi-infusion with integrated multiple pressure sensing allows earlier detection of line occlusions | *Sensors* (MDPI) / PMC8555317 | **MANUAL REQUEST REQUIRED** (Author Request) | N/A (Gated) | Multi-lumen downstream inline pressure | Downstream line occlusion dynamics, CUSUM/EWMA threshold validation |
| **`force_failure`** | Development of a Real-time Force-based Algorithm for Infusion Failure Detection | *J Diabetes Sci Technol* / PMC11535327 | **MANUAL REQUEST REQUIRED** (Author Request) | N/A (Gated) | Plunger drive force ($N$), inline pressure ($psi$) | Cross-sensor physical consistency (force vs. hydraulic pressure) |
| **`smart_pump_logs`** | Determinants of Intravenous Infusion Longevity and Infusion Failure (Spain) | *JMIR AI* / PMC10582697 | **RESTRICTED** (Hospital GDPR / BD Ethics) | N/A (Gated) | Event timestamps, alarms (occlusion, air-in-line), VTBI, VI | Operational priors, clinical alarm frequencies, Digital Twin state transitions |
| **`sensor_defects`** | Typical Sensor Defects Dataset | Zenodo Record `56358` | **ACQUIRED** (13 CSV files + PDF) | 86.95 MB | 10x Temperature ($^\circ\text{C}$), Pressure ($Pa$), Smoke, 1 Hz Time ($s$) | Statistical drift benchmark (EWMA/CUSUM), offset, excessive noise, Isolation Forest |

---

## 2. In-Depth Dataset Profiling & Limitations

### DATASET A: DosePI (Peristaltic Pump Accuracy Enhancement)
- **Publication:** Nature Scientific Data (2025). DOI: [10.1038/s41597-025-05902-z](https://doi.org/10.1038/s41597-025-05902-z)
- **Repository:** Figshare DOI: [10.6084/m9.figshare.28391645](https://doi.org/10.6084/m9.figshare.28391645)
- **License:** CC BY 4.0
- **Actual Files Downloaded:**
  - `dosepi-datasets/core/`: `0.2 ml.csv` (2,652 rows), `0.3 ml.csv` (16,221 rows), `1.2 ml.csv` (16,249 rows), `2.0 ml.csv` (7,824 rows).
  - `dosepi-datasets/compensated/AR-onlineTraining/`: `p10-0.1ml.csv`, `p10-0.2ml.csv`, `p10-1.2ml.csv`, `p10-2.0ml.csv`.
  - `dosepi-datasets/compensated/GRU-onlineTraining/`: `6neur-0.1ml.csv`, `6neur-0.3ml.csv`, `6neur-1.2ml.csv`.
  - `dosepi-datasets/compensated/training-data/`: `0.3 ml-AR10-ADCS-active.csv`, `1.2 ml-AR10-ADCS-active.csv`.
- **Actual Columns Found:**
  - Core files: Single column `p [g]` (delivered mass in grams measured by analytical balance).
  - Compensated files: `Set volume`, `Actual weight`, `Prediction`.
- **Data Quality & Encoding Nuances:**
  - Recalibration markers: In core datasets, a value of `-99.0` denotes an operator recalibration event split. In compensated files, `-1.0` denotes initial uncompensated warm-up steps. Naive models will fail if these sentinel values are not handled as state transitions.
- **Medical Device Limitations:**
  - **CRITICAL DISTINCTION:** DosePI is a **pharmaceutical vial filling/dispensing** dataset (high speed 600 RPM, discrete boluses into vials), **not** a continuous clinical intravenous infusion pump delivering 1–10 mL/h over 24 hours to a vein.
  - Does not contain downstream hydraulic line pressure, patient venous backpressure, or cannula load cell data.
  - **Role:** High-fidelity ground truth for pump delivery error estimation ($E = \text{measured} - \text{commanded}$) and wear compensation.

---

### DATASET B: Multi-Infusion Pressure Sensing & Line Occlusion Study
- **Publication:** Sensors (2021); 21(20): 6934. DOI: [10.3390/s21206934](https://doi.org/10.3390/s21206934), PMCID: [PMC8555317](https://pmc.ncbi.nlm.nih.gov/articles/PMC8555317/)
- **License:** Open Access article; dataset available on reasonable request.
- **Acquisition Status:** **Author Request Required**. The published manuscript states:
  > *"The datasets used and/or analyzed during the current study are available from the corresponding author on reasonable request."*
- **Manual Request Protocol:**
  - Corresponding author: Prof. Dr. Tim Kaufmann (`t.kaufmann@umcg.nl`), Department of Anesthesiology, University Medical Center Groningen, Netherlands.
  - Expected files: In vitro multi-infusion line pressure time-series records during baseline and clamp occlusion experiments.
  - Place acquired files into: `data/raw/pressure_occlusion/`.
- **Role & Limitations:**
  - Pressure rise dynamics during downstream occlusion obey fluid hydraulic compliance: $\Delta P(t) = \frac{Q}{C} \cdot t$ (linear/exponential ramp). In contrast, sensor drift is monotonic independent of fluid flow rate. This physical distinction is fundamental for our fault classification engine.

---

### DATASET C: Real-Time Force-Based Infusion Failure Detection
- **Publication:** Journal of Diabetes Science and Technology (2024). DOI: [10.1177/19322968241247530](https://doi.org/10.1177/19322968241247530), PMCID: [PMC11535327](https://pmc.ncbi.nlm.nih.gov/articles/PMC11535327/)
- **Acquisition Status:** **Author Request Required**.
- **Manual Request Protocol:**
  - Corresponding author: Prof. Dr. Bryan P. Boudouris (`boudouris@purdue.edu`), Purdue University.
  - Supplementary material provided with paper (`sj-docx-1-dst-10.1177_19322968241247530.docx`) contains force-to-pressure scaling functions.
  - Place continuous load-cell records into: `data/raw/force_failure/`.
- **Role & Limitations:**
  - Establishes the cross-sensor relationship: drive force $F_{\text{drive}} = P_{\text{fluid}} \cdot A_{\text{plunger}} + F_{\text{friction}}$. In genuine occlusion, both $F$ and $P$ rise concurrently; if only $P$ drifts while $F$ remains normal (or vice versa), the anomaly is isolated to the sensor, not the patient fluid pathway.

---

### DATASET D: Hospitals Infusion Data Set: Spain (Smart Pump Event Logs)
- **Publication:** JMIR AI (2023); 2: e48628. DOI: [10.2196/48628](https://doi.org/10.2196/48628), PMCID: [PMC10582697](https://pmc.ncbi.nlm.nih.gov/articles/PMC10582697/)
- **Acquisition Status:** **Clinical Data Governance Restricted**.
- **Governance Context:**
  - Analyzed 116,913 intravenous infusions across multiple hospitals in Spain using Becton Dickinson smart infusion pumps.
  - Data contains hospital telemetry, patient administration times, and proprietary device logs protected under GDPR and institutional ethics boards.
- **Manual Access Protocol:**
  - Requests require formal data transfer agreement (DTA) and clinical ethics board clearance with BD Medical Affairs and the University of Limerick.
  - Place approved event log CSV files into: `data/raw/smart_pump_logs/`.
- **CRITICAL DISTINCTION:**
  - Smart pump event logs are **discrete operational records** (e.g., `OCCLUSION_ALARM_TRIGGERED`, `INFUSION_STOPPED`, `VTBI_REACHED`), **not continuous 100 Hz sensor voltages**. They provide operational priors for failure frequency, but cannot be used for raw sensor signal filtering or EWMA drift tracking.

---

### DATASET E: Typical Sensor Defects Dataset (Zenodo Record 56358)
- **Publication:** Institute Industrial IT (inIT), TH OWL (2016). DOI: [10.5281/zenodo.56358](https://doi.org/10.5281/zenodo.56358)
- **Repository:** Zenodo Record 56358
- **License:** Open Data Commons Attribution License (ODC-By) v1.0
- **Actual Files Downloaded:**
  - `Configuration.pdf` (60.8 KB)
  - `data.zip` (4.05 MB) extracted to `data/raw/sensor_defects/extracted/`:
    - `data_standard.csv` (6.63 MB, 50,000+ rows) - Defect-free multi-channel baseline.
    - `data_drift_0_001.csv` (6.80 MB) - 1‰/h linear drift on target sensor.
    - `data_drift_0_0025.csv` (6.80 MB) - 2.5‰/h linear drift.
    - `data_drift_0_005.csv` (6.80 MB) - 5‰/h linear drift.
    - `data_offset_1.csv`, `data_offset_2.csv`, `data_offset_5.csv` - Fixed biases (+1°C, +2°C, +5°C).
    - `data_noise_0dB.csv`, `data_noise_10dB.csv` - Calibrated Gaussian noise.
    - `data_peak_1.csv`, `data_peak_2.csv`, `data_peak_5.csv`, `data_peak_10.csv` - Transient spike defects.
- **Actual Columns Found:**
  - `Time [sec]`: Integer seconds (0 to 49,999), 1 Hz strictly monotonic sampling.
  - 10x Internal temperature sensors: `Temp_Inside_0 [°C]` through `Temp_Inside_9 [°C]`.
  - External and ventilation temperatures: `Temp_Outside_0 [°C]`, `Temp_Outside_1 [°C]`, `Temp_Ventilation_In [°C]`, `Temp_Ventilation_Out [°C]`.
  - Process channels: `Pressure_Ventilation_Out`, `Gas_Ventilation_Out`, `Smoke_Inside_0`, `Smoke_Inside_1`, `Leakage_0`, `Leakage_1`.
  - `Unnamed: 23`: Empty artifact column resulting from trailing commas in CSV headers.
- **Data Quality & Encoding Nuances:**
  - Encoded in `latin1` / `cp1252` (byte `0xb0` for `°C`). Attempting naive `utf-8` decoding raises a `UnicodeDecodeError`.
  - Trailing comma produces a column with 100% NaN values that must be dropped during preprocessing.
- **Medical Device Limitations:**
  - **CRITICAL DISTINCTION:** This is an **industrial container sensor validation dataset**, **not an infusion pump**.
  - **Role:** Serves as the ground-truth benchmark for statistical drift detection algorithms (EWMA, CUSUM) and anomaly detection baselines where known defect labels are unambiguously defined.

---

## 3. Data Integration Strategy & Safeguards

```
[ Research Datasets ]
       │
       ├── DosePI (Figshare) ──────────────► [ Gravimetric Error Estimation (XGBoost Regressor) ]
       │                                     - Predicts Delivery Deviation: E = V_delivered - V_commanded
       │
       ├── Sensor Defects (Zenodo 56358) ──► [ Statistical Drift Engine (EWMA / CUSUM) ]
       │                                     - Evaluates Detection Delay, False Alarm Rate, Offset Tracking
       │
       ├── Clinical Occlusion/Force ──────► [ Cross-Sensor Physical Consistency Rules ]
       │   (PMC8555317 / PMC11535327)        - Hydraulic coupling: ΔP vs ΔF under occlusion
       │
       └── Modular Custom / Synthetic ─────► [ End-to-End Pipeline & Fault Classifier (XGBoost) ]
           (ISO 60601-2-24 Compliant)        - Multi-class classification: Normal, Drift, Offset, Occlusion
```

### Inviolable Engineering Rules:
1. **Never combine incompatible datasets into a single unprincipled matrix:**
   Container temperature columns must never be joined horizontally with peristaltic vial weights.
2. **Never fabricate ground truth labels:**
   Datasets without explicit failure annotations must remain unsupervised or be evaluated strictly on their verified physical signals.
3. **No automatic device calibration:**
   All ML predictions produce recommendations and health scores; medical device safety prohibits automated in-line calibration without verified engineering sign-off.

