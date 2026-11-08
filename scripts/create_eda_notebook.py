"""Script to create a clean, executable 01_dataset_exploration.ipynb notebook."""

import json
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title
cells.append(nbf.v4.new_markdown_cell("""# 01. Research Dataset Exploration & Baseline Analysis

**Project:** Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins  
**Target Device:** Infusion Pump  
**Engineering Phase:** Phase 1 - Environment, Acquisition, Inspection & Baseline Analysis  

---

## Executive Summary
This notebook performs rigorous, reproducible exploratory data analysis across the public research datasets acquired for our Infusion Pump ML system.

### Key Objectives:
1. Inspect available public datasets (DosePI, Typical Sensor Defects) and document clinical access restrictions for the remaining studies.
2. Characterize dosing accuracy distributions and gravimetric residuals in peristaltic pumping (DosePI).
3. Validate multi-sensor dynamics, temperature correlations, and ground-truth defect patterns (drift, offset, noise, peaks).
4. Establish statistical and physical data quality baselines to guide feature engineering and drift detection.
"""))

# Cell 2: Code Imports
cells.append(nbf.v4.new_code_cell("""# Environment & Imports
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json

from src.utils.config import DATASET_METADATA, RAW_DATA_DIR, REPORTS_DIR
from src.data.dataset_loader import load_dosepi, load_sensor_defects

print("All libraries imported successfully.")
print(f"Project root: {PROJECT_ROOT}")
"""))

# Cell 3: Markdown Section 1
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Inventory & Availability

We inspect the dataset inventory report generated during the automated acquisition phase.
"""))

# Cell 4: Code Inventory View
cells.append(nbf.v4.new_code_cell("""# Load and display inventory summary
inventory_path = PROJECT_ROOT / "reports" / "dataset_inventory.json"
if inventory_path.exists():
    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    
    summary_data = []
    for ds_id, meta in inv.items():
        summary_data.append({
            "Dataset ID": ds_id,
            "Name": meta["name"][:45] + "...",
            "Status": meta["status"],
            "Files": len(meta.get("files", [])),
            "Size (MB)": meta.get("total_size_mb", 0.0),
            "Sensors": ", ".join(meta.get("sensors_supported", []))
        })
    inv_df = pd.DataFrame(summary_data)
    display(inv_df)
else:
    print("Run scripts/run_eda.py first to generate the dataset inventory.")
"""))

# Cell 5: Markdown Section 2 - DosePI
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset A: DosePI (Peristaltic Dosing & Accuracy)

DosePI (*Nature Scientific Data*, 2025) provides high-resolution gravimetric mass readings from a Flexicon peristaltic filling pump under controlled pharmaceutical dosing cycles.

### Distinctions & Relevance:
- **Device:** Flexicon peristaltic pump with MC100 controller.
- **Physical Channel:** Gravimetric balance measured weight ($p [g]$) vs target set volume ($V_{set} [mL]$).
- **Recalibration Markers:** Measured weight of `-99.0` denotes calibration events.
- **Role in ML System:** Used to model pump delivery compensation, mechanical tubing wear over repeated pump head actuations, and reference residual error estimation.
"""))

# Cell 6: Code DosePI Core Analysis
cells.append(nbf.v4.new_code_cell("""# Inspect DosePI Core Files
core_dir = RAW_DATA_DIR / "dosepi" / "dosepi-datasets" / "core"
core_files = sorted(list(core_dir.glob("*.csv")))

print(f"Found {len(core_files)} DosePI core files: {[f.name for f in core_files]}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

dosepi_summaries = []

for idx, f in enumerate(core_files):
    df_core = pd.read_csv(f)
    # Filter out calibration split markers (-99.0)
    valid_mass = df_core[df_core['p [g]'] > 0]['p [g]']
    
    # Target volume from file name
    target_vol = float(f.stem.replace(" ml", "").replace("ml", ""))
    error = valid_mass - target_vol
    rel_error_pct = (error / target_vol) * 100.0
    
    dosepi_summaries.append({
        "File": f.name,
        "Target (mL)": target_vol,
        "Valid Dosings": len(valid_mass),
        "Mean Mass (g)": valid_mass.mean(),
        "Std Mass (g)": valid_mass.std(),
        "Mean Error (g)": error.mean(),
        "Mean Rel Error (%)": rel_error_pct.mean(),
        "Max Rel Error (%)": rel_error_pct.abs().max()
    })
    
    ax = axes[idx]
    ax.hist(valid_mass, bins=50, color='royalblue', alpha=0.7, edgecolor='black')
    ax.axvline(target_vol, color='red', linestyle='--', linewidth=2, label=f'Target ({target_vol} mL)')
    ax.axvline(valid_mass.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean ({valid_mass.mean():.4f} g)')
    ax.set_title(f"DosePI Dosing Distribution: {f.name}")
    ax.set_xlabel("Measured Delivered Mass (g)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "dosepi_core_distributions.png", dpi=150)
plt.show()

display(pd.DataFrame(dosepi_summaries))
"""))

# Cell 7: Markdown Section 3 - Sensor Defects
cells.append(nbf.v4.new_markdown_cell(r"""## 3. Dataset E: Typical Sensor Defects (Zenodo Record 56358)

The Typical Sensor Defects dataset provides a multi-sensor baseline recorded in an industrial testing container along with 12 synthetically injected defect regimes in `Temp_Sensor_2`.

### Channels Available:
- **Baseline:** Temperature channels (`Temp_Inside_0` to `Temp_Inside_9`, `Temp_Outside_0`, `Temp_Outside_1`, `Temp_Ventilation_...`), `Pressure_Ventilation_Out`, `Gas_Ventilation_Out`, optical smoke detectors.
- **Sampling:** 1 Hz continuous regular time-series ($T = 1.0\text{ s}$).
- **Injected Fault Modes:**
  - `data_drift_0_001.csv` ($1‰$/h drift)
  - `data_drift_0_0025.csv` ($2.5‰$/h drift)
  - `data_drift_0_005.csv` ($5‰$/h drift)
  - `data_offset_1.csv` ($+1^\circ\text{C}$ bias)
  - `data_offset_2.csv` ($+2^\circ\text{C}$ bias)
  - `data_offset_5.csv` ($+5^\circ\text{C}$ bias)
  - `data_noise_0dB.csv` / `data_noise_10dB.csv`
  - `data_peak_1.csv` to `data_peak_10.csv`
"""))

# Cell 8: Code Sensor Defects Baseline vs Drift
cells.append(nbf.v4.new_code_cell("""# Load Baseline and Injected Drift Datasets
extracted_dir = RAW_DATA_DIR / "sensor_defects" / "extracted"

df_std = pd.read_csv(extracted_dir / "data_standard.csv", encoding="latin1")
df_drift = pd.read_csv(extracted_dir / "data_drift_0_005.csv", encoding="latin1")
df_offset = pd.read_csv(extracted_dir / "data_offset_2.csv", encoding="latin1")

# Clean trailing empty column if present
for df_tmp in [df_std, df_drift, df_offset]:
    if "Unnamed: 23" in df_tmp.columns:
        df_tmp.drop(columns=["Unnamed: 23"], inplace=True)

print(f"Standard shape: {df_std.shape}")
print(f"Drift 5% shape: {df_drift.shape}")
print(f"Offset 2C shape: {df_offset.shape}")

# Subsample 15,000 steps for clean visualization
n_vis = 15000
time_sec = df_std['Time [sec]'].iloc[:n_vis]

plt.figure(figsize=(15, 6))
plt.plot(time_sec / 3600.0, df_std['Temp_Inside_2 [°C]'].iloc[:n_vis], label="Reference Channel (Temp_Inside_2)", color='green', alpha=0.8)
plt.plot(time_sec / 3600.0, df_std['Temp_Inside_0 [°C]'].iloc[:n_vis], label="Baseline Target (Temp_Inside_0)", color='black', linestyle=':', alpha=0.7)
plt.plot(time_sec / 3600.0, df_drift['Temp_Inside_0 [°C]'].iloc[:n_vis], label="Gradual Drift (5‰/h)", color='crimson', linewidth=2)
plt.plot(time_sec / 3600.0, df_offset['Temp_Inside_0 [°C]'].iloc[:n_vis], label="Sudden Offset (+2°C)", color='darkorange', linestyle='--', linewidth=1.5)

plt.title("Sensor Defect Profiles vs. Baseline Truth (Typical Sensor Defects)")
plt.xlabel("Time (Hours)")
plt.ylabel("Temperature (°C)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "sensor_defect_profiles.png", dpi=150)
plt.show()
"""))

# Cell 9: Code Residual and Cross-Sensor Analysis
cells.append(nbf.v4.new_code_cell("""# Cross-Sensor Residual Analysis
# Under normal operation, co-located sensors share strong physical correlations.
# Sensor fault breaks this cross-sensor consistency.

residual_standard = df_std['Temp_Inside_0 [°C]'].iloc[:n_vis] - df_std['Temp_Inside_2 [°C]'].iloc[:n_vis]
residual_drift = df_drift['Temp_Inside_0 [°C]'].iloc[:n_vis] - df_std['Temp_Inside_2 [°C]'].iloc[:n_vis]

plt.figure(figsize=(14, 5))
plt.plot(time_sec / 3600.0, residual_standard, label="Baseline Residual (Normal Noise Band)", color='navy', alpha=0.6)
plt.plot(time_sec / 3600.0, residual_drift, label="Drifted Residual (Persistent Linear Divergence)", color='red', linewidth=2)
plt.axhline(residual_standard.mean(), color='navy', linestyle='--', label=f"Baseline Residual Mean ({residual_standard.mean():.3f}°C)")
plt.axhline(residual_standard.mean() + 3 * residual_standard.std(), color='gray', linestyle=':', label="3-Sigma Normal Noise Limit")
plt.axhline(residual_standard.mean() - 3 * residual_standard.std(), color='gray', linestyle=':')

plt.title("Cross-Sensor Residuals: Distinguishing Normal Noise from Sensor Drift")
plt.xlabel("Time (Hours)")
plt.ylabel("Residual [Target - Co-located Reference] (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "cross_sensor_residual_analysis.png", dpi=150)
plt.show()

print(f"Normal noise 3-sigma band: ±{3 * residual_standard.std():.4f} °C")
print(f"Drift exceeds 3-sigma band at hour: {((residual_drift - residual_standard.mean()).abs() > 3 * residual_standard.std()).idxmax() / 3600.0:.2f} h")
"""))

# Cell 10: Markdown Section 4 - Limitations & Synthesis
cells.append(nbf.v4.new_markdown_cell("""## 4. Synthesis & Scientific Grounding for Infusion Pump ML

### Key Findings from Phase 1 Inspection:
1. **Public Data Landscape:**
   - Public open-access datasets in the biomedical/pump engineering domain are constrained: DosePI provides gravimetric filling pump output; Zenodo 56358 provides industrial multi-sensor defect regimes.
   - Clinical datasets (PMC8555317, PMC11535327, JMIR AI e48628) require formal researcher data access agreements due to patient confidentiality and device maker IP protections.
2. **Data Cleanliness & Encoding:**
   - European/German scientific datasets frequently use non-UTF-8 encodings (`latin1`) for degree symbols (`°C`, byte `0xb0`) and trailing commas producing empty columns (`Unnamed: 23`). Preprocessing pipelines must be resilient to these real-world formats.
3. **Implications for Phase 2 (Preprocessing, Features & Baseline Models):**
   - **Never mix incompatible datasets:** Do not merge container temperature readings with pharmaceutical pump weights into a single unprincipled matrix.
   - **Modular Engines:** Build independent, pluggable pipelines:
     - Statistical Drift Engine (EWMA & CUSUM) validated on Zenodo 56358.
     - Gravimetric Dosing Accuracy & Error Estimator validated on DosePI.
     - Physically Grounded Infusion Pump Synthetic Engine (generating fluid-dynamic compliant pressure, force, and occlusion curves conforming to ISO 60601-2-24 standards).
"""))

# Write notebook file
notebook_path = Path(r"D:\techgium\notebooks\01_dataset_exploration.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as nf:
    nbf.write(nb, nf)

print(f"Notebook generated successfully at: {notebook_path}")
