"""Generate EDA plots and statistical summaries directly."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local writable matplotlib cache
os.environ["MPLCONFIGDIR"] = str(Path("data/interim/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import RAW_DATA_DIR, REPORTS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("generate_eda_figures")


def main():
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. DosePI Analysis
    logger.info("Analyzing DosePI Core files...")
    core_dir = RAW_DATA_DIR / "dosepi" / "dosepi-datasets" / "core"
    core_files = sorted(list(core_dir.glob("*.csv")))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, f in enumerate(core_files):
        df_core = pd.read_csv(f)
        valid_mass = df_core[df_core["p [g]"] > 0]["p [g]"]
        target_vol = float(f.stem.replace(" ml", "").replace("ml", ""))

        ax = axes[idx]
        ax.hist(valid_mass, bins=50, color="royalblue", alpha=0.75, edgecolor="black")
        ax.axvline(target_vol, color="red", linestyle="--", linewidth=2, label=f"Target ({target_vol} mL)")
        ax.axvline(valid_mass.mean(), color="orange", linestyle="-", linewidth=2, label=f"Mean ({valid_mass.mean():.4f} g)")
        ax.set_title(f"DosePI Dosing Distribution: {f.name} (N={len(valid_mass):,})")
        ax.set_xlabel("Delivered Mass (g)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = figures_dir / "dosepi_core_distributions.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    logger.info(f"Saved: {p1}")

    # 2. Typical Sensor Defects Analysis
    logger.info("Analyzing Typical Sensor Defects...")
    extracted_dir = RAW_DATA_DIR / "sensor_defects" / "extracted"

    df_std = pd.read_csv(extracted_dir / "data_standard.csv", encoding="latin1")
    df_drift = pd.read_csv(extracted_dir / "data_drift_0_005.csv", encoding="latin1")
    df_offset = pd.read_csv(extracted_dir / "data_offset_2.csv", encoding="latin1")

    for df_tmp in [df_std, df_drift, df_offset]:
        if "Unnamed: 23" in df_tmp.columns:
            df_tmp.drop(columns=["Unnamed: 23"], inplace=True)

    n_vis = 15000
    time_h = df_std["Time [sec]"].iloc[:n_vis] / 3600.0

    plt.figure(figsize=(15, 6))
    plt.plot(time_h, df_std["Temp_Inside_2 [°C]"].iloc[:n_vis], label="Reference Channel (Temp_Inside_2)", color="green", alpha=0.8)
    plt.plot(time_h, df_std["Temp_Inside_0 [°C]"].iloc[:n_vis], label="Baseline Target (Temp_Inside_0)", color="black", linestyle=":", alpha=0.7)
    plt.plot(time_h, df_drift["Temp_Inside_0 [°C]"].iloc[:n_vis], label="Gradual Drift (5‰/h Injection)", color="crimson", linewidth=2)
    plt.plot(time_h, df_offset["Temp_Inside_0 [°C]"].iloc[:n_vis], label="Sudden Offset (+2°C Injection)", color="darkorange", linestyle="--", linewidth=1.5)

    plt.title("Sensor Defect Profiles vs. Baseline Truth (Typical Sensor Defects)")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Temperature (°C)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = figures_dir / "sensor_defect_profiles.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    logger.info(f"Saved: {p2}")

    # 3. Residual & Noise Band Analysis
    residual_standard = df_std["Temp_Inside_0 [°C]"].iloc[:n_vis] - df_std["Temp_Inside_2 [°C]"].iloc[:n_vis]
    residual_drift = df_drift["Temp_Inside_0 [°C]"].iloc[:n_vis] - df_std["Temp_Inside_2 [°C]"].iloc[:n_vis]

    plt.figure(figsize=(14, 5))
    plt.plot(time_h, residual_standard, label="Baseline Residual (Normal Noise Band)", color="navy", alpha=0.6)
    plt.plot(time_h, residual_drift, label="Drifted Residual (Persistent Linear Divergence)", color="red", linewidth=2)
    plt.axhline(residual_standard.mean(), color="navy", linestyle="--", label=f"Mean Residual ({residual_standard.mean():.3f}°C)")
    plt.axhline(residual_standard.mean() + 3 * residual_standard.std(), color="gray", linestyle=":", label="±3-Sigma Normal Noise Limit")
    plt.axhline(residual_standard.mean() - 3 * residual_standard.std(), color="gray", linestyle=":")

    plt.title("Cross-Sensor Residuals: Distinguishing Normal Noise from Persistent Sensor Drift")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Residual [Target - Co-located Reference] (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = figures_dir / "cross_sensor_residual_analysis.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    logger.info(f"Saved: {p3}")

    print("\n[ALL EDA FIGURES GENERATED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
