"""Proof of Concept (PoC) Validation Visualizer.

Simulates an end-to-end infusion pump telemetry run demonstrating:
1. Normal operation (t = 0 to 30s)
2. Gradual Sensor Drift (t = 30 to 75s: Sensor failure, process safe)
3. Genuine Downstream Occlusion (t = 75 to 120s: Process failure, sensor healthy)

Generates a publication-grade, 4-panel validation dashboard image:
'reports/figures/poc_validation_dashboard.png'
"""

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Local writable matplotlib cache
os.environ["MPLCONFIGDIR"] = str(Path("data/interim/.matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api.app import (
    load_production_models,
    observe_stream,
    reset_session,
)
from api.schemas.payloads import (
    SequentialObservationRequest,
    SessionResetRequest,
)

def run_coro(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def main():
    print("=" * 70)
    print("Generating Proof of Concept (PoC) Validation Visualizer...")
    print("=" * 70)

    load_production_models()

    session_id = "poc_dashboard_demo"
    run_coro(reset_session(SessionResetRequest(session_id=session_id)))

    timestamps = []
    pressures = []
    dt_pressures = []
    forces = []
    expected_forces = []
    flows = []
    sensor_healths = []
    process_risks = []
    predicted_classes = []
    calibrated_confs = []
    prob_normal = []
    prob_drift = []
    prob_occlusion = []

    # 120 seconds simulation
    for t in range(120):
        # Scenario phasing
        if t < 35:
            # Phase 1: Normal steady-state delivery
            p = 80.0 + np.random.normal(0, 0.25)
            f = 4.85 + np.random.normal(0, 0.02)
            q = 50.0 + np.random.normal(0, 0.1)
        elif t < 75:
            # Phase 2: Sensor Positive Drift (Sensor fault, process normal)
            drift_dt = t - 35
            p = 80.0 + 1.4 * drift_dt + np.random.normal(0, 0.3)
            f = 4.85 + np.random.normal(0, 0.02) # Force stays constant -> transducer mismatch
            q = 50.0 + np.random.normal(0, 0.1)  # Flow delivers normally
        else:
            # Phase 3: Genuine Downstream Line Occlusion (Process fault, sensor healthy)
            occ_dt = t - 75
            p = min(360.0, 80.0 + 7.5 * occ_dt + np.random.normal(0, 0.4))
            f = (p * 0.0133322 * 3.14) + 1.5 + np.random.normal(0, 0.05) # Force surges with pressure
            q = max(0.0, 50.0 - 2.5 * occ_dt)                             # Flow collapses

        req = SequentialObservationRequest(
            session_id=session_id,
            sensor_id="PRESS_LINE_01",
            timestamp=float(t),
            sensor_pressure_mmhg=round(float(p), 2),
            sensor_force_n=round(float(f), 3),
            sensor_flow_ml_h=round(float(q), 2),
            sensor_temperature_c=22.0,
            commanded_rate_ml_h=50.0,
            dt_predicted_pressure_mmhg=80.0,
        )

        res = run_coro(observe_stream(req))

        timestamps.append(t)
        pressures.append(p)
        dt_pressures.append(80.0)
        forces.append(f)
        expected_forces.append((p * 0.0133322 * 3.14) + 1.5)
        flows.append(q)
        sensor_healths.append(res.sensor_health_score)
        process_risks.append(res.process_risk_score)
        predicted_classes.append(res.predicted_fault_class)
        calibrated_confs.append(res.calibrated_confidence)
        prob_normal.append(res.calibrated_probabilities.get("NORMAL", 0.0))
        prob_drift.append(res.calibrated_probabilities.get("POSITIVE_DRIFT", 0.0) + res.calibrated_probabilities.get("SUDDEN_OFFSET", 0.0))
        prob_occlusion.append(res.calibrated_probabilities.get("GENUINE_OCCLUSION", 0.0))

    # Plot 4-panel dashboard
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Panel 1: Pressure & Digital Twin
    ax1 = axes[0]
    ax1.plot(timestamps, pressures, "b-", linewidth=2, label="Measured Pressure $P_{sensor}$ (mmHg)")
    ax1.plot(timestamps, dt_pressures, "k--", linewidth=1.5, label="Digital Twin Expectation $P_{DT}$ (mmHg)")
    ax1.axvline(35, color="gray", linestyle=":", alpha=0.7)
    ax1.axvline(75, color="gray", linestyle=":", alpha=0.7)
    ax1.text(17, 300, "Phase 1: Normal", horizontalalignment="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    ax1.text(55, 300, "Phase 2: Sensor Drift", horizontalalignment="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax1.text(97, 300, "Phase 3: Genuine Occlusion", horizontalalignment="center", fontweight="bold", bbox=dict(boxstyle="round", facecolor="salmon", alpha=0.5))
    ax1.set_ylabel("Pressure (mmHg)", fontweight="bold")
    ax1.set_title("Infusion Pump ML Sensor Intelligence — Proof of Concept Validation Dashboard", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Plunger Force & Flow Delivery
    ax2 = axes[1]
    ax2_flow = ax2.twinx()
    l1 = ax2.plot(timestamps, forces, "g-", linewidth=2, label="Measured Plunger Force (N)")
    l2 = ax2.plot(timestamps, expected_forces, "purple", linestyle="--", linewidth=1.5, label="Expected Hydraulic Force $P \cdot A + F_f$ (N)")
    l3 = ax2_flow.plot(timestamps, flows, "tab:cyan", linewidth=2, label="Measured Flow (mL/h)")
    ax2.axvline(35, color="gray", linestyle=":", alpha=0.7)
    ax2.axvline(75, color="gray", linestyle=":", alpha=0.7)
    ax2.set_ylabel("Plunger Force (N)", fontweight="bold", color="darkgreen")
    ax2_flow.set_ylabel("Flow (mL/h)", fontweight="bold", color="tab:cyan")
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Decoupled Sensor Health vs Process Risk
    ax3 = axes[2]
    ax3.plot(timestamps, sensor_healths, "tab:green", linewidth=2.5, label="Sensor Health Score (0-100: Transducer Integrity)")
    ax3.plot(timestamps, process_risks, "tab:red", linewidth=2.5, linestyle="-", label="Clinical Process Risk Score (0-100: Patient Hazard)")
    ax3.axvline(35, color="gray", linestyle=":", alpha=0.7)
    ax3.axvline(75, color="gray", linestyle=":", alpha=0.7)
    ax3.set_ylabel("Score (0 - 100)", fontweight="bold")
    ax3.set_ylim(-5, 105)
    ax3.axhline(80, color="green", linestyle=":", alpha=0.4, label="Healthy Threshold (>= 80)")
    ax3.axhline(60, color="red", linestyle=":", alpha=0.4, label="Risk Alarm Threshold (>= 60)")
    ax3.legend(loc="center left")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Calibrated Multi-Class Fault Probabilities
    ax4 = axes[3]
    ax4.plot(timestamps, prob_normal, "tab:blue", linewidth=2, label="P(NORMAL)")
    ax4.plot(timestamps, prob_drift, "tab:orange", linewidth=2, label="P(DRIFT / OFFSET)")
    ax4.plot(timestamps, prob_occlusion, "tab:red", linewidth=2, label="P(GENUINE_OCCLUSION)")
    ax4.axvline(35, color="gray", linestyle=":", alpha=0.7)
    ax4.axvline(75, color="gray", linestyle=":", alpha=0.7)
    ax4.set_xlabel("Time (seconds)", fontweight="bold")
    ax4.set_ylabel("Calibrated Probability", fontweight="bold")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    out_path = Path("reports/figures/poc_validation_dashboard.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Validation dashboard successfully saved to: {out_path.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()

