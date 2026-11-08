"""Automated Exploratory Data Analysis (EDA) and reporting script."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data.inspect import inspect_raw_datasets, export_inventory_reports
from src.utils.config import RAW_DATA_DIR, REPORTS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger("run_eda")


def run_comprehensive_eda():
    """Execute complete EDA across all raw datasets and produce Markdown report."""
    logger.info("Inspecting raw datasets across data/raw/ ...")
    inventory = inspect_raw_datasets()

    logger.info("Exporting dataset inventory reports...")
    export_inventory_reports(inventory)

    eda_report_path = REPORTS_DIR / "EDA_REPORT.md"
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(eda_report_path, "w", encoding="utf-8") as rf:
        rf.write("# Exploratory Data Analysis (EDA) & Data Quality Report\n\n")
        rf.write("**Project:** Self-Calibrating and Self-Healing Medical Devices Using AI and Digital Twins (Infusion Pump Target)\n")
        rf.write("**Author:** Senior ML & Biomedical AI Team\n")
        rf.write("**Phase:** Phase 1 - Environment, Acquisition, Inspection & Baseline Analysis\n\n")
        rf.write("---\n\n")

        rf.write("## 1. Executive Summary\n\n")
        total_datasets = len(inventory)
        acquired = [k for k, v in inventory.items() if v["status"] == "ACQUIRED"]
        pending = [k for k, v in inventory.items() if v["status"] != "ACQUIRED"]

        rf.write(f"- **Total Target Datasets:** {total_datasets}\n")
        rf.write(f"- **Datasets Acquired / Local Files Inspected:** {len(acquired)} (`{', '.join(acquired)}`)\n")
        rf.write(f"- **Datasets Requiring Manual / Author Request:** {len(pending)} (`{', '.join(pending)}`)\n\n")

        rf.write("### Critical ML Engineering Distinctions:\n\n")
        rf.write("1. **No Continuous Raw Intravenous Pump Waveform in Open Domain:** None of the public datasets provide an open-access, continuous multi-sensor intravenous infusion stream with simultaneous pressure, force, flow, and motor encoder signals over months of aging.\n")
        rf.write("2. **DosePI Domain Specifics:** DosePI is an exceptional high-resolution dataset, but it represents pharmaceutical peristaltic dosing/filling cycles (dispensing target volumes into vials with high-precision balance readouts), not continuous clinical IV delivery in an ambulatory patient setting.\n")
        rf.write("3. **Typical Sensor Defects Role:** Zenodo 56358 provides clean, controlled synthetic injections (drift, offset, peaks, noise) atop real multi-channel sensor baseline in a container. It serves as a gold standard for validating our statistical drift and offset detection engines (EWMA/CUSUM), but is not an infusion pump dataset.\n")
        rf.write("4. **Clinical Data Governance Restrictions:** Datasets B (PMC8555317), C (PMC11535327), and D (JMIR AI e48628) represent vital infusion domain literature, but their continuous raw sensor archives are protected by hospital clinical governance and author-access restrictions.\n\n")

        rf.write("---\n\n")
        rf.write("## 2. In-Depth Inspection by Dataset\n\n")

        for ds_id, data in inventory.items():
            rf.write(f"### Dataset: {data['name']} (`{ds_id}`)\n\n")
            rf.write(f"- **DOI / Link:** [{data.get('doi', 'Link')}]({data['paper_url']})\n")
            rf.write(f"- **Acquisition Status:** `{data['status']}`\n")
            rf.write(f"- **Domain Context:** {data['type']}\n")
            rf.write(f"- **Analytical Purpose in ML System:** {data['purpose']}\n")
            rf.write(f"- **Supported Sensors:** `{', '.join(data['sensors_supported'])}`\n\n")

            files = data.get("files", [])
            if not files:
                rf.write("> **Status:** No local files found. Refer to `DATASET_GUIDE.md` for manual acquisition or author request procedures.\n\n")
                continue

            rf.write(f"#### Inspected Files ({len(files)} total):\n\n")
            for f in files:
                if "error" in f:
                    rf.write(f"- **{f['file_name']}**: Read error: {f['error']}\n")
                    continue

                rf.write(f"##### File: `{f['file_name']}`\n")
                rf.write(f"- **Path:** `{f.get('relative_path', f['file_name'])}`\n")
                rf.write(f"- **Size:** {f.get('file_size_mb', 0):.2f} MB ({f.get('file_size_bytes', 0):,} bytes)\n")

                if "total_columns" in f:
                    rf.write(f"- **Rows Inspected:** {f['total_rows_inspected']:,}\n")
                    rf.write(f"- **Columns ({f['total_columns']}):** `{', '.join(f['columns'])}`\n")
                    rf.write(f"- **Duplicate Rows:** {f.get('duplicate_rows', 0)}\n\n")

                    rf.write("| Column | Data Type | Null Count (%) | Unique Values | Min | Max | Mean ± Std |\n")
                    rf.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for c_name, c_prof in f.get("column_profiles", {}).items():
                        min_val = f"{c_prof.get('min'):.4f}" if "min" in c_prof else "N/A"
                        max_val = f"{c_prof.get('max'):.4f}" if "max" in c_prof else "N/A"
                        mean_std = f"{c_prof.get('mean'):.4f} ± {c_prof.get('std'):.4f}" if "mean" in c_prof else "N/A"
                        rf.write(f"| `{c_name}` | `{c_prof['dtype']}` | {c_prof['null_count']} ({c_prof['null_pct']}%) | {c_prof['unique_values']:,} | {min_val} | {max_val} | {mean_std} |\n")
                    rf.write("\n")

                    if f.get("timestamp_analysis"):
                        ts_info = f["timestamp_analysis"]
                        rf.write(f"**Sampling & Temporal Dynamics:**\n")
                        rf.write(f"- Primary Timestamp Column: `{ts_info.get('column_identified', 'N/A')}`\n")
                        if "sampling_interval_median" in ts_info:
                            rf.write(f"- Median Sampling Interval: {ts_info['sampling_interval_median']}\n")
                            rf.write(f"- Min/Max Interval: {ts_info.get('sampling_interval_min')} / {ts_info.get('sampling_interval_max')}\n")
                        if "is_strictly_monotonic" in ts_info:
                            rf.write(f"- Strictly Monotonic Increasing: `{ts_info['is_strictly_monotonic']}`\n")
                        rf.write("\n")

            rf.write("\n")

    logger.info(f"Comprehensive EDA report generated at {eda_report_path}")
    print(f"\n[EDA COMPLETED] Full report available at: {eda_report_path}")


if __name__ == "__main__":
    run_comprehensive_eda()

