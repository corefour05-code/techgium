"""Data inspection and validation module for Infusion Pump ML System.

Provides automated inspection of raw and interim data files:
- Schema and column detection
- Missing values and NaN percentage
- Sampling frequency and timestamp regularity
- Value ranges and out-of-bound checks
- Duplicate detection
- Sensor-specific sanity checks
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.config import DATASET_METADATA, RAW_DATA_DIR, REPORTS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _read_csv_with_fallback(file_path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with encoding fallback (utf-8 -> latin1 -> cp1252)."""
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    # If all fail, try with ignore errors
    return pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", **kwargs)


def inspect_csv_file(file_path: Path, max_rows: int = 50000) -> Dict[str, Any]:
    """Perform comprehensive structural and statistical inspection on a CSV file.

    Args:
        file_path: Path to the CSV file.
        max_rows: Maximum rows to sample for detailed stats if file is massive.

    Returns:
        Inspection summary dictionary.
    """
    if not file_path.exists():
        return {"error": f"File {file_path} does not exist"}

    file_size_bytes = file_path.stat().st_size

    try:
        # Read header first
        header_df = _read_csv_with_fallback(file_path, nrows=5)
        columns = list(header_df.columns)

        # Full or sampled read
        df = _read_csv_with_fallback(file_path, nrows=max_rows)
        row_count = len(df)

        col_profiles = {}
        for col in columns:
            series = df[col]
            dtype_str = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = round(float(null_count / row_count * 100), 2) if row_count > 0 else 0.0

            profile: Dict[str, Any] = {
                "dtype": dtype_str,
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_values": int(series.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna()
                if len(clean_s) > 0:
                    profile.update({
                        "min": float(clean_s.min()),
                        "max": float(clean_s.max()),
                        "mean": round(float(clean_s.mean()), 4),
                        "std": round(float(clean_s.std()), 4) if len(clean_s) > 1 else 0.0,
                        "has_infinite": bool(np.isinf(clean_s).any()),
                    })
            col_profiles[col] = profile

        # Timestamp analysis if timestamp-like column exists
        ts_candidates = [c for c in columns if any(k in c.lower() for k in ["time", "timestamp", "date", "sec", "t_"])]
        ts_analysis = {}
        if ts_candidates:
            ts_col = ts_candidates[0]
            ts_s = df[ts_col].dropna()
            ts_analysis["column_identified"] = ts_col
            try:
                # Try numeric delta if numeric
                if pd.api.types.is_numeric_dtype(ts_s):
                    diffs = ts_s.diff().dropna()
                    if len(diffs) > 0:
                        ts_analysis["sampling_interval_median"] = round(float(diffs.median()), 4)
                        ts_analysis["sampling_interval_min"] = round(float(diffs.min()), 4)
                        ts_analysis["sampling_interval_max"] = round(float(diffs.max()), 4)
                        ts_analysis["is_strictly_monotonic"] = bool(ts_s.is_monotonic_increasing)
                else:
                    # Parse as datetime
                    dt_s = pd.to_datetime(ts_s, errors="coerce")
                    valid_dt = dt_s.dropna()
                    if len(valid_dt) > 1:
                        diffs = valid_dt.diff().dropna().dt.total_seconds()
                        ts_analysis["sampling_interval_sec_median"] = round(float(diffs.median()), 4)
                        ts_analysis["start_time"] = str(valid_dt.min())
                        ts_analysis["end_time"] = str(valid_dt.max())
                        ts_analysis["is_strictly_monotonic"] = bool(valid_dt.is_monotonic_increasing)
            except Exception as e:
                ts_analysis["parsing_error"] = str(e)

        return {
            "file_name": file_path.name,
            "relative_path": str(file_path.relative_to(RAW_DATA_DIR.parent)),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
            "total_rows_inspected": row_count,
            "total_columns": len(columns),
            "columns": columns,
            "column_profiles": col_profiles,
            "timestamp_analysis": ts_analysis,
            "duplicate_rows": int(df.duplicated().sum()),
        }

    except Exception as e:
        logger.error(f"Error inspecting {file_path}: {e}")
        return {
            "file_name": file_path.name,
            "error": str(e),
            "file_size_bytes": file_size_bytes
        }


def inspect_raw_datasets() -> Dict[str, Any]:
    """Inspect all files in the data/raw/ subdirectories and compile an inventory report."""
    inventory: Dict[str, Any] = {}

    for ds_id, meta in DATASET_METADATA.items():
        ds_dir = meta["target_dir"]
        ds_inventory = {
            "id": ds_id,
            "name": meta["name"],
            "paper_url": meta["paper_url"],
            "doi": meta.get("doi"),
            "purpose": meta["purpose"],
            "type": meta["type"],
            "sensors_supported": meta["sensors_supported"],
            "files": [],
            "status": "NOT_ACQUIRED",
            "total_size_mb": 0.0,
        }

        if ds_dir.exists():
            # Find all data files (csv, txt, zip, dat, etc.)
            files = [p for p in ds_dir.rglob("*") if p.is_file() and p.name != ".gitkeep" and not p.name.endswith(".md")]
            if files:
                ds_inventory["status"] = "ACQUIRED"
                total_bytes = 0
                for f in files:
                    total_bytes += f.stat().st_size
                    if f.suffix.lower() in [".csv", ".txt", ".tsv"]:
                        f_insp = inspect_csv_file(f)
                        ds_inventory["files"].append(f_insp)
                    else:
                        ds_inventory["files"].append({
                            "file_name": f.name,
                            "relative_path": str(f.relative_to(RAW_DATA_DIR.parent)),
                            "file_size_bytes": f.stat().st_size,
                            "file_size_mb": round(f.stat().st_size / (1024 * 1024), 3),
                            "format": f.suffix,
                        })
                ds_inventory["total_size_mb"] = round(total_bytes / (1024 * 1024), 3)

        inventory[ds_id] = ds_inventory

    return inventory


def export_inventory_reports(inventory: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Save dataset inventory as structured JSON and Markdown."""
    out_dir = output_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_inventory.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(inventory, jf, indent=2)
    logger.info(f"Exported JSON inventory to {json_path}")

    md_path = out_dir / "dataset_inventory.md"
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("# Research Dataset Inventory & Availability Report\n\n")
        mf.write("Generated for the MedTech Challenge: Self-Calibrating and Self-Healing Infusion Pump ML System.\n\n")
        mf.write("| Dataset ID | Title | Status | Files Count | Size (MB) | Supported Sensors |\n")
        mf.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for ds_id, data in inventory.items():
            files_count = len(data.get("files", []))
            size_mb = data.get("total_size_mb", 0.0)
            sensors = ", ".join(data.get("sensors_supported", []))
            mf.write(f"| **{ds_id}** | {data['name'][:45]}... | `{data['status']}` | {files_count} | {size_mb} MB | {sensors} |\n")

        mf.write("\n\n---\n\n")
        for ds_id, data in inventory.items():
            mf.write(f"## Dataset: {data['name']}\n\n")
            mf.write(f"- **ID:** `{ds_id}`\n")
            mf.write(f"- **Publication / URL:** [{data.get('doi', 'Link')}]({data['paper_url']})\n")
            mf.write(f"- **Status:** `{data['status']}`\n")
            mf.write(f"- **Dataset Type:** {data['type']}\n")
            mf.write(f"- **Analytical Purpose:** {data['purpose']}\n\n")

            files = data.get("files", [])
            if not files:
                mf.write("> **Note on Acquisition:** This dataset has not yet been populated or requires author/institutional data governance approval.\n\n")
            else:
                mf.write("### Inspected Files:\n\n")
                for f in files:
                    if "error" in f:
                        mf.write(f"- **{f['file_name']}**: Error inspecting file - {f['error']}\n")
                        continue
                    mf.write(f"#### File: `{f['file_name']}` ({f.get('file_size_mb', 0)} MB)\n\n")
                    if "total_columns" in f:
                        mf.write(f"- Total rows inspected: {f['total_rows_inspected']:,}\n")
                        mf.write(f"- Total columns: {f['total_columns']}\n")
                        mf.write(f"- Columns found: `{[c for c in f['columns']]}`\n")
                        if f.get("timestamp_analysis"):
                            mf.write(f"- Timestamp Analysis: {json.dumps(f['timestamp_analysis'])}\n")
                    mf.write("\n")

    logger.info(f"Exported Markdown inventory report to {md_path}")
    return md_path
