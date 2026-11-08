"""Dataset loader module for Infusion Pump ML System.

Provides consistent, validated Pandas loading interfaces for heterogeneous datasets:
- DosePI (Gravimetric peristaltic filling dataset)
- Typical Sensor Defects (Zenodo multi-sensor baseline and defect injection files)
- Pressure Occlusion / Force Failure / Smart Pump Logs (when provided or acquired)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.utils.config import DATASET_METADATA, RAW_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_dosepi(raw_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load DosePI dataset CSV files from data/raw/dosepi/.

    Returns:
        Dictionary mapping filename (or experiment key) to parsed DataFrame.
    """
    target_dir = raw_dir or DATASET_METADATA["dosepi"]["target_dir"]
    if not target_dir.exists():
        raise FileNotFoundError(f"DosePI directory {target_dir} not found.")

    csv_files = list(target_dir.glob("*.csv")) + list(target_dir.glob("*.tsv"))
    if not csv_files:
        logger.warning(f"No CSV/TSV files found in {target_dir}.")
        return {}

    dfs = {}
    for f in csv_files:
        logger.info(f"Loading DosePI file: {f.name}")
        df = pd.read_csv(f)
        dfs[f.stem] = df

    return dfs


def load_sensor_defects(
    raw_dir: Optional[Path] = None,
    include_defects: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load Typical Sensor Defects dataset from data/raw/sensor_defects/.

    Handles both root directory and unzipped 'extracted' subfolder.

    Args:
        raw_dir: Custom path to raw directory.
        include_defects: List of specific defect keywords to load (e.g. ['drift', 'offset', 'noise']).
                         If None, loads all available CSV files.

    Returns:
        Dictionary mapping defect file names to DataFrames.
    """
    target_dir = raw_dir or DATASET_METADATA["sensor_defects"]["target_dir"]
    if not target_dir.exists():
        raise FileNotFoundError(f"Sensor Defects directory {target_dir} not found.")

    # Find CSV files in directory or extracted subfolder
    csv_files = list(target_dir.glob("*.csv")) + list(target_dir.rglob("extracted/**/*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {target_dir}.")
        return {}

    loaded = {}
    for f in csv_files:
        # Check if matches filter
        if include_defects:
            if not any(d.lower() in f.name.lower() for d in include_defects):
                continue

        logger.info(f"Loading Sensor Defects file: {f.name}")
        try:
            # Zenodo sensor defects CSV uses ';' or ',' delimiter and latin1 encoding for [°C] symbol
            encodings = ["latin1", "utf-8", "cp1252"]
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(f, sep=None, engine="python", encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is not None:
                loaded[f.stem] = df
            else:
                logger.error(f"Could not decode {f.name} with standard encodings.")
        except Exception as e:
            logger.error(f"Error reading {f.name}: {e}")

    return loaded


def load_pressure_occlusion(raw_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load Pressure Occlusion records if present in data/raw/pressure_occlusion/."""
    target_dir = raw_dir or DATASET_METADATA["pressure_occlusion"]["target_dir"]
    csvs = list(target_dir.glob("*.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])


def load_force_failure(raw_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load Force-based Failure records if present in data/raw/force_failure/."""
    target_dir = raw_dir or DATASET_METADATA["force_failure"]["target_dir"]
    csvs = list(target_dir.glob("*.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])


def load_smart_pump_logs(raw_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load Smart Pump Event Logs if present in data/raw/smart_pump_logs/."""
    target_dir = raw_dir or DATASET_METADATA["smart_pump_logs"]["target_dir"]
    csvs = list(target_dir.glob("*.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])
