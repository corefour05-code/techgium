"""Configuration management for Infusion Pump ML System."""

from enum import Enum
from pathlib import Path
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CUSTOM_DATA_DIR = DATA_DIR / "custom"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_SEED = 42


class DataProvenance(str, Enum):
    """Explicit provenance tag for all data records to ensure scientific traceability."""
    REAL_PUBLIC_DATA = "REAL_PUBLIC_DATA"
    REAL_CUSTOM_DATA = "REAL_CUSTOM_DATA"
    SYNTHETIC_DATA = "SYNTHETIC_DATA"


class SensorType(str, Enum):
    """Supported sensor types across heterogeneous pump architectures."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW = "flow"
    FORCE = "force"
    MOTOR_ENCODER = "motor_encoder"
    CUSTOM = "custom"


class SensorHealthStatus(str, Enum):
    """Interpretable sensor health status classifications."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DRIFT_SUSPECTED = "DRIFT_SUSPECTED"
    FAULT_SUSPECTED = "FAULT_SUSPECTED"
    MAINTENANCE_REQUIRED = "MAINTENANCE_REQUIRED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class FaultClass(int, Enum):
    """Configurable fault classification targets."""
    NORMAL = 0
    POSITIVE_DRIFT = 1
    NEGATIVE_DRIFT = 2
    SUDDEN_OFFSET = 3
    EXCESSIVE_NOISE = 4
    SENSOR_DROPOUT = 5
    SENSOR_FREEZE = 6
    TEMPERATURE_VARIATION = 7
    GENUINE_DOWNSTREAM_OCCLUSION = 8
    OTHER_ABNORMAL = 9


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "dosepi": {
        "id": "dosepi",
        "name": "DosePI: Extensive Dataset for Peristaltic Pump Accuracy Enhancement",
        "paper_url": "https://www.nature.com/articles/s41597-025-05902-z",
        "repository_url": "https://doi.org/10.6084/m9.figshare.28391645",
        "figshare_article_id": "28391645",
        "doi": "10.1038/s41597-025-05902-z",
        "target_dir": RAW_DATA_DIR / "dosepi",
        "sensors_supported": ["flow", "weight_dosing", "motor_steps", "operating_parameters"],
        "purpose": "Pump delivery accuracy, dosing variation, compensation modelling, pump output behaviour.",
        "type": "Pharmaceutical peristaltic filling pump dataset (not clinical intravenous continuous infusion)."
    },
    "pressure_occlusion": {
        "id": "pressure_occlusion",
        "name": "Multi-infusion with integrated multiple pressure sensing allows earlier detection of line occlusions",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8555317/",
        "pmcid": "PMC8555317",
        "doi": "10.3390/s21206934",
        "target_dir": RAW_DATA_DIR / "pressure_occlusion",
        "sensors_supported": ["pressure"],
        "purpose": "Pressure behaviour, normal infusion patterns, genuine occlusion events, pressure anomaly analysis.",
        "type": "In vitro multi-infusion clinical pressure-event research study."
    },
    "force_failure": {
        "id": "force_failure",
        "name": "Development of a Real-time Force-based Algorithm for Infusion Failure Detection",
        "paper_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11535327/",
        "pmcid": "PMC11535327",
        "doi": "10.1177/19322968241247530",
        "target_dir": RAW_DATA_DIR / "force_failure",
        "sensors_supported": ["force", "pressure"],
        "purpose": "Pressure and force relationships, infusion failure classification, cross-sensor consistency analysis.",
        "type": "In vitro force and line-pressure experimental failure study."
    },
    "smart_pump_logs": {
        "id": "smart_pump_logs",
        "name": "Hospitals Infusion Data Set: Spain (Determinants of Intravenous Infusion Longevity and Infusion Failure)",
        "paper_url": "https://ai.jmir.org/2023/1/e48628/",
        "doi": "10.2196/48628",
        "target_dir": RAW_DATA_DIR / "smart_pump_logs",
        "sensors_supported": ["event_logs", "operating_conditions", "alarm_records"],
        "purpose": "Infusion operating conditions, pump event information, occlusion and infusion termination context.",
        "type": "Retrospective hospital smart pump event logs (operational discrete events, not continuous raw waveforms)."
    },
    "sensor_defects": {
        "id": "sensor_defects",
        "name": "Typical Sensor Defects Dataset",
        "paper_url": "https://zenodo.org/records/56358",
        "repository_url": "https://zenodo.org/records/56358",
        "zenodo_record_id": "56358",
        "doi": "10.5281/zenodo.56358",
        "target_dir": RAW_DATA_DIR / "sensor_defects",
        "sensors_supported": ["temperature", "smoke_defect_channels"],
        "purpose": "Sensor drift, offset, noise, peaks and abnormal sensor behaviour benchmark.",
        "type": "Generic multi-sensor defect benchmark dataset (hazardous material storage container, not an infusion pump)."
    }
}

