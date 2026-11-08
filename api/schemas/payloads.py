"""Pydantic request and response schemas for Infusion Pump ML REST API."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """System and environment status payload."""
    status: str = Field(..., description="Overall API health status ('OK', 'DEGRADED')")
    service: str = Field("Infusion Pump ML Intelligence Engine", description="Service name")
    version: str = Field("1.0.0-phase4", description="ML system release version")
    timestamp: str = Field(..., description="UTC ISO8601 timestamp")
    device_target: str = Field("Infusion Pump", description="Target medical device")
    models_loaded: Dict[str, bool] = Field(..., description="Loaded state of production models")
    safety_disclaimer: str = Field(..., description="Mandatory biomedical advisory disclaimer")


class ModelStatusResponse(BaseModel):
    """Production model inventory and operational metadata."""
    timestamp: str
    provenance: str = "SYNTHETIC_DATA"
    models: Dict[str, Any] = Field(..., description="Detailed model artifacts and parameters")


class FaultPredictionRequest(BaseModel):
    """Payload for real-time sensor fault classification."""
    session_id: Optional[str] = Field("pump_01", description="Optional session identifier")
    timestamp: Optional[float] = Field(None, description="Optional timestamp")
    sensor_id: str = Field("PRESS_LINE_01", description="Identifier of the target sensor transducer")
    sensor_pressure_mmhg: Union[float, List[float]] = Field(..., description="Window or reading of pressure (mmHg)")
    sensor_force_n: Optional[Union[float, List[float]]] = Field(None, description="Optional plunger force readings (N)")
    sensor_flow_ml_h: Optional[Union[float, List[float]]] = Field(None, description="Optional measured delivery flow (mL/h)")
    sensor_temperature_c: Optional[Union[float, List[float]]] = Field(None, description="Optional ambient temperature readings (°C)")
    commanded_rate_ml_h: Optional[float] = Field(50.0, description="Pump commanded infusion rate setpoint (mL/h)")
    dt_predicted_pressure_mmhg: Optional[float] = Field(None, description="Digital Twin expected line pressure (mmHg)")

    @field_validator(
        "sensor_pressure_mmhg",
        "sensor_force_n",
        "sensor_flow_ml_h",
        "sensor_temperature_c",
        mode="before",
    )
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return [float(v)]
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Input list cannot be empty")
        return v


class FaultPredictionResponse(BaseModel):
    """Fault attribution response with SHAP feature explanations."""
    sensor_id: str
    timestamp: str
    provenance: str
    predicted_fault_class: str
    class_index: int
    model_confidence: float
    class_probabilities: Dict[str, float]
    is_sensor_fault: bool
    is_process_anomaly: bool
    shap_explanation: Optional[Dict[str, Any]] = None
    disclaimer: str


class SensorErrorRequest(BaseModel):
    """Payload for measurement error estimation."""
    session_id: Optional[str] = Field("pump_01", description="Optional session identifier")
    timestamp: Optional[float] = Field(None, description="Optional timestamp")
    sensor_id: str = Field("PRESS_LINE_01", description="Transducer identifier")
    sensor_type: str = Field("pressure", description="Sensor type: 'pressure', 'flow'")
    sensor_pressure_mmhg: Union[float, List[float]] = Field(..., description="Recent pressure signal window")
    observed_sensor_value: Optional[float] = Field(None, description="Current uncompensated sensor reading")
    independent_reference_value: Optional[float] = Field(None, description="Independent physical reference reading")
    sensor_force_n: Optional[Union[float, List[float]]] = None
    sensor_flow_ml_h: Optional[Union[float, List[float]]] = None
    sensor_temperature_c: Optional[Union[float, List[float]]] = None
    commanded_rate_ml_h: Optional[float] = 50.0
    dt_predicted_pressure_mmhg: Optional[float] = None

    @field_validator(
        "sensor_pressure_mmhg",
        "sensor_force_n",
        "sensor_flow_ml_h",
        "sensor_temperature_c",
        mode="before",
    )
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return [float(v)]
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Input list cannot be empty")
        return v


class SensorErrorResponse(BaseModel):
    """Sensor measurement error estimation report."""
    sensor_id: str
    sensor_type: str
    units: str
    status: str  # 'ESTIMATION_AVAILABLE' or 'INSUFFICIENT_REFERENCE_DATA'
    observed_sensor_value: Optional[float] = None
    estimated_sensor_error: Optional[float] = None
    estimated_calibrated_value: Optional[float] = None
    confidence_interval_95: Optional[List[float]] = None
    model_residual_rmse: Optional[float] = None
    advisory_recommendation: str
    disclaimer: str


class SensorHealthRequest(BaseModel):
    """Payload for integrated sensor health and process risk scoring."""
    session_id: Optional[str] = Field("pump_01", description="Optional session identifier")
    timestamp: Optional[float] = Field(None, description="Optional timestamp")
    sensor_id: str = Field("PRESS_LINE_01", description="Target sensor identifier")
    sensor_type: str = Field("pressure", description="Type of sensor channel")
    sensor_pressure_mmhg: Union[float, List[float]] = Field(..., description="Time-series pressure window")
    sensor_force_n: Optional[Union[float, List[float]]] = None
    sensor_flow_ml_h: Optional[Union[float, List[float]]] = None
    sensor_temperature_c: Optional[Union[float, List[float]]] = None
    commanded_rate_ml_h: Optional[float] = 50.0
    dt_predicted_pressure_mmhg: Optional[float] = None
    observed_sensor_value: Optional[float] = None
    independent_reference_value: Optional[float] = None

    @field_validator(
        "sensor_pressure_mmhg",
        "sensor_force_n",
        "sensor_flow_ml_h",
        "sensor_temperature_c",
        mode="before",
    )
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return [float(v)]
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Input list cannot be empty")
        return v


class SensorHealthResponse(BaseModel):
    """Integrated interpretable sensor health and clinical delivery risk report."""
    sensor_id: str
    sensor_type: str
    status: str
    sensor_health_score: float = Field(..., ge=0.0, le=100.0, description="Transducer physical integrity (0-100)")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Alias for sensor health score")
    process_risk_score: float = Field(..., ge=0.0, le=100.0, description="Clinical delivery risk (0-100)")
    process_risk_level: str = Field(..., description="'LOW', 'MODERATE', 'HIGH', 'CRITICAL'")
    model_confidence: float = Field(..., ge=0.0, le=100.0)
    anomaly_detected: bool
    anomaly_severity_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly score from Isolation Forest (not a failure probability)")
    drift_detected: bool
    drift_direction: Optional[str] = None
    reference_available: bool
    estimated_measurement_error: Optional[float] = None
    recommended_action: str
    evidence_breakdown: List[str]
    deductions: Dict[str, float]
    process_risk_factors: Dict[str, float]
    disclaimer: str


class BatchPredictionRequest(BaseModel):
    """Batch evaluation payload containing multiple windows or devices."""
    items: List[FaultPredictionRequest] = Field(..., min_length=1)


class BatchPredictionResponse(BaseModel):
    """Batch evaluation response."""
    timestamp: str
    total_processed: int
    results: List[FaultPredictionResponse]


# ---------------------------------------------------------------------------
# Phase 6: Sequential Telemetry & Stateful Stream Schemas
# ---------------------------------------------------------------------------

class SequentialObservationRequest(BaseModel):
    """Payload for sequential, sample-by-sample sensor observation ingestion."""
    session_id: str = Field("pump_session_01", min_length=1, description="Unique session/channel identifier for stateful buffering")
    sensor_id: str = Field("PRESS_LINE_01", min_length=1, description="Identifier of the target sensor transducer")
    timestamp: float = Field(..., description="Timestamp in seconds (epoch or relative)")
    sensor_pressure_mmhg: Union[float, List[float]] = Field(..., description="Current transducer pressure reading (mmHg)")
    sensor_force_n: Optional[Union[float, List[float]]] = Field(None, description="Optional measured plunger force (N)")
    sensor_flow_ml_h: Optional[Union[float, List[float]]] = Field(None, description="Optional measured delivery flow (mL/h)")
    sensor_temperature_c: Optional[Union[float, List[float]]] = Field(None, description="Optional ambient temperature (°C)")
    commanded_rate_ml_h: float = Field(50.0, ge=0.0, le=1500.0, description="Pump delivery setpoint (mL/h)")
    dt_predicted_pressure_mmhg: Optional[float] = Field(None, description="Digital Twin expected line pressure (mmHg)")
    independent_reference_value: Optional[float] = Field(None, description="Independent physical reference reading (if available)")
    baseline_pressure_mmhg: Optional[float] = Field(None, description="Optional known catheter baseline pressure (mmHg)")

    @field_validator(
        "sensor_pressure_mmhg",
        "sensor_force_n",
        "sensor_flow_ml_h",
        "sensor_temperature_c",
        mode="before",
    )
    @classmethod
    def coerce_to_scalar(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input list cannot be empty")
            return float(v[-1])
        return float(v)

    @field_validator("sensor_pressure_mmhg")
    @classmethod
    def validate_pressure_bounds(cls, v):
        if v < -50.0 or v > 2000.0:
            raise ValueError(f"Pressure {v} out of bounds [-50, 2000] mmHg")
        return v

    @field_validator("sensor_force_n")
    @classmethod
    def validate_force_bounds(cls, v):
        if v is not None and (v < 0.0 or v > 200.0):
            raise ValueError(f"Force {v} out of bounds [0, 200] N")
        return v

    @field_validator("sensor_flow_ml_h")
    @classmethod
    def validate_flow_bounds(cls, v):
        if v is not None and (v < 0.0 or v > 2000.0):
            raise ValueError(f"Flow {v} out of bounds [0, 2000] mL/h")
        return v

    @field_validator("sensor_temperature_c")
    @classmethod
    def validate_temp_bounds(cls, v):
        if v is not None and (v < -20.0 or v > 100.0):
            raise ValueError(f"Temperature {v} out of bounds [-20, 100] C")
        return v


class SequentialPredictionResponse(BaseModel):
    """Real-time multi-task diagnostic response for sequential stream observation."""
    session_id: str
    sensor_id: str
    timestamp: float
    provenance: str = "SYNTHETIC_DATA"
    stream_quality: str
    warmup_status: str
    buffered_samples: int
    predicted_fault_class: str
    class_index: int
    model_confidence: float
    calibrated_confidence: Optional[float] = None
    calibrated_probabilities: Optional[Dict[str, float]] = None
    raw_class_probabilities: Dict[str, float]
    is_sensor_fault: bool
    is_process_anomaly: bool
    sensor_health_score: float = Field(..., ge=0.0, le=100.0)
    process_risk_score: float = Field(..., ge=0.0, le=100.0)
    process_risk_level: str
    anomaly_detected: bool
    anomaly_severity_score: float = Field(..., ge=0.0, le=1.0)
    estimated_sensor_error: Optional[float] = None
    conformal_error_interval_95: Optional[List[float]] = None
    error_status: str
    advisory_recommendation: str
    disclaimer: str


class SessionResetRequest(BaseModel):
    """Payload to reset a stateful stream buffer session."""
    session_id: str = Field(..., min_length=1, description="Session identifier to reset")


class SessionResetResponse(BaseModel):
    """Acknowledgement of stream session reset."""
    session_id: str
    reset_success: bool
    timestamp: str
    message: str


class SessionStatusResponse(BaseModel):
    """Session buffer status and telemetry health indicators."""
    session_id: str
    buffered_samples: int
    max_buffer_size: int
    warmup_threshold: int
    warmup_status: str
    total_observations_received: int
    last_timestamp: Optional[float] = None
    telemetry_quality_issues: Dict[str, int]

