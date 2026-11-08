"""Sequential Sensor Stream Buffer and Session Management for Infusion Pump ML.

Provides stateful FIFO buffering for sequential telemetry feeds (e.g. 1 Hz live streaming
from Digital Twins or physical pump hardware).
Tracks warm-up sufficiency, timestamp ordering, and telemetry quality gaps without
breaking batch or stateless window compatibility.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.feature_pipeline import SensorFeaturePipeline
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class StreamQualityStatus(str, Enum):
    """Telemetry stream timing and integrity quality indicators."""
    OK = "OK"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUT_OF_ORDER_TIMESTAMP = "OUT_OF_ORDER_TIMESTAMP"
    STALE_TELEMETRY_GAP = "STALE_TELEMETRY_GAP"
    OUT_OF_RANGE_VALUE = "OUT_OF_RANGE_VALUE"


class WarmupStatus(str, Enum):
    """Pipeline initialization status reflecting rolling window feature maturity."""
    WARMUP = "WARMUP"  # Insufficient samples for multi-scale rolling feature extraction (< 15 steps)
    READY = "READY"    # Sufficient samples accumulated for full feature representation (>= 15 steps)


@dataclass
class SensorObservation:
    """Individual time-series observation from sensor telemetry feed."""
    timestamp: float  # Epoch seconds or relative test-run seconds
    sensor_pressure_mmhg: float
    sensor_force_n: Optional[float] = None
    sensor_flow_ml_h: Optional[float] = None
    sensor_temperature_c: Optional[float] = None
    commanded_rate_ml_h: float = 50.0
    dt_predicted_pressure_mmhg: Optional[float] = None
    independent_reference_value: Optional[float] = None
    sensor_id: str = "PRESS_LINE_01"


class SensorStreamBuffer:
    """Thread-safe stateful buffer maintaining rolling sensor telemetry for one session.
    
    Accumulates sequential observations and converts them to multi-feature dataframes
    matching the training pipeline requirements.
    """

    def __init__(
        self,
        session_id: str,
        max_buffer_size: int = 120,
        warmup_threshold: int = 15,
        stale_gap_threshold_s: float = 5.0,
    ):
        self.session_id = session_id
        self.max_buffer_size = max_buffer_size
        self.warmup_threshold = warmup_threshold
        self.stale_gap_threshold_s = stale_gap_threshold_s

        self._lock = threading.Lock()
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._last_timestamp: Optional[float] = None
        self._total_observations: int = 0
        self._quality_issues_count: Dict[str, int] = {
            StreamQualityStatus.DUPLICATE_TIMESTAMP.value: 0,
            StreamQualityStatus.OUT_OF_ORDER_TIMESTAMP.value: 0,
            StreamQualityStatus.STALE_TELEMETRY_GAP.value: 0,
        }

    def add_observation(
        self,
        obs: SensorObservation,
    ) -> Tuple[StreamQualityStatus, WarmupStatus]:
        """Ingest a single sensor observation and return stream quality and warmup status."""
        with self._lock:
            quality = StreamQualityStatus.OK

            if self._last_timestamp is not None:
                delta_t = obs.timestamp - self._last_timestamp
                if abs(delta_t) < 1e-6:
                    quality = StreamQualityStatus.DUPLICATE_TIMESTAMP
                    self._quality_issues_count[StreamQualityStatus.DUPLICATE_TIMESTAMP.value] += 1
                elif delta_t < 0.0:
                    quality = StreamQualityStatus.OUT_OF_ORDER_TIMESTAMP
                    self._quality_issues_count[StreamQualityStatus.OUT_OF_ORDER_TIMESTAMP.value] += 1
                elif delta_t > self.stale_gap_threshold_s:
                    quality = StreamQualityStatus.STALE_TELEMETRY_GAP
                    self._quality_issues_count[StreamQualityStatus.STALE_TELEMETRY_GAP.value] += 1

            self._buffer.append(obs)
            self._last_timestamp = obs.timestamp
            self._total_observations += 1

            warmup = (
                WarmupStatus.READY
                if len(self._buffer) >= self.warmup_threshold
                else WarmupStatus.WARMUP
            )

            return quality, warmup

    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffered observations to a pandas DataFrame for feature extraction."""
        with self._lock:
            if not self._buffer:
                return pd.DataFrame()

            records = []
            for obs in self._buffer:
                row = {
                    "timestamp": obs.timestamp,
                    "sensor_pressure_mmhg": obs.sensor_pressure_mmhg,
                    "commanded_rate_ml_h": obs.commanded_rate_ml_h,
                }
                if obs.sensor_force_n is not None:
                    row["sensor_force_n"] = obs.sensor_force_n
                if obs.sensor_flow_ml_h is not None:
                    row["sensor_flow_ml_h"] = obs.sensor_flow_ml_h
                if obs.sensor_temperature_c is not None:
                    row["sensor_temperature_c"] = obs.sensor_temperature_c
                if obs.dt_predicted_pressure_mmhg is not None:
                    row["dt_predicted_pressure_mmhg"] = obs.dt_predicted_pressure_mmhg
                records.append(row)

            return pd.DataFrame(records)

    def extract_features(
        self,
        pipeline: SensorFeaturePipeline,
        baseline_mmhg: Optional[float] = None,
    ) -> pd.DataFrame:
        """Extract multi-scale rolling feature vector for the most recent observation."""
        df = self.to_dataframe()
        if len(df) == 0:
            raise ValueError(f"Buffer for session '{self.session_id}' is empty.")

        # Transform dataframe through pipeline; extract last timestep
        feats_all = pipeline.transform_dataframe(df, baseline_mmhg=baseline_mmhg)
        return feats_all.iloc[[-1]]

    def get_latest_observation(self) -> Optional[SensorObservation]:
        """Return the most recently ingested observation."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def reset(self) -> None:
        """Clear buffer and reset session counters."""
        with self._lock:
            self._buffer.clear()
            self._last_timestamp = None
            self._total_observations = 0
            for k in self._quality_issues_count:
                self._quality_issues_count[k] = 0
            logger.info(f"Session buffer '{self.session_id}' reset.")

    def get_status(self) -> Dict[str, Any]:
        """Return operational metadata and buffer health for the session."""
        with self._lock:
            buf_len = len(self._buffer)
            warmup = WarmupStatus.READY if buf_len >= self.warmup_threshold else WarmupStatus.WARMUP
            return {
                "session_id": self.session_id,
                "buffered_samples": buf_len,
                "max_buffer_size": self.max_buffer_size,
                "warmup_threshold": self.warmup_threshold,
                "warmup_status": warmup.value,
                "total_observations_received": self._total_observations,
                "last_timestamp": self._last_timestamp,
                "telemetry_quality_issues": dict(self._quality_issues_count),
            }


class StreamSessionManager:
    """Singleton/Registry managing multi-device, multi-session streaming telemetry buffers."""

    def __init__(self, default_max_buffer_size: int = 120, default_warmup_threshold: int = 15):
        self._sessions: Dict[str, SensorStreamBuffer] = {}
        self._lock = threading.Lock()
        self.default_max_buffer_size = default_max_buffer_size
        self.default_warmup_threshold = default_warmup_threshold

    def get_or_create_session(
        self,
        session_id: str,
        max_buffer_size: Optional[int] = None,
        warmup_threshold: Optional[int] = None,
    ) -> SensorStreamBuffer:
        """Retrieve existing stream buffer or instantiate a new session."""
        with self._lock:
            if session_id not in self._sessions:
                buf_size = max_buffer_size or self.default_max_buffer_size
                w_thresh = warmup_threshold or self.default_warmup_threshold
                self._sessions[session_id] = SensorStreamBuffer(
                    session_id=session_id,
                    max_buffer_size=buf_size,
                    warmup_threshold=w_thresh,
                )
                logger.info(f"Created new streaming session buffer: '{session_id}'")
            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SensorStreamBuffer]:
        """Retrieve session buffer if exists, else None."""
        with self._lock:
            return self._sessions.get(session_id)

    def reset_session(self, session_id: str) -> bool:
        """Reset specific session buffer."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].reset()
                return True
            return False

    def remove_session(self, session_id: str) -> bool:
        """Remove session entirely."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        """List active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

