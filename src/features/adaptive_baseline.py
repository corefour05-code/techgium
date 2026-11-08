"""Adaptive Baseline Estimation Module for Infusion Pump Pressure Transducers.

Addresses the fixed-baseline weakness identified in Phase 4.1:
Previous architectures assumed a static 40 mmHg catheter backpressure (P_dt = 40 + R*Q),
which generated false positive drift and noise alarms when catheters operated at 15 mmHg (CVP)
or 65 mmHg (neonatal/high-resistance lines).

Medical Device AI Safety Constraints:
1. Baseline estimation is permitted ONLY during a defined pre-infusion tare period
   or when verified fluidic stability criteria are met.
2. Baseline estimation is REJECTED if:
   - Pressure is unstable (high variance or steep derivative).
   - Active high-flow infusion is underway (hydrodynamic backpressure is not hydrostatic baseline).
   - Transducer hardware faults are suspected (freeze or dropout).
   - An occlusion or severe pressure surge is developing.
3. NEVER continuously auto-zero an active pressure signal:
   Continuous auto-zeroing would mathematically erase a genuine downstream occlusion!
4. Raw sensor measurements are strictly preserved; the estimated baseline is stored as a separate channel.
5. Estimating an offset does NOT prove true physical sensor accuracy without an independent reference standard.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class BaselineQualityStatus(str, Enum):
    """Quality classification of the pressure baseline estimate."""
    VALIDATED = "VALIDATED"                        # Explicitly verified by clinical protocol/reference
    ESTIMATED_STABLE = "ESTIMATED_STABLE"          # Pre-infusion stable tare window
    REJECTED_UNSTABLE = "REJECTED_UNSTABLE"        # Pressure variance too high for reliable baseline
    REJECTED_ACTIVE_INFUSION = "REJECTED_ACTIVE_INFUSION" # Pumping active; cannot isolate hydrostatic baseline
    REJECTED_TRANSDUCER_FAULT = "REJECTED_TRANSDUCER_FAULT" # Sensor freeze or dropout detected
    REJECTED_OCCLUSION_RISK = "REJECTED_OCCLUSION_RISK"     # Pressure surging; auto-zero strictly blocked
    UNINITIALIZED = "UNINITIALIZED"                # Default state before any estimation window


@dataclass
class BaselineEstimate:
    """Immutable snapshot of the estimated catheter baseline state."""
    baseline_pressure_mmhg: float
    confidence: float
    quality_status: BaselineQualityStatus
    sample_count: int
    stability_std_mmhg: float
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_pressure_mmhg": round(self.baseline_pressure_mmhg, 2),
            "confidence": round(self.confidence, 3),
            "quality_status": self.quality_status.value,
            "sample_count": self.sample_count,
            "stability_std_mmhg": round(self.stability_std_mmhg, 3),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class AdaptiveBaselineEstimator:
    """Configurable, safety-gated baseline estimation engine."""

    def __init__(
        self,
        nominal_fallback_baseline_mmhg: float = 40.0,
        max_acceptable_stability_std_mmhg: float = 1.5,
        max_acceptable_slope_mmhg_s: float = 0.05,
        active_flow_threshold_ml_h: float = 2.0,
        occlusion_risk_threshold_mmhg: float = 180.0,
        min_tare_window_samples: int = 15,
    ):
        self.nominal_fallback = nominal_fallback_baseline_mmhg
        self.max_std = max_acceptable_stability_std_mmhg
        self.max_slope = max_acceptable_slope_mmhg_s
        self.active_flow_thresh = active_flow_threshold_ml_h
        self.occlusion_risk_thresh = occlusion_risk_threshold_mmhg
        self.min_samples = min_tare_window_samples

        # Internal state
        self._current_baseline = self.nominal_fallback
        self._confidence = 0.50
        self._status = BaselineQualityStatus.UNINITIALIZED
        self._stability_std = 0.0
        self._sample_count = 0
        self._is_locked = False

    @property
    def current_baseline_mmhg(self) -> float:
        return self._current_baseline

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def status(self) -> BaselineQualityStatus:
        return self._status

    @property
    def is_valid(self) -> bool:
        return self._status in (BaselineQualityStatus.VALIDATED, BaselineQualityStatus.ESTIMATED_STABLE)

    def set_validated_baseline(
        self,
        baseline_mmhg: float,
        confidence: float = 1.0,
    ) -> BaselineEstimate:
        """Lock an explicitly validated reference baseline (e.g. from pre-infusion arterial zeroing)."""
        if baseline_mmhg <= 0.0 or baseline_mmhg > 150.0:
            raise ValueError(f"Clinically implausible baseline pressure: {baseline_mmhg} mmHg")

        self._current_baseline = float(baseline_mmhg)
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._status = BaselineQualityStatus.VALIDATED
        self._stability_std = 0.0
        self._sample_count = 999
        self._is_locked = True

        logger.info(f"Validated baseline locked at {self._current_baseline:.2f} mmHg (confidence={self._confidence:.2f})")
        return self.get_estimate()

    def estimate_from_pre_infusion(
        self,
        pressure_window: Union[List[float], np.ndarray, pd.Series],
        flow_window: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        force_window: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> BaselineEstimate:
        """Estimate baseline from a pre-infusion priming/tare window.

        Safety Rules:
        - Rejects if pressure variance > max_std.
        - Rejects if flow > active_flow_thresh.
        - Rejects if pressure > occlusion_risk_thresh.
        - Rejects if sensor freeze (variance < 1e-5) or dropout (reading <= 0.1).
        """
        p_arr = np.asarray(pressure_window, dtype=np.float64)
        if len(p_arr) < self.min_samples:
            return BaselineEstimate(
                baseline_pressure_mmhg=self._current_baseline,
                confidence=0.30,
                quality_status=BaselineQualityStatus.REJECTED_UNSTABLE,
                sample_count=len(p_arr),
                stability_std_mmhg=float(np.std(p_arr)) if len(p_arr) > 1 else 99.0,
                is_valid=False,
                rejection_reason=f"Insufficient sample count: {len(p_arr)} < {self.min_samples}",
            )

        mean_p = float(np.mean(p_arr))
        std_p = float(np.std(p_arr))
        p_diff = np.diff(p_arr)
        slope_p = float(np.abs(np.mean(p_diff))) if len(p_diff) > 0 else 0.0

        # Safety Check 1: Transducer Faults (Freeze or Dropout)
        if mean_p <= 0.5:
            return BaselineEstimate(
                baseline_pressure_mmhg=self.nominal_fallback,
                confidence=0.0,
                quality_status=BaselineQualityStatus.REJECTED_TRANSDUCER_FAULT,
                sample_count=len(p_arr),
                stability_std_mmhg=std_p,
                is_valid=False,
                rejection_reason="Sensor dropout detected (pressure near 0.0 mmHg)",
            )

        if std_p < 1e-4 and len(p_arr) >= 10:
            return BaselineEstimate(
                baseline_pressure_mmhg=self.nominal_fallback,
                confidence=0.0,
                quality_status=BaselineQualityStatus.REJECTED_TRANSDUCER_FAULT,
                sample_count=len(p_arr),
                stability_std_mmhg=std_p,
                is_valid=False,
                rejection_reason="Sensor freeze detected (zero variance in tare window)",
            )

        # Safety Check 2: Occlusion or Severe Backpressure Surge
        if mean_p >= self.occlusion_risk_thresh:
            return BaselineEstimate(
                baseline_pressure_mmhg=self.nominal_fallback,
                confidence=0.10,
                quality_status=BaselineQualityStatus.REJECTED_OCCLUSION_RISK,
                sample_count=len(p_arr),
                stability_std_mmhg=std_p,
                is_valid=False,
                rejection_reason=f"Pressure {mean_p:.1f} mmHg exceeds occlusion risk limit ({self.occlusion_risk_thresh} mmHg). Auto-zeroing blocked.",
            )

        # Safety Check 3: Active Flow Infusion
        if flow_window is not None:
            q_arr = np.asarray(flow_window, dtype=np.float64)
            if len(q_arr) > 0 and np.mean(q_arr) > self.active_flow_thresh:
                return BaselineEstimate(
                    baseline_pressure_mmhg=self.nominal_fallback,
                    confidence=0.20,
                    quality_status=BaselineQualityStatus.REJECTED_ACTIVE_INFUSION,
                    sample_count=len(p_arr),
                    stability_std_mmhg=std_p,
                    is_valid=False,
                    rejection_reason=f"Active delivery flow ({np.mean(q_arr):.1f} mL/h) exceeds tare threshold. Hydrodynamic pressure is not baseline.",
                )

        # Safety Check 4: Stability & Variance Criteria
        if std_p > self.max_std or slope_p > self.max_slope:
            return BaselineEstimate(
                baseline_pressure_mmhg=self.nominal_fallback,
                confidence=0.25,
                quality_status=BaselineQualityStatus.REJECTED_UNSTABLE,
                sample_count=len(p_arr),
                stability_std_mmhg=std_p,
                is_valid=False,
                rejection_reason=f"Pressure unstable: std={std_p:.2f} mmHg (limit {self.max_std}), slope={slope_p:.3f} mmHg/s",
            )

        # All criteria satisfied: Accept and lock baseline
        self._current_baseline = round(mean_p, 2)
        # Confidence scales inversely with noise variance and directly with sample size
        conf = float(np.clip((1.0 - (std_p / (self.max_std * 1.5))) * min(1.0, len(p_arr) / 30.0), 0.60, 0.98))
        self._confidence = round(conf, 3)
        self._status = BaselineQualityStatus.ESTIMATED_STABLE
        self._stability_std = round(std_p, 3)
        self._sample_count = len(p_arr)

        logger.info(f"Stable pre-infusion baseline estimated: {self._current_baseline} mmHg (std={self._stability_std}, conf={self._confidence})")
        return self.get_estimate()

    def get_estimate(self) -> BaselineEstimate:
        """Return current snapshot of baseline state."""
        return BaselineEstimate(
            baseline_pressure_mmhg=self._current_baseline,
            confidence=self._confidence,
            quality_status=self._status,
            sample_count=self._sample_count,
            stability_std_mmhg=self._stability_std,
            is_valid=self.is_valid,
        )

    def compute_baseline_features(
        self,
        p_series: pd.Series,
        commanded_rate_series: Optional[pd.Series] = None,
        line_hydraulic_resistance: float = 0.8,
    ) -> pd.DataFrame:
        """Compute robust baseline-relative pressure features causally across time-series.

        Returns DataFrame with:
        - p_baseline_applied: The active baseline value used
        - p_dev_from_baseline: P_raw - P_baseline (pure pressure rise)
        - p_expected_dynamic: P_baseline + (R_line * Q_cmd)
        - p_dynamic_residual: P_raw - p_expected_dynamic (deviations from hydrodynamic expectation)
        - baseline_confidence: Confidence score of the active baseline
        - baseline_is_valid: Binary flag (1.0 = valid, 0.0 = fallback default)
        """
        n = len(p_series)
        b_val = self._current_baseline
        b_conf = self._confidence
        b_valid = 1.0 if self.is_valid else 0.0

        if commanded_rate_series is not None:
            q_cmd = pd.to_numeric(commanded_rate_series, errors="coerce").fillna(50.0)
        else:
            q_cmd = pd.Series(50.0, index=p_series.index)

        p_dev = p_series - b_val
        p_expected = b_val + (line_hydraulic_resistance * q_cmd)
        p_dyn_res = p_series - p_expected

        feats = pd.DataFrame(index=p_series.index)
        feats["p_baseline_applied"] = b_val
        feats["p_dev_from_baseline"] = p_dev
        feats["p_expected_dynamic"] = p_expected
        feats["p_dynamic_residual"] = p_dyn_res
        feats["baseline_confidence"] = b_conf
        feats["baseline_is_valid"] = b_valid

        return feats

