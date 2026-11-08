"""Exponentially Weighted Moving Average (EWMA) Drift Detection Engine.

Implements mathematically rigorous EWMA control chart for detecting gradual sensor drift:
- Time-varying and asymptotic variance bounds
- Empirical or analytical calibration using in-control baseline data
- Avoids arbitrary threshold selection by tuning L to achieve target false-alarm rates
- Distinguishes isolated noise spikes from persistent drift via confirmation rules
- Identifies drift direction (positive vs negative) and true detection delay
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class EWMAResult:
    """Detailed output from EWMA drift detection monitoring."""
    is_drift_detected: bool
    drift_direction: Optional[str]  # "POSITIVE", "NEGATIVE", or None
    first_detection_index: Optional[int]
    detection_delay_steps: Optional[int]
    total_alarm_steps: int
    distinct_alarm_events: int
    false_alarm_rate: float
    false_alarm_events_count: int
    detection_rate: float
    smoothed_statistic: np.ndarray
    upper_control_limit: np.ndarray
    lower_control_limit: np.ndarray
    in_control_mean: float
    in_control_std: float
    lambda_param: float
    l_param: float
    is_currently_alarming: bool = False
    alarm_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_drift_detected": self.is_drift_detected,
            "drift_direction": self.drift_direction,
            "first_detection_index": self.first_detection_index,
            "detection_delay_steps": self.detection_delay_steps,
            "total_alarm_steps": self.total_alarm_steps,
            "distinct_alarm_events": self.distinct_alarm_events,
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "false_alarm_events_count": self.false_alarm_events_count,
            "detection_rate": round(self.detection_rate, 4),
            "in_control_mean": round(self.in_control_mean, 4),
            "in_control_std": round(self.in_control_std, 4),
            "lambda_param": self.lambda_param,
            "l_param": round(self.l_param, 4),
        }


class EWMADetector:
    """EWMA Control Chart for gradual sensor drift detection."""

    def __init__(
        self,
        lambda_param: float = 0.15,
        l_param: Optional[float] = None,
        use_time_varying_limits: bool = True,
        consecutive_alarms_to_confirm: int = 3,
    ):
        """
        Args:
            lambda_param: Smoothing constant in (0, 1].
            l_param: Width factor of control limits in terms of sigma.
            use_time_varying_limits: If True, uses exact time-varying variance.
            consecutive_alarms_to_confirm: Number of consecutive alarm steps required
                                           to confirm persistent drift vs transient noise spike.
        """
        if not (0.0 < lambda_param <= 1.0):
            raise ValueError(f"lambda_param must be in (0, 1], got {lambda_param}")
        self.lambda_param = lambda_param
        self.l_param = l_param or 3.0
        self.use_time_varying_limits = use_time_varying_limits
        self.consecutive_alarms_to_confirm = max(1, consecutive_alarms_to_confirm)

        self.mu_0_: Optional[float] = None
        self.sigma_0_: Optional[float] = None
        self.is_calibrated_: bool = False

    def calibrate_baseline(
        self,
        baseline_series: Union[pd.Series, np.ndarray],
        target_false_alarm_rate: float = 0.005,
    ) -> "EWMADetector":
        """Calibrate in-control mean, std, and L factor from verified baseline data."""
        arr = np.asarray(baseline_series, dtype=np.float64)
        clean = arr[~np.isnan(arr)]
        if len(clean) < 30:
            raise ValueError("Baseline dataset requires at least 30 valid points for calibration.")

        self.mu_0_ = float(np.median(clean))
        mad = np.median(np.abs(clean - self.mu_0_))
        self.sigma_0_ = float(1.4826 * mad) if mad > 1e-6 else float(np.std(clean))
        if self.sigma_0_ < 1e-6:
            self.sigma_0_ = 1.0

        # Simulate EWMA on baseline to tune L empirically
        z = np.zeros(len(clean), dtype=np.float64)
        z[0] = self.mu_0_
        for t in range(1, len(clean)):
            z[t] = self.lambda_param * clean[t] + (1.0 - self.lambda_param) * z[t - 1]

        t_idx = np.arange(1, len(clean) + 1, dtype=np.float64)
        if self.use_time_varying_limits:
            sigma_z = self.sigma_0_ * np.sqrt(
                (self.lambda_param / (2.0 - self.lambda_param)) * (1.0 - (1.0 - self.lambda_param) ** (2.0 * t_idx))
            )
        else:
            sigma_z = np.full(len(clean), self.sigma_0_ * np.sqrt(self.lambda_param / (2.0 - self.lambda_param)))

        std_residuals = np.abs(z - self.mu_0_) / np.maximum(sigma_z, 1e-6)

        # Select L as (1 - target_false_alarm_rate) quantile (e.g. 99.5th percentile for FAR <= 0.5%)
        empirical_l = float(np.quantile(std_residuals, 1.0 - target_false_alarm_rate))
        self.l_param = float(np.clip(empirical_l, 2.5, 4.5))

        self.is_calibrated_ = True
        logger.info(
            f"EWMA Calibrated: mu_0={self.mu_0_:.4f}, sigma_0={self.sigma_0_:.4f}, "
            f"lambda={self.lambda_param}, L={self.l_param:.3f} (target FAR={target_false_alarm_rate})"
        )
        return self

    def monitor(
        self,
        series: Union[pd.Series, np.ndarray],
        fault_start_idx: Optional[int] = None,
    ) -> EWMAResult:
        """Run EWMA drift detection over time-series."""
        if not self.is_calibrated_:
            arr_temp = np.asarray(series)
            split_idx = max(50, int(len(arr_temp) * 0.2))
            self.calibrate_baseline(arr_temp[:split_idx])

        arr = np.asarray(series, dtype=np.float64)
        n = len(arr)

        z = np.zeros(n, dtype=np.float64)
        ucl = np.zeros(n, dtype=np.float64)
        lcl = np.zeros(n, dtype=np.float64)

        z[0] = arr[0] if not np.isnan(arr[0]) else self.mu_0_

        for t in range(n):
            val = arr[t]
            if np.isnan(val):
                val = z[t - 1] if t > 0 else self.mu_0_

            if t == 0:
                z[t] = self.lambda_param * val + (1.0 - self.lambda_param) * self.mu_0_
            else:
                z[t] = self.lambda_param * val + (1.0 - self.lambda_param) * z[t - 1]

            step = t + 1
            if self.use_time_varying_limits:
                sigma_zt = self.sigma_0_ * np.sqrt(
                    (self.lambda_param / (2.0 - self.lambda_param))
                    * (1.0 - (1.0 - self.lambda_param) ** (2.0 * step))
                )
            else:
                sigma_zt = self.sigma_0_ * np.sqrt(self.lambda_param / (2.0 - self.lambda_param))

            ucl[t] = self.mu_0_ + (self.l_param * sigma_zt)
            lcl[t] = self.mu_0_ - (self.l_param * sigma_zt)

        pos_alarms_mask = z > ucl
        neg_alarms_mask = z < lcl
        raw_alarm_mask = pos_alarms_mask | neg_alarms_mask

        # Apply persistence / confirmation rule (M consecutive steps)
        confirmed_alarm_mask = np.zeros(n, dtype=bool)
        consec = 0
        for t in range(n):
            if raw_alarm_mask[t]:
                consec += 1
                if consec >= self.consecutive_alarms_to_confirm:
                    confirmed_alarm_mask[t] = True
            else:
                consec = 0

        # Count distinct alarm excursions
        diff = np.diff(raw_alarm_mask.astype(int))
        distinct_events = int(np.sum(diff == 1)) + (1 if raw_alarm_mask[0] else 0)

        # Dominant drift direction
        pos_count = int(np.sum(pos_alarms_mask))
        neg_count = int(np.sum(neg_alarms_mask))
        if pos_count > neg_count:
            direction = "POSITIVE"
        elif neg_count > pos_count:
            direction = "NEGATIVE"
        else:
            direction = None

        total_alarm_steps = int(np.sum(confirmed_alarm_mask))
        is_detected = total_alarm_steps > 0

        first_det = None
        delay = None
        pre_fault_alarms = 0
        pre_fault_events = 0
        far = 0.0
        det_rate = 0.0

        if fault_start_idx is not None:
            pre_mask = confirmed_alarm_mask[:fault_start_idx]
            pre_fault_alarms = int(np.sum(pre_mask))
            if len(pre_mask) > 1:
                p_diff = np.diff(pre_mask.astype(int))
                pre_fault_events = int(np.sum(p_diff == 1)) + (1 if pre_mask[0] else 0)
            far = float(pre_fault_alarms / max(1, fault_start_idx))

            post_fault_steps = n - fault_start_idx
            post_mask = confirmed_alarm_mask[fault_start_idx:]
            post_fault_alarms = int(np.sum(post_mask))
            det_rate = float(post_fault_alarms / max(1, post_fault_steps))

            post_fault_detections = np.where(post_mask)[0]
            if len(post_fault_detections) > 0:
                first_det = int(fault_start_idx + post_fault_detections[0])
                delay = int(post_fault_detections[0])
            else:
                first_det = None
                delay = None
            is_detected = (post_fault_alarms > 0)
        else:
            all_dets = np.where(confirmed_alarm_mask)[0]
            first_det = int(all_dets[0]) if len(all_dets) > 0 else None
            is_detected = len(all_dets) > 0
            far = float(total_alarm_steps / max(1, n)) if not is_detected else 0.0
            det_rate = 1.0 if is_detected else 0.0

        return EWMAResult(
            is_drift_detected=is_detected,
            drift_direction=direction,
            first_detection_index=first_det,
            detection_delay_steps=delay,
            total_alarm_steps=total_alarm_steps,
            distinct_alarm_events=distinct_events,
            false_alarm_rate=far,
            false_alarm_events_count=pre_fault_events,
            detection_rate=det_rate,
            smoothed_statistic=z,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            in_control_mean=self.mu_0_,
            in_control_std=self.sigma_0_,
            lambda_param=self.lambda_param,
            l_param=self.l_param,
            is_currently_alarming=bool(confirmed_alarm_mask[-1]) if n > 0 else False,
            alarm_mask=confirmed_alarm_mask,
        )
