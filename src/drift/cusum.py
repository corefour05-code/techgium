"""Cumulative Sum (CUSUM) Control Chart for Persistent Shift and Drift Detection.

Implements tabular two-sided CUSUM:
- Tracks cumulative positive (S+) and negative (S-) deviations from in-control mean
- Baseline calibration tuning decision interval H based on target False Alarm Rate (FAR)
- Reset-on-alarm capability to prevent single excursion run inflation
- Sustained drift confirmation rule (requiring M consecutive alarms) to distinguish noise spikes from persistent drift
- Explicit reporting of both per-sample alarm rate and event-level false alarms
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class CUSUMResult:
    """Detailed output from CUSUM drift detection monitoring."""
    is_drift_detected: bool
    drift_direction: Optional[str]  # "POSITIVE", "NEGATIVE", or None
    first_detection_index: Optional[int]
    detection_delay_steps: Optional[int]
    total_alarm_steps: int
    distinct_alarm_events: int
    false_alarm_rate: float            # Per-sample pre-fault alarm rate
    false_alarm_events_count: int      # Count of distinct pre-fault false alarm excursions
    detection_rate: float              # Post-fault detection rate
    s_positive: np.ndarray
    s_negative: np.ndarray
    decision_threshold: float
    reference_slack_k: float
    in_control_mean: float
    in_control_std: float
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
            "decision_threshold": round(self.decision_threshold, 4),
            "reference_slack_k": round(self.reference_slack_k, 4),
            "in_control_mean": round(self.in_control_mean, 4),
            "in_control_std": round(self.in_control_std, 4),
        }


class CUSUMDetector:
    """Tabular Two-Sided CUSUM Detector with Event-Level False Alarm Control."""

    def __init__(
        self,
        k_factor: float = 0.5,
        h_factor: Optional[float] = None,
        consecutive_alarms_to_confirm: int = 3,
        reset_on_alarm: bool = True,
    ):
        """
        Args:
            k_factor: Slack/reference factor in terms of sigma (delta / 2).
                      k=0.5 is optimal for detecting a 1.0-sigma shift.
            h_factor: Decision threshold multiplier in terms of sigma (H = h * sigma).
            consecutive_alarms_to_confirm: Number of consecutive steps S > H required
                                           to confirm a persistent drift vs transient noise spike.
            reset_on_alarm: Reset cumulative sum to 0 after an alarm event to enable independent monitoring.
        """
        self.k_factor = k_factor
        self.h_factor = h_factor or 4.5
        self.consecutive_alarms_to_confirm = max(1, consecutive_alarms_to_confirm)
        self.reset_on_alarm = reset_on_alarm

        self.mu_0_: Optional[float] = None
        self.sigma_0_: Optional[float] = None
        self.k_val_: Optional[float] = None
        self.h_val_: Optional[float] = None
        self.is_calibrated_: bool = False

    def calibrate_baseline(
        self,
        baseline_series: Union[pd.Series, np.ndarray],
        target_false_alarm_rate: float = 0.005,
    ) -> "CUSUMDetector":
        """Calibrate in-control mean, std, and decision threshold H from baseline data."""
        arr = np.asarray(baseline_series, dtype=np.float64)
        clean = arr[~np.isnan(arr)]
        if len(clean) < 30:
            raise ValueError("Baseline dataset requires at least 30 valid points for calibration.")

        self.mu_0_ = float(np.median(clean))
        mad = np.median(np.abs(clean - self.mu_0_))
        self.sigma_0_ = float(1.4826 * mad) if mad > 1e-6 else float(np.std(clean))
        if self.sigma_0_ < 1e-6:
            self.sigma_0_ = 1.0

        self.k_val_ = self.k_factor * self.sigma_0_

        # Simulate CUSUM with reset_on_alarm on clean baseline to choose H
        # Target: False alarm event rate <= target_false_alarm_rate
        # For standard CUSUM with k=0.5, h=4.5 to 5.0 provides in-control ARL > 500-1000 steps
        candidate_h_factors = np.linspace(3.5, 6.5, 31)
        best_h_factor = 5.0

        for h_cand in candidate_h_factors:
            h_cand_val = h_cand * self.sigma_0_
            s_pos = 0.0
            s_neg = 0.0
            alarms = 0
            for x in clean:
                s_pos = max(0.0, s_pos + (x - self.mu_0_) - self.k_val_)
                s_neg = max(0.0, s_neg - (x - self.mu_0_) - self.k_val_)
                if s_pos > h_cand_val:
                    alarms += 1
                    s_pos = 0.0 if self.reset_on_alarm else s_pos
                elif s_neg > h_cand_val:
                    alarms += 1
                    s_neg = 0.0 if self.reset_on_alarm else s_neg

            far_cand = alarms / len(clean)
            if far_cand <= target_false_alarm_rate:
                best_h_factor = float(h_cand)
                break
            best_h_factor = float(h_cand)

        self.h_factor = best_h_factor
        self.h_val_ = self.h_factor * self.sigma_0_
        self.is_calibrated_ = True

        logger.info(
            f"CUSUM Calibrated: mu_0={self.mu_0_:.4f}, sigma_0={self.sigma_0_:.4f}, "
            f"K={self.k_val_:.4f}, H={self.h_val_:.4f} (h={self.h_factor:.2f})"
        )
        return self

    def monitor(
        self,
        series: Union[pd.Series, np.ndarray],
        fault_start_idx: Optional[int] = None,
    ) -> CUSUMResult:
        """Run CUSUM monitoring on sensor time-series."""
        if not self.is_calibrated_:
            arr_temp = np.asarray(series)
            split_idx = max(50, int(len(arr_temp) * 0.2))
            self.calibrate_baseline(arr_temp[:split_idx])

        arr = np.asarray(series, dtype=np.float64)
        n = len(arr)

        s_pos = np.zeros(n, dtype=np.float64)
        s_neg = np.zeros(n, dtype=np.float64)
        raw_alarm_mask = np.zeros(n, dtype=bool)

        curr_pos = 0.0
        curr_neg = 0.0

        for t in range(n):
            val = arr[t]
            if np.isnan(val):
                val = self.mu_0_

            curr_pos = max(0.0, curr_pos + (val - self.mu_0_) - self.k_val_)
            curr_neg = max(0.0, curr_neg - (val - self.mu_0_) - self.k_val_)

            s_pos[t] = curr_pos
            s_neg[t] = curr_neg

            alarm_pos = curr_pos > self.h_val_
            alarm_neg = curr_neg > self.h_val_

            if alarm_pos or alarm_neg:
                raw_alarm_mask[t] = True
                if self.reset_on_alarm:
                    if alarm_pos:
                        curr_pos = 0.0
                    if alarm_neg:
                        curr_neg = 0.0

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

        # Count distinct alarm excursion events
        diff = np.diff(raw_alarm_mask.astype(int))
        distinct_events = int(np.sum(diff == 1)) + (1 if raw_alarm_mask[0] else 0)

        # Direction analysis based on cumulative sums
        pos_alarms = np.sum(s_pos > self.h_val_)
        neg_alarms = np.sum(s_neg > self.h_val_)
        if pos_alarms > neg_alarms:
            direction = "POSITIVE"
        elif neg_alarms > pos_alarms:
            direction = "NEGATIVE"
        else:
            direction = None

        first_det = None
        delay = None
        pre_fault_alarms = 0
        pre_fault_events = 0
        far = 0.0
        det_rate = 0.0
        total_alarm_steps = int(np.sum(confirmed_alarm_mask))
        is_detected = total_alarm_steps > 0

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

        return CUSUMResult(
            is_drift_detected=is_detected,
            drift_direction=direction,
            first_detection_index=first_det,
            detection_delay_steps=delay,
            total_alarm_steps=total_alarm_steps,
            distinct_alarm_events=distinct_events,
            false_alarm_rate=far,
            false_alarm_events_count=pre_fault_events,
            detection_rate=det_rate,
            s_positive=s_pos,
            s_negative=s_neg,
            decision_threshold=self.h_val_,
            reference_slack_k=self.k_val_,
            in_control_mean=self.mu_0_,
            in_control_std=self.sigma_0_,
            is_currently_alarming=bool(confirmed_alarm_mask[-1]) if n > 0 else False,
            alarm_mask=confirmed_alarm_mask,
        )
