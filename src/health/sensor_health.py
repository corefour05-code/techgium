"""Interpretable Evidence-Based Sensor Health & Process Risk Engine.

Separates:
1. Sensor Health Score (0-100): Reflects physical and electrical transducer integrity.
2. Device/Process Risk Score (0-100): Reflects the operational safety and delivery status of the infusion pathway.

Critical Medical AI Rule:
- A genuine downstream line occlusion does NOT reduce sensor health because the sensor
  is operating correctly by accurately measuring the elevated hydraulic backpressure.
  Instead, occlusion dramatically escalates the Process Risk Score (to CRITICAL).
- Isolation Forest severity is strictly an Anomaly Severity Score (distance from baseline envelope),
  NOT a probability of failure.
- An ML model CANNOT assert true accuracy without independent reference measurements.
- Recommendations are advisory; NO automated device calibration is permitted.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from src.utils.config import SensorHealthStatus
from src.sensor_fusion.consistency_engine import ConsistencyVerdict, ConsistencyAssessment
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class HealthAssessmentReport:
    """Consolidated interpretable sensor health and process risk report."""
    sensor_id: str
    sensor_type: str
    status: SensorHealthStatus
    sensor_health_score: float        # 0.0 to 100.0 (transducer integrity)
    health_score: float               # Alias for backward compatibility (= sensor_health_score)
    process_risk_score: float         # 0.0 to 100.0 (clinical/delivery risk: 0=nominal, 100=critical)
    process_risk_level: str           # "LOW", "MODERATE", "HIGH", "CRITICAL"
    model_confidence: float           # 0.0 to 1.0
    anomaly_detected: bool
    anomaly_severity_score: float     # In [0, 1] - NOT a probability of failure
    drift_detected: bool
    drift_direction: Optional[str]
    reference_available: bool
    estimated_measurement_error: Optional[float]
    recommended_action: str
    evidence_breakdown: List[str] = field(default_factory=list)
    deductions: Dict[str, float] = field(default_factory=dict)
    process_risk_factors: Dict[str, float] = field(default_factory=dict)
    disclaimer: str = (
        "Medical Device AI Safety Notice: Recommendations are advisory only. "
        "No automated in-line device calibration is permitted without validated biomedical engineering sign-off. "
        "True measurement accuracy cannot be confirmed without an independent reference measurement."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "status": self.status.value,
            "sensor_health_score": round(self.sensor_health_score, 1),
            "health_score": round(self.health_score, 1),
            "process_risk_score": round(self.process_risk_score, 1),
            "process_risk_level": self.process_risk_level,
            "model_confidence": round(self.model_confidence, 2),
            "anomaly_detected": self.anomaly_detected,
            "anomaly_severity_score": round(self.anomaly_severity_score, 3),
            "drift_detected": self.drift_detected,
            "drift_direction": self.drift_direction,
            "reference_available": self.reference_available,
            "estimated_measurement_error": self.estimated_measurement_error,
            "recommended_action": self.recommended_action,
            "evidence_breakdown": self.evidence_breakdown,
            "deductions": self.deductions,
            "process_risk_factors": self.process_risk_factors,
            "disclaimer": self.disclaimer,
        }


class SensorHealthEngine:
    """Evaluates multi-modal evidence to produce decoupled sensor health and process risk metrics."""

    def evaluate_health(
        self,
        sensor_id: str,
        sensor_type: str,
        consistency_assessment: ConsistencyAssessment,
        anomaly_detected: bool = False,
        anomaly_severity: float = 0.0,
        ewma_drift_detected: bool = False,
        ewma_drift_direction: Optional[str] = None,
        cusum_drift_detected: bool = False,
        cusum_drift_direction: Optional[str] = None,
        independent_reference_value: Optional[float] = None,
        observed_sensor_value: Optional[float] = None,
        has_missing_data: bool = False,
        classifier_fault: Optional[str] = None,
    ) -> HealthAssessmentReport:
        """Compute decoupled sensor health score and process risk score.

        Separation Rules:
        - Sensor Health: 100.0 baseline, reduced by transducer defects (dropout, freeze, drift, reference error).
          A genuine occlusion does NOT reduce sensor health because the sensor is correctly reading high backpressure.
        - Process Risk: 0.0 baseline (nominal infusion). Escalated by clinical pathway faults (occlusion: 95,
          telemetry loss: 90, drift: 50).
        - Isolation Forest score is strictly treated as an anomaly severity metric, not a failure probability.
        """
        sensor_score = 100.0
        process_risk = 5.0
        deductions: Dict[str, float] = {}
        process_risk_factors: Dict[str, float] = {}
        evidence: List[str] = []

        verdict = consistency_assessment.verdict
        drift_active = ewma_drift_detected or cusum_drift_detected
        drift_dir = ewma_drift_direction or cusum_drift_direction

        # 1. Independent Reference Ground Truth
        ref_available = independent_reference_value is not None and observed_sensor_value is not None
        measured_error = None
        if ref_available:
            measured_error = float(observed_sensor_value - independent_reference_value)
            evidence.append(f"Independent reference available: measured error is {measured_error:+.2f} units.")
            if abs(measured_error) > 15.0:
                ded = min(40.0, abs(measured_error) * 1.5)
                deductions["reference_error_penalty"] = round(ded, 1)
                sensor_score -= ded
                process_risk += min(30.0, abs(measured_error))
        else:
            evidence.append("No independent reference measurement available; model cannot establish true physical accuracy.")

        # 2. Transducer Integrity vs Clinical Process Evaluation
        if verdict == ConsistencyVerdict.SENSOR_DROPOUT_SUSPECTED:
            deductions["sensor_dropout"] = 80.0
            sensor_score -= 80.0
            process_risk = max(process_risk, 90.0)
            process_risk_factors["telemetry_loss"] = 90.0
            status = SensorHealthStatus.FAULT_SUSPECTED
            action = "Immediate inspection required: Sensor disconnected or dropped to rail."
            evidence.extend(consistency_assessment.evidence)

        elif verdict == ConsistencyVerdict.SENSOR_FREEZE_SUSPECTED:
            deductions["sensor_freeze"] = 75.0
            sensor_score -= 75.0
            process_risk = max(process_risk, 85.0)
            process_risk_factors["frozen_transducer"] = 85.0
            status = SensorHealthStatus.FAULT_SUSPECTED
            action = "Sensor transducer locked: Cycle device power and inspect ADC bus."
            evidence.extend(consistency_assessment.evidence)

        elif verdict == ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED:
            # SENSOR IS FULLY FUNCTIONAL: No sensor health deduction!
            deductions["process_occlusion"] = 0.0  # Recorded for audit trail without penalizing transducer
            process_risk = 95.0
            process_risk_factors["fluid_pathway_occlusion"] = 95.0
            status = SensorHealthStatus.WARNING
            action = "CLINICAL ALERT: Downstream line occlusion confirmed. Inspect catheter/tubing immediately."
            evidence.append("Sensor transducer is fully functional and accurately detecting fluid pathway blockage.")
            evidence.extend(consistency_assessment.evidence)

        elif verdict == ConsistencyVerdict.INSUFFICIENT_EVIDENCE:
            status = SensorHealthStatus.INSUFFICIENT_DATA
            process_risk = max(process_risk, 40.0)
            process_risk_factors["unverified_single_channel"] = 40.0
            action = "Insufficient multi-sensor evidence: Cannot cross-validate transducer without secondary channel."
            evidence.append("Single-channel configuration lacks redundant sensor validation.")

        elif verdict == ConsistencyVerdict.CONFLICTING_EVIDENCE:
            status = SensorHealthStatus.WARNING
            process_risk = max(process_risk, 55.0)
            process_risk_factors["ambiguous_sensor_signals"] = 55.0
            action = "Conflicting multi-sensor evidence: Sensor relationships ambiguous. Biomedical engineering review advised."
            evidence.extend(consistency_assessment.evidence)

        elif verdict == ConsistencyVerdict.TEMPERATURE_INFLUENCE_SUSPECTED:
            deductions["thermal_influence"] = 25.0
            sensor_score -= 25.0
            process_risk = max(process_risk, 35.0)
            process_risk_factors["thermal_divergence"] = 35.0
            status = SensorHealthStatus.WARNING
            action = "Temperature-induced variance detected: Verify ambient operating temperature range (15-30°C)."
            evidence.extend(consistency_assessment.evidence)

        elif verdict == ConsistencyVerdict.SENSOR_DRIFT_SUSPECTED or drift_active:
            deductions["statistical_drift"] = 45.0
            sensor_score -= 45.0
            process_risk = max(process_risk, 50.0)
            process_risk_factors["dosing_inaccuracy_risk"] = 50.0
            status = SensorHealthStatus.DRIFT_SUSPECTED
            action = "Gradual sensor drift suspected: Perform independent reference calibration check before next infusion."
            evidence.append(f"Statistical drift confirmed in {drift_dir or 'unspecified'} direction.")
            evidence.extend(consistency_assessment.evidence)

        elif has_missing_data:
            deductions["missing_data"] = 20.0
            sensor_score -= 20.0
            process_risk = max(process_risk, 30.0)
            status = SensorHealthStatus.WARNING
            action = "Missing readings detected in sensor stream: Check telemetry bus."
            evidence.append("Intermittent missing values observed.")

        else:
            status = SensorHealthStatus.HEALTHY
            action = "Continue normal operation: Sensor signals within in-control parameters."
            evidence.append("All physical and statistical checks normal.")

        # 3. Multivariate Anomaly Penalty (Treated as Anomaly Severity Score)
        if anomaly_detected:
            if_ded = min(30.0, anomaly_severity * 30.0)
            deductions["isolation_forest_anomaly"] = round(if_ded, 1)
            # If genuine occlusion, anomaly is process-related, not sensor failure
            if verdict != ConsistencyVerdict.GENUINE_OCCLUSION_CONFIRMED:
                sensor_score -= if_ded
            process_risk = max(process_risk, float(anomaly_severity * 60.0))
            evidence.append(f"Multivariate anomaly detected by Isolation Forest (anomaly severity score: {anomaly_severity:.2f}).")
            if status == SensorHealthStatus.HEALTHY:
                status = SensorHealthStatus.WARNING

        # 4. Supervised Classifier Evidence Fusion
        if classifier_fault is not None:
            if classifier_fault == "EXCESSIVE_NOISE":
                deductions["excessive_noise"] = 30.0
                sensor_score -= 30.0
                process_risk = max(process_risk, 35.0)
                evidence.append("Supervised classifier confirmed excessive transducer noise.")
                if status == SensorHealthStatus.HEALTHY:
                    status = SensorHealthStatus.WARNING
                    action = "Excessive transducer noise detected: Inspect cabling and shielding."

            elif classifier_fault in ("POSITIVE_DRIFT", "NEGATIVE_DRIFT"):
                if "statistical_drift" not in deductions:
                    deductions["classifier_drift"] = 35.0
                    sensor_score -= 35.0
                    process_risk = max(process_risk, 35.0)
                    evidence.append(f"Supervised classifier identified {classifier_fault}.")
                    if status == SensorHealthStatus.HEALTHY:
                        status = SensorHealthStatus.DRIFT_SUSPECTED
                        action = f"Gradual sensor {classifier_fault.lower()} detected by classifier: Verification advised."

            elif classifier_fault == "SUDDEN_OFFSET":
                if "sudden_offset" not in deductions:
                    deductions["sudden_offset"] = 40.0
                    sensor_score -= 40.0
                    process_risk = max(process_risk, 40.0)
                    evidence.append("Supervised classifier detected sudden sensor offset.")
                    if status == SensorHealthStatus.HEALTHY:
                        status = SensorHealthStatus.FAULT_SUSPECTED
                        action = "Sudden sensor offset detected: Recalibration or transducer replacement required."

            elif classifier_fault == "GENUINE_OCCLUSION":
                # Process anomaly: does NOT deduct sensor health!
                process_risk = max(process_risk, 90.0)
                process_risk_factors["fluid_pathway_occlusion"] = 90.0
                evidence.append("Classifier confirmed genuine downstream line occlusion.")
                if status == SensorHealthStatus.HEALTHY:
                    status = SensorHealthStatus.WARNING
                    action = "CLINICAL ALERT: Downstream line occlusion confirmed. Inspect catheter/tubing immediately."

            elif classifier_fault == "TEMPERATURE_INFLUENCE":
                if "thermal_influence" not in deductions:
                    deductions["thermal_influence"] = 15.0
                    sensor_score -= 15.0
                process_risk = max(process_risk, 20.0)
                process_risk_factors["thermal_divergence"] = 20.0
                evidence.append("Classifier detected temperature influence on transducer reading.")

        final_sensor_health = float(np.clip(sensor_score, 0.0, 100.0))
        final_process_risk = float(np.clip(process_risk, 0.0, 100.0))

        # Categorize risk level
        if final_process_risk >= 80.0:
            risk_level = "CRITICAL"
        elif final_process_risk >= 50.0:
            risk_level = "HIGH"
        elif final_process_risk >= 25.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        confidence = float(np.clip(consistency_assessment.confidence, 0.0, 1.0))

        return HealthAssessmentReport(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            status=status,
            sensor_health_score=final_sensor_health,
            health_score=final_sensor_health,
            process_risk_score=final_process_risk,
            process_risk_level=risk_level,
            model_confidence=confidence,
            anomaly_detected=anomaly_detected,
            anomaly_severity_score=float(anomaly_severity),
            drift_detected=drift_active,
            drift_direction=drift_dir,
            reference_available=ref_available,
            estimated_measurement_error=measured_error,
            recommended_action=action,
            evidence_breakdown=evidence,
            deductions=deductions,
            process_risk_factors=process_risk_factors,
        )
