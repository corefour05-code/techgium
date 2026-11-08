"""FastAPI Production REST API for Infusion Pump ML System.

Endpoints:
- GET /health
- GET /models/status
- POST /predict/fault
- POST /predict/sensor-error
- POST /predict/sensor-health
- POST /predict/batch
- POST /stream/observe
- POST /session/reset
- GET /session/status/{session_id}
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import MODELS_DIR, DataProvenance
from src.features.feature_pipeline import SensorFeaturePipeline
from src.anomaly.isolation_forest import MultivariateAnomalyDetector
from src.classification.xgboost_classifier import (
    InfusionPumpFaultClassifier,
    CLASS_NAMES,
)
from src.classification.probability_calibration import MulticlassCalibrator
from src.regression.xgboost_regressor import SensorErrorEstimator
from src.regression.conformal_regressor import ConformalSensorErrorEstimator
from src.explainability.shap_explainer import InfusionSHAPExplainer
from src.sensor_fusion.consistency_engine import (
    CrossSensorConsistencyEngine,
    ConsistencyVerdict,
)
from src.health.sensor_health import SensorHealthEngine, HealthAssessmentReport
from src.session.stream_buffer import (
    SensorObservation,
    StreamQualityStatus,
    WarmupStatus,
    SensorStreamBuffer,
    StreamSessionManager,
)
from api.schemas.payloads import (
    HealthResponse,
    ModelStatusResponse,
    FaultPredictionRequest,
    FaultPredictionResponse,
    SensorErrorRequest,
    SensorErrorResponse,
    SensorHealthRequest,
    SensorHealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    SequentialObservationRequest,
    SequentialPredictionResponse,
    SessionResetRequest,
    SessionResetResponse,
    SessionStatusResponse,
)
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Model Registry
models_state: Dict[str, Any] = {
    "iso_forest": None,
    "classifier": None,
    "regressor_pressure": None,
    "shap_explainer": None,
    "pipeline": None,
    "consistency_engine": None,
    "health_engine": None,
    "calibrator": None,
    "conformal_regressor": None,
    "session_manager": None,
}


def load_production_models():
    """Load persisted models and inference engines."""
    logger.info("Initializing ML models and feature pipelines...")
    models_state["pipeline"] = SensorFeaturePipeline()
    models_state["consistency_engine"] = CrossSensorConsistencyEngine()
    models_state["health_engine"] = SensorHealthEngine()
    models_state["session_manager"] = StreamSessionManager()

    # 1. Isolation Forest
    iso_path = MODELS_DIR / "anomaly_detection" / "isolation_forest.joblib"
    if iso_path.exists():
        try:
            models_state["iso_forest"] = MultivariateAnomalyDetector.load(iso_path)
            logger.info("Loaded Isolation Forest detector.")
        except Exception as e:
            logger.error(f"Failed loading Isolation Forest: {e}")

    # 2. XGBoost Classifier
    clf_path = MODELS_DIR / "classification" / "xgboost_classifier.joblib"
    if clf_path.exists():
        try:
            clf = InfusionPumpFaultClassifier.load(clf_path)
            models_state["classifier"] = clf
            models_state["shap_explainer"] = InfusionSHAPExplainer(clf)
            logger.info("Loaded XGBoost Classifier and SHAP Explainer.")
        except Exception as e:
            logger.error(f"Failed loading XGBoost Classifier: {e}")

    # 2b. Probability Calibrator
    cal_path = MODELS_DIR / "classification" / "probability_calibrator.joblib"
    if cal_path.exists() and models_state["classifier"] is not None:
        try:
            models_state["calibrator"] = MulticlassCalibrator.load(
                cal_path, base_classifier=models_state["classifier"]
            )
            logger.info("Loaded Multiclass Calibrator.")
        except Exception as e:
            logger.error(f"Failed loading Calibrator: {e}")

    # 3. XGBoost Regressor
    reg_path = MODELS_DIR / "regression" / "xgboost_regressor_pressure.joblib"
    if reg_path.exists():
        try:
            reg = SensorErrorEstimator.load(reg_path)
            models_state["regressor_pressure"] = reg
            logger.info("Loaded Pressure Error Regressor.")
        except Exception as e:
            logger.error(f"Failed loading Pressure Regressor: {e}")

    # 3b. Conformal Regressor
    conf_path = MODELS_DIR / "regression" / "conformal_regressor_pressure.joblib"
    if conf_path.exists() and models_state["regressor_pressure"] is not None:
        try:
            models_state["conformal_regressor"] = ConformalSensorErrorEstimator.load(
                conf_path, base_regressor=models_state["regressor_pressure"]
            )
            logger.info("Loaded Conformal Sensor Error Estimator.")
        except Exception as e:
            logger.error(f"Failed loading Conformal Regressor: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    load_production_models()
    yield
    logger.info("Shutting down ML API service.")


app = FastAPI(
    title="Infusion Pump AI & Digital Twin ML Intelligence API",
    description=(
        "Production REST API for multi-sensor drift detection, multivariate anomaly detection, "
        "occlusion disambiguation, XGBoost fault classification, SHAP explanations, "
        "and evidence-based sensor health scoring."
    ),
    version="1.0.0-phase4",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Health check endpoint indicating API and model status."""
    now_iso = datetime.now(timezone.utc).isoformat()
    loaded_dict = {
        "isolation_forest": models_state["iso_forest"] is not None,
        "xgboost_classifier": models_state["classifier"] is not None,
        "xgboost_regressor_pressure": models_state["regressor_pressure"] is not None,
        "shap_explainer": models_state["shap_explainer"] is not None,
        "probability_calibrator": models_state["calibrator"] is not None,
        "conformal_regressor_pressure": models_state["conformal_regressor"] is not None,
    }
    overall_status = "OK" if all(loaded_dict.values()) else "DEGRADED"

    return HealthResponse(
        status=overall_status,
        service="Infusion Pump ML Intelligence Engine",
        version="1.0.0-phase4",
        timestamp=now_iso,
        device_target="Infusion Pump",
        models_loaded=loaded_dict,
        safety_disclaimer=(
            "Advisory Medical AI Notice: Recommendations are advisory only. "
            "No automated device calibration is permitted without validated biomedical engineering sign-off. "
            "True sensor accuracy cannot be confirmed without independent reference measurements."
        ),
    )


@app.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Detailed metadata of all loaded machine learning models."""
    now_iso = datetime.now(timezone.utc).isoformat()
    models_info = {}

    clf: Optional[InfusionPumpFaultClassifier] = models_state["classifier"]
    if clf and clf.is_fitted_:
        models_info["xgboost_classifier"] = {
            "type": "XGBClassifier (Hist)",
            "supported_classes": CLASS_NAMES,
            "num_features": len(clf.feature_names_),
            "is_fitted": clf.is_fitted_,
        }

    reg: Optional[SensorErrorEstimator] = models_state["regressor_pressure"]
    if reg and reg.is_fitted_:
        models_info["xgboost_regressor_pressure"] = {
            "type": "XGBRegressor",
            "sensor_type": reg.sensor_type,
            "units": reg.units,
            "residual_rmse": reg.residual_rmse_,
            "is_fitted": reg.is_fitted_,
        }

    iso: Optional[MultivariateAnomalyDetector] = models_state["iso_forest"]
    if iso and iso.is_fitted_:
        models_info["isolation_forest"] = {
            "type": "IsolationForest",
            "contamination": iso.contamination,
            "score_bounds": [iso.score_min_, iso.score_max_],
            "is_fitted": iso.is_fitted_,
        }

    cal: Optional[MulticlassCalibrator] = models_state["calibrator"]
    if cal and cal.is_fitted_:
        models_info["probability_calibrator"] = {
            "type": f"MulticlassCalibrator ({cal.method})",
            "temperature": getattr(cal, "temperature_", None),
            "is_fitted": cal.is_fitted_,
        }

    conf: Optional[ConformalSensorErrorEstimator] = models_state["conformal_regressor"]
    if conf and conf.is_calibrated_:
        models_info["conformal_regressor_pressure"] = {
            "type": "ConformalSensorErrorEstimator (Split Conformal)",
            "alpha": conf.alpha,
            "conformal_quantile_mmhg": conf.conformal_quantile_,
            "calibration_samples": conf.calibration_sample_count_,
            "is_calibrated": conf.is_calibrated_,
        }

    return ModelStatusResponse(
        timestamp=now_iso,
        provenance=DataProvenance.SYNTHETIC_DATA.value,
        models=models_info,
    )


@app.post("/predict/fault", response_model=FaultPredictionResponse)
async def predict_fault(req: FaultPredictionRequest):
    """Predict fault classification with confidence and SHAP attribution."""
    clf: Optional[InfusionPumpFaultClassifier] = models_state["classifier"]
    pipeline: Optional[SensorFeaturePipeline] = models_state["pipeline"]
    shap_exp: Optional[InfusionSHAPExplainer] = models_state["shap_explainer"]

    if clf is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier model is not loaded or initialized.",
        )

    try:
        # Extract features from input window
        window_dict = req.model_dump()
        feat_df = pipeline.transform_window_dict(window_dict)

        # Classification prediction
        detail = clf.predict_detailed(feat_df)[0]

        # SHAP local attribution
        shap_res = None
        if shap_exp is not None:
            shap_res = shap_exp.explain_instance(feat_df)

        now_iso = datetime.now(timezone.utc).isoformat()

        return FaultPredictionResponse(
            sensor_id=req.sensor_id,
            timestamp=now_iso,
            provenance=DataProvenance.SYNTHETIC_DATA.value,
            predicted_fault_class=detail["predicted_class"],
            class_index=detail["predicted_class_index"],
            model_confidence=detail["confidence"],
            class_probabilities=detail["class_probabilities"],
            is_sensor_fault=detail["is_fault"],
            is_process_anomaly=detail["is_process_anomaly"],
            shap_explanation=shap_res,
            disclaimer=(
                "Medical Device AI Safety Notice: Model attribution is statistical, not causal. "
                "Advisory recommendation only; clinical engineering verification required."
            ),
        )
    except Exception as e:
        logger.error(f"Error during fault prediction: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/predict/sensor-error", response_model=SensorErrorResponse)
async def predict_sensor_error(req: SensorErrorRequest):
    """Estimate transducer measurement error using supervised regression."""
    if req.sensor_type.lower() != "pressure":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown or unsupported sensor type '{req.sensor_type}'. Currently supported: 'pressure'.",
        )

    reg: Optional[SensorErrorEstimator] = models_state["regressor_pressure"]
    pipeline: Optional[SensorFeaturePipeline] = models_state["pipeline"]

    if reg is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Regression model is not loaded.",
        )

    try:
        window_dict = req.model_dump()
        feat_df = pipeline.transform_window_dict(window_dict)

        ref_avail = req.independent_reference_value is not None
        obs_val = req.observed_sensor_value or float(req.sensor_pressure_mmhg[-1])

        res = reg.estimate_sensor_error(
            feat_df,
            observed_sensor_value=obs_val,
            reference_measurement_available=ref_avail,
        )

        return SensorErrorResponse(
            sensor_id=req.sensor_id,
            sensor_type=reg.sensor_type,
            units=reg.units,
            status=res["status"],
            observed_sensor_value=res.get("observed_sensor_value"),
            estimated_sensor_error=res.get("estimated_sensor_error"),
            estimated_calibrated_value=res.get("estimated_calibrated_value"),
            confidence_interval_95=res.get("confidence_interval_95"),
            model_residual_rmse=res.get("model_residual_rmse"),
            advisory_recommendation=res.get("advisory_recommendation", res.get("message", "")),
            disclaimer=res.get("disclaimer", "Advisory only; no automated modification permitted."),
        )
    except Exception as e:
        logger.error(f"Error during error estimation: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/predict/sensor-health", response_model=SensorHealthResponse)
async def predict_sensor_health(req: SensorHealthRequest):
    """Integrated sensor health and clinical process risk assessment."""
    pipeline: Optional[SensorFeaturePipeline] = models_state["pipeline"]
    consistency_engine: Optional[CrossSensorConsistencyEngine] = models_state["consistency_engine"]
    health_engine: Optional[SensorHealthEngine] = models_state["health_engine"]
    iso: Optional[MultivariateAnomalyDetector] = models_state["iso_forest"]

    if consistency_engine is None or health_engine is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Consistency and health engines not initialized.",
        )

    try:
        # Build mini DataFrame from window
        window_dict = req.model_dump()
        length = len(req.sensor_pressure_mmhg)
        df_win = pd.DataFrame({
            "sensor_pressure_mmhg": req.sensor_pressure_mmhg,
        })
        if req.sensor_force_n is not None:
            df_win["sensor_force_n"] = req.sensor_force_n
        if req.sensor_flow_ml_h is not None:
            df_win["sensor_flow_ml_h"] = req.sensor_flow_ml_h
        if req.sensor_temperature_c is not None:
            df_win["sensor_temperature_c"] = req.sensor_temperature_c
        df_win["commanded_rate_ml_h"] = req.commanded_rate_ml_h or 50.0
        if req.dt_predicted_pressure_mmhg is not None:
            df_win["dt_predicted_pressure_mmhg"] = req.dt_predicted_pressure_mmhg

        # 1. Cross-sensor consistency check
        c_assessment = consistency_engine.assess_run(
            df_win,
            pressure_col="sensor_pressure_mmhg",
            force_col="sensor_force_n" if req.sensor_force_n else None,
            flow_col="sensor_flow_ml_h" if req.sensor_flow_ml_h else None,
            temperature_col="sensor_temperature_c" if req.sensor_temperature_c else None,
            commanded_rate_col="commanded_rate_ml_h",
            dt_pressure_col="dt_predicted_pressure_mmhg" if req.dt_predicted_pressure_mmhg else None,
        )

        # 2. Multivariate Isolation Forest anomaly check (if loaded)
        anomaly_flag = False
        anomaly_sev = 0.0
        if iso is not None and iso.is_fitted_:
            feats = pipeline.transform_dataframe(df_win)
            if len(feats) > 0:
                # Use features present in iso model
                common_feats = [f for f in iso.feature_names_ if f in feats.columns]
                if len(common_feats) == len(iso.feature_names_):
                    preds_iso = iso.predict(feats[iso.feature_names_])
                    anomaly_flag = bool(preds_iso["anomaly_detected"].iloc[-1])
                    anomaly_sev = float(preds_iso["anomaly_severity"].iloc[-1])

        # 3. Health Engine Evaluation
        obs_val = req.observed_sensor_value or float(req.sensor_pressure_mmhg[-1])
        h_report = health_engine.evaluate_health(
            sensor_id=req.sensor_id,
            sensor_type=req.sensor_type,
            consistency_assessment=c_assessment,
            anomaly_detected=anomaly_flag,
            anomaly_severity=anomaly_sev,
            observed_sensor_value=obs_val,
            independent_reference_value=req.independent_reference_value,
        )

        return SensorHealthResponse(
            sensor_id=h_report.sensor_id,
            sensor_type=h_report.sensor_type,
            status=h_report.status.value,
            sensor_health_score=h_report.sensor_health_score,
            health_score=h_report.health_score,
            process_risk_score=h_report.process_risk_score,
            process_risk_level=h_report.process_risk_level,
            model_confidence=h_report.model_confidence,
            anomaly_detected=h_report.anomaly_detected,
            anomaly_severity_score=h_report.anomaly_severity_score,
            drift_detected=h_report.drift_detected,
            drift_direction=h_report.drift_direction,
            reference_available=h_report.reference_available,
            estimated_measurement_error=h_report.estimated_measurement_error,
            recommended_action=h_report.recommended_action,
            evidence_breakdown=h_report.evidence_breakdown,
            deductions=h_report.deductions,
            process_risk_factors=h_report.process_risk_factors,
            disclaimer=h_report.disclaimer,
        )
    except Exception as e:
        logger.error(f"Error during health evaluation: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest):
    """Batch prediction across multiple sensor windows."""
    results = []
    for item in req.items:
        res = await predict_fault(item)
        results.append(res)

    now_iso = datetime.now(timezone.utc).isoformat()
    return BatchPredictionResponse(
        timestamp=now_iso,
        total_processed=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Phase 6: Sequential Streaming & Stateful Session Endpoints
# ---------------------------------------------------------------------------

@app.post("/stream/observe", response_model=SequentialPredictionResponse)
async def observe_stream(req: SequentialObservationRequest):
    """Ingest a single sequential telemetry observation and perform real-time multi-task inference.
    
    Provides stateful FIFO buffering for live streaming feeds (1 Hz nominal) without losing
    rolling window context (15-step and 60-step dynamics).
    """
    session_mgr: Optional[StreamSessionManager] = models_state.get("session_manager")
    pipeline: Optional[SensorFeaturePipeline] = models_state.get("pipeline")
    clf: Optional[InfusionPumpFaultClassifier] = models_state.get("classifier")
    calibrator: Optional[MulticlassCalibrator] = models_state.get("calibrator")
    conf_reg: Optional[ConformalSensorErrorEstimator] = models_state.get("conformal_regressor")
    reg: Optional[SensorErrorEstimator] = models_state.get("regressor_pressure")
    consistency_engine: Optional[CrossSensorConsistencyEngine] = models_state.get("consistency_engine")
    health_engine: Optional[SensorHealthEngine] = models_state.get("health_engine")
    iso: Optional[MultivariateAnomalyDetector] = models_state.get("iso_forest")

    if session_mgr is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Streaming buffer manager or feature pipeline is not initialized.",
        )

    # 1. Ingest observation into stateful session buffer
    obs = SensorObservation(
        timestamp=req.timestamp,
        sensor_pressure_mmhg=req.sensor_pressure_mmhg,
        sensor_force_n=req.sensor_force_n,
        sensor_flow_ml_h=req.sensor_flow_ml_h,
        sensor_temperature_c=req.sensor_temperature_c,
        commanded_rate_ml_h=req.commanded_rate_ml_h,
        dt_predicted_pressure_mmhg=req.dt_predicted_pressure_mmhg,
        independent_reference_value=req.independent_reference_value,
        sensor_id=req.sensor_id,
    )
    buffer = session_mgr.get_or_create_session(req.session_id)
    quality_status, warmup_status = buffer.add_observation(obs)
    buf_len = len(buffer._buffer)

    # 2. Check Warmup status
    if warmup_status == WarmupStatus.WARMUP:
        return SequentialPredictionResponse(
            session_id=req.session_id,
            sensor_id=req.sensor_id,
            timestamp=req.timestamp,
            provenance=DataProvenance.SYNTHETIC_DATA.value,
            stream_quality=quality_status.value,
            warmup_status=warmup_status.value,
            buffered_samples=buf_len,
            predicted_fault_class="NORMAL",
            class_index=0,
            model_confidence=1.0,
            calibrated_confidence=1.0,
            calibrated_probabilities={c: (1.0 if c == "NORMAL" else 0.0) for c in CLASS_NAMES},
            raw_class_probabilities={c: (1.0 if c == "NORMAL" else 0.0) for c in CLASS_NAMES},
            is_sensor_fault=False,
            is_process_anomaly=False,
            sensor_health_score=100.0,
            process_risk_score=0.0,
            process_risk_level="LOW",
            anomaly_detected=False,
            anomaly_severity_score=0.0,
            estimated_sensor_error=None,
            conformal_error_interval_95=None,
            error_status="INSUFFICIENT_REFERENCE_DATA" if req.independent_reference_value is None else "WARMUP",
            advisory_recommendation=f"Stream warming up ({buf_len}/{buffer.warmup_threshold} samples). Multi-scale feature dynamics maturing.",
            disclaimer="Advisory Medical AI Notice: Pre-infusion warmup in progress.",
        )

    # 3. Ready state: Extract features from session buffer
    try:
        feat_df = buffer.extract_features(pipeline, baseline_mmhg=req.baseline_pressure_mmhg)
    except Exception as e:
        logger.error(f"Feature extraction failed for session {req.session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Feature extraction error: {e}")

    # 4. Fault Classification & Probability Calibration
    predicted_class = "NORMAL"
    class_idx = 0
    raw_confidence = 1.0
    cal_confidence = 1.0
    raw_probs = {c: (1.0 if c == "NORMAL" else 0.0) for c in CLASS_NAMES}
    cal_probs = dict(raw_probs)
    is_sensor_fault = False
    is_process_anomaly = False

    if clf is not None and clf.is_fitted_:
        detail = clf.predict_detailed(feat_df)[0]
        predicted_class = detail["predicted_class"]
        class_idx = detail["predicted_class_index"]
        raw_confidence = detail["confidence"]
        raw_probs = detail["class_probabilities"]
        is_sensor_fault = detail["is_fault"]
        is_process_anomaly = detail["is_process_anomaly"]

        if calibrator is not None and calibrator.is_fitted_:
            try:
                cal_p_arr = calibrator.predict_proba(feat_df)[0]
                cal_confidence = round(float(np.max(cal_p_arr)), 4)
                cal_probs = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(cal_p_arr)}
            except Exception as e:
                logger.warning(f"Calibrator prediction failed, falling back to raw probabilities: {e}")
                cal_confidence = raw_confidence
                cal_probs = raw_probs
        else:
            cal_confidence = raw_confidence
            cal_probs = raw_probs

    # 5. Isolation Forest Anomaly Detection
    anomaly_flag = False
    anomaly_sev = 0.0
    if iso is not None and iso.is_fitted_:
        try:
            common_feats = [f for f in iso.feature_names_ if f in feat_df.columns]
            if len(common_feats) == len(iso.feature_names_):
                preds_iso = iso.predict(feat_df[iso.feature_names_])
                anomaly_flag = bool(preds_iso["anomaly_detected"].iloc[-1])
                anomaly_sev = float(preds_iso["anomaly_severity"].iloc[-1])
        except Exception as e:
            logger.warning(f"Isolation Forest inference error: {e}")

    # 6. Cross-Sensor Consistency & Health / Risk Engine
    sensor_health_score = 100.0
    process_risk_score = 0.0
    process_risk_level = "LOW"
    advisory_rec = "Normal infusion telemetry. Transducer operating nominally."

    if consistency_engine is not None and health_engine is not None:
        try:
            df_buf = buffer.to_dataframe()
            c_assessment = consistency_engine.assess_run(
                df_buf,
                pressure_col="sensor_pressure_mmhg",
                force_col="sensor_force_n" if req.sensor_force_n is not None else None,
                flow_col="sensor_flow_ml_h" if req.sensor_flow_ml_h is not None else None,
                temperature_col="sensor_temperature_c" if req.sensor_temperature_c is not None else None,
                commanded_rate_col="commanded_rate_ml_h",
                dt_pressure_col="dt_predicted_pressure_mmhg" if req.dt_predicted_pressure_mmhg is not None else None,
            )
            h_report = health_engine.evaluate_health(
                sensor_id=req.sensor_id,
                sensor_type="pressure",
                consistency_assessment=c_assessment,
                anomaly_detected=anomaly_flag,
                anomaly_severity=anomaly_sev,
                observed_sensor_value=req.sensor_pressure_mmhg,
                independent_reference_value=req.independent_reference_value,
                classifier_fault=predicted_class,
            )
            sensor_health_score = h_report.sensor_health_score
            process_risk_score = h_report.process_risk_score
            process_risk_level = h_report.process_risk_level
            advisory_rec = h_report.recommended_action
        except Exception as e:
            logger.warning(f"Health engine evaluation error: {e}")

    # 7. Measurement Error & Conformal Uncertainty Interval
    est_error = None
    conformal_interval = None
    error_status = "INSUFFICIENT_REFERENCE_DATA"

    if req.independent_reference_value is not None:
        if conf_reg is not None and conf_reg.is_calibrated_:
            try:
                res_conf = conf_reg.predict_conformal_interval(
                    feat_df,
                    observed_sensor_value=req.sensor_pressure_mmhg,
                    reference_measurement_available=True,
                )
                est_error = res_conf.get("estimated_sensor_error")
                conformal_interval = res_conf.get("conformal_interval_95")
                error_status = res_conf.get("status", "ESTIMATION_AVAILABLE")
            except Exception as e:
                logger.warning(f"Conformal regressor estimation error: {e}")
        elif reg is not None and reg.is_fitted_:
            try:
                res_reg = reg.estimate_sensor_error(
                    feat_df,
                    observed_sensor_value=req.sensor_pressure_mmhg,
                    reference_measurement_available=True,
                )
                est_error = res_reg.get("estimated_sensor_error")
                conformal_interval = res_reg.get("confidence_interval_95")
                error_status = res_reg.get("status", "ESTIMATION_AVAILABLE")
            except Exception as e:
                logger.warning(f"Regressor error estimation error: {e}")

    return SequentialPredictionResponse(
        session_id=req.session_id,
        sensor_id=req.sensor_id,
        timestamp=req.timestamp,
        provenance=DataProvenance.SYNTHETIC_DATA.value,
        stream_quality=quality_status.value,
        warmup_status=warmup_status.value,
        buffered_samples=buf_len,
        predicted_fault_class=predicted_class,
        class_index=class_idx,
        model_confidence=round(raw_confidence, 4),
        calibrated_confidence=round(cal_confidence, 4),
        calibrated_probabilities=cal_probs,
        raw_class_probabilities=raw_probs,
        is_sensor_fault=is_sensor_fault,
        is_process_anomaly=is_process_anomaly,
        sensor_health_score=round(sensor_health_score, 1),
        process_risk_score=round(process_risk_score, 1),
        process_risk_level=process_risk_level,
        anomaly_detected=anomaly_flag,
        anomaly_severity_score=round(anomaly_sev, 4),
        estimated_sensor_error=round(est_error, 3) if est_error is not None else None,
        conformal_error_interval_95=conformal_interval,
        error_status=error_status,
        advisory_recommendation=advisory_rec,
        disclaimer=(
            "Advisory Medical AI Notice: Recommendations are advisory only. "
            "No automated device calibration is permitted without validated biomedical engineering sign-off."
        ),
    )


@app.post("/session/reset", response_model=SessionResetResponse)
async def reset_session(req: SessionResetRequest):
    """Reset a streaming telemetry buffer session."""
    session_mgr: Optional[StreamSessionManager] = models_state.get("session_manager")
    if session_mgr is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized.",
        )
    success = session_mgr.reset_session(req.session_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    return SessionResetResponse(
        session_id=req.session_id,
        reset_success=success,
        timestamp=now_iso,
        message=f"Session '{req.session_id}' buffer reset successfully." if success else f"Session '{req.session_id}' not found.",
    )


@app.get("/session/status/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Query current buffer status and quality metrics for a session."""
    session_mgr: Optional[StreamSessionManager] = models_state.get("session_manager")
    if session_mgr is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session manager not initialized.",
        )
    session = session_mgr.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    status_dict = session.get_status()
    return SessionStatusResponse(**status_dict)


