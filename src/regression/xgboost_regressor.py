"""Supervised Sensor Measurement Error Estimator using XGBoost Regression.

Target:
sensor_error = sensor_reading - independent_reference_reading

Critical Medical AI Constraints:
- Trained strictly on paired sensor and independent physical reference measurements
  (e.g. verified manometer or gravimetric analytical balance).
- NEVER substitutes Digital Twin theoretical predictions for independent ground truth.
- Kept strictly separate by sensor type (e.g. pressure in mmHg, delivery in mL/h).
- Returns INSUFFICIENT_REFERENCE_DATA when independent reference measurements are unavailable.
- Model predictions are advisory; NO automated in-line calibration is permitted without engineering sign-off.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb

from src.utils.config import RANDOM_SEED, MODELS_DIR, SensorType, DataProvenance
from src.features.feature_pipeline import FEATURE_NAMES
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class SensorErrorEstimator:
    """Supervised XGBoost Regressor for estimating transducer measurement error."""

    def __init__(
        self,
        sensor_type: str = "pressure",
        units: str = "mmHg",
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        random_state: int = RANDOM_SEED,
    ):
        self.sensor_type = sensor_type
        self.units = units
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model_: Optional[xgb.XGBRegressor] = None
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.residual_rmse_: float = 0.5  # Estimated test RMSE for confidence interval
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_error: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val_error: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "SensorErrorEstimator":
        """Train XGBoost error estimator on paired sensor-reference features."""
        y_train_arr = np.asarray(y_train_error, dtype=np.float64)
        X_train_clean = X_train[self.feature_names_].copy()

        self.model_ = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            tree_method="hist",
            eval_metric="rmse",
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val_error is not None:
            eval_set = [(X_val[self.feature_names_], np.asarray(y_val_error, dtype=np.float64))]

        self.model_.fit(X_train_clean, y_train_arr, eval_set=eval_set, verbose=False)

        # Baseline training RMSE
        train_preds = self.model_.predict(X_train_clean)
        self.residual_rmse_ = float(root_mean_squared_error(y_train_arr, train_preds))
        self.is_fitted_ = True

        logger.info(
            f"SensorErrorEstimator [{self.sensor_type}] fitted on {len(X_train)} samples. "
            f"Train RMSE: {self.residual_rmse_:.4f} {self.units}"
        )
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected sensor error (sensor - reference)."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("SensorErrorEstimator must be fitted before predict.")
        X_clean = X[self.feature_names_].copy()
        return self.model_.predict(X_clean)

    def estimate_sensor_error(
        self,
        features: pd.DataFrame,
        observed_sensor_value: Optional[float] = None,
        reference_measurement_available: bool = False,
    ) -> Dict[str, Any]:
        """Produce structured error estimation report with clinical safety disclaimer."""
        if not reference_measurement_available:
            return {
                "status": "INSUFFICIENT_REFERENCE_DATA",
                "sensor_type": self.sensor_type,
                "units": self.units,
                "estimated_sensor_error": None,
                "estimated_calibrated_value": None,
                "confidence_interval_95": None,
                "message": (
                    "Independent reference measurements are not available. "
                    "The system cannot establish true sensor accuracy or compute error without independent reference data."
                ),
            }

        # Predict error
        err_preds = self.predict_error(features)
        pred_error = float(err_preds[-1])

        ci_lower = pred_error - 1.96 * self.residual_rmse_
        ci_upper = pred_error + 1.96 * self.residual_rmse_

        calibrated_val = None
        if observed_sensor_value is not None:
            # Calibrated expectation = Observed - Estimated Error
            calibrated_val = round(observed_sensor_value - pred_error, 2)

        return {
            "status": "ESTIMATION_AVAILABLE",
            "sensor_type": self.sensor_type,
            "units": self.units,
            "observed_sensor_value": observed_sensor_value,
            "estimated_sensor_error": round(pred_error, 3),
            "estimated_calibrated_value": calibrated_val,
            "confidence_interval_95": [round(ci_lower, 3), round(ci_upper, 3)],
            "model_residual_rmse": round(self.residual_rmse_, 3),
            "advisory_recommendation": (
                f"Predicted transducer error: {pred_error:+.2f} {self.units}. "
                "Advisory recommendation only; validated engineering sign-off required for device calibration."
            ),
            "disclaimer": (
                "Safety Notice: Machine Learning cannot certify true accuracy without active independent reference verification. "
                "Automatic device modification is strictly prohibited."
            ),
        }

    def evaluate(self, X_test: pd.DataFrame, y_test_error: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Compute regression metrics: MAE, RMSE, and R2 score."""
        y_true = np.asarray(y_test_error, dtype=np.float64)
        y_pred = self.predict_error(X_test)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        self.residual_rmse_ = rmse

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "units": self.units,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist error estimator artifact bundle."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Cannot save unfitted regressor.")
        out_path = filepath or (MODELS_DIR / "regression" / f"xgboost_regressor_{self.sensor_type}.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "feature_names": self.feature_names_,
                "sensor_type": self.sensor_type,
                "units": self.units,
                "residual_rmse": self.residual_rmse_,
                "is_fitted": self.is_fitted_,
                "provenance": DataProvenance.SYNTHETIC_DATA.value,
            },
            out_path,
        )
        logger.info(f"Saved SensorErrorEstimator to {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Path) -> "SensorErrorEstimator":
        """Load persisted regressor artifact bundle."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        bundle = joblib.load(filepath)
        estimator = cls(sensor_type=bundle["sensor_type"], units=bundle["units"])
        estimator.model_ = bundle["model"]
        estimator.feature_names_ = bundle["feature_names"]
        estimator.residual_rmse_ = bundle["residual_rmse"]
        estimator.is_fitted_ = bundle["is_fitted"]
        logger.info(f"Loaded SensorErrorEstimator from {filepath}")
        return estimator
