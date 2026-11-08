"""Split Conformal Prediction Engine for Transducer Measurement Error Regression.

Replaces the mathematically invalid heuristic assumption (±1.96 * RMSE)
with finite-sample, distribution-free Split Conformal Prediction.

Key Theoretical Principles:
1. Under exchangeability, Split Conformal Prediction guarantees marginal coverage:
   P(Y_test ∈ [f(X) - q_alpha, f(X) + q_alpha]) >= 1 - alpha
2. Does NOT assume normal, homoscedastic, or symmetric errors.
3. Calibration is performed STRICTLY on a dedicated held-out calibration set (X_cal, y_cal).
4. Clearly tags all outputs as SYNTHETIC_DATA; real physical validation remains PENDING.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.utils.config import MODELS_DIR, RANDOM_SEED, DataProvenance
from src.regression.xgboost_regressor import SensorErrorEstimator
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ConformalSensorErrorEstimator:
    """Split Conformal Prediction wrapper for sensor measurement error estimation."""

    def __init__(self, base_regressor: SensorErrorEstimator, alpha: float = 0.05):
        self.base_regressor = base_regressor
        self.alpha = alpha  # 1 - alpha is nominal coverage (e.g. 0.95 for 95%)
        self.conformal_quantile_: float = 0.0
        self.signed_q_low_: float = 0.0
        self.signed_q_high_: float = 0.0
        self.calibration_sample_count_: int = 0
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal_error: Union[pd.Series, np.ndarray],
    ) -> "ConformalSensorErrorEstimator":
        """Calibrate conformal quantile threshold strictly on held-out calibration set."""
        y_true = np.asarray(y_cal_error, dtype=np.float64)
        y_pred = self.base_regressor.predict_error(X_cal)

        residuals_signed = y_true - y_pred
        abs_residuals = np.abs(residuals_signed)

        n_cal = len(abs_residuals)
        if n_cal < 50:
            raise ValueError(f"Insufficient calibration samples: {n_cal} < 50.")

        # Conformal quantile calculation with finite-sample correction
        # q_alpha = ceil((n + 1)(1 - alpha)) / n quantile
        target_quantile = min(1.0, np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal)
        self.conformal_quantile_ = float(np.quantile(abs_residuals, target_quantile))

        # Asymmetric empirical residual bounds (to properly handle skewness)
        self.signed_q_low_ = float(np.quantile(residuals_signed, self.alpha / 2.0))
        self.signed_q_high_ = float(np.quantile(residuals_signed, 1.0 - (self.alpha / 2.0)))

        self.calibration_sample_count_ = n_cal
        self.is_calibrated_ = True

        logger.info(
            f"Conformal calibrator fitted on {n_cal} samples: "
            f"conformal threshold q_95 = ±{self.conformal_quantile_:.3f} mmHg, "
            f"empirical asymmetric bounds = [{self.signed_q_low_:.3f}, {self.signed_q_high_:.3f}] mmHg"
        )
        return self

    def predict_conformal_interval(
        self,
        X: pd.DataFrame,
        observed_sensor_value: Optional[float] = None,
        reference_measurement_available: bool = True,
    ) -> Dict[str, Any]:
        """Produce point estimate and mathematically grounded conformal prediction intervals."""
        if not reference_measurement_available:
            return {
                "status": "INSUFFICIENT_REFERENCE_DATA",
                "estimated_sensor_error": None,
                "conformal_interval_95": None,
                "message": (
                    "Independent reference measurements are not available. "
                    "Cannot establish true sensor accuracy or compute error intervals without independent reference data."
                ),
            }

        if not self.is_calibrated_:
            raise RuntimeError("Estimator must be calibrated before generating conformal prediction intervals.")

        y_pred = self.base_regressor.predict_error(X)
        pred_error = float(y_pred[-1])

        # Symmetric Conformal Prediction Interval: [y_hat - q, y_hat + q]
        conformal_lower = round(pred_error - self.conformal_quantile_, 3)
        conformal_upper = round(pred_error + self.conformal_quantile_, 3)

        # Asymmetric Quantile Interval: [y_hat + q_low, y_hat + q_high]
        asymmetric_lower = round(pred_error + self.signed_q_low_, 3)
        asymmetric_upper = round(pred_error + self.signed_q_high_, 3)

        calibrated_val = None
        if observed_sensor_value is not None:
            calibrated_val = round(observed_sensor_value - pred_error, 2)

        return {
            "status": "ESTIMATION_AVAILABLE",
            "provenance": DataProvenance.SYNTHETIC_DATA.value,
            "sensor_type": self.base_regressor.sensor_type,
            "units": self.base_regressor.units,
            "observed_sensor_value": observed_sensor_value,
            "estimated_sensor_error": round(pred_error, 3),
            "estimated_calibrated_value": calibrated_val,
            "conformal_interval_95": [conformal_lower, conformal_upper],
            "asymmetric_quantile_interval_95": [asymmetric_lower, asymmetric_upper],
            "conformal_halfwidth": round(self.conformal_quantile_, 3),
            "confidence_level": round(1.0 - self.alpha, 2),
            "calibration_sample_count": self.calibration_sample_count_,
            "uncertainty_method": "Split Conformal Prediction (Finite-sample marginal coverage)",
            "advisory_recommendation": (
                f"Predicted transducer error: {pred_error:+.2f} {self.base_regressor.units}. "
                f"95% Conformal Interval: [{conformal_lower:.2f}, {conformal_upper:.2f}] {self.base_regressor.units}. "
                "Advisory recommendation only; validated biomedical engineering sign-off required for device calibration."
            ),
            "disclaimer": (
                "Safety Notice: Conformal intervals guarantee mathematical coverage under exchangeability, "
                "but physical accuracy cannot be established without active NIST-traceable reference verification. "
                "Automatic device recalibration is strictly prohibited."
            ),
        }

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test_error: Union[pd.Series, np.ndarray],
        baseline_group: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Compute empirical coverage percentage across test dataset and operating subgroups."""
        if not self.is_calibrated_:
            raise RuntimeError("Estimator must be calibrated before evaluating coverage.")

        y_true = np.asarray(y_test_error, dtype=np.float64)
        y_pred = self.base_regressor.predict_error(X_test)

        # Conformal interval coverage
        lower_conf = y_pred - self.conformal_quantile_
        upper_conf = y_pred + self.conformal_quantile_
        in_conformal = (y_true >= lower_conf) & (y_true <= upper_conf)
        conformal_coverage = float(np.mean(in_conformal))

        # Naive Gaussian 1.96 * RMSE coverage for comparison
        rmse_val = float(root_mean_squared_error(y_true, y_pred))
        naive_halfwidth = 1.96 * rmse_val
        lower_naive = y_pred - naive_halfwidth
        upper_naive = y_pred + naive_halfwidth
        in_naive = (y_true >= lower_naive) & (y_true <= upper_naive)
        naive_coverage = float(np.mean(in_naive))

        # Asymmetric interval coverage
        lower_asym = y_pred + self.signed_q_low_
        upper_asym = y_pred + self.signed_q_high_
        in_asym = (y_true >= lower_asym) & (y_true <= upper_asym)
        asym_coverage = float(np.mean(in_asym))

        results = {
            "target_nominal_coverage": 1.0 - self.alpha,
            "conformal_prediction": {
                "empirical_coverage": round(conformal_coverage, 4),
                "interval_width_mmhg": round(2.0 * self.conformal_quantile_, 3),
                "halfwidth_mmhg": round(self.conformal_quantile_, 3),
            },
            "asymmetric_empirical_quantile": {
                "empirical_coverage": round(asym_coverage, 4),
                "interval_width_mmhg": round(self.signed_q_high_ - self.signed_q_low_, 3),
                "bounds_mmhg": [round(self.signed_q_low_, 3), round(self.signed_q_high_, 3)],
            },
            "naive_gaussian_comparison": {
                "empirical_coverage": round(naive_coverage, 4),
                "interval_width_mmhg": round(2.0 * naive_halfwidth, 3),
                "rmse_mmhg": round(rmse_val, 3),
            },
        }

        # Subgroup coverage if baseline condition provided
        if baseline_group is not None:
            subgroup_cov = {}
            for b_val in np.unique(baseline_group):
                mask = (baseline_group.values == b_val)
                sub_cov = float(np.mean(in_conformal[mask]))
                subgroup_cov[f"baseline_{b_val}_mmhg"] = {
                    "samples": int(np.sum(mask)),
                    "conformal_coverage": round(sub_cov, 4),
                }
            results["subgroup_coverage"] = subgroup_cov

        return results

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist conformal calibrator parameters."""
        if not self.is_calibrated_:
            raise RuntimeError("Cannot save uncalibrated conformal estimator.")
        out_path = filepath or (MODELS_DIR / "regression" / "conformal_regressor_pressure.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "alpha": self.alpha,
                "conformal_quantile": self.conformal_quantile_,
                "signed_q_low": self.signed_q_low_,
                "signed_q_high": self.signed_q_high_,
                "calibration_sample_count": self.calibration_sample_count_,
                "base_regressor_path": str(MODELS_DIR / "regression" / f"xgboost_regressor_{self.base_regressor.sensor_type}.joblib"),
            },
            out_path,
        )
        logger.info(f"Saved conformal regressor bundle to {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Path, base_regressor: SensorErrorEstimator) -> "ConformalSensorErrorEstimator":
        """Load persisted conformal calibrator."""
        data = joblib.load(filepath)
        conf = cls(base_regressor=base_regressor, alpha=data["alpha"])
        conf.conformal_quantile_ = data["conformal_quantile"]
        conf.signed_q_low_ = data["signed_q_low"]
        conf.signed_q_high_ = data["signed_q_high"]
        conf.calibration_sample_count_ = data["calibration_sample_count"]
        conf.is_calibrated_ = True
        return conf
