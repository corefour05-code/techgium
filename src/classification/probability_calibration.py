"""Multiclass Probability Calibration Engine for Infusion Pump ML.

Calibrates raw decision tree leaf probability outputs to reflect reliable empirical frequencies.
Supports:
1. Temperature Scaling: Optimizes scalar T > 0 on validation logits using Cross-Entropy minimization.
2. One-vs-Rest Platt Sigmoid / Isotonic Calibration via CalibratedClassifierCV.

Strict Validation Requirement:
Calibration parameters are fitted STRICTLY on held-out validation data (X_val, y_val),
never on training or test datasets.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from src.utils.config import MODELS_DIR, RANDOM_SEED
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_ece(
    probas: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for top predicted class."""
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if np.sum(in_bin) > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return round(float(ece), 4)


def compute_brier_score(probas: np.ndarray, y_true: np.ndarray) -> float:
    """Compute multi-class Brier score: mean squared difference from one-hot ground truth."""
    N, K = probas.shape
    y_onehot = np.zeros((N, K), dtype=float)
    for i, val in enumerate(y_true):
        y_onehot[i, int(val)] = 1.0
    return round(float(np.mean(np.sum((probas - y_onehot) ** 2, axis=1))), 5)


class MulticlassCalibrator:
    """Multiclass probability calibration wrapper for XGBoost classifiers."""

    def __init__(self, method: str = "temperature"):
        self.method = method  # "temperature", "sigmoid", "isotonic"
        self.temperature_: float = 1.0
        self.calibrated_model_: Optional[Any] = None
        self.is_fitted_: bool = False
        self.base_classifier_ = None

    def fit(
        self,
        base_classifier,
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray],
    ) -> "MulticlassCalibrator":
        """Fit calibration mapping strictly on validation dataset."""
        self.base_classifier_ = base_classifier
        y_val_arr = np.asarray(y_val, dtype=int)

        if self.method == "temperature":
            # Extract uncalibrated validation probabilities (or raw margins if available)
            raw_probas = base_classifier.predict_proba(X_val)
            # Convert clipped probabilities to pseudo-logits: z = log(p)
            eps = 1e-7
            p_clipped = np.clip(raw_probas, eps, 1.0 - eps)
            logits = np.log(p_clipped)

            # Define loss function for temperature optimization
            def nll_loss(T_param):
                T = max(1e-3, T_param[0])
                scaled_logits = logits / T
                # Softmax normalization
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                scaled_probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                return log_loss(y_val_arr, scaled_probas)

            opt_res = minimize(nll_loss, x0=[1.5], bounds=[(0.1, 10.0)], method="L-BFGS-B")
            self.temperature_ = float(opt_res.x[0])
            self.is_fitted_ = True
            logger.info(f"Fitted Temperature Scaling calibrator: T = {self.temperature_:.4f} (NLL = {opt_res.fun:.4f})")

        elif self.method in ("sigmoid", "isotonic"):
            # Use scikit-learn CalibratedClassifierCV with prefit base estimator
            # Note: base_classifier.model_ is the raw XGBClassifier
            self.calibrated_model_ = CalibratedClassifierCV(
                estimator=base_classifier.model_,
                method=self.method,
                cv="prefit",
            )
            clean_X_val = X_val[base_classifier.feature_names_].copy()
            self.calibrated_model_.fit(clean_X_val, y_val_arr)
            self.is_fitted_ = True
            logger.info(f"Fitted {self.method} CalibratedClassifierCV on {len(X_val)} validation samples.")

        else:
            raise ValueError(f"Unknown calibration method: '{self.method}'")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict well-calibrated class probability matrix."""
        if not self.is_fitted_:
            raise RuntimeError("Calibrator must be fitted before predict_proba.")

        if self.method == "temperature":
            raw_probas = self.base_classifier_.predict_proba(X)
            eps = 1e-7
            p_clipped = np.clip(raw_probas, eps, 1.0 - eps)
            logits = np.log(p_clipped)

            scaled_logits = logits / self.temperature_
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            calibrated_probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return calibrated_probas

        elif self.method in ("sigmoid", "isotonic"):
            clean_X = X[self.base_classifier_.feature_names_].copy()
            return self.calibrated_model_.predict_proba(clean_X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class index based on calibrated probabilities."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def evaluate_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute calibration improvement metrics: uncalibrated vs calibrated."""
        y_arr = np.asarray(y_test, dtype=int)
        raw_probas = self.base_classifier_.predict_proba(X_test)
        cal_probas = self.predict_proba(X_test)

        raw_brier = compute_brier_score(raw_probas, y_arr)
        cal_brier = compute_brier_score(cal_probas, y_arr)

        raw_ece = compute_ece(raw_probas, y_arr)
        cal_ece = compute_ece(cal_probas, y_arr)

        return {
            "uncalibrated": {
                "brier_score": raw_brier,
                "expected_calibration_error": raw_ece,
            },
            "calibrated": {
                "brier_score": cal_brier,
                "expected_calibration_error": cal_ece,
                "temperature": round(self.temperature_, 4) if self.method == "temperature" else None,
            },
            "ece_reduction_percent": round(((raw_ece - cal_ece) / max(1e-5, raw_ece)) * 100.0, 2),
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist calibrator bundle."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted calibrator.")
        out_path = filepath or (MODELS_DIR / "classification" / "probability_calibrator.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.method,
                "temperature": self.temperature_,
                "calibrated_model": self.calibrated_model_,
                "feature_names": self.base_classifier_.feature_names_,
            },
            out_path,
        )
        logger.info(f"Saved calibrator bundle to {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Path, base_classifier) -> "MulticlassCalibrator":
        """Load persisted calibrator."""
        data = joblib.load(filepath)
        cal = cls(method=data["method"])
        cal.temperature_ = data["temperature"]
        cal.calibrated_model_ = data["calibrated_model"]
        cal.base_classifier_ = base_classifier
        cal.is_fitted_ = True
        return cal

