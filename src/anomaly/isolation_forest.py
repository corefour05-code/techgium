"""Unsupervised Multivariate Anomaly Detection using Isolation Forest.

Detects abnormal multivariate departures from in-control feature space:
- Trained strictly on verified in-control baseline features (preventing data leakage)
- Generates raw decision scores, normalized anomaly severity in [0, 1], and statistical confidence
- Integrates rolling statistics, rate of change, and cross-sensor residuals
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config import RANDOM_SEED, MODELS_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class MultivariateAnomalyDetector:
    """Unsupervised Isolation Forest detector for sensor time-series."""

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 150,
        random_state: int = RANDOM_SEED,
    ):
        """
        Args:
            contamination: Expected proportion of outliers in in-control baseline (typically 0.5% - 1%).
            n_estimators: Number of isolation trees in ensemble.
            random_state: Fixed seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names_: List[str] = []
        self.model_: Optional[IsolationForest] = None
        self.score_min_: float = -0.5
        self.score_max_: float = 0.5
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "MultivariateAnomalyDetector":
        """Fit Isolation Forest on in-control baseline feature vectors."""
        clean_X = X.dropna()
        if len(clean_X) < 50:
            raise ValueError("At least 50 valid feature rows required to fit Isolation Forest.")

        self.feature_names_ = list(clean_X.columns)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(clean_X)

        # Baseline decision scores to calibrate normalization bounds
        base_scores = self.model_.decision_function(clean_X)
        self.score_min_ = float(np.min(base_scores))
        self.score_max_ = float(np.max(base_scores))
        self.is_fitted_ = True

        logger.info(
            f"IsolationForest fitted on {len(clean_X)} samples across {len(self.feature_names_)} features. "
            f"Decision bounds: [{self.score_min_:.4f}, {self.score_max_:.4f}]"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score feature vectors and return anomaly flags and severity metrics."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Anomaly detector must be fitted before predict.")

        # Ensure feature alignment
        missing_feats = [f for f in self.feature_names_ if f not in X.columns]
        if missing_feats:
            raise ValueError(f"Input DataFrame is missing fitted features: {missing_feats}")

        X_eval = X[self.feature_names_].copy()
        # Handle transient NaNs via forward-fill
        X_eval = X_eval.ffill().bfill()

        # -1 = anomaly, 1 = normal
        raw_preds = self.model_.predict(X_eval)
        # Higher decision function = more normal; negative = anomalous
        decision_scores = self.model_.decision_function(X_eval)

        # Invert and normalize to [0, 1] severity scale (0 = perfectly normal, 1 = extreme anomaly)
        norm_severity = (self.score_max_ - decision_scores) / max(1e-4, self.score_max_ - self.score_min_)
        norm_severity = np.clip(norm_severity, 0.0, 1.0)

        # Confidence metric: distance from decision boundary (0.0)
        confidence = np.clip(np.abs(decision_scores) / (self.score_max_ - self.score_min_ + 1e-6), 0.0, 1.0)

        res = pd.DataFrame(index=X.index)
        res["anomaly_detected"] = (raw_preds == -1)
        res["anomaly_score_raw"] = decision_scores
        res["anomaly_severity"] = norm_severity
        res["confidence"] = confidence
        return res

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist fitted model artifacts."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Cannot save unfitted model.")
        out_path = filepath or (MODELS_DIR / "anomaly_detection" / "isolation_forest.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "feature_names": self.feature_names_,
                "contamination": self.contamination,
                "score_min": self.score_min_,
                "score_max": self.score_max_,
                "is_fitted": self.is_fitted_,
            },
            out_path,
        )
        logger.info(f"Saved IsolationForest model to {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Path) -> "MultivariateAnomalyDetector":
        """Load persisted model artifacts."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        detector = cls(contamination=data["contamination"])
        detector.model_ = data["model"]
        detector.feature_names_ = data["feature_names"]
        detector.score_min_ = data["score_min"]
        detector.score_max_ = data["score_max"]
        detector.is_fitted_ = data["is_fitted"]
        logger.info(f"Loaded IsolationForest model from {filepath}")
        return detector

