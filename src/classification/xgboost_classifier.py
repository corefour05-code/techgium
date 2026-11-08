"""Supervised Multiclass Fault Classifier using XGBoost.

Differentiates 9 distinct operational and failure states:
0: NORMAL
1: POSITIVE_DRIFT
2: NEGATIVE_DRIFT
3: SUDDEN_OFFSET
4: EXCESSIVE_NOISE
5: SENSOR_FREEZE
6: SENSOR_DROPOUT
7: GENUINE_OCCLUSION
8: TEMPERATURE_INFLUENCE

Medical Device AI Constraints:
- Trained on physics-informed synthetic benchmark runs with run-level data separation.
- Class weighting applied to prevent majority class dominance.
- Does NOT claim real-world clinical pump accuracy based solely on synthetic training data.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from src.utils.config import RANDOM_SEED, MODELS_DIR, DataProvenance
from src.features.feature_pipeline import FEATURE_NAMES
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

CLASS_NAMES = [
    "NORMAL",
    "POSITIVE_DRIFT",
    "NEGATIVE_DRIFT",
    "SUDDEN_OFFSET",
    "EXCESSIVE_NOISE",
    "SENSOR_FREEZE",
    "SENSOR_DROPOUT",
    "GENUINE_OCCLUSION",
    "TEMPERATURE_INFLUENCE",
]

CLASS_NAME_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}


class InfusionPumpFaultClassifier:
    """Supervised XGBoost multi-class classifier for sensor faults and process occlusion."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        use_class_weights: bool = True,
    ) -> "InfusionPumpFaultClassifier":
        """Train XGBoost model with balanced sample weights and optional validation monitoring."""
        y_train_arr = np.asarray(y_train, dtype=int)
        X_train_clean = X_train[self.feature_names_].copy()

        # Compute balanced weights to handle class imbalance
        sample_weights = None
        if use_class_weights:
            sample_weights = compute_sample_weight("balanced", y_train_arr)

        self.model_ = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val[self.feature_names_], np.asarray(y_val, dtype=int))]

        self.model_.fit(
            X_train_clean,
            y_train_arr,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted_ = True
        logger.info(f"XGBClassifier fitted on {len(X_train)} samples across {len(CLASS_NAMES)} classes.")
        return self

    def _apply_integrity_overrides(self, X: pd.DataFrame, probas: np.ndarray) -> np.ndarray:
        """Apply deterministic physical integrity safety rules (ADC freeze, electrical dropout, sudden offset jump)."""
        probas_out = probas.copy()
        if not isinstance(X, pd.DataFrame):
            return probas_out

        freeze_idx = CLASS_NAME_TO_INDEX["SENSOR_FREEZE"]
        dropout_idx = CLASS_NAME_TO_INDEX["SENSOR_DROPOUT"]
        offset_idx = CLASS_NAME_TO_INDEX["SUDDEN_OFFSET"]

        for i in range(len(X)):
            row = X.iloc[i]
            freeze_cnt = float(row.get("sensor_freeze_counter", 0.0))
            drop_cnt = float(row.get("sensor_dropout_counter", 0.0))
            p_std = float(row.get("p_roll_std_60", 1.0))
            p_res = abs(float(row.get("p_dt_residual", 0.0)))
            f_res = abs(float(row.get("force_hydraulic_residual", 0.0)))

            if drop_cnt >= 10:
                probas_out[i, :] = 0.0001
                probas_out[i, dropout_idx] = 0.999
                probas_out[i, :] /= np.sum(probas_out[i, :])
            elif freeze_cnt >= 15 and p_std < 0.05:
                probas_out[i, :] = 0.0001
                probas_out[i, freeze_idx] = 0.999
                probas_out[i, :] /= np.sum(probas_out[i, :])
            elif p_res > 15.0 and p_std > 5.0 and f_res > 0.2:
                probas_out[i, :] = 0.0001
                probas_out[i, offset_idx] = 0.999
                probas_out[i, :] /= np.sum(probas_out[i, :])

        return probas_out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict integer class indices."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probability matrix (N, 9) with physical integrity safety checks."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Classifier must be fitted before predict_proba.")
        X_clean = X[self.feature_names_].copy()
        raw_probas = self.model_.predict_proba(X_clean)
        return self._apply_integrity_overrides(X, raw_probas)

    def predict_detailed(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return human-readable prediction report with confidence and calibrated probabilities."""
        probas = self.predict_proba(X)
        preds = np.argmax(probas, axis=1)

        results = []
        for i, (pred_idx, p_row) in enumerate(zip(preds, probas)):
            pred_class = INDEX_TO_CLASS_NAME[pred_idx]
            conf = float(p_row[pred_idx])
            class_prob_dict = {CLASS_NAMES[c]: round(float(p_row[c]), 4) for c in range(len(CLASS_NAMES))}

            results.append({
                "predicted_class_index": int(pred_idx),
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
                "class_probabilities": class_prob_dict,
                "is_fault": pred_class not in ("NORMAL", "GENUINE_OCCLUSION"),
                "is_process_anomaly": pred_class == "GENUINE_OCCLUSION",
            })
        return results

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Compute comprehensive multiclass performance metrics."""
        y_true = np.asarray(y_test, dtype=int)
        y_pred = self.predict(X_test)

        bacc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        # False alarm rate on normal in-control class
        # (Predicted non-normal when true label is NORMAL)
        normal_idx = CLASS_NAME_TO_INDEX["NORMAL"]
        normal_total = np.sum(y_true == normal_idx)
        normal_false_alarms = np.sum((y_true == normal_idx) & (y_pred != normal_idx))
        far = float(normal_false_alarms / max(1, normal_total))

        return {
            "balanced_accuracy": round(bacc, 4),
            "false_alarm_rate_normal": round(far, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "classes_evaluated": CLASS_NAMES,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist classifier artifact bundle."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Cannot save unfitted classifier.")
        out_path = filepath or (MODELS_DIR / "classification" / "xgboost_classifier.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "feature_names": self.feature_names_,
                "class_names": CLASS_NAMES,
                "is_fitted": self.is_fitted_,
                "provenance": DataProvenance.SYNTHETIC_DATA.value,
                "disclaimer": "Benchmark model trained on physics-informed synthetic data; not clinically certified.",
            },
            out_path,
        )
        logger.info(f"Saved XGBClassifier to {out_path}")
        return out_path

    @classmethod
    def load(cls, filepath: Path) -> "InfusionPumpFaultClassifier":
        """Load persisted classifier bundle."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        bundle = joblib.load(filepath)
        classifier = cls()
        classifier.model_ = bundle["model"]
        classifier.feature_names_ = bundle["feature_names"]
        classifier.is_fitted_ = bundle["is_fitted"]
        logger.info(f"Loaded XGBClassifier from {filepath}")
        return classifier

