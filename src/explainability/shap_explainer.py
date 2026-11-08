"""SHAP (SHapley Additive exPlanations) Engine for Infusion Pump ML.

Provides local and global feature attribution for tree-based ensemble models.

Medical Device AI Disclaimer:
SHAP values quantify statistical feature attributions within the tree ensemble and
must NOT be interpreted as clinical causal proof. They explain why the model made a prediction,
not necessarily the underlying biological or pathophysiological etiology.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import shap

from src.classification.xgboost_classifier import CLASS_NAMES, InfusionPumpFaultClassifier
from src.features.feature_pipeline import FEATURE_NAMES
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class InfusionSHAPExplainer:
    """Interpretable SHAP explainer for XGBoost classification predictions."""

    def __init__(self, classifier: InfusionPumpFaultClassifier):
        if not classifier.is_fitted_ or classifier.model_ is None:
            raise ValueError("Classifier must be fitted before creating SHAP explainer.")
        self.classifier = classifier
        self.feature_names = classifier.feature_names_
        # TreeExplainer is fast, exact, and native for XGBoost
        self.explainer = shap.TreeExplainer(classifier.model_)
        logger.info("Initialized TreeExplainer for XGBoost classifier.")

    def explain_instance(
        self,
        features: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Produce human-readable explanation for a single prediction row.

        Args:
            features: DataFrame with 1 or more rows aligned to self.feature_names.
            top_k: Number of top contributing features to include in explanation.

        Returns:
            Structured explanation dictionary with contributing features and direction.
        """
        X_clean = features[self.feature_names].copy().iloc[[-1]]
        raw_vals = X_clean.iloc[0].to_dict()

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X_clean)

        # Get predicted class and probabilities
        probas = self.classifier.predict_proba(X_clean)[0]
        pred_idx = int(np.argmax(probas))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probas[pred_idx])

        # Handle SHAP multi-class output shape
        # In modern SHAP, for multiclass TreeExplainer, raw_shap is shape (N, n_features, n_classes)
        # or a list of len n_classes with arrays of shape (N, n_features)
        if isinstance(raw_shap, list):
            class_shap_vals = raw_shap[pred_idx][0]
        elif len(raw_shap.shape) == 3:
            class_shap_vals = raw_shap[0, :, pred_idx]
        else:
            class_shap_vals = raw_shap[0]

        # Rank features by absolute SHAP attribution
        abs_shap = np.abs(class_shap_vals)
        ranked_indices = np.argsort(abs_shap)[::-1][:top_k]

        contributions: List[Dict[str, Any]] = []
        explanation_phrases: List[str] = []

        for idx in ranked_indices:
            feat_name = self.feature_names[idx]
            feat_val = float(raw_vals[feat_name])
            shap_val = float(class_shap_vals[idx])
            direction = "PUSHED_TOWARD" if shap_val > 0 else "PUSHED_AWAY"

            contributions.append({
                "feature": feat_name,
                "value": round(feat_val, 4),
                "shap_attribution": round(shap_val, 4),
                "influence_direction": direction,
            })

            dir_text = "increased likelihood of" if shap_val > 0 else "reduced likelihood of"
            explanation_phrases.append(f"{feat_name} ({feat_val:.2f}) {dir_text} {pred_class} (attribution: {shap_val:+.3f})")

        summary_text = (
            f"Prediction: {pred_class} (confidence: {confidence:.1%}). "
            f"Primary factors: {'; '.join(explanation_phrases[:3])}."
        )

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "top_contributing_features": contributions,
            "explanation_summary": summary_text,
            "disclaimer": (
                "Interpretability Notice: SHAP feature attributions reflect statistical model associations "
                "within the decision trees and must NOT be interpreted as causal physiological proof."
            ),
        }

    def compute_global_summary_data(self, X_sample: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Compute global SHAP values over a representative sample for visualization."""
        X_clean = X_sample[self.feature_names].copy()
        shap_vals = self.explainer.shap_values(X_clean)
        return shap_vals, self.feature_names
