"""SHAP explainability helpers for ExplainInvest predictions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class ExplainabilityEngine:
    """Provide per-trade SHAP explanations and top feature contribution summaries."""

    def __init__(self, model, background_data: np.ndarray, feature_names: List[str]) -> None:
        """Initialize SHAP explainer, preferring GradientExplainer for deep sequence models."""
        import shap  # Imported here to keep module import lightweight when SHAP is unavailable.

        self.model = model
        self.feature_names = feature_names
        self.time_steps = background_data.shape[1]
        self.num_features = background_data.shape[2]
        self.flat_feature_names = self._build_flat_feature_names(feature_names, self.time_steps)

        self.explainer_mode = "kernel"
        self.background_flat = background_data.reshape(background_data.shape[0], -1)
        try:
            self.explainer = shap.GradientExplainer(self.model, background_data)
            self.explainer_mode = "gradient"
        except Exception:
            # Fallback for environments where GradientExplainer cannot trace the model graph.
            self.explainer = shap.KernelExplainer(self._model_predict_flat, self.background_flat)

    def _build_flat_feature_names(self, feature_names: List[str], time_steps: int) -> List[str]:
        """Create names for flattened time-step features consumed by KernelExplainer."""
        flat_names = []
        for t in range(time_steps):
            for feature in feature_names:
                flat_names.append(f"t-{time_steps - t}_{feature}")
        return flat_names

    def _model_predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        """Predict model probabilities from flattened SHAP input arrays."""
        X = X_flat.reshape(-1, self.time_steps, self.num_features)
        return self.model.predict(X, verbose=0)

    def explain_single_trade(
        self,
        X_single: np.ndarray,
        predicted_class: int,
        latest_feature_values: Dict[str, float],
        top_k: int = 5,
        nsamples: int = 120,
    ) -> List[Dict[str, float | str]]:
        """Compute SHAP values for one trade and return top feature-level contributions."""
        if self.explainer_mode == "gradient":
            shap_values_all = self.explainer.shap_values(X_single)
        else:
            X_flat = X_single.reshape(1, -1)
            shap_values_all = self.explainer.shap_values(X_flat, nsamples=nsamples)

        # SHAP can return list[class] or ndarray depending on SHAP version.
        if isinstance(shap_values_all, list):
            class_shap_arr = np.asarray(shap_values_all[predicted_class])
            class_shap_2d = np.asarray(class_shap_arr[0], dtype=np.float32)
            if class_shap_2d.ndim == 1:
                class_shap_2d = class_shap_2d.reshape(self.time_steps, self.num_features)
        else:
            shap_arr = np.asarray(shap_values_all)
            if shap_arr.ndim == 4 and shap_arr.shape[-1] >= 3:
                # Shape: (samples, time_steps, features, classes)
                class_shap_2d = np.asarray(shap_arr[0, :, :, predicted_class], dtype=np.float32)
            elif shap_arr.ndim == 3 and shap_arr.shape[-1] >= 3:
                # Shape: (samples, flat_features, classes)
                class_shap_flat = shap_arr[0, :, predicted_class]
                class_shap_2d = np.asarray(class_shap_flat, dtype=np.float32).reshape(
                    self.time_steps, self.num_features
                )
            elif shap_arr.ndim == 3 and shap_arr.shape[0] >= 3:
                # Shape: (classes, samples, flat_features)
                class_shap_flat = shap_arr[predicted_class, 0, :]
                class_shap_2d = np.asarray(class_shap_flat, dtype=np.float32).reshape(
                    self.time_steps, self.num_features
                )
            else:
                raise ValueError(f"Unexpected SHAP output shape: {shap_arr.shape}")

        # Aggregate over time so explanations are feature-centric for readability.
        feature_signed = np.sum(class_shap_2d, axis=0)
        feature_magnitude = np.abs(feature_signed)
        top_indices = np.argsort(-feature_magnitude)[:top_k]

        top_items: List[Dict[str, float | str]] = []
        for idx in top_indices:
            feature = self.feature_names[idx]
            shap_value = float(feature_signed[idx])
            direction = "↑" if shap_value >= 0 else "↓"
            top_items.append(
                {
                    "feature": feature,
                    "value": float(latest_feature_values.get(feature, 0.0)),
                    "shap_value": shap_value,
                    "direction": direction,
                }
            )

        return top_items


def format_top_features(
    top_features: List[Dict[str, float | str]],
    signal_name: str,
) -> List[str]:
    """Format SHAP top-feature dictionaries into human-readable output lines."""
    lines: List[str] = []
    for item in top_features:
        feat = str(item["feature"])
        value = float(item["value"])
        shap_val = float(item["shap_value"])
        direction = str(item["direction"])

        if shap_val >= 0:
            effect = f"pushed {signal_name}"
        else:
            effect = f"pulled away from {signal_name}"

        lines.append(f"{feat} = {value:.4f} -> {shap_val:+.4f} ({effect} {direction})")

    return lines
