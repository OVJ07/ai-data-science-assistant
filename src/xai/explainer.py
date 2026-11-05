"""
src/xai/explainer.py
─────────────────────
SHAP/LIME explanation module.
Produces global feature importance and local prediction explanations
with correct feature names from the fitted pipeline.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Output of the XAI explainer for a dataset + fitted pipeline."""

    # Global importance (feature_name -> mean |SHAP|)
    global_importance: Dict[str, float] = field(default_factory=dict)
    global_importance_df: Optional[pd.DataFrame] = None

    # Local explanation for one observation
    local_shap: Optional[Dict[str, float]] = None     # feature -> SHAP value
    local_lime: Optional[Dict[str, float]] = None     # feature -> LIME weight
    observation_index: int = -1
    prediction: Any = None
    prediction_probability: Optional[float] = None
    predicted_class_label: Optional[str] = None
    base_value: Optional[float] = None
    explained_output_index: Optional[int] = None
    explained_output_label: Optional[str] = None
    explained_output_probability: Optional[float] = None
    output_space: str = ""
    additivity_error: Optional[float] = None
    additivity_verified: Optional[bool] = None

    # Feature names (from fitted preprocessor)
    feature_names: List[str] = field(default_factory=list)

    # Method used
    shap_method: str = ""
    lime_available: bool = False
    error: Optional[str] = None

    def is_successful(self) -> bool:
        return self.error is None and (self.global_importance or self.local_shap)


class XAIExplainer:
    """
    Generates SHAP (primary) and LIME (fallback/supplementary) explanations
    for a fitted scikit-learn Pipeline.

    Correct feature names are extracted from the fitted ColumnTransformer.
    """

    def __init__(self) -> None:
        self._shap_explainer = None
        self._shap_values_cache: Optional[np.ndarray] = None
        self._transformed_X_train: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._label_encoder = None

    def fit(
        self,
        pipeline,
        X_train: pd.DataFrame,
        feature_names: List[str],
        problem_type: str = "binary_classification",
        label_encoder=None,
    ) -> None:
        """
        Fit the SHAP explainer on training data.

        Parameters
        ----------
        pipeline : fitted sklearn Pipeline
        X_train : pd.DataFrame (raw, not yet transformed)
        feature_names : list of feature names after preprocessing
        problem_type : str
        """
        self._feature_names = feature_names
        self._label_encoder = label_encoder
        self._problem_type = problem_type

        # Transform training data using the fitted preprocessor
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            raise ValueError("Pipeline does not have a 'preprocessor' step.")

        X_transformed = preprocessor.transform(X_train)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        self._transformed_X_train = X_transformed

        # Get the model step (last step after preprocessor)
        step_names = list(pipeline.named_steps.keys())
        model_step = step_names[-1]
        model = pipeline.named_steps[model_step]

        self._pipeline = pipeline
        self._model = model
        self._preprocessor = preprocessor

        # ── Choose SHAP explainer ─────────────────────────────────────────────
        self._shap_explainer = None
        self._shap_method = ""

        try:
            import shap

            model_type = type(model).__name__
            tree_models = {
                "RandomForestClassifier", "RandomForestRegressor",
                "DecisionTreeClassifier", "DecisionTreeRegressor",
                "GradientBoostingClassifier", "GradientBoostingRegressor",
                "XGBClassifier", "XGBRegressor",
            }
            linear_models = {
                "LogisticRegression", "LinearRegression",
                "Ridge", "Lasso", "ElasticNet",
            }

            if model_type in tree_models:
                self._shap_explainer = shap.TreeExplainer(model)
                self._shap_method = "TreeExplainer"
            elif model_type in linear_models:
                self._shap_explainer = shap.LinearExplainer(
                    model, X_transformed, feature_perturbation="correlation_dependent"
                )
                self._shap_method = "LinearExplainer"
            else:
                # KernelExplainer is slow — use a background sample
                background = shap.sample(X_transformed, min(50, len(X_transformed)))
                predict_fn = pipeline.predict_proba if hasattr(pipeline, "predict_proba") else pipeline.predict
                self._shap_explainer = shap.KernelExplainer(predict_fn, background)
                self._shap_method = "KernelExplainer"

            logger.info("SHAP explainer initialized: %s", self._shap_method)

        except Exception as exc:
            logger.warning("Could not initialize SHAP explainer: %s", exc)
            self._shap_method = "unavailable"

    def explain_global(
        self,
        X_sample: Optional[pd.DataFrame] = None,
        max_samples: int = 200,
    ) -> ExplanationResult:
        """
        Compute global feature importance via mean |SHAP values|.

        Parameters
        ----------
        X_sample : pd.DataFrame — subset of data for computing SHAP (use training data)
        max_samples : int — cap for KernelExplainer performance
        """
        result = ExplanationResult(feature_names=self._feature_names)
        result.shap_method = self._shap_method

        if self._shap_explainer is None:
            result.error = "SHAP explainer not initialized."
            return self._fallback_global_importance(result)

        try:
            import shap

            X_eval = self._transformed_X_train
            if X_sample is not None:
                X_eval = self._preprocessor.transform(X_sample)
                if hasattr(X_eval, "toarray"):
                    X_eval = X_eval.toarray()

            # Cap samples for performance
            if len(X_eval) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_eval), max_samples, replace=False)
                X_eval = X_eval[idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = self._shap_explainer.shap_values(X_eval)

            # SHAP has used both list[class](rows, features) and ndarray
            # (rows, features, classes) representations across versions.
            # Reduce every non-feature axis, preserving the transformed
            # feature axis rather than accidentally returning one value/class.
            mean_abs_shap = self._global_mean_abs(shap_values, X_eval.shape[1])

            # Map to feature names
            feat_names = self._feature_names
            if len(mean_abs_shap) != len(feat_names):
                feat_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

            result.global_importance = {
                f: float(v) for f, v in zip(feat_names, mean_abs_shap)
            }
            result.global_importance_df = pd.DataFrame({
                "feature": feat_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            self._shap_values_cache = shap_values
            logger.info("Global SHAP computed for %d features.", len(feat_names))

        except Exception as exc:
            logger.error("Global SHAP failed: %s", exc, exc_info=True)
            result.error = str(exc)
            return self._fallback_global_importance(result)

        return result

    def explain_local(
        self,
        X_row: pd.DataFrame,
        observation_index: int,
    ) -> ExplanationResult:
        """
        Compute local SHAP explanation for a single observation.

        Parameters
        ----------
        X_row : pd.DataFrame — single row (not yet transformed)
        observation_index : int — display index
        """
        result = ExplanationResult(
            feature_names=self._feature_names,
            observation_index=observation_index,
            shap_method=self._shap_method,
        )

        # ── Prediction ────────────────────────────────────────────────────────
        try:
            pred_raw = self._pipeline.predict(X_row)[0]
            result.prediction = pred_raw

            class_idx = self._predicted_class_index(pred_raw)
            result.explained_output_index = class_idx
            if hasattr(self._pipeline, "predict_proba"):
                proba = self._pipeline.predict_proba(X_row)[0]
                result.prediction_probability = float(proba[class_idx])
                result.explained_output_probability = result.prediction_probability

            if self._label_encoder is not None and hasattr(self._label_encoder, "classes_"):
                try:
                    result.predicted_class_label = str(self._label_encoder.classes_[int(pred_raw)])
                except (IndexError, ValueError):
                    result.predicted_class_label = str(pred_raw)
            else:
                result.predicted_class_label = str(pred_raw)
            result.explained_output_label = result.predicted_class_label

        except Exception as exc:
            logger.warning("Prediction failed: %s", exc)
            result.error = f"Prediction failed: {exc}"
            return result

        # ── SHAP local explanation ────────────────────────────────────────────
        if self._shap_explainer is not None:
            try:
                X_transformed = self._preprocessor.transform(X_row)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shap_vals = self._shap_explainer.shap_values(X_transformed)

                local_vals, output_idx = self._select_local_output(
                    shap_vals, result.explained_output_index or 0, X_transformed.shape[1]
                )
                result.explained_output_index = output_idx
                result.explained_output_label = self._class_label_for_index(output_idx)
                if hasattr(self._pipeline, "predict_proba"):
                    probabilities = self._pipeline.predict_proba(X_row)[0]
                    result.explained_output_probability = float(probabilities[output_idx])

                feat_names = self._feature_names
                if len(local_vals) != len(feat_names):
                    feat_names = [f"feature_{i}" for i in range(len(local_vals))]

                result.local_shap = {f: float(v) for f, v in zip(feat_names, local_vals)}

                # Base value
                try:
                    ev = self._shap_explainer.expected_value
                    result.base_value = self._select_expected_value(
                        ev, result.explained_output_index or 0
                    )
                except Exception:
                    result.base_value = None

                self._verify_additivity(result, X_row)

            except Exception as exc:
                logger.warning("Local SHAP failed: %s", exc)
                result.error = f"SHAP local failed: {exc}"

        # ── LIME explanation (supplementary or fallback) ───────────────────────
        try:
            lime_result = self._compute_lime(X_row, result)
            if lime_result:
                result.local_lime = lime_result
                result.lime_available = True
        except Exception as exc:
            logger.debug("LIME explanation skipped: %s", exc)

        return result

    def _predicted_class_index(self, prediction: Any) -> int:
        """Map a model label to its probability/SHAP output position."""
        classes = getattr(self._model, "classes_", None)
        if classes is not None:
            matches = np.flatnonzero(np.asarray(classes) == prediction)
            if len(matches):
                return int(matches[0])
        return int(prediction) if isinstance(prediction, (int, np.integer)) else 0

    def _class_label_for_index(self, index: int) -> str:
        classes = getattr(self._model, "classes_", None)
        if classes is not None and 0 <= index < len(classes):
            raw = classes[index]
            if self._label_encoder is not None and hasattr(self._label_encoder, "classes_"):
                return str(self._label_encoder.classes_[int(raw)])
            return str(raw)
        return str(index)

    def _select_local_output(self, shap_values: Any, requested_index: int, n_features: int) -> Tuple[np.ndarray, int]:
        """Normalise legacy and current SHAP output shapes for one row."""
        if isinstance(shap_values, list):
            output_idx = min(requested_index, len(shap_values) - 1)
            return np.asarray(shap_values[output_idx])[0], output_idx
        arr = np.asarray(shap_values)
        if arr.ndim == 1:
            return arr, requested_index
        if arr.ndim == 2:
            # A two-dimensional binary explanation conventionally represents
            # the positive (class-1) output.  Label it accurately rather than
            # claiming it explains a predicted class-0 output.
            output_idx = 1 if "classification" in self._problem_type else 0
            return arr[0], output_idx
        if arr.ndim == 3:
            feature_axis = next((i for i, size in enumerate(arr.shape) if size == n_features and i != 0), 1)
            output_axis = 2 if feature_axis == 1 else 1
            output_idx = min(requested_index, arr.shape[output_axis] - 1)
            return arr[0, :, output_idx] if feature_axis == 1 else arr[0, output_idx, :], output_idx
        raise ValueError(f"Unsupported SHAP local output shape: {arr.shape}")

    @staticmethod
    def _select_expected_value(expected_value: Any, output_index: int) -> float:
        values = np.asarray(expected_value)
        return float(values[min(output_index, values.size - 1)]) if values.ndim else float(values)

    def _verify_additivity(self, result: ExplanationResult, X_row: pd.DataFrame, tolerance: float = 1e-4) -> None:
        """Check base + local SHAP against the same selected model output."""
        if result.base_value is None or not result.local_shap:
            return
        shap_sum = result.base_value + sum(result.local_shap.values())
        candidates: List[Tuple[str, float]] = []
        if hasattr(self._pipeline, "predict_proba") and result.explained_output_index is not None:
            candidates.append(("probability", float(self._pipeline.predict_proba(X_row)[0][result.explained_output_index])))
        if hasattr(self._pipeline, "decision_function"):
            score = np.asarray(self._pipeline.decision_function(X_row)).reshape(-1)
            candidates.append(("decision score", float(score[0] if len(score) == 1 else score[result.explained_output_index or 0])))
        if "regression" in self._problem_type:
            candidates.append(("predicted value", float(self._pipeline.predict(X_row)[0])))
        if candidates:
            result.output_space, output = min(candidates, key=lambda item: abs(shap_sum - item[1]))
            result.additivity_error = float(abs(shap_sum - output))
            result.additivity_verified = result.additivity_error <= tolerance

    @staticmethod
    def _global_mean_abs(shap_values: Any, n_features: int) -> np.ndarray:
        """Return one mean absolute contribution per transformed feature."""
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values)  # classes, rows, features
            return np.abs(arr).mean(axis=(0, 1))

        arr = np.asarray(shap_values)
        if arr.ndim == 1:
            return np.abs(arr)
        if arr.ndim == 2:
            return np.abs(arr).mean(axis=0)
        if arr.ndim == 3:
            # Find the feature dimension defensively; current SHAP is usually
            # rows, features, outputs, but older integrations can transpose it.
            feature_axes = [i for i, size in enumerate(arr.shape) if size == n_features]
            feature_axis = feature_axes[0] if feature_axes else 1
            return np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i != feature_axis))
        raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")

    def _compute_lime(
        self,
        X_row: pd.DataFrame,
        shap_result: ExplanationResult,
    ) -> Optional[Dict[str, float]]:
        """Compute a LIME explanation for a single observation."""
        from lime.lime_tabular import LimeTabularExplainer
        from config.settings import LIME_NUM_FEATURES, LIME_NUM_SAMPLES

        if self._transformed_X_train is None:
            return None

        problem_type = getattr(self, "_problem_type", "binary_classification")
        mode = "regression" if "regression" in problem_type else "classification"

        explainer = LimeTabularExplainer(
            training_data=self._transformed_X_train,
            feature_names=self._feature_names,
            mode=mode,
            random_state=42,
        )

        X_transformed = self._preprocessor.transform(X_row)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        if mode == "classification" and hasattr(self._pipeline, "predict_proba"):
            def predict_fn(x):
                # LIME passes numpy arrays; construct a temporary DataFrame
                return self._model.predict_proba(x)
            exp = explainer.explain_instance(
                X_transformed[0], predict_fn, num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )
        else:
            def predict_fn(x):
                return self._model.predict(x)
            exp = explainer.explain_instance(
                X_transformed[0], predict_fn, num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )

        return {feat: float(weight) for feat, weight in exp.as_list()}

    def _fallback_global_importance(self, result: ExplanationResult) -> ExplanationResult:
        """
        Use model's built-in feature_importances_ or coef_ as fallback.
        """
        try:
            model = self._model
            feat_names = self._feature_names

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    importances = np.abs(coef).mean(axis=0)
                else:
                    importances = np.abs(coef)
            else:
                result.error = (result.error or "") + " No fallback importance available."
                return result

            if len(importances) != len(feat_names):
                feat_names = [f"feature_{i}" for i in range(len(importances))]

            result.global_importance = {
                f: float(v) for f, v in zip(feat_names, importances)
            }
            result.global_importance_df = pd.DataFrame({
                "feature": feat_names,
                "mean_abs_shap": importances,
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            result.shap_method = "model_native_importance (fallback)"
            result.error = None

        except Exception as exc:
            logger.error("Fallback importance also failed: %s", exc)
            result.error = str(exc)

        return result
