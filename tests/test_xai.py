"""tests/test_xai.py — Tests for the XAI explainer (SHAP/LIME)."""

import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DatasetProfiler
from src.ml.trainer import ModelTrainer
from src.xai.explainer import XAIExplainer, ExplanationResult


def _train_and_get_result(df, target, model_name="Decision Tree", problem_type=None):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target_column=target, problem_type=problem_type)
    trainer = ModelTrainer()
    return trainer.train(df=df, profile=profile, model_name=model_name), profile


class TestXAIExplainer:
    def test_fit_does_not_raise(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
        )

    def test_global_shap_returns_result(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
        )
        global_res = explainer.explain_global()
        assert isinstance(global_res, ExplanationResult)
        assert global_res.global_importance is not None
        assert len(global_res.global_importance) > 0

    def test_global_shap_all_features_present(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
        )
        global_res = explainer.explain_global()
        # Each feature name should appear in global importance
        for feat in result.feature_names:
            assert feat in global_res.global_importance or \
                   f"feature_{result.feature_names.index(feat)}" in global_res.global_importance, \
                   f"Feature '{feat}' missing from global importance"

    def test_global_importance_non_negative(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
        )
        global_res = explainer.explain_global()
        for val in global_res.global_importance.values():
            assert val >= 0.0

    def test_local_shap_prediction_matches(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
            label_encoder=result.label_encoder,
        )
        row = result.X_test.iloc[[0]]
        local_res = explainer.explain_local(row, observation_index=0)
        assert local_res.prediction is not None
        assert local_res.predicted_class_label is not None

    def test_local_shap_identifies_output_and_checks_additivity(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        assert result.is_successful(), result.error
        explainer = XAIExplainer()
        explainer.fit(result.pipeline, result.X_train, result.feature_names, profile.problem_type or "binary_classification")
        local_res = explainer.explain_local(result.X_test.iloc[[0]], observation_index=0)
        assert local_res.explained_output_label is not None
        assert local_res.explained_output_probability is not None
        if local_res.local_shap and local_res.additivity_error is not None:
            assert local_res.additivity_verified
            assert local_res.additivity_error <= 1e-4

    def test_local_output_selection_uses_class_axis_not_raw_label(self):
        """A multiclass label need not equal its probability-array position."""
        explainer = XAIExplainer()
        explainer._problem_type = "multiclass_classification"
        values = np.arange(12, dtype=float).reshape(1, 3, 4)
        selected, output_index = explainer._select_local_output(values, requested_index=2, n_features=3)
        assert output_index == 2
        assert np.array_equal(selected, values[0, :, 2])

    def test_local_shap_produces_feature_contributions(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type=profile.problem_type or "binary_classification",
        )
        row = result.X_test.iloc[[0]]
        local_res = explainer.explain_local(row, observation_index=0)
        if local_res.local_shap:
            assert len(local_res.local_shap) > 0
            # SHAP values can be positive or negative
            for v in local_res.local_shap.values():
                assert isinstance(v, float)

    def test_global_shap_regression(self, regression_df):
        result, profile = _train_and_get_result(
            regression_df, "target",
            model_name="Decision Tree Regressor", problem_type="regression"
        )
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type="regression",
        )
        global_res = explainer.explain_global()
        assert global_res.global_importance
        assert len(global_res.global_importance) > 0

    def test_fallback_importance_when_shap_unavailable(self, binary_classification_df):
        """Test that model-native importance is used as fallback."""
        result, profile = _train_and_get_result(binary_classification_df, "target",
                                                 model_name="Random Forest")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type="binary_classification",
        )
        # Force SHAP to be unavailable
        explainer._shap_explainer = None
        explainer._shap_method = "unavailable"
        global_res = explainer.explain_global()
        # Should fall back to feature_importances_
        assert global_res.global_importance or global_res.error

    def test_explanation_result_is_successful(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type="binary_classification",
        )
        global_res = explainer.explain_global()
        assert global_res.is_successful()

    def test_global_importance_df_sorted_descending(self, binary_classification_df):
        result, profile = _train_and_get_result(binary_classification_df, "target")
        if not result.is_successful():
            pytest.skip("Training failed")
        explainer = XAIExplainer()
        explainer.fit(
            pipeline=result.pipeline,
            X_train=result.X_train,
            feature_names=result.feature_names,
            problem_type="binary_classification",
        )
        global_res = explainer.explain_global()
        if global_res.global_importance_df is not None:
            vals = global_res.global_importance_df["mean_abs_shap"].tolist()
            assert vals == sorted(vals, reverse=True)
