"""tests/test_trainer.py — Tests for ML pipeline training and comparison."""

import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DatasetProfiler
from src.ml.trainer import ModelTrainer, TrainingResult
from src.ml.baseline import BaselineStrategy
from src.ml.comparison import ModelComparator
from src.rag.recommendation import SearchSpaceConfig


# ── ModelTrainer tests ────────────────────────────────────────────────────────

class TestModelTrainer:
    def test_train_binary_no_grid(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="Logistic Regression",
            param_grid=None,
        )
        assert result.is_successful()
        assert result.pipeline is not None
        assert result.y_pred is not None
        assert len(result.y_pred) == len(result.y_test)

    def test_train_binary_with_grid(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        param_grid = {
            "classifier__C": [0.1, 1.0],
            "classifier__max_iter": [1000],
        }
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="Logistic Regression",
            param_grid=param_grid,
        )
        assert result.is_successful()
        assert result.best_params
        assert result.best_cv_score is not None
        assert result.n_param_combinations == 2

    def test_adult_like_top_k_ohe_logistic_cv_end_to_end(self, adult_income_sample_df):
        """Regression test for pandas string targets in CV plus RAG top-K OHE."""
        profile = DatasetProfiler().profile(adult_income_sample_df, target_column="income")
        assert "native.country" in profile.high_cardinality_features

        result = ModelTrainer().train(
            df=adult_income_sample_df,
            profile=profile,
            model_name="Logistic Regression",
            param_grid={"classifier__C": [0.5, 1.0], "classifier__max_iter": [1000]},
            cv_scoring="f1_weighted",
            numerical_imputer_strategy="median",
            categorical_imputer_strategy="most_frequent",
            scale_numerical=True,
            max_ohe_categories=15,
            is_rag_guided=True,
        )
        assert result.is_successful(), result.error
        assert result.best_cv_score is not None
        assert result.n_param_combinations == 2
        assert len(result.y_pred) == len(result.y_test)
        assert isinstance(result.y_train, np.ndarray)

        encoder = result.preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        assert encoder.max_categories == 15
        assert encoder.handle_unknown == "infrequent_if_exist"
        unseen = result.X_test.iloc[[0]].copy()
        unseen["native.country"] = "Neverland"
        assert result.pipeline.predict(unseen).shape == (1,)

    @pytest.mark.parametrize("model_name", ["Decision Tree", "Random Forest"])
    def test_adult_like_string_target_trains_other_classifiers(
        self, adult_income_sample_df, model_name
    ):
        profile = DatasetProfiler().profile(adult_income_sample_df, target_column="income")
        result = ModelTrainer().train(
            df=adult_income_sample_df,
            profile=profile,
            model_name=model_name,
            max_ohe_categories=15,
        )
        assert result.is_successful(), result.error
        assert len(result.y_pred) == len(result.y_test)

    def test_comparison_uses_independent_runs_and_same_test_rows(self, imbalanced_classification_df):
        profile = DatasetProfiler().profile(imbalanced_classification_df, target_column="target")
        rag_config = SearchSpaceConfig(
            model_name="Logistic Regression",
            param_grid={"classifier__C": [0.1, 1.0], "classifier__class_weight": [None, "balanced"]},
            cv_scoring="balanced_accuracy",
            use_stratified=True,
            is_rag_guided=True,
        )
        comparison = ModelComparator().compare(
            imbalanced_classification_df, profile, "Logistic Regression", rag_config
        )
        assert comparison.baseline_result is not comparison.rag_result
        assert comparison.baseline_result.pipeline is not comparison.rag_result.pipeline
        assert comparison.test_indices_match
        assert comparison.baseline_grid_size == comparison.baseline_result.n_param_combinations
        assert comparison.rag_grid_size == comparison.rag_result.n_param_combinations
        assert comparison.prediction_agreement is not None
        assert comparison.predictions_identical is not None

    def test_comparison_labels_identical_configurations_honestly(self, binary_classification_df):
        profile = DatasetProfiler().profile(binary_classification_df, target_column="target")
        baseline = BaselineStrategy()
        rag_config = SearchSpaceConfig(
            model_name="Logistic Regression",
            param_grid=baseline.get_param_grid("Logistic Regression"),
            cv_scoring=baseline.get_cv_scoring(profile.problem_type or ""),
            use_stratified=True,
            is_rag_guided=True,
        )
        comparison = ModelComparator().compare(
            binary_classification_df, profile, "Logistic Regression", rag_config
        )
        assert not comparison.configurations_differ
        assert "No RAG-specific training configuration" in comparison.honest_assessment

    def test_train_regression(self, regression_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(regression_df, target_column="target",
                                   problem_type="regression")
        trainer = ModelTrainer()
        result = trainer.train(
            df=regression_df,
            profile=profile,
            model_name="Random Forest Regressor",
        )
        assert result.is_successful()
        assert "rmse" in result.test_metrics
        assert "r2" in result.test_metrics
        assert result.test_metrics["rmse"] >= 0

    def test_train_with_missing_values(self, df_with_missing):
        profiler = DatasetProfiler()
        profile = profiler.profile(df_with_missing, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=df_with_missing,
            profile=profile,
            model_name="Decision Tree",
        )
        assert result.is_successful(), result.error

    def test_train_imbalanced_uses_stratified_cv(self, imbalanced_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(imbalanced_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=imbalanced_classification_df,
            profile=profile,
            model_name="Logistic Regression",
            use_stratified=True,
        )
        assert result.is_successful(), result.error

    def test_train_classification_metrics(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="Random Forest",
        )
        assert result.is_successful()
        metrics = result.test_metrics
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "balanced_accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_feature_names_extracted(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="Logistic Regression",
        )
        assert result.is_successful()
        assert len(result.feature_names) > 0
        assert all(isinstance(f, str) for f in result.feature_names)

    def test_train_invalid_model_fails_gracefully(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="NonExistent Model XYZ",
        )
        assert not result.is_successful()
        assert result.error is not None

    def test_rag_guided_flag_propagated(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        trainer = ModelTrainer()
        result = trainer.train(
            df=binary_classification_df,
            profile=profile,
            model_name="Decision Tree",
            is_rag_guided=True,
        )
        assert result.is_rag_guided is True

    def test_multiclass_training(self, multiclass_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(multiclass_df, target_column="target",
                                   problem_type="multiclass_classification")
        trainer = ModelTrainer()
        result = trainer.train(
            df=multiclass_df,
            profile=profile,
            model_name="Decision Tree",
        )
        assert result.is_successful(), result.error
        assert "f1_weighted" in result.test_metrics


# ── BaselineStrategy tests ────────────────────────────────────────────────────

class TestBaselineStrategy:
    def test_returns_grid_for_known_model(self):
        bs = BaselineStrategy()
        grid = bs.get_param_grid("Logistic Regression")
        assert isinstance(grid, dict)

    def test_returns_empty_grid_for_linear_regression(self):
        bs = BaselineStrategy()
        grid = bs.get_param_grid("Linear Regression")
        assert isinstance(grid, dict)

    def test_cv_scoring_classification(self):
        bs = BaselineStrategy()
        scoring = bs.get_cv_scoring("binary_classification")
        assert isinstance(scoring, str)

    def test_cv_scoring_regression(self):
        bs = BaselineStrategy()
        scoring = bs.get_cv_scoring("regression")
        assert "rmse" in scoring or "mean_squared" in scoring

    def test_describe_returns_string(self):
        bs = BaselineStrategy()
        desc = bs.describe("Random Forest")
        assert isinstance(desc, str)
        assert len(desc) > 0


# ── ModelComparator tests ─────────────────────────────────────────────────────

class TestModelComparator:
    def _make_search_config(self, profile) -> SearchSpaceConfig:
        return SearchSpaceConfig(
            model_name="Decision Tree",
            param_grid={"classifier__max_depth": [3, 5]},
            cv_scoring="f1_weighted",
            use_stratified=True,
            reasoning=["test reason"],
            is_rag_guided=True,
        )

    def test_compare_runs_both(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        comparator = ModelComparator()
        config = self._make_search_config(profile)
        result = comparator.compare(
            df=binary_classification_df,
            profile=profile,
            model_name="Decision Tree",
            rag_search_config=config,
        )
        assert result.baseline_result is not None
        assert result.rag_result is not None

    def test_compare_produces_honest_assessment(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        comparator = ModelComparator()
        config = self._make_search_config(profile)
        result = comparator.compare(
            df=binary_classification_df,
            profile=profile,
            model_name="Decision Tree",
            rag_search_config=config,
        )
        assert isinstance(result.honest_assessment, str)
        assert len(result.honest_assessment) > 0

    def test_compare_metric_differences_computed(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        comparator = ModelComparator()
        config = self._make_search_config(profile)
        result = comparator.compare(
            df=binary_classification_df,
            profile=profile,
            model_name="Decision Tree",
            rag_search_config=config,
        )
        assert isinstance(result.metric_differences, dict)
