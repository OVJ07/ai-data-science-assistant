"""
src/ml/trainer.py
──────────────────
Handles full training pipeline including preprocessing, model fitting,
GridSearchCV hyperparameter search, and prediction.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config.settings import CV_FOLDS, RANDOM_STATE, TEST_SIZE
from src.data.profiler import DatasetProfile
from src.ml.evaluation import evaluate_classification, evaluate_regression
from src.ml.model_factory import ModelFactory
from src.ml.preprocessing import PreprocessingBuilder, safe_align

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Complete training result including pipeline, metrics, and metadata."""

    # Pipeline artifacts
    pipeline: Any = None                    # fitted sklearn Pipeline
    preprocessor: Any = None               # fitted ColumnTransformer
    label_encoder: Any = None

    # Split data
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None

    # Predictions
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None

    # Metrics
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    cv_results: Dict[str, Any] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    cv_score_std: Optional[float] = None
    cv_scoring: str = ""

    # Feature info
    feature_names: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # Metadata
    model_name: str = ""
    problem_type: str = ""
    training_time_seconds: float = 0.0
    is_rag_guided: bool = False
    n_param_combinations: int = 0
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    effective_estimator_params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def is_successful(self) -> bool:
        return self.pipeline is not None and self.error is None


class ModelTrainer:
    """
    Trains an ML pipeline with optional GridSearchCV.
    Enforces train/test isolation throughout.
    """

    def __init__(self) -> None:
        self._prep_builder = PreprocessingBuilder()

    def train(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        model_name: str,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_scoring: str = "f1_weighted",
        use_stratified: bool = True,
        numerical_imputer_strategy: str = "median",
        categorical_imputer_strategy: str = "most_frequent",
        scale_numerical: bool = True,
        exclude_columns: Optional[List[str]] = None,
        test_size: float = TEST_SIZE,
        is_rag_guided: bool = False,
        max_ohe_categories: Optional[int] = None,
    ) -> TrainingResult:
        """
        Full training workflow:
        1. Prepare features and target
        2. Train/test split
        3. Build preprocessing pipeline
        4. GridSearchCV on training data
        5. Evaluate on held-out test set

        Parameters
        ----------
        param_grid : dict or None — if None, train with default hyperparameters
        """
        result = TrainingResult(
            model_name=model_name,
            problem_type=profile.problem_type or "unknown",
            is_rag_guided=is_rag_guided,
        )

        start_time = time.time()
        try:
            # ── Prepare target ────────────────────────────────────────────────
            y_series, le = self._prep_builder.prepare_target(
                df, profile.target_column, profile.problem_type or "regression"
            )
            result.label_encoder = le

            X_df, y_aligned = safe_align(df, y_series)

            # ── Train/test split ──────────────────────────────────────────────
            stratify = None
            if use_stratified and "classification" in (profile.problem_type or ""):
                # Do not hand pandas extension arrays to sklearn's indexing
                # internals even before the later CV step.
                stratify = np.asarray(y_aligned)

            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_df,
                y_aligned,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )

            # sklearn's splitters use NumPy index arrays.  Normalise targets at
            # this boundary so pandas extension arrays (notably Arrow strings
            # from current CSV ingestion) are never passed to GridSearchCV.
            y_train_array = np.asarray(y_train)
            y_test_array = np.asarray(y_test)
            result.y_train = y_train_array
            result.y_test = y_test_array

            # ── Build preprocessor ────────────────────────────────────────────
            preprocessor, _ = self._prep_builder.build(
                df,
                profile,
                numerical_imputer_strategy=numerical_imputer_strategy,
                categorical_imputer_strategy=categorical_imputer_strategy,
                scale_numerical=scale_numerical,
                max_ohe_categories=max_ohe_categories,
                exclude_columns=exclude_columns,
            )

            # ── Create model ──────────────────────────────────────────────────
            estimator = ModelFactory.create(model_name)
            step_name = ModelFactory.get_step_name(model_name)

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                (step_name, estimator),
            ])

            # ── Store training data for XAI ───────────────────────────────────
            result.X_train = X_train_df
            result.X_test = X_test_df

            # ── GridSearchCV or simple fit ────────────────────────────────────
            validated_grid = {}
            if param_grid:
                validated_grid = ModelFactory.validate_param_grid(model_name, param_grid)
                # Ensure required default params are in grid keys
                result.n_param_combinations = _count_combinations(validated_grid)

            if validated_grid:
                cv_splitter = self._get_cv_splitter(
                    profile.problem_type, use_stratified, y_train_array
                )

                grid_search = GridSearchCV(
                    pipeline,
                    validated_grid,
                    cv=cv_splitter,
                    scoring=cv_scoring,
                    n_jobs=-1,
                    refit=True,
                    return_train_score=False,
                    verbose=0,
                )
                grid_search.fit(X_train_df, y_train_array)

                result.pipeline = grid_search.best_estimator_
                result.best_params = grid_search.best_params_
                result.best_cv_score = float(grid_search.best_score_)
                result.cv_scoring = cv_scoring

                # Extract CV score std for best params
                best_idx = grid_search.best_index_
                std_key = f"std_test_score"
                if std_key in grid_search.cv_results_:
                    result.cv_score_std = float(grid_search.cv_results_[std_key][best_idx])
                result.cv_results = {
                    k: v.tolist() if hasattr(v, "tolist") else list(v)
                    for k, v in grid_search.cv_results_.items()
                }
            else:
                # No grid search — simple fit
                pipeline.fit(X_train_df, y_train_array)
                result.pipeline = pipeline
                result.best_params = {}
                result.cv_scoring = cv_scoring

            # ── Extract feature names ─────────────────────────────────────────
            fitted_preprocessor = result.pipeline.named_steps["preprocessor"]
            result.preprocessor = fitted_preprocessor
            result.feature_names = self._prep_builder.get_feature_names(fitted_preprocessor)
            result.numerical_features = self._prep_builder.numerical_features
            result.categorical_features = self._prep_builder.categorical_features
            result.preprocessing_config = {
                "numerical_imputer_strategy": numerical_imputer_strategy,
                "categorical_imputer_strategy": categorical_imputer_strategy,
                "scale_numerical": scale_numerical,
                "max_ohe_categories": max_ohe_categories,
                "excluded_columns": sorted(exclude_columns or []),
                "numerical_features": result.numerical_features,
                "categorical_features": result.categorical_features,
                "transformed_feature_count": len(result.feature_names),
            }
            fitted_estimator = result.pipeline.named_steps[step_name]
            result.effective_estimator_params = {
                key: value for key, value in fitted_estimator.get_params(deep=False).items()
                if key in {param.split("__", 1)[-1] for param in result.best_params}
            }

            # ── Evaluate on test set ──────────────────────────────────────────
            y_pred = result.pipeline.predict(X_test_df)
            result.y_pred = y_pred

            y_proba = None
            if hasattr(result.pipeline, "predict_proba"):
                try:
                    y_proba = result.pipeline.predict_proba(X_test_df)
                    result.y_proba = y_proba
                except Exception:
                    pass

            is_binary = profile.problem_type == "binary_classification"
            class_names = None
            if le and le.classes_ is not None:
                class_names = [str(c) for c in le.classes_]

            if "classification" in (profile.problem_type or ""):
                result.test_metrics = evaluate_classification(
                    result.y_test, y_pred, y_proba, class_names, is_binary
                )
            else:
                result.test_metrics = evaluate_regression(result.y_test, y_pred)

        except Exception as exc:
            logger.error("Training failed for %s: %s", model_name, exc, exc_info=True)
            result.error = str(exc)

        result.training_time_seconds = time.time() - start_time
        logger.info(
            "Training '%s' completed in %.2fs (error=%s)",
            model_name,
            result.training_time_seconds,
            result.error,
        )
        return result

    @staticmethod
    def _get_cv_splitter(
        problem_type: Optional[str],
        use_stratified: bool,
        y_train: np.ndarray,
    ):
        """Return the appropriate CV splitter for the problem type."""
        if use_stratified and "classification" in (problem_type or ""):
            # Check if stratification is feasible
            from collections import Counter
            counts = Counter(y_train)
            min_count = min(counts.values())
            if min_count >= CV_FOLDS:
                return StratifiedKFold(
                    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
                )
            else:
                # Reduce folds if class count is too small
                safe_folds = max(2, min_count)
                logger.warning(
                    "Minority class has %d samples; reducing CV folds to %d.",
                    min_count,
                    safe_folds,
                )
                return StratifiedKFold(
                    n_splits=safe_folds, shuffle=True, random_state=RANDOM_STATE
                )
        return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _count_combinations(param_grid: Dict[str, List[Any]]) -> int:
    """Count the total number of hyperparameter combinations."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total
