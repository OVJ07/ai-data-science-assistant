"""
src/ml/comparison.py
─────────────────────
Fair reproducible comparison between baseline and RAG-guided training.
Uses identical train/test splits and comparable CV budgets.

Bug fix: Each training run uses its own ModelTrainer instance to prevent
shared state (PreprocessingBuilder label encoder, etc.) from bleeding
between baseline and RAG-guided runs.
"""

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.profiler import DatasetProfile
from src.ml.baseline import BaselineStrategy
from src.ml.diagnostics import (
    DatasetFingerprint,
    compute_config_hash,
    compute_prediction_agreement,
    compute_prediction_hash,
)
from src.ml.trainer import ModelTrainer, TrainingResult
from src.rag.recommendation import SearchSpaceConfig

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Full comparison between baseline and RAG-guided training."""

    model_name: str = ""
    problem_type: str = ""

    baseline_result: Optional[TrainingResult] = None
    rag_result: Optional[TrainingResult] = None

    # Differences (RAG - baseline), with sign convention documented
    metric_differences: Dict[str, float] = field(default_factory=dict)
    metric_differences_pct: Dict[str, float] = field(default_factory=dict)

    # Interpretation
    improvement_summary: str = ""
    honest_assessment: str = ""

    # Metadata
    dataset_fingerprint: str = ""
    baseline_config_hash: str = ""
    rag_config_hash: str = ""
    baseline_prediction_hash: str = ""
    rag_prediction_hash: str = ""
    baseline_grid_size: int = 0
    rag_grid_size: int = 0
    test_indices_match: bool = False
    predictions_identical: Optional[bool] = None
    prediction_agreement: Optional[float] = None
    max_prediction_difference: Optional[float] = None
    configurations_differ: bool = False
    baseline_configuration: Dict[str, Any] = field(default_factory=dict)
    rag_configuration: Dict[str, Any] = field(default_factory=dict)
    rag_guidance_notes: List[str] = field(default_factory=list)
    comparison_timestamp: str = ""
    random_state_used: int = 0


class ModelComparator:
    """
    Runs baseline-vs-RAG comparison ensuring identical experimental conditions:
    - Same dataset
    - Same target
    - Same train/test split (controlled by RANDOM_STATE in both trainers)
    - Comparable CV budgets
    - Identical evaluation metrics
    - INDEPENDENT trainer instances to prevent shared state
    """

    def __init__(self) -> None:
        self._baseline_strategy = BaselineStrategy()

    def compare(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        model_name: str,
        rag_search_config: SearchSpaceConfig,
        exclude_columns: Optional[List[str]] = None,
        numerical_imputer_strategy: str = "median",
        categorical_imputer_strategy: str = "most_frequent",
        align_cv_scoring: bool = True,
    ) -> ComparisonResult:
        """
        Run baseline and RAG-guided training on the same data.

        Each run uses a fresh ModelTrainer to prevent shared PreprocessingBuilder
        state from affecting results.

        Parameters
        ----------
        rag_search_config : SearchSpaceConfig from the recommendation engine
        align_cv_scoring : bool (default True)
            Enforces identical primary CV evaluation metric between baseline and RAG
            to eliminate confounding metric mismatches.
        """
        from config.settings import MAX_ONE_HOT_CATEGORIES, RANDOM_STATE

        result = ComparisonResult(
            model_name=model_name,
            problem_type=profile.problem_type or "unknown",
            comparison_timestamp=datetime.datetime.now().isoformat(),
            random_state_used=RANDOM_STATE,
            dataset_fingerprint=DatasetFingerprint.from_profile(profile).fingerprint_hash,
        )

        # ── Determine CV scoring ──────────────────────────────────────────────
        raw_baseline_scoring = self._baseline_strategy.get_cv_scoring(
            profile.problem_type or ""
        )
        effective_baseline_scoring = (
            rag_search_config.cv_scoring
            if align_cv_scoring and rag_search_config.cv_scoring
            else raw_baseline_scoring
        )
        effective_rag_scoring = rag_search_config.cv_scoring or raw_baseline_scoring

        # ── Preprocessing isolation ───────────────────────────────────────────
        baseline_num_imputer = numerical_imputer_strategy
        baseline_cat_imputer = categorical_imputer_strategy
        baseline_max_ohe = None  # Baseline does not cap one-hot categories

        rag_preprocessing = rag_search_config.preprocessing_config
        rag_num_imputer = rag_preprocessing.get("numerical_imputer_strategy", baseline_num_imputer)
        rag_cat_imputer = rag_preprocessing.get("categorical_imputer_strategy", baseline_cat_imputer)
        rag_max_ohe = rag_preprocessing.get("max_ohe_categories")

        baseline_grid = self._baseline_strategy.get_param_grid(model_name)

        result.baseline_configuration = {
            "param_grid": baseline_grid,
            "cv_scoring": effective_baseline_scoring,
            "numerical_imputer_strategy": baseline_num_imputer,
            "categorical_imputer_strategy": baseline_cat_imputer,
            "max_ohe_categories": baseline_max_ohe,
        }
        result.rag_configuration = {
            "param_grid": rag_search_config.param_grid,
            "cv_scoring": effective_rag_scoring,
            "numerical_imputer_strategy": rag_num_imputer,
            "categorical_imputer_strategy": rag_cat_imputer,
            "max_ohe_categories": rag_max_ohe,
        }
        result.baseline_config_hash = compute_config_hash(
            baseline_grid, result.baseline_configuration, effective_baseline_scoring
        )
        result.rag_config_hash = compute_config_hash(
            rag_search_config.param_grid, result.rag_configuration, effective_rag_scoring
        )
        result.rag_guidance_notes = list(rag_search_config.application_notes) + list(rag_search_config.reasoning)
        result.configurations_differ = result.baseline_config_hash != result.rag_config_hash

        # ── Baseline training (fresh trainer) ─────────────────────────────────
        logger.info("Running BASELINE training for: %s", model_name)
        baseline_trainer = ModelTrainer()
        result.baseline_result = baseline_trainer.train(
            df=df,
            profile=profile,
            model_name=model_name,
            param_grid=baseline_grid,
            cv_scoring=effective_baseline_scoring,
            use_stratified=True,
            numerical_imputer_strategy=baseline_num_imputer,
            categorical_imputer_strategy=baseline_cat_imputer,
            exclude_columns=exclude_columns,
            is_rag_guided=False,
        )

        # ── RAG-guided training (fresh trainer) ───────────────────────────────
        logger.info("Running RAG-GUIDED training for: %s", model_name)
        rag_trainer = ModelTrainer()
        result.rag_result = rag_trainer.train(
            df=df,
            profile=profile,
            model_name=model_name,
            param_grid=rag_search_config.param_grid,
            cv_scoring=effective_rag_scoring,
            use_stratified=rag_search_config.use_stratified,
            numerical_imputer_strategy=rag_num_imputer,
            categorical_imputer_strategy=rag_cat_imputer,
            exclude_columns=exclude_columns,
            is_rag_guided=True,
            max_ohe_categories=rag_max_ohe,
        )

        # Report what GridSearchCV actually evaluated, not merely the size of
        # the requested grids. This also makes invalid/filtered parameters
        # visible to callers.
        result.baseline_grid_size = result.baseline_result.n_param_combinations
        result.rag_grid_size = result.rag_result.n_param_combinations
        self._compare_predictions(result)

        # ── Compute differences ───────────────────────────────────────────────
        self._compute_differences(result)

        return result

    @staticmethod
    def _compare_predictions(result: ComparisonResult) -> None:
        """Record whether independently fitted runs made the same predictions."""
        baseline, rag = result.baseline_result, result.rag_result
        if not (baseline and rag and baseline.is_successful() and rag.is_successful()):
            return
        result.baseline_prediction_hash = compute_prediction_hash(baseline.y_pred)
        result.rag_prediction_hash = compute_prediction_hash(rag.y_pred)
        result.test_indices_match = bool(
            baseline.X_test.index.equals(rag.X_test.index)
            and np.array_equal(baseline.y_test, rag.y_test)
        )
        if not result.test_indices_match:
            logger.error("Comparison runs did not use identical held-out observations.")
            return
        b_pred, r_pred = np.asarray(baseline.y_pred), np.asarray(rag.y_pred)
        if b_pred.shape != r_pred.shape:
            return
        if "classification" in result.problem_type:
            result.prediction_agreement = float(np.mean(b_pred == r_pred))
            result.predictions_identical = bool(np.array_equal(b_pred, r_pred))
        else:
            result.max_prediction_difference = float(np.max(np.abs(b_pred - r_pred))) if len(b_pred) else 0.0
            result.predictions_identical = bool(np.array_equal(b_pred, r_pred))

    def _compute_differences(self, result: ComparisonResult) -> None:
        """Compute metric differences and generate honest assessment."""
        b_ok = result.baseline_result and result.baseline_result.is_successful()
        r_ok = result.rag_result and result.rag_result.is_successful()

        if not (b_ok and r_ok):
            failed = []
            if not b_ok:
                failed.append(f"baseline ({getattr(result.baseline_result, 'error', 'unknown')})")
            if not r_ok:
                failed.append(f"RAG-guided ({getattr(result.rag_result, 'error', 'unknown')})")
            result.honest_assessment = (
                f"Comparison could not be completed. Failed: {', '.join(failed)}"
            )
            return

        if not result.test_indices_match:
            result.honest_assessment = "Comparison invalid: runs did not use identical held-out observations."
            return

        b_metrics = result.baseline_result.test_metrics
        r_metrics = result.rag_result.test_metrics

        numeric_keys = [
            k for k in b_metrics
            if isinstance(b_metrics.get(k), (int, float))
            and isinstance(r_metrics.get(k), (int, float))
            and b_metrics[k] is not None
            and r_metrics[k] is not None
            and k not in ("confusion_matrix",)
        ]

        for key in numeric_keys:
            b_val = float(b_metrics[key])
            r_val = float(r_metrics[key])
            diff = r_val - b_val
            result.metric_differences[key] = round(diff, 6)
            # Relative change (avoid div by zero)
            if abs(b_val) > 1e-9:
                result.metric_differences_pct[key] = round(diff / abs(b_val) * 100, 2)

        # Primary metric for summary
        primary = _primary_metric(result.problem_type)
        diff_primary = result.metric_differences.get(primary)

        if diff_primary is not None:
            b_val = b_metrics.get(primary)
            r_val = r_metrics.get(primary)
            pct = result.metric_differences_pct.get(primary, 0.0)

            if not result.configurations_differ:
                result.honest_assessment = (
                    "No RAG-specific training configuration was applicable to this dataset. "
                    "Baseline and RAG-guided therefore used the same search space, scoring, and preprocessing; "
                    "identical metrics are expected and are not evidence of an independent RAG improvement. "
                    "Choose a dataset with an applicable issue (for example material imbalance, high cardinality, "
                    "or a RAG-adjusted search space) to evaluate a treatment difference."
                )
                result.improvement_summary = result.honest_assessment
                return

            # Check if search spaces differ enough to expect a difference
            same_grid = result.baseline_grid_size == result.rag_grid_size
            same_scoring = (
                result.baseline_result.cv_scoring == result.rag_result.cv_scoring
            )
            configs_note = ""
            if same_grid and same_scoring:
                configs_note = (
                    " Note: baseline and RAG-guided used identical grid sizes and scoring — "
                    "performance differences, if any, reflect stochastic variation."
                )

            if abs(diff_primary) < 0.005:
                prediction_note = (
                    " Both independently fitted pipelines produced identical held-out predictions."
                    if result.predictions_identical else
                    f" Prediction agreement was {result.prediction_agreement:.1%}."
                    if result.prediction_agreement is not None else ""
                )
                result.honest_assessment = (
                    f"RAG-guided and baseline configurations achieved similar {primary} "
                    f"(baseline={b_val:.4f}, RAG={r_val:.4f}, delta={diff_primary:+.6f}, "
                    f"{pct:+.2f}%). "
                    "Dataset characteristics did not lead to a measurable performance difference "
                    f"on this train/test split.{prediction_note}{configs_note}"
                )
            elif diff_primary > 0:
                result.honest_assessment = (
                    f"RAG-guided configuration outperformed baseline on {primary} "
                    f"(baseline={b_val:.4f}, RAG={r_val:.4f}, delta={diff_primary:+.6f}, "
                    f"{pct:+.2f}%). "
                    "Caution: this is a single train/test split — differences may not "
                    f"generalise across different datasets or splits.{configs_note}"
                )
            else:
                result.honest_assessment = (
                    f"Baseline configuration outperformed RAG-guided on {primary} "
                    f"(baseline={b_val:.4f}, RAG={r_val:.4f}, delta={diff_primary:+.6f}, "
                    f"{pct:+.2f}%). "
                    "RAG-guided hyperparameters do not guarantee improvement; "
                    "empirical validation on this dataset is the final arbiter. "
                    f"The RAG approach adapts to dataset characteristics, which may not always "
                    f"benefit this specific data.{configs_note}"
                )
        else:
            result.honest_assessment = "Primary metric not available for comparison."

        result.improvement_summary = result.honest_assessment


def _grid_size(grid: Dict[str, list]) -> int:
    size = 1
    for v in grid.values():
        size *= max(len(v), 1)
    return size


def _primary_metric(problem_type: str) -> str:
    if "classification" in problem_type:
        return "balanced_accuracy" if "binary" in problem_type else "f1_weighted"
    return "rmse"
