"""
src/ml/baseline.py
───────────────────
Conventional (non-RAG) baseline hyperparameter search strategy.
Used as the comparison benchmark in baseline-vs-RAG experiments.
"""

import logging
from typing import Any, Dict, List

from src.ml.model_factory import BASELINE_GRIDS, ModelFactory

logger = logging.getLogger(__name__)


class BaselineStrategy:
    """
    Provides conventional, dataset-agnostic hyperparameter search spaces.
    These are fixed search spaces that do NOT use dataset characteristics.
    Used as the baseline in baseline-vs-RAG comparison.
    """

    BASELINE_SCORING: Dict[str, str] = {
        "binary_classification": "f1_weighted",
        "multiclass_classification": "f1_weighted",
        "regression": "neg_root_mean_squared_error",
    }

    def get_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """Return the conventional baseline param grid for a given model."""
        grid = BASELINE_GRIDS.get(model_name, {})
        return ModelFactory.validate_param_grid(model_name, grid)

    def get_cv_scoring(self, problem_type: str) -> str:
        """Return standard CV scoring for the problem type (no imbalance adjustment)."""
        return self.BASELINE_SCORING.get(problem_type, "accuracy")

    def describe(self, model_name: str) -> str:
        """Human-readable description of the baseline strategy."""
        grid = self.get_param_grid(model_name)
        if not grid:
            return (
                f"Baseline: train {model_name} with default hyperparameters. "
                "No dataset-aware adjustments."
            )
        param_str = "; ".join(f"{k}={v}" for k, v in grid.items())
        return (
            f"Baseline: conventional grid search for {model_name} "
            f"without dataset-aware adjustments. Grid: [{param_str}]"
        )
