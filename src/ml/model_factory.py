"""
src/ml/model_factory.py
────────────────────────
Model registry: maps model name strings to scikit-learn estimator instances.
Supports classification and regression, including auto-selection logic.
"""

import logging
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)

# ── Model registries ──────────────────────────────────────────────────────────

CLASSIFICATION_MODELS: Dict[str, type] = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Extra Trees": ExtraTreesClassifier,
    "Hist Gradient Boosting": HistGradientBoostingClassifier,
}

REGRESSION_MODELS: Dict[str, type] = {
    "Ridge Regression": Ridge,
    "Linear Regression": LinearRegression,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Extra Trees Regressor": ExtraTreesRegressor,
    "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor,
}

ALL_MODELS: Dict[str, type] = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}

# Default hyperparameters ensuring reproducibility
DEFAULT_KWARGS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {"random_state": RANDOM_STATE, "max_iter": 1000},
    "Decision Tree": {"random_state": RANDOM_STATE},
    "Random Forest": {"random_state": RANDOM_STATE, "n_jobs": -1},
    "Extra Trees": {"random_state": RANDOM_STATE, "n_jobs": -1},
    "Hist Gradient Boosting": {"random_state": RANDOM_STATE},
    "Ridge Regression": {},
    "Linear Regression": {},
    "Decision Tree Regressor": {"random_state": RANDOM_STATE},
    "Random Forest Regressor": {"random_state": RANDOM_STATE, "n_jobs": -1},
    "Extra Trees Regressor": {"random_state": RANDOM_STATE, "n_jobs": -1},
    "Hist Gradient Boosting Regressor": {"random_state": RANDOM_STATE},
}

# Pipeline step name (prefix used in param_grid keys)
PIPELINE_STEP_NAMES: Dict[str, str] = {
    "Logistic Regression": "classifier",
    "Decision Tree": "classifier",
    "Random Forest": "classifier",
    "Extra Trees": "classifier",
    "Hist Gradient Boosting": "classifier",
    "Ridge Regression": "regressor",
    "Linear Regression": "regressor",
    "Decision Tree Regressor": "regressor",
    "Random Forest Regressor": "regressor",
    "Extra Trees Regressor": "regressor",
    "Hist Gradient Boosting Regressor": "regressor",
}

# Models that natively handle missing values (no imputation needed, but still allowed)
NATIVE_MISSING_SUPPORT: set = {"Hist Gradient Boosting", "Hist Gradient Boosting Regressor"}

# Models requiring numerical scaling
REQUIRES_SCALING: set = {"Logistic Regression", "Ridge Regression"}


class ModelFactory:
    """Creates scikit-learn estimator instances by name."""

    @staticmethod
    def create(model_name: str, **kwargs: Any) -> BaseEstimator:
        """
        Instantiate a model by name.

        Parameters
        ----------
        model_name : str — must be a key in ALL_MODELS
        **kwargs : any additional parameters to pass to the constructor

        Returns
        -------
        sklearn estimator instance
        """
        if model_name not in ALL_MODELS:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Supported: {list(ALL_MODELS.keys())}"
            )
        cls = ALL_MODELS[model_name]
        defaults = DEFAULT_KWARGS.get(model_name, {})
        final_kwargs = {**defaults, **kwargs}

        # Only pass random_state if the constructor accepts it
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        if "random_state" not in valid_params:
            final_kwargs.pop("random_state", None)
        if "n_jobs" not in valid_params:
            final_kwargs.pop("n_jobs", None)

        estimator = cls(**final_kwargs)
        logger.debug("Created model: %s(%s)", model_name, final_kwargs)
        return estimator

    @staticmethod
    def get_classification_names() -> List[str]:
        return list(CLASSIFICATION_MODELS.keys())

    @staticmethod
    def get_regression_names() -> List[str]:
        return list(REGRESSION_MODELS.keys())

    @staticmethod
    def get_step_name(model_name: str) -> str:
        return PIPELINE_STEP_NAMES.get(model_name, "model")

    @staticmethod
    def is_classification_model(model_name: str) -> bool:
        return model_name in CLASSIFICATION_MODELS

    @staticmethod
    def is_regression_model(model_name: str) -> bool:
        return model_name in REGRESSION_MODELS

    @staticmethod
    def requires_scaling(model_name: str) -> bool:
        return model_name in REQUIRES_SCALING

    @staticmethod
    def supports_missing_natively(model_name: str) -> bool:
        return model_name in NATIVE_MISSING_SUPPORT

    @staticmethod
    def validate_param_grid(
        model_name: str,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """
        Remove invalid hyperparameter keys for the given model.
        Returns a validated (possibly reduced) param_grid.
        """
        if model_name not in ALL_MODELS:
            return param_grid

        cls = ALL_MODELS[model_name]
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        validated: Dict[str, List[Any]] = {}
        for key, values in param_grid.items():
            # Keys should be like "classifier__C"
            parts = key.split("__", 1)
            if len(parts) == 2:
                _, param_name = parts
                if param_name in valid_params:
                    validated[key] = values
                else:
                    logger.debug(
                        "Removed invalid param '%s' for model '%s'", key, model_name
                    )
            else:
                validated[key] = values

        return validated


# ── Default search grids for each model ──────────────────────────────────────

BASELINE_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__max_iter": [1000],
    },
    "Decision Tree": {
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    },
    "Random Forest": {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [None, 10],
        "classifier__max_features": ["sqrt"],
    },
    "Extra Trees": {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [None, 10],
        "classifier__max_features": ["sqrt"],
    },
    "Hist Gradient Boosting": {
        "classifier__max_iter": [100, 200],
        "classifier__learning_rate": [0.1],
        "classifier__max_depth": [None, 6],
    },
    "Ridge Regression": {
        "regressor__alpha": [0.1, 1.0, 10.0],
    },
    "Linear Regression": {},
    "Decision Tree Regressor": {
        "regressor__max_depth": [5, 10, None],
        "regressor__min_samples_split": [2, 5],
    },
    "Random Forest Regressor": {
        "regressor__n_estimators": [100],
        "regressor__max_depth": [None, 10],
        "regressor__max_features": ["sqrt"],
    },
    "Extra Trees Regressor": {
        "regressor__n_estimators": [100],
        "regressor__max_depth": [None, 10],
        "regressor__max_features": ["sqrt"],
    },
    "Hist Gradient Boosting Regressor": {
        "regressor__max_iter": [100, 200],
        "regressor__learning_rate": [0.1],
        "regressor__max_depth": [None, 6],
    },
}
