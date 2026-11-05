"""
src/ml/auto_selector.py
────────────────────────
Automatic ML estimator selection based on dataset characteristics.
Selects appropriate candidate models, scores them via cross-validation,
and returns the best model name with full selection metadata.

Design principles:
- Never uses the test set for selection.
- All scoring happens inside cross-validation on training data only.
- Selection logic is transparent and logged.
- Falls back to a sensible default if CV fails.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from config.settings import CV_FOLDS, RANDOM_STATE
from src.data.profiler import DatasetProfile
from src.ml.model_factory import (
    BASELINE_GRIDS,
    ModelFactory,
    REQUIRES_SCALING,
)
from src.ml.preprocessing import PreprocessingBuilder, safe_align

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Metadata for one model candidate during selection."""
    model_name: str
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    cv_scoring: str = ""
    training_time_s: float = 0.0
    excluded_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def eligible(self) -> bool:
        return self.excluded_reason is None


@dataclass
class AutoSelectionResult:
    """Full result of automatic model selection."""
    selected_model: str = ""
    selection_reason: str = ""
    cv_scoring: str = ""
    candidates: List[ModelCandidate] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    selection_time_s: float = 0.0
    problem_type: str = ""
    n_rows_train: int = 0


def _get_cv_scoring(profile: DatasetProfile) -> str:
    """Determine appropriate CV scoring metric from dataset profile."""
    pt = profile.problem_type or ""
    if "binary_classification" in pt:
        return "balanced_accuracy" if profile.is_imbalanced else "f1_weighted"
    if "multiclass_classification" in pt:
        return "f1_weighted"
    return "neg_root_mean_squared_error"


def _eligible_models(profile: DatasetProfile) -> List[Tuple[str, Optional[str]]]:
    """
    Return list of (model_name, exclusion_reason_or_None) tuples for the
    given dataset profile.

    Exclusion criteria (transparent):
    - Problem type mismatch
    - Dataset too small for complex ensemble
    - Too many categories for linear model without encoding cap
    """
    pt = profile.problem_type or ""
    is_classification = "classification" in pt
    n = profile.n_rows
    has_many_cats = bool(profile.high_cardinality_features)

    if is_classification:
        candidates = ModelFactory.get_classification_names()
    else:
        candidates = ModelFactory.get_regression_names()

    result = []
    for name in candidates:
        reason = None
        # Exclude Hist Gradient Boosting on very small datasets (poor fit)
        if "Hist Gradient Boosting" in name and n < 150:
            reason = f"Dataset too small ({n} rows) for gradient boosting"
        # Logistic Regression: note that it works but may be slow with many OHE features
        if name in ("Logistic Regression", "Ridge Regression") and has_many_cats and n < 100:
            reason = f"High-cardinality features with small dataset may cause convergence issues"
        result.append((name, reason))

    return result


def select_model_auto(
    df: pd.DataFrame,
    profile: DatasetProfile,
    exclude_columns: Optional[List[str]] = None,
    numerical_imputer_strategy: str = "median",
    categorical_imputer_strategy: str = "most_frequent",
    max_candidates: int = 4,
    cv_folds: int = CV_FOLDS,
) -> AutoSelectionResult:
    """
    Automatically select the best ML estimator via cross-validation.

    Uses only the training partition (no test set leakage).
    Returns an AutoSelectionResult with full audit trail.

    Parameters
    ----------
    df : pd.DataFrame — full dataset (will be split internally)
    profile : DatasetProfile
    exclude_columns : columns to drop before training
    numerical_imputer_strategy : imputation for numerical columns
    categorical_imputer_strategy : imputation for categorical columns
    max_candidates : max number of models to actually CV-evaluate
    cv_folds : number of cross-validation folds
    """
    t0 = time.time()
    result = AutoSelectionResult(problem_type=profile.problem_type or "")
    cv_scoring = _get_cv_scoring(profile)
    result.cv_scoring = cv_scoring

    # ── Build training data (80% as used by trainer) ──────────────────────────
    from sklearn.model_selection import train_test_split
    from src.ml.preprocessing import safe_align

    prep_builder = PreprocessingBuilder()
    try:
        y_series, le = prep_builder.prepare_target(
            df, profile.target_column, profile.problem_type or "regression"
        )
    except Exception as exc:
        result.selected_model = _default_model(profile)
        result.fallback_used = True
        result.fallback_reason = f"Target preparation failed: {exc}"
        result.selection_reason = "Fallback: target preparation error"
        result.selection_time_s = time.time() - t0
        return result

    X_df, y_aligned = safe_align(df, y_series)
    stratify = y_aligned.values if "classification" in (profile.problem_type or "") else None

    try:
        X_train_df, _, y_train, _ = train_test_split(
            X_df, y_aligned,
            test_size=0.2, random_state=RANDOM_STATE, stratify=stratify,
        )
    except Exception as exc:
        result.selected_model = _default_model(profile)
        result.fallback_used = True
        result.fallback_reason = f"Train/test split failed: {exc}"
        result.selection_time_s = time.time() - t0
        return result

    result.n_rows_train = len(X_train_df)

    # ── Build preprocessor once (fitted on training data inside CV) ───────────
    try:
        preprocessor, _ = prep_builder.build(
            df, profile,
            numerical_imputer_strategy=numerical_imputer_strategy,
            categorical_imputer_strategy=categorical_imputer_strategy,
            scale_numerical=False,  # handled per-model below
            exclude_columns=exclude_columns,
        )
    except Exception as exc:
        result.selected_model = _default_model(profile)
        result.fallback_used = True
        result.fallback_reason = f"Preprocessor build failed: {exc}"
        result.selection_time_s = time.time() - t0
        return result

    # ── Determine eligible candidates ─────────────────────────────────────────
    eligibility = _eligible_models(profile)
    candidates: List[ModelCandidate] = []

    for name, excl_reason in eligibility:
        cand = ModelCandidate(model_name=name, cv_scoring=cv_scoring,
                              excluded_reason=excl_reason)
        candidates.append(cand)

    result.candidates = candidates
    eligible_names = [c.model_name for c in candidates if c.eligible]

    # Limit candidates for speed
    eligible_names = eligible_names[:max_candidates]

    # ── CV scoring ────────────────────────────────────────────────────────────
    is_classification = "classification" in (profile.problem_type or "")
    from collections import Counter
    counts = Counter(y_train.values)
    min_count = min(counts.values()) if counts else cv_folds

    safe_folds = min(cv_folds, max(2, min_count))
    if is_classification:
        cv_splitter = StratifiedKFold(
            n_splits=safe_folds, shuffle=True, random_state=RANDOM_STATE
        )
    else:
        cv_splitter = KFold(n_splits=safe_folds, shuffle=True, random_state=RANDOM_STATE)

    scored = []
    for name in eligible_names:
        cand = next(c for c in candidates if c.model_name == name)
        t_start = time.time()
        try:
            needs_scale = name in REQUIRES_SCALING
            # Rebuild preprocessor with correct scaling
            pb = PreprocessingBuilder()
            pre, _ = pb.build(
                df, profile,
                numerical_imputer_strategy=numerical_imputer_strategy,
                categorical_imputer_strategy=categorical_imputer_strategy,
                scale_numerical=needs_scale,
                exclude_columns=exclude_columns,
            )
            estimator = ModelFactory.create(name)
            step_name = ModelFactory.get_step_name(name)
            pipeline = Pipeline(steps=[("preprocessor", pre), (step_name, estimator)])

            scores = cross_val_score(
                pipeline, X_train_df, y_train.values,
                cv=cv_splitter, scoring=cv_scoring, n_jobs=1,
            )
            cand.cv_score = float(np.mean(scores))
            cand.cv_std = float(np.std(scores))
            cand.training_time_s = time.time() - t_start
            scored.append(cand)
            logger.info(
                "AutoSelect CV: %s => %.4f +/- %.4f (%.1fs)",
                name, cand.cv_score, cand.cv_std, cand.training_time_s,
            )
        except Exception as exc:
            cand.error = str(exc)
            cand.training_time_s = time.time() - t_start
            logger.warning("AutoSelect CV failed for %s: %s", name, exc)

    # ── Pick best ─────────────────────────────────────────────────────────────
    # For RMSE (neg), higher is better. For accuracy/f1, higher is better.
    scored_with_score = [c for c in scored if c.cv_score is not None]

    if not scored_with_score:
        result.selected_model = _default_model(profile)
        result.fallback_used = True
        result.fallback_reason = "All candidates failed cross-validation"
        result.selection_reason = f"Fallback default: {result.selected_model}"
    else:
        best = max(scored_with_score, key=lambda c: c.cv_score)
        result.selected_model = best.model_name
        result.selection_reason = (
            f"Best CV {cv_scoring}: {best.model_name} "
            f"({best.cv_score:.4f} +/- {best.cv_std:.4f})"
        )

    result.selection_time_s = time.time() - t0
    logger.info(
        "AutoSelect complete: selected='%s' in %.1fs",
        result.selected_model, result.selection_time_s,
    )
    return result


def _default_model(profile: DatasetProfile) -> str:
    """Return a safe default model name."""
    if "classification" in (profile.problem_type or ""):
        return "Logistic Regression"
    return "Ridge Regression"
