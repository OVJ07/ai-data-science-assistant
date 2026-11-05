"""
src/data/profiler.py
────────────────────
Dataset profiling: produces a DatasetProfile dataclass that captures all
relevant dataset characteristics needed by the RAG query generator and
recommendation engine.

Improved feature classification:
- Distinguishes true ID columns (integer sequence), high-cardinality
  categoricals, text/name fields, and potentially informative features.
- Does not silently remove columns with potentially useful information.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    CLASS_IMBALANCE_RATIO,
    HIGH_CARDINALITY_THRESHOLD,
    ID_LIKE_UNIQUE_RATIO,
    MIN_SAMPLES_FOR_ML,
)

logger = logging.getLogger(__name__)

# ── Name/text column heuristics ───────────────────────────────────────────────
# Columns whose names suggest free-form text or names — not true identifiers
TEXT_LIKE_KEYWORDS = {
    "name", "title", "description", "comment", "text", "message",
    "review", "notes", "bio", "summary", "address", "email",
}

# Columns whose names strongly suggest they are database IDs
ID_KEYWORDS = {
    "id", "idx", "index", "key", "uid", "uuid", "passengerid",
    "customerid", "userid", "productid", "orderid", "recordid",
}


def _is_likely_id_column(col: str, series: pd.Series, unique_ratio: float) -> bool:
    """
    Return True if this column is likely a database ID or meaningless index.

    Criteria (all must be met):
    - High unique ratio (>= ID_LIKE_UNIQUE_RATIO)
    - Column name matches ID keywords, OR series looks like a sequential integer index
    - Not a text/name field (those are handled separately)
    """
    col_lower = col.lower().strip()

    # Explicit text/name columns — NOT IDs
    for kw in TEXT_LIKE_KEYWORDS:
        if kw in col_lower:
            return False

    # Check name matches ID keywords
    name_is_id = any(
        col_lower == kw or col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_")
        for kw in ID_KEYWORDS
    )

    if name_is_id and unique_ratio >= ID_LIKE_UNIQUE_RATIO:
        return True

    # Integer sequential index detection (even without keyword match)
    if unique_ratio >= ID_LIKE_UNIQUE_RATIO and pd.api.types.is_integer_dtype(series.dtype):
        vals = series.dropna().sort_values().values
        if len(vals) > 10:
            diffs = np.diff(vals)
            # Sequential integers: all diffs are equal and small
            if len(set(diffs)) <= 2 and max(diffs) <= 5:
                return True

    return False


def _is_text_field(col: str, series: pd.Series, unique_ratio: float, n_unique: int) -> bool:
    """
    Return True if this column appears to be a free-text or name field.
    These are high-cardinality but potentially informative (e.g., passenger names
    may encode titles like Mr., Mrs., Dr. that are useful features).
    """
    col_lower = col.lower().strip()
    is_text_name = any(kw in col_lower for kw in TEXT_LIKE_KEYWORDS)

    if is_text_name and unique_ratio > 0.5 and n_unique > 20:
        return True

    # Check if values look like strings with spaces (names/sentences)
    if unique_ratio > 0.8 and pd.api.types.is_object_dtype(series.dtype):
        sample = series.dropna().head(20).astype(str)
        # If most values contain a space, likely a name/text field
        has_space = sample.str.contains(r"\s").mean()
        if has_space > 0.5:
            return True

    return False


@dataclass
class DatasetProfile:
    """Complete dataset characterisation used across the system."""

    # Basics
    n_rows: int = 0
    n_cols: int = 0
    n_duplicates: int = 0

    # Features
    all_columns: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    datetime_features: List[str] = field(default_factory=list)
    constant_features: List[str] = field(default_factory=list)
    high_cardinality_features: List[str] = field(default_factory=list)
    id_like_features: List[str] = field(default_factory=list)
    text_like_features: List[str] = field(default_factory=list)   # NEW: text/name fields

    # Missing values
    missing_counts: Dict[str, int] = field(default_factory=dict)
    missing_percentages: Dict[str, float] = field(default_factory=dict)
    has_missing: bool = False
    missing_rate_overall: float = 0.0

    # Target
    target_column: Optional[str] = None
    problem_type: Optional[str] = None   # 'binary_classification', 'multiclass_classification', 'regression'
    n_classes: Optional[int] = None
    class_distribution: Optional[Dict] = None
    is_imbalanced: bool = False
    imbalance_ratio: Optional[float] = None
    target_dtype: Optional[str] = None

    # Numerical stats
    numerical_stats: Dict[str, Any] = field(default_factory=dict)

    # Unique value counts
    unique_counts: Dict[str, int] = field(default_factory=dict)
    unique_ratios: Dict[str, float] = field(default_factory=dict)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Dataset summary
    memory_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON-serialization or display."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_duplicates": self.n_duplicates,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "constant_features": self.constant_features,
            "high_cardinality_features": self.high_cardinality_features,
            "id_like_features": self.id_like_features,
            "text_like_features": self.text_like_features,
            "has_missing": self.has_missing,
            "missing_rate_overall": round(self.missing_rate_overall, 4),
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "n_classes": self.n_classes,
            "is_imbalanced": self.is_imbalanced,
            "imbalance_ratio": self.imbalance_ratio,
            "warnings": self.warnings,
        }


class DatasetProfiler:
    """Produces a DatasetProfile from a raw DataFrame and target selection."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._profile: Optional[DatasetProfile] = None

    def profile(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
    ) -> DatasetProfile:
        """
        Compute a full DatasetProfile.

        Parameters
        ----------
        df : pd.DataFrame
        target_column : str, optional
        problem_type : str, optional
            If supplied, overrides auto-detection.
            One of 'binary_classification', 'multiclass_classification', 'regression'.
        """
        self._df = df.copy()
        p = DatasetProfile()

        p.n_rows = len(df)
        p.n_cols = len(df.columns)
        p.all_columns = list(df.columns)
        p.n_duplicates = int(df.duplicated().sum())
        p.memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

        # ── Feature type detection ────────────────────────────────────────────
        for col in df.columns:
            if col == target_column:
                continue
            dtype = df[col].dtype
            series = df[col]
            n_unique = series.nunique(dropna=True)
            unique_ratio = n_unique / max(len(df), 1)

            p.unique_counts[col] = int(n_unique)
            p.unique_ratios[col] = round(float(unique_ratio), 4)

            if n_unique <= 1:
                p.constant_features.append(col)

            if pd.api.types.is_datetime64_any_dtype(dtype):
                p.datetime_features.append(col)

            elif pd.api.types.is_numeric_dtype(dtype):
                p.numerical_features.append(col)
                # Check for ID-like numeric (sequential integer, named *id*)
                if unique_ratio >= ID_LIKE_UNIQUE_RATIO and n_unique > 20:
                    if _is_likely_id_column(col, series, unique_ratio):
                        p.id_like_features.append(col)

            else:
                # Check text/name fields before flagging as ID
                if _is_text_field(col, series, unique_ratio, n_unique):
                    p.text_like_features.append(col)
                    p.categorical_features.append(col)
                    if n_unique > HIGH_CARDINALITY_THRESHOLD:
                        p.high_cardinality_features.append(col)
                elif unique_ratio >= ID_LIKE_UNIQUE_RATIO and n_unique > 20:
                    if _is_likely_id_column(col, series, unique_ratio):
                        p.id_like_features.append(col)
                    else:
                        p.categorical_features.append(col)
                        p.high_cardinality_features.append(col)
                elif n_unique > HIGH_CARDINALITY_THRESHOLD:
                    p.categorical_features.append(col)
                    p.high_cardinality_features.append(col)
                else:
                    p.categorical_features.append(col)

        # ── Missing values ────────────────────────────────────────────────────
        missing = df.isnull().sum()
        p.missing_counts = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        p.missing_percentages = {
            col: round(cnt / len(df) * 100, 2)
            for col, cnt in p.missing_counts.items()
        }
        total_cells = df.size
        p.missing_rate_overall = float(missing.sum() / total_cells) if total_cells > 0 else 0.0
        p.has_missing = bool(p.missing_counts)

        # ── Numerical stats ───────────────────────────────────────────────────
        num_cols = [c for c in p.numerical_features if c in df.columns]
        if num_cols:
            stats = df[num_cols].describe().to_dict()
            p.numerical_stats = {
                col: {k: round(float(v), 6) if pd.notna(v) else None for k, v in col_stats.items()}
                for col, col_stats in stats.items()
            }

        # ── Target analysis ───────────────────────────────────────────────────
        if target_column and target_column in df.columns:
            p.target_column = target_column
            p.target_dtype = str(df[target_column].dtype)
            self._analyze_target(df, p, problem_type)
        else:
            p.warnings.append("No target column selected. Model training will not be possible.")

        # ── Data quality warnings ─────────────────────────────────────────────
        if p.n_rows < MIN_SAMPLES_FOR_ML:
            p.warnings.append(
                f"Dataset has only {p.n_rows} rows. Model training results may be unreliable."
            )
        if p.constant_features:
            p.warnings.append(
                f"Constant features detected: {p.constant_features}. "
                "These carry no information and will not help models."
            )
        if p.id_like_features:
            p.warnings.append(
                f"Likely ID columns detected: {p.id_like_features}. "
                "These should typically be excluded from training as they are arbitrary identifiers."
            )
        if p.text_like_features:
            p.warnings.append(
                f"Text/name fields detected: {p.text_like_features}. "
                "These may contain useful information (e.g., titles in passenger names) "
                "but require feature engineering before use. Consider excluding or engineering them."
            )
        if p.n_duplicates > 0:
            p.warnings.append(
                f"{p.n_duplicates} duplicate rows found. "
                "Duplicates may inflate performance metrics."
            )
        if p.has_missing:
            worst = max(p.missing_percentages.items(), key=lambda x: x[1])
            p.warnings.append(
                f"Missing values present. Worst: '{worst[0]}' ({worst[1]:.1f}% missing). "
                "Imputation is required."
            )

        self._profile = p
        return p

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _analyze_target(
        df: pd.DataFrame,
        p: DatasetProfile,
        override_problem_type: Optional[str],
    ) -> None:
        target = p.target_column
        series = df[target].dropna()

        is_numeric = pd.api.types.is_numeric_dtype(df[target].dtype)
        n_unique = series.nunique()
        unique_ratio = n_unique / max(len(series), 1)

        # ── Problem type detection ────────────────────────────────────────────
        if override_problem_type:
            p.problem_type = override_problem_type
        else:
            if not is_numeric:
                p.problem_type = (
                    "binary_classification" if n_unique == 2 else "multiclass_classification"
                )
            elif n_unique == 2:
                p.problem_type = "binary_classification"
            elif n_unique <= 20 and unique_ratio < 0.05:
                p.problem_type = "multiclass_classification"
            elif is_numeric and (unique_ratio > 0.05 or n_unique > 20):
                p.problem_type = "regression"
            else:
                p.problem_type = "binary_classification"
                p.warnings.append(
                    f"Problem type auto-detected as binary classification for target '{target}' "
                    f"(unique={n_unique}, ratio={unique_ratio:.3f}). "
                    "You may override this if incorrect."
                )

        # ── Class distribution for classification ─────────────────────────────
        if "classification" in (p.problem_type or ""):
            value_counts = series.value_counts()
            p.class_distribution = {str(k): int(v) for k, v in value_counts.items()}
            p.n_classes = int(n_unique)

            if len(value_counts) >= 2:
                minority = int(value_counts.iloc[-1])
                majority = int(value_counts.iloc[0])
                ratio = minority / majority if majority > 0 else 1.0
                p.imbalance_ratio = round(ratio, 4)
                p.is_imbalanced = ratio < CLASS_IMBALANCE_RATIO

                if p.is_imbalanced:
                    p.warnings.append(
                        f"Class imbalance detected: minority/majority ratio = {ratio:.3f}. "
                        "Consider class_weight='balanced' and appropriate scoring metrics."
                    )

        # ── Unsuitable target warnings ────────────────────────────────────────
        if n_unique <= 1:
            p.warnings.append(
                f"Target column '{target}' has only {n_unique} unique value(s). "
                "Cannot train a meaningful model."
            )
        if is_numeric and series.isnull().any():
            p.warnings.append(
                f"Target column '{target}' contains missing values. "
                "Rows with missing targets will be excluded from training."
            )
