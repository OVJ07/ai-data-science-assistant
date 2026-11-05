"""
src/ml/preprocessing.py
────────────────────────
Builds scikit-learn preprocessing pipelines from a DatasetProfile.
All transformations are enclosed within the pipeline to prevent data leakage.
Feature names are preserved throughout via get_feature_names_out().
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from config.settings import HIGH_CARDINALITY_THRESHOLD, MAX_ONE_HOT_CATEGORIES
from src.data.profiler import DatasetProfile

logger = logging.getLogger(__name__)


class PreprocessingBuilder:
    """
    Builds a ColumnTransformer preprocessing pipeline based on dataset profile
    and recommendations. Feature names are explicitly tracked.
    """

    def __init__(self) -> None:
        self._numerical_features: List[str] = []
        self._categorical_features: List[str] = []
        self._feature_names_out: Optional[List[str]] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def build(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        numerical_imputer_strategy: str = "median",
        categorical_imputer_strategy: str = "most_frequent",
        scale_numerical: bool = True,
        max_ohe_categories: Optional[int] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple["ColumnTransformer", List[str]]:
        """
        Build a ColumnTransformer pipeline.

        Parameters
        ----------
        df : pd.DataFrame — the full dataset (not yet split)
        profile : DatasetProfile
        numerical_imputer_strategy : str
        categorical_imputer_strategy : str
        scale_numerical : bool — if True, apply StandardScaler after imputation
        max_ohe_categories : int — limit OHE to top-N categories
        exclude_columns : list — columns to exclude from features

        Returns
        -------
        (preprocessor: ColumnTransformer, feature_columns: List[str])
        """
        exclude = set(exclude_columns or [])
        if profile.target_column:
            exclude.add(profile.target_column)
        exclude.update(profile.constant_features)

        # Determine active numerical/categorical feature columns
        num_cols = [
            c for c in profile.numerical_features
            if c in df.columns and c not in exclude
        ]
        cat_cols = [
            c for c in profile.categorical_features
            if c in df.columns and c not in exclude
        ]

        self._numerical_features = num_cols
        self._categorical_features = cat_cols

        # ── Numerical pipeline ────────────────────────────────────────────────
        num_steps = [("imputer", SimpleImputer(strategy=numerical_imputer_strategy))]
        if scale_numerical:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)

        # ── Categorical pipeline ──────────────────────────────────────────────
        max_cats = max_ohe_categories
        ohe = OneHotEncoder(
            # With ``max_categories`` this preserves a fitted infrequent bucket
            # for both low-frequency and previously unseen values.  If no
            # infrequent bucket is fitted, sklearn still safely emits all-zero
            # columns for an unseen category.
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            max_categories=max_cats if max_cats else None,
        )
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy, fill_value="Unknown")),
            ("encoder", ohe),
        ])

        # ── ColumnTransformer ─────────────────────────────────────────────────
        transformers = []
        if num_cols:
            transformers.append(("numerical", num_pipeline, num_cols))
        if cat_cols:
            transformers.append(("categorical", cat_pipeline, cat_cols))

        if not transformers:
            raise ValueError(
                "No valid feature columns found. "
                "All columns are either the target, constant, or excluded."
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )

        all_input_cols = num_cols + cat_cols
        logger.info(
            "Built preprocessor: %d numerical, %d categorical features",
            len(num_cols),
            len(cat_cols),
        )
        return preprocessor, all_input_cols

    def get_feature_names(self, preprocessor: "ColumnTransformer") -> List[str]:
        """
        Extract feature names from a fitted ColumnTransformer.
        Handles both sklearn >= 1.0 and older versions.
        """
        try:
            return list(preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older sklearn
            names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    try:
                        names.extend(transformer.get_feature_names_out(cols))
                    except Exception:
                        names.extend([f"{name}__{c}" for c in cols])
                else:
                    names.extend([f"{name}__{c}" for c in cols])
            return names

    def prepare_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: str,
    ) -> Tuple[pd.Series, Optional[LabelEncoder]]:
        """
        Prepare the target series.
        For classification with string labels, encode to integers.
        Returns (y, label_encoder_or_None).
        """
        y = df[target_column].dropna()
        le = None

        if "classification" in problem_type:
            # pandas 3 commonly represents CSV strings as ``str`` backed by an
            # ArrowStringArray.  That extension array cannot be fancy-indexed
            # by sklearn's NumPy CV indices.  Encode every non-numeric target
            # into a NumPy-backed integer Series rather than checking only the
            # legacy ``object`` and ``category`` dtypes.
            if not pd.api.types.is_numeric_dtype(y.dtype):
                le = LabelEncoder()
                y = pd.Series(
                    np.asarray(le.fit_transform(y.astype(str)), dtype=np.int64),
                    index=y.index,
                    name=target_column,
                )
                self._label_encoder = le
                logger.info(
                    "Target encoded: %d classes → %s",
                    len(le.classes_),
                    list(le.classes_),
                )
        return y, le

    @property
    def numerical_features(self) -> List[str]:
        return self._numerical_features

    @property
    def categorical_features(self) -> List[str]:
        return self._categorical_features

    @property
    def label_encoder(self) -> Optional[LabelEncoder]:
        return self._label_encoder


def safe_align(
    df: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows where target is NaN, align indices, and remove the target.

    ``y`` retains its original column name when it is made from a DataFrame
    column.  Removing that named column here is a final defence against target
    leakage for every caller (trainer, auto-selector, and future workflows).
    """
    common_idx = df.index.intersection(y.index)
    X = df.loc[common_idx].copy()
    if y.name is not None and y.name in X.columns:
        X = X.drop(columns=[y.name])
    return X, y.loc[common_idx]
