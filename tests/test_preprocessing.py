"""tests/test_preprocessing.py — Tests for the ML preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.profiler import DatasetProfiler
from src.ml.preprocessing import PreprocessingBuilder, safe_align


class TestPreprocessingBuilder:
    def test_build_returns_pipeline(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        builder = PreprocessingBuilder()
        preprocessor, warnings = builder.build(binary_classification_df, profile)
        assert preprocessor is not None

    def test_no_data_leakage(self, binary_classification_df):
        """Preprocessor fitted only on training data, not full dataset."""
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        X = binary_classification_df.drop(columns=["target"])
        y = binary_classification_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        builder = PreprocessingBuilder()
        preprocessor, _ = builder.build(binary_classification_df, profile)
        preprocessor.fit(X_train)
        # Should not raise
        X_test_transformed = preprocessor.transform(X_test)
        assert X_test_transformed.shape[0] == len(X_test)

    def test_feature_names_match_output_dims(self, binary_classification_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(binary_classification_df, target_column="target")
        builder = PreprocessingBuilder()
        preprocessor, _ = builder.build(binary_classification_df, profile)
        X = binary_classification_df.drop(columns=["target"])
        preprocessor.fit(X)
        feature_names = builder.get_feature_names(preprocessor)
        X_transformed = preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        assert len(feature_names) == X_transformed.shape[1], \
            f"Feature names ({len(feature_names)}) != output dims ({X_transformed.shape[1]})"

    def test_handles_missing_values(self, df_with_missing):
        profiler = DatasetProfiler()
        profile = profiler.profile(df_with_missing, target_column="target")
        builder = PreprocessingBuilder()
        preprocessor, _ = builder.build(df_with_missing, profile)
        X = df_with_missing.drop(columns=["target"])
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        # No NaN values after imputation
        assert not np.isnan(X_transformed).any()

    def test_exclude_columns(self, high_cardinality_df):
        profiler = DatasetProfiler()
        profile = profiler.profile(high_cardinality_df, target_column="target")
        builder = PreprocessingBuilder()
        preprocessor, _ = builder.build(
            high_cardinality_df, profile, exclude_columns=["product_id"]
        )
        X = high_cardinality_df.drop(columns=["target", "product_id"])
        preprocessor.fit(X)
        feature_names = builder.get_feature_names(preprocessor)
        assert all("product_id" not in f for f in feature_names)

    def test_prepare_target_binary(self, binary_classification_df):
        builder = PreprocessingBuilder()
        y, le = builder.prepare_target(
            binary_classification_df, "target", "binary_classification"
        )
        assert len(y) == len(binary_classification_df)
        # Binary targets should be 0 or 1
        assert set(y.unique()).issubset({0, 1})

    def test_prepare_target_regression(self, regression_df):
        builder = PreprocessingBuilder()
        y, le = builder.prepare_target(regression_df, "target", "regression")
        assert le is None
        assert y.dtype in [np.float64, np.float32, float]

    def test_prepare_target_encodes_arrow_string_target(self, adult_income_sample_df):
        builder = PreprocessingBuilder()
        assert not pd.api.types.is_numeric_dtype(adult_income_sample_df["income"].dtype)
        y, le = builder.prepare_target(
            adult_income_sample_df, "income", "binary_classification"
        )
        assert le is not None
        assert pd.api.types.is_integer_dtype(y.dtype)
        assert isinstance(y.to_numpy(), np.ndarray)


class TestSafeAlign:
    def test_align_removes_target_column(self, binary_classification_df):
        y = binary_classification_df["target"]
        X, y_aligned = safe_align(binary_classification_df, y)
        assert "target" not in X.columns

    def test_align_preserves_index(self, binary_classification_df):
        # Drop a row to create index mismatch
        df_sub = binary_classification_df.iloc[::2].copy()
        y_sub = df_sub["target"]
        X, y_aligned = safe_align(df_sub, y_sub)
        assert len(X) == len(y_aligned)
        assert list(X.index) == list(y_aligned.index)
