"""tests/test_analyzer.py — Tests for dataset loading, profiling, and analysis."""

import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DatasetProfiler
from src.data.analyzer import DatasetAnalyzer
from src.data.loader import DataLoader
import io


class TestDatasetProfiler:
    def test_basic_profile(self, binary_classification_df):
        profiler = DatasetProfiler()
        p = profiler.profile(binary_classification_df, target_column="target")
        assert p.n_rows == 120
        assert p.n_cols == 5
        assert "age" in p.numerical_features
        assert "education" in p.categorical_features
        assert p.target_column == "target"

    def test_problem_type_binary(self, binary_classification_df):
        profiler = DatasetProfiler()
        p = profiler.profile(binary_classification_df, target_column="target")
        assert p.problem_type == "binary_classification"
        assert p.n_classes == 2

    def test_problem_type_multiclass(self, multiclass_df):
        profiler = DatasetProfiler()
        p = profiler.profile(multiclass_df, target_column="target")
        assert p.problem_type in ("multiclass_classification", "binary_classification")
        # Accept auto-detection with override
        p2 = profiler.profile(multiclass_df, target_column="target",
                               problem_type="multiclass_classification")
        assert p2.problem_type == "multiclass_classification"

    def test_problem_type_regression(self, regression_df):
        profiler = DatasetProfiler()
        p = profiler.profile(regression_df, target_column="target")
        assert p.problem_type == "regression"

    def test_missing_values_detected(self, df_with_missing):
        profiler = DatasetProfiler()
        p = profiler.profile(df_with_missing, target_column="target")
        assert p.has_missing
        assert "age" in p.missing_counts or "city" in p.missing_counts
        assert p.missing_rate_overall > 0

    def test_imbalance_detected(self, imbalanced_classification_df):
        profiler = DatasetProfiler()
        p = profiler.profile(imbalanced_classification_df, target_column="target")
        assert p.is_imbalanced
        assert p.imbalance_ratio is not None
        assert p.imbalance_ratio < 0.20

    def test_high_cardinality_detected(self, high_cardinality_df):
        profiler = DatasetProfiler()
        p = profiler.profile(high_cardinality_df, target_column="target")
        # product_id should be flagged as id-like or high-cardinality
        assert (
            "product_id" in p.id_like_features
            or "product_id" in p.high_cardinality_features
        )

    def test_no_target(self, binary_classification_df):
        profiler = DatasetProfiler()
        p = profiler.profile(binary_classification_df)
        assert p.target_column is None
        assert any("target" in w.lower() or "no target" in w.lower() for w in p.warnings)

    def test_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4], "target": [0, 0, 1]})
        profiler = DatasetProfiler()
        p = profiler.profile(df, target_column="target")
        assert p.n_duplicates == 1


class TestDatasetAnalyzer:
    def test_analyze_returns_profile(self, binary_classification_df):
        analyzer = DatasetAnalyzer()
        p = analyzer.analyze(binary_classification_df, target_column="target")
        assert p is not None
        assert p.n_rows > 0

    def test_correlation_matrix(self, binary_classification_df):
        analyzer = DatasetAnalyzer()
        analyzer.analyze(binary_classification_df, target_column="target")
        corr = analyzer.get_correlation_matrix()
        assert corr is not None
        assert corr.shape[0] > 0

    def test_missing_report_empty(self, binary_classification_df):
        analyzer = DatasetAnalyzer()
        analyzer.analyze(binary_classification_df, target_column="target")
        report = analyzer.get_missing_value_report()
        assert report.empty  # no missing values in fixture

    def test_missing_report_populated(self, df_with_missing):
        analyzer = DatasetAnalyzer()
        analyzer.analyze(df_with_missing, target_column="target")
        report = analyzer.get_missing_value_report()
        assert not report.empty
        assert "Missing %" in report.columns


class TestDataLoader:
    def test_load_from_bytes(self, binary_classification_df):
        csv_bytes = binary_classification_df.to_csv(index=False).encode()
        loader = DataLoader()
        df = loader.load_from_bytes(csv_bytes, "test.csv")
        assert df.shape == binary_classification_df.shape

    def test_empty_file_raises(self):
        loader = DataLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load_from_bytes(b"", "empty.csv")

    def test_single_column_raises(self):
        loader = DataLoader()
        csv_bytes = b"col\n1\n2\n3"
        with pytest.raises(ValueError, match="column"):
            loader.load_from_bytes(csv_bytes, "single.csv")
