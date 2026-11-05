"""
tests/test_multi_retriever_comparison.py
────────────────────────────────────────
Tests for MultiRetrieverComparator (Baseline, TF-IDF, Dense, Hybrid).
Mocks Ollama embeddings so it executes locally without Ollama running.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DatasetProfile
from src.ml.multi_retriever_comparison import (
    MultiRetrieverComparator,
    MultiRetrieverComparisonResult,
)


@pytest.fixture
def sample_dataset():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat1": rng.choice(["A", "B", "C"], n),
        "target": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # Introduce some missing values
    df.loc[0:5, "num1"] = np.nan
    return df


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        n_rows=100,
        n_cols=4,
        problem_type="binary_classification",
        target_column="target",
        has_missing=True,
        missing_rate_overall=0.05,
        is_imbalanced=True,
        imbalance_ratio=0.25,
        numerical_features=["num1", "num2"],
        categorical_features=["cat1"],
    )


def test_multi_retriever_runs_and_records_all_four(sample_dataset, sample_profile, tmp_path):
    comparator = MultiRetrieverComparator(cache_dir=str(tmp_path))

    emb = np.random.default_rng(42).normal(0, 1, 768).tolist()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embedding": emb}
    resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=resp):
        res = comparator.run_comparison(
            df=sample_dataset,
            profile=sample_profile,
            model_name="Decision Tree",
        )

    assert isinstance(res, MultiRetrieverComparisonResult)
    assert res.baseline_run.strategy_name == "Baseline (No RAG)"
    assert not res.baseline_run.is_rag_guided
    assert res.tfidf_run.is_rag_guided
    assert res.dense_run.is_rag_guided
    assert res.hybrid_run.is_rag_guided

    # Verify honest summary string exists and is non-empty
    assert res.honest_summary != ""
    assert len(res.comparison_table) == 4

    # Verify baseline is independent
    assert res.baseline_run.training_result is not None
    assert res.tfidf_run.training_result is not None
    assert res.baseline_run.training_result is not res.tfidf_run.training_result


def test_multi_retriever_handles_dense_down(sample_dataset, sample_profile, tmp_path):
    import requests as req_lib
    comparator = MultiRetrieverComparator()

    with patch("requests.post", side_effect=req_lib.exceptions.ConnectionError("refused")):
        res = comparator.run_comparison(
            df=sample_dataset,
            profile=sample_profile,
            model_name="Decision Tree",
            embedding_model="non_cached_offline_model",
        )

    assert isinstance(res, MultiRetrieverComparisonResult)
    assert res.baseline_run.training_result.is_successful()
    assert res.tfidf_run.training_result.is_successful()
    # Dense should record error rather than crashing
    assert res.dense_run.error is not None
    assert res.hybrid_run.error is not None
