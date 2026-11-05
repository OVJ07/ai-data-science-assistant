"""
tests/test_rag_pipeline_audit.py
─────────────────────────────────
Comprehensive end-to-end audit regression tests for RAG pipeline influence:
1. Verifies that different retrieval outputs produce distinct recommendations and search spaces.
2. Verifies that baseline is strictly decoupled from RAG recommendations.
3. Verifies aligned CV scoring protocol to prevent objective divergence.
4. Verifies configuration and prediction hash diagnostics.
5. Verifies end-to-end provenance integrity.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DatasetProfile, DatasetProfiler
from src.ml.comparison import ModelComparator
from src.ml.diagnostics import (
    DatasetFingerprint,
    compute_config_hash,
    compute_prediction_agreement,
    compute_prediction_hash,
)
from src.rag.recommendation import RecommendationEngine, SearchSpaceConfig
from src.rag.retriever import RetrievedDocument


@pytest.fixture
def imbalanced_profile():
    return DatasetProfile(
        n_rows=150,
        n_cols=4,
        problem_type="binary_classification",
        target_column="target",
        has_missing=True,
        missing_rate_overall=0.08,
        is_imbalanced=True,
        imbalance_ratio=0.15,
        numerical_features=["feat1", "feat2"],
        categorical_features=["cat1"],
        high_cardinality_features=["cat1"],
    )


@pytest.fixture
def mock_retriever():
    class DummyRetriever:
        def retrieve_multiple_queries(self, queries, **kwargs):
            return []
    return DummyRetriever()


def test_different_retrieval_docs_produce_distinct_recommendations(imbalanced_profile, mock_retriever):
    """When retriever A retrieves class_imbalance and retriever B does not, recommendations must differ."""
    engine = RecommendationEngine(mock_retriever)

    docs_with_imbalance = [
        RetrievedDocument(
            doc_id="class_imbalance",
            title="Class Imbalance",
            content="Use class_weight='balanced'",
            relevance_score=0.9,
            source_file="class_imbalance.md",
        )
    ]

    docs_without_imbalance = [
        RetrievedDocument(
            doc_id="decision_tree",
            title="Decision Tree",
            content="Tune criterion and max_depth",
            relevance_score=0.85,
            source_file="decision_tree.md",
        )
    ]

    recs_a, _ = engine.generate_recommendations(imbalanced_profile, [], retrieved_docs=docs_with_imbalance)
    recs_b, _ = engine.generate_recommendations(imbalanced_profile, [], retrieved_docs=docs_without_imbalance)

    has_cw_a = any("class_weight" in r.technique.lower() for r in recs_a)
    has_cw_b = any("class_weight" in r.technique.lower() for r in recs_b)

    assert has_cw_a is True, "Retriever with class_imbalance doc must recommend class_weight"
    assert has_cw_b is False, "Retriever lacking class_imbalance doc must NOT recommend class_weight"

    # Search space must also differ
    space_a = engine.generate_search_space(imbalanced_profile, "Decision Tree", recs_a, docs_with_imbalance)
    space_b = engine.generate_search_space(imbalanced_profile, "Decision Tree", recs_b, docs_without_imbalance)

    assert "classifier__class_weight" in space_a.param_grid
    assert "classifier__class_weight" not in space_b.param_grid
    assert space_a.cv_scoring == "balanced_accuracy"
    assert space_b.cv_scoring == "f1_weighted"


def test_model_knowledge_influences_hyperparameter_grid(imbalanced_profile, mock_retriever):
    """Retrieving decision_tree or random_forest expands the model search space accordingly."""
    engine = RecommendationEngine(mock_retriever)

    dt_doc = [
        RetrievedDocument(
            doc_id="decision_tree",
            title="Decision Tree",
            content="Evaluate criterion gini and entropy",
            relevance_score=0.9,
            source_file="decision_tree.md",
        )
    ]

    recs, docs = engine.generate_recommendations(imbalanced_profile, [], retrieved_docs=dt_doc)
    space = engine.generate_search_space(imbalanced_profile, "Decision Tree", recs, docs, is_rag_guided=True)

    assert "classifier__criterion" in space.param_grid
    assert "gini" in space.param_grid["classifier__criterion"]
    assert "entropy" in space.param_grid["classifier__criterion"]


def test_baseline_isolated_from_rag_recommendations(binary_classification_df):
    """ModelComparator baseline run must not inherit RAG preprocessing or param grid modifications."""
    profiler = DatasetProfiler()
    profile = profiler.profile(binary_classification_df, target_column="target")

    rag_config = SearchSpaceConfig(
        model_name="Decision Tree",
        param_grid={"classifier__max_depth": [2, 4], "classifier__min_samples_split": [10]},
        cv_scoring="balanced_accuracy",
        preprocessing_config={
            "numerical_imputer_strategy": "mean",
            "categorical_imputer_strategy": "constant",
            "max_ohe_categories": 12,
        },
        is_rag_guided=True,
    )

    comparator = ModelComparator()
    comp_result = comparator.compare(
        df=binary_classification_df,
        profile=profile,
        model_name="Decision Tree",
        rag_search_config=rag_config,
        align_cv_scoring=False,
    )

    # Baseline configuration must remain pure baseline defaults
    assert comp_result.baseline_configuration["param_grid"] != rag_config.param_grid
    assert comp_result.baseline_configuration["numerical_imputer_strategy"] == "median"
    assert comp_result.baseline_configuration["categorical_imputer_strategy"] == "most_frequent"
    assert comp_result.configurations_differ is True


def test_aligned_cv_scoring_ensures_fair_comparison(binary_classification_df):
    """When align_cv_scoring=True, both baseline and RAG evaluate under the same CV scoring metric."""
    profiler = DatasetProfiler()
    profile = profiler.profile(binary_classification_df, target_column="target")

    rag_config = SearchSpaceConfig(
        model_name="Decision Tree",
        param_grid={"classifier__max_depth": [3, 5]},
        cv_scoring="balanced_accuracy",
        is_rag_guided=True,
    )

    comparator = ModelComparator()
    comp_result = comparator.compare(
        df=binary_classification_df,
        profile=profile,
        model_name="Decision Tree",
        rag_search_config=rag_config,
        align_cv_scoring=True,
    )

    assert comp_result.baseline_result.cv_scoring == "balanced_accuracy"
    assert comp_result.rag_result.cv_scoring == "balanced_accuracy"
    assert comp_result.baseline_configuration["cv_scoring"] == comp_result.rag_configuration["cv_scoring"]


def test_config_and_prediction_hashes():
    """Verify deterministic hashing of configurations and predictions."""
    grid1 = {"classifier__max_depth": [3, 5], "classifier__C": [0.1, 1.0]}
    grid2 = {"classifier__C": [0.1, 1.0], "classifier__max_depth": [3, 5]}  # Different key order
    grid3 = {"classifier__max_depth": [5, 10]}

    prep = {"numerical_imputer_strategy": "median"}

    hash1 = compute_config_hash(grid1, prep, "accuracy")
    hash2 = compute_config_hash(grid2, prep, "accuracy")
    hash3 = compute_config_hash(grid3, prep, "accuracy")

    assert hash1 == hash2, "Identical configurations with different key order must yield the same hash"
    assert hash1 != hash3, "Different configurations must yield different hashes"

    preds_a = np.array([0, 1, 0, 1, 1])
    preds_b = np.array([0, 1, 0, 1, 1])
    preds_c = np.array([1, 1, 0, 1, 0])

    assert compute_prediction_hash(preds_a) == compute_prediction_hash(preds_b)
    assert compute_prediction_hash(preds_a) != compute_prediction_hash(preds_c)
    assert compute_prediction_agreement(preds_a, preds_b) == 1.0
    assert compute_prediction_agreement(preds_a, preds_c) == 0.6


def test_dataset_fingerprint(binary_classification_df):
    """Verify dataset fingerprint creation."""
    profiler = DatasetProfiler()
    profile = profiler.profile(binary_classification_df, target_column="target")
    fp = DatasetFingerprint.from_profile(profile)

    assert fp.fingerprint_hash
    assert len(fp.fingerprint_hash) == 12
    assert fp.n_rows == len(binary_classification_df)

