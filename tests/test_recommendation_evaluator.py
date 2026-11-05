"""
tests/test_recommendation_evaluator.py
──────────────────────────────────────
Tests for RecommendationEvaluator across retrieval strategies.
"""

import pytest
from src.data.profiler import DatasetProfile
from src.rag.recommendation import Recommendation, SearchSpaceConfig
from src.rag.recommendation_evaluator import (
    RecommendationEvaluator,
    StrategyRecommendationReport,
    SUPPORTED_PIPELINE_KEYS,
)
from src.rag.retriever import RetrievedDocument


@pytest.fixture
def evaluator():
    return RecommendationEvaluator()


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        n_rows=100,
        n_cols=5,
        problem_type="binary_classification",
        target_column="target",
        has_missing=True,
        missing_rate_overall=0.08,
        is_imbalanced=True,
        imbalance_ratio=0.15,
        high_cardinality_features=["zip_code"],
        id_like_features=["user_id"],
    )


@pytest.fixture
def retrieved_docs():
    return [
        RetrievedDocument(
            doc_id="missing_values",
            title="Missing Values",
            content="Content",
            relevance_score=0.85,
            source_file="missing_values.md",
        ),
        RetrievedDocument(
            doc_id="class_imbalance",
            title="Class Imbalance",
            content="Content",
            relevance_score=0.75,
            source_file="class_imbalance.md",
        ),
    ]


def test_evaluate_recommendations_supported_and_unsupported(evaluator, sample_profile, retrieved_docs):
    recs = [
        Recommendation(
            technique="Numerical Imputation (median)",
            trigger="Missing values",
            reason="Handle missing values",
            supporting_doc="missing_values",
            config={"numerical_imputer_strategy": "median"},
        ),
        Recommendation(
            technique="Hyperparameter Tuning",
            trigger="General",
            reason="Optimize model",
            supporting_doc="hyperparameter_tuning",  # Not in retrieved_docs
            config={"max_depth": [3, 5]},
        ),
    ]

    report = evaluator.evaluate_recommendations(
        strategy_name="TF-IDF",
        profile=sample_profile,
        retrieved_docs=retrieved_docs,
        recommendations=recs,
    )

    assert isinstance(report, StrategyRecommendationReport)
    assert report.total_recommendations == 2
    assert report.supported_count == 1
    assert report.support_rate == 0.5
    assert report.validity_rate == 1.0


def test_invalid_config_flagged(evaluator, sample_profile, retrieved_docs):
    recs = [
        Recommendation(
            technique="Invalid Imputation",
            trigger="Missing values",
            reason="test",
            supporting_doc="missing_values",
            config={"numerical_imputer_strategy": "invalid_mode_strategy"},
        )
    ]

    report = evaluator.evaluate_recommendations(
        strategy_name="Dense",
        profile=sample_profile,
        retrieved_docs=retrieved_docs,
        recommendations=recs,
    )

    assert report.valid_count == 0
    assert report.validity_rate == 0.0
    assert "Invalid numerical imputer strategy" in report.evaluations[0].notes[0]


def test_irrelevant_recommendation_flagged(evaluator, retrieved_docs):
    # Balanced dataset with no missing values
    balanced_profile = DatasetProfile(
        n_rows=500,
        n_cols=5,
        problem_type="binary_classification",
        has_missing=False,
        is_imbalanced=False,
    )

    recs = [
        Recommendation(
            technique="class_weight balanced",
            trigger="False alarm",
            reason="Imbalance",
            supporting_doc="class_imbalance",
            config={"class_weight": [None, "balanced"]},
        ),
        Recommendation(
            technique="Mean Imputation",
            trigger="False alarm",
            reason="Missing",
            supporting_doc="missing_values",
            config={"numerical_imputer_strategy": "mean"},
        ),
    ]

    report = evaluator.evaluate_recommendations(
        strategy_name="Hybrid",
        profile=balanced_profile,
        retrieved_docs=retrieved_docs,
        recommendations=recs,
    )

    assert report.relevant_count == 0
    assert report.relevance_rate == 0.0


def test_applied_status_mapped_from_search_config(evaluator, sample_profile, retrieved_docs):
    recs = [
        Recommendation(
            technique="Numerical Imputation (strategy='median')",
            trigger="Missing",
            reason="Missing",
            supporting_doc="missing_values",
            config={"numerical_imputer_strategy": "median"},
        )
    ]

    search_config = SearchSpaceConfig(
        model_name="Random Forest",
        recommendation_outcomes=[{
            "recommendation": "Numerical Imputation (strategy='median')",
            "status": "applied",
            "reason": "Mapped to training preprocessor.",
        }],
    )

    report = evaluator.evaluate_recommendations(
        strategy_name="Hybrid",
        profile=sample_profile,
        retrieved_docs=retrieved_docs,
        recommendations=recs,
        search_config=search_config,
    )

    assert report.applied_count == 1
    assert report.application_rate == 1.0
    assert report.evaluations[0].application_status == "applied"

