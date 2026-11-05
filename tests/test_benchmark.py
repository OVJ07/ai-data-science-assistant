"""
tests/test_benchmark.py
────────────────────────
Tests for the retrieval benchmark module.
All retriever calls are mocked so this runs without Ollama.
"""

import pytest
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_retrieved_doc(doc_id: str, score: float):
    """Create a minimal mock RetrievedDocument."""
    from src.rag.retriever import RetrievedDocument
    return RetrievedDocument(
        doc_id=doc_id,
        title=doc_id.replace("_", " ").title(),
        content=f"Content for {doc_id}",
        relevance_score=score,
        source_file=f"{doc_id}.md",
    )


# ── Tests: metric functions ───────────────────────────────────────────────────

class TestMetricFunctions:
    """Test the standalone metric calculation functions."""

    def test_precision_at_k_all_relevant(self):
        from src.rag.benchmark import _compute_precision_at_k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        assert _compute_precision_at_k(retrieved, relevant, k=5) == 1.0

    def test_precision_at_k_none_relevant(self):
        from src.rag.benchmark import _compute_precision_at_k
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert _compute_precision_at_k(retrieved, relevant, k=5) == 0.0

    def test_precision_at_k_partial(self):
        from src.rag.benchmark import _compute_precision_at_k
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        # 3 hits in top 5 → 3/5
        assert abs(_compute_precision_at_k(retrieved, relevant, k=5) - 0.6) < 1e-9

    def test_precision_at_k_zero(self):
        from src.rag.benchmark import _compute_precision_at_k
        assert _compute_precision_at_k([], set(), k=0) == 0.0

    def test_recall_at_k_all_retrieved(self):
        from src.rag.benchmark import _compute_recall_at_k
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        assert _compute_recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_empty_relevant(self):
        from src.rag.benchmark import _compute_recall_at_k
        retrieved = ["a", "b"]
        relevant = set()
        assert _compute_recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_recall_at_k_partial(self):
        from src.rag.benchmark import _compute_recall_at_k
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        # 1/2 relevant retrieved in top 5
        assert abs(_compute_recall_at_k(retrieved, relevant, k=5) - 0.5) < 1e-9

    def test_mrr_first_hit_at_rank_1(self):
        from src.rag.benchmark import _compute_mrr
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert _compute_mrr(retrieved, relevant) == 1.0

    def test_mrr_first_hit_at_rank_2(self):
        from src.rag.benchmark import _compute_mrr
        retrieved = ["x", "a", "c"]
        relevant = {"a"}
        assert abs(_compute_mrr(retrieved, relevant) - 0.5) < 1e-9

    def test_mrr_no_relevant(self):
        from src.rag.benchmark import _compute_mrr
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert _compute_mrr(retrieved, relevant) == 0.0

    def test_ndcg_at_k_perfect(self):
        from src.rag.benchmark import _compute_ndcg_at_k
        retrieved = ["a", "b"]
        grades = {"a": 2, "b": 1}
        # Perfect ranking → nDCG = 1.0
        score = _compute_ndcg_at_k(retrieved, grades, k=5)
        assert abs(score - 1.0) < 1e-9

    def test_ndcg_at_k_zero_when_no_relevant(self):
        from src.rag.benchmark import _compute_ndcg_at_k
        retrieved = ["x", "y"]
        grades = {}
        assert _compute_ndcg_at_k(retrieved, grades, k=5) == 0.0

    def test_ndcg_at_k_reversed_ordering(self):
        from src.rag.benchmark import _compute_ndcg_at_k
        # Worst ranking (lowest grade first)
        retrieved = ["b", "a"]
        grades = {"a": 2, "b": 1}
        score = _compute_ndcg_at_k(retrieved, grades, k=5)
        assert 0.0 < score < 1.0  # worse than perfect but > 0

    def test_ndcg_at_k_between_0_and_1(self):
        from src.rag.benchmark import _compute_ndcg_at_k
        retrieved = ["a", "x", "b", "y", "c"]
        grades = {"a": 2, "b": 1, "d": 2}
        score = _compute_ndcg_at_k(retrieved, grades, k=5)
        assert 0.0 <= score <= 1.0


# ── Tests: BENCHMARK_QUERIES ──────────────────────────────────────────────────

class TestBenchmarkQueries:
    """Validate the ground truth query set."""

    def test_benchmark_queries_not_empty(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        assert len(BENCHMARK_QUERIES) > 0

    def test_benchmark_queries_have_required_fields(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        for q in BENCHMARK_QUERIES:
            assert q.query_id, f"Empty query_id: {q}"
            assert q.query_text, f"Empty query_text: {q.query_id}"
            assert q.topic, f"Empty topic: {q.query_id}"

    def test_benchmark_query_ids_unique(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        ids = [q.query_id for q in BENCHMARK_QUERIES]
        assert len(ids) == len(set(ids))

    def test_relevant_doc_ids_match_known_kb_docs(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        known_docs = {
            "categorical_encoding", "class_imbalance", "cross_validation",
            "decision_tree", "evaluation_metrics", "explainable_ai",
            "high_cardinality", "hyperparameter_tuning", "logistic_regression",
            "missing_values", "random_forest", "regression",
        }
        for q in BENCHMARK_QUERIES:
            for doc_id in q.relevant_doc_ids:
                assert doc_id in known_docs, (
                    f"Query {q.query_id} references unknown doc_id '{doc_id}'"
                )

    def test_relevance_grades_consistent_with_relevant_doc_ids(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        for q in BENCHMARK_QUERIES:
            for doc_id in q.relevant_doc_ids:
                assert doc_id in q.relevance_grades, (
                    f"Query {q.query_id}: doc '{doc_id}' in relevant_doc_ids but not in relevance_grades"
                )

    def test_relevance_grades_are_1_or_2(self):
        from src.rag.benchmark import BENCHMARK_QUERIES
        for q in BENCHMARK_QUERIES:
            for doc_id, grade in q.relevance_grades.items():
                assert grade in (1, 2), (
                    f"Query {q.query_id}: invalid grade {grade} for doc '{doc_id}'"
                )


# ── Tests: BenchmarkReport ────────────────────────────────────────────────────

class TestBenchmarkReport:
    def test_report_fields_present(self):
        from src.rag.benchmark import BenchmarkReport
        r = BenchmarkReport(
            strategy_name="tfidf",
            n_queries=12,
            n_queries_with_relevant=12,
            mean_precision_at_5=0.5,
            mean_recall_at_5=0.6,
            mrr=0.7,
            mean_ndcg_at_5=0.65,
            mean_latency_ms=5.0,
            index_build_time_seconds=0.1,
            cache_hit_rate=None,
        )
        assert r.strategy_name == "tfidf"
        assert r.cache_hit_rate is None

    def test_dense_report_has_cache_hit_rate(self):
        from src.rag.benchmark import BenchmarkReport
        r = BenchmarkReport(
            strategy_name="dense:nomic-embed-text:latest",
            n_queries=12,
            n_queries_with_relevant=12,
            mean_precision_at_5=0.5,
            mean_recall_at_5=0.6,
            mrr=0.7,
            mean_ndcg_at_5=0.65,
            mean_latency_ms=50.0,
            index_build_time_seconds=5.0,
            cache_hit_rate=0.75,
        )
        assert r.cache_hit_rate == 0.75


# ── Tests: RetrievalBenchmark with mocked TF-IDF ─────────────────────────────

class TestTFIDFBenchmark:
    """Run TF-IDF benchmark using the real knowledge base."""

    def test_run_tfidf_returns_report(self):
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR)
        report = bench.run_tfidf()
        assert report.strategy_name == "tfidf"
        assert report.n_queries > 0
        assert 0.0 <= report.mean_precision_at_5 <= 1.0
        assert 0.0 <= report.mean_recall_at_5 <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert 0.0 <= report.mean_ndcg_at_5 <= 1.0
        assert report.mean_latency_ms >= 0
        assert report.index_build_time_seconds >= 0
        assert report.cache_hit_rate is None  # TF-IDF has no cache

    def test_run_tfidf_per_query_results(self):
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR)
        report = bench.run_tfidf()
        assert len(report.per_query_results) > 0
        for qr in report.per_query_results:
            assert qr.query_id
            assert isinstance(qr.retrieved_doc_ids, list)
            assert 0.0 <= qr.precision_at_5 <= 1.0
            assert 0.0 <= qr.recall_at_5 <= 1.0
            assert 0.0 <= qr.mrr <= 1.0
            assert 0.0 <= qr.ndcg_at_5 <= 1.0
            assert qr.latency_ms >= 0

    def test_run_tfidf_limitation_notes_present(self):
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR)
        report = bench.run_tfidf()
        assert len(report.limitation_notes) > 0

    def test_run_tfidf_reproducible(self):
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR)
        r1 = bench.run_tfidf()
        r2 = bench.run_tfidf()
        # TF-IDF is deterministic
        assert r1.mean_precision_at_5 == r2.mean_precision_at_5
        assert r1.mrr == r2.mrr


# ── Tests: Dense benchmark with mocked Ollama ────────────────────────────────

class TestDenseBenchmark:
    def test_run_dense_ollama_down_returns_error_report(self, tmp_path):
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        import requests as req_lib
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR, cache_dir=empty_cache)
        with patch("requests.post", side_effect=req_lib.exceptions.ConnectionError("refused")):
            report = bench.run_dense(embedding_model="nonexistent_test_model")
        assert len(report.errors) > 0 or not report.n_queries > 0

    def test_run_dense_with_mock_embeddings(self, tmp_path):
        import numpy as np
        from src.rag.benchmark import RetrievalBenchmark
        from config.settings import KNOWLEDGE_BASE_DIR
        cache = tmp_path / "cache"
        cache.mkdir()
        bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR, cache_dir=cache)
        emb = np.random.default_rng(42).normal(0, 1, 768).tolist()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"embedding": emb}
        resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=resp):
            report = bench.run_dense(embedding_model="nomic-embed-text:latest")
        assert report.strategy_name == "dense:nomic-embed-text:latest"
        assert report.n_queries > 0


# ── Tests: compare_reports ────────────────────────────────────────────────────

class TestCompareReports:
    def test_compare_reports_returns_string(self):
        from src.rag.benchmark import RetrievalBenchmark, BenchmarkReport, compare_reports
        r1 = BenchmarkReport(
            strategy_name="tfidf", n_queries=12, n_queries_with_relevant=12,
            mean_precision_at_5=0.4, mean_recall_at_5=0.5, mrr=0.6,
            mean_ndcg_at_5=0.55, mean_latency_ms=3.0,
            index_build_time_seconds=0.05, cache_hit_rate=None,
        )
        r2 = BenchmarkReport(
            strategy_name="dense:nomic-embed-text:latest", n_queries=12, n_queries_with_relevant=12,
            mean_precision_at_5=0.5, mean_recall_at_5=0.6, mrr=0.7,
            mean_ndcg_at_5=0.65, mean_latency_ms=50.0,
            index_build_time_seconds=3.0, cache_hit_rate=0.0,
        )
        table = compare_reports({"tfidf": r1, "dense": r2})
        assert isinstance(table, str)
        assert "tfidf" in table
        assert "dense" in table
        # Should have some form of table
        assert "|" in table or "Strategy" in table

    def test_compare_reports_marks_best(self):
        from src.rag.benchmark import BenchmarkReport, compare_reports
        r1 = BenchmarkReport(
            strategy_name="tfidf", n_queries=12, n_queries_with_relevant=12,
            mean_precision_at_5=0.4, mean_recall_at_5=0.5, mrr=0.6,
            mean_ndcg_at_5=0.55, mean_latency_ms=3.0,
            index_build_time_seconds=0.05, cache_hit_rate=None,
        )
        r2 = BenchmarkReport(
            strategy_name="dense", n_queries=12, n_queries_with_relevant=12,
            mean_precision_at_5=0.8, mean_recall_at_5=0.9, mrr=0.95,
            mean_ndcg_at_5=0.9, mean_latency_ms=100.0,
            index_build_time_seconds=10.0, cache_hit_rate=0.0,
        )
        table = compare_reports({"tfidf": r1, "dense": r2})
        # The best marker should appear somewhere
        assert "★" in table or "*" in table or "best" in table.lower()

