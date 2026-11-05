"""
src/rag/benchmark.py
─────────────────────
Reproducible retrieval benchmark comparing TF-IDF, dense, and hybrid strategies.

Metrics:
- Recall@5: |relevant ∩ retrieved@5| / |relevant|
- Precision@5: |relevant ∩ retrieved@5| / 5
- MRR: mean(1 / rank_of_first_relevant) over queries with ≥1 relevant doc
- nDCG@5: normalized discounted cumulative gain at cutoff 5 (graded relevance)
- Latency: wall-clock time per retrieve() call in milliseconds
- Index build time: seconds
- Cache hit rate (dense/hybrid only)

Queries with no relevant documents are excluded from Recall, Precision, MRR,
and nDCG calculations but are explicitly reported.
"""

import datetime
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from config.settings import EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR
from src.rag.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)

# ── Ground-truth query definitions ────────────────────────────────────────────


@dataclass
class BenchmarkQuery:
    """A single evaluation query with hand-labeled ground truth."""

    query_id: str
    query_text: str
    relevant_doc_ids: List[str]       # doc IDs from knowledge base (ground truth)
    relevance_grades: Dict[str, int]  # doc_id -> grade (2=highly relevant, 1=relevant)
    topic: str
    notes: str = ""


BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    BenchmarkQuery(
        query_id="Q01",
        query_text="high cardinality categorical features one-hot encoding dimensionality",
        relevant_doc_ids=["high_cardinality", "categorical_encoding"],
        relevance_grades={"high_cardinality": 2, "categorical_encoding": 2},
        topic="high_cardinality",
    ),
    BenchmarkQuery(
        query_id="Q02",
        query_text="missing values imputation strategy mean median KNN",
        relevant_doc_ids=["missing_values"],
        relevance_grades={"missing_values": 2},
        topic="missing_values",
    ),
    BenchmarkQuery(
        query_id="Q03",
        query_text="class imbalance minority class oversampling class_weight balanced",
        relevant_doc_ids=["class_imbalance", "evaluation_metrics"],
        relevance_grades={"class_imbalance": 2, "evaluation_metrics": 1},
        topic="class_imbalance",
    ),
    BenchmarkQuery(
        query_id="Q04",
        query_text="categorical encoding ordinal label encoding preprocessing",
        relevant_doc_ids=["categorical_encoding", "high_cardinality"],
        relevance_grades={"categorical_encoding": 2, "high_cardinality": 1},
        topic="categorical_encoding",
    ),
    BenchmarkQuery(
        query_id="Q05",
        query_text="feature scaling StandardScaler MinMaxScaler normalization numerical",
        relevant_doc_ids=["missing_values", "categorical_encoding"],
        relevance_grades={"missing_values": 1, "categorical_encoding": 1},
        topic="feature_scaling",
        notes="No dedicated scaling document; preprocessing docs are most relevant",
    ),
    BenchmarkQuery(
        query_id="Q06",
        query_text="model selection random forest logistic regression decision tree comparison",
        relevant_doc_ids=["random_forest", "logistic_regression", "decision_tree"],
        relevance_grades={"random_forest": 2, "logistic_regression": 2, "decision_tree": 2},
        topic="model_selection",
    ),
    BenchmarkQuery(
        query_id="Q07",
        query_text="hyperparameter tuning GridSearchCV search space optimization",
        relevant_doc_ids=["hyperparameter_tuning", "cross_validation"],
        relevance_grades={"hyperparameter_tuning": 2, "cross_validation": 1},
        topic="hyperparameter_tuning",
    ),
    BenchmarkQuery(
        query_id="Q08",
        query_text="cross-validation stratified k-fold train test split",
        relevant_doc_ids=["cross_validation", "hyperparameter_tuning"],
        relevance_grades={"cross_validation": 2, "hyperparameter_tuning": 1},
        topic="cross_validation",
    ),
    BenchmarkQuery(
        query_id="Q09",
        query_text="evaluation metrics accuracy F1 ROC AUC precision recall",
        relevant_doc_ids=["evaluation_metrics", "class_imbalance"],
        relevance_grades={"evaluation_metrics": 2, "class_imbalance": 1},
        topic="evaluation_metrics",
    ),
    BenchmarkQuery(
        query_id="Q10",
        query_text="SHAP feature importance explainability interpretable model",
        relevant_doc_ids=["explainable_ai"],
        relevance_grades={"explainable_ai": 2},
        topic="explainability",
    ),
    BenchmarkQuery(
        query_id="Q11",
        query_text="data leakage train test separation preprocessing pipeline",
        relevant_doc_ids=["cross_validation", "missing_values"],
        relevance_grades={"cross_validation": 2, "missing_values": 1},
        topic="data_leakage",
    ),
    BenchmarkQuery(
        query_id="Q12",
        query_text="regression model RMSE MAE R-squared continuous target",
        relevant_doc_ids=["regression", "evaluation_metrics"],
        relevance_grades={"regression": 2, "evaluation_metrics": 2},
        topic="regression",
    ),
]

# ── Result dataclasses ─────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Results for a single query evaluation."""

    query_id: str
    query_text: str
    topic: str
    retrieved_doc_ids: List[str]    # in rank order (top to bottom)
    retrieved_scores: List[float]
    relevant_doc_ids: List[str]     # ground truth
    precision_at_5: float
    recall_at_5: float
    mrr: float                      # 1/rank of first relevant, 0 if none
    ndcg_at_5: float
    latency_ms: float
    has_relevant_docs: bool         # False if relevant_doc_ids is empty
    notes: str = ""


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results for one retrieval strategy."""

    strategy_name: str                   # "tfidf" | "dense:model_name" | "hybrid:model_name"
    n_queries: int
    n_queries_with_relevant: int         # queries where has_relevant_docs=True
    mean_precision_at_5: float
    mean_recall_at_5: float
    mrr: float
    mean_ndcg_at_5: float
    mean_latency_ms: float
    index_build_time_seconds: float
    cache_hit_rate: Optional[float]      # None for TF-IDF
    per_query_results: List[QueryResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    limitation_notes: List[str] = field(default_factory=list)
    timestamp: str = ""


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _compute_precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Precision@k = |relevant ∩ retrieved[:k]| / k.  Returns 0.0 if k=0."""
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def _compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 5) -> float:
    """Recall@k = |relevant ∩ retrieved[:k]| / |relevant|.  Returns 0.0 if |relevant|=0."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def _compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """MRR contribution for one query.  Returns 0 if no relevant doc found."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _compute_ndcg_at_k(
    retrieved: List[str],
    relevance_grades: Dict[str, int],
    k: int = 5,
) -> float:
    """
    nDCG@k using graded relevance.

    DCG  = Σ_i( (2^grade_i - 1) / log2(i+1) )  for i in 1..k
    IDCG = DCG of ideal ranking (grades sorted descending)
    nDCG = DCG / IDCG.  Returns 0.0 if IDCG=0.
    """

    def dcg(ranked_grades: List[int]) -> float:
        return sum(
            (2 ** g - 1) / math.log2(i + 2)
            for i, g in enumerate(ranked_grades[:k])
        )

    retrieved_grades = [relevance_grades.get(doc_id, 0) for doc_id in retrieved[:k]]
    actual_dcg = dcg(retrieved_grades)

    ideal_grades = sorted(relevance_grades.values(), reverse=True)
    ideal_dcg = dcg(ideal_grades)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ── Main benchmark class ───────────────────────────────────────────────────────


class RetrievalBenchmark:
    """
    Runs retrieval benchmarks across strategies on fixed queries.

    Usage
    -----
    bench = RetrievalBenchmark(kb_dir=KNOWLEDGE_BASE_DIR)
    report = bench.run_tfidf()
    report = bench.run_dense(embedding_model="nomic-embed-text:latest")
    report = bench.run_hybrid(embedding_model="nomic-embed-text:latest")
    all_reports = bench.run_all(embedding_model="nomic-embed-text:latest")
    """

    _LIMITATION_NOTE = (
        "Ground truth relevance labels are human-defined for 12 queries covering 12 topics. "
        "Completeness is limited by knowledge base size."
    )

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.kb_dir: Path = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

    # ── Internal evaluation helpers ────────────────────────────────────────────

    def _evaluate_query(
        self,
        bq: BenchmarkQuery,
        retrieved_docs: list,
        latency_ms: float,
    ) -> QueryResult:
        """Compute per-query metrics given a list of RetrievedDocument objects."""
        retrieved_ids = [doc.doc_id for doc in retrieved_docs]
        retrieved_scores = [doc.relevance_score for doc in retrieved_docs]
        relevant_set: Set[str] = set(bq.relevant_doc_ids)
        has_relevant = bool(relevant_set)

        precision = _compute_precision_at_k(retrieved_ids, relevant_set, k=5)
        recall = _compute_recall_at_k(retrieved_ids, relevant_set, k=5)
        mrr = _compute_mrr(retrieved_ids, relevant_set)
        ndcg = _compute_ndcg_at_k(retrieved_ids, bq.relevance_grades, k=5)

        return QueryResult(
            query_id=bq.query_id,
            query_text=bq.query_text,
            topic=bq.topic,
            retrieved_doc_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_doc_ids=bq.relevant_doc_ids,
            precision_at_5=precision,
            recall_at_5=recall,
            mrr=mrr,
            ndcg_at_5=ndcg,
            latency_ms=latency_ms,
            has_relevant_docs=has_relevant,
            notes=bq.notes,
        )

    @staticmethod
    def _aggregate(
        per_query: List[QueryResult],
        strategy_name: str,
        build_time: float,
        cache_hit_rate: Optional[float],
        errors: List[str],
    ) -> BenchmarkReport:
        """Aggregate per-query results into a BenchmarkReport."""
        n_queries = len(per_query)
        queries_with_relevant = [q for q in per_query if q.has_relevant_docs]
        n_with_relevant = len(queries_with_relevant)

        if n_with_relevant > 0:
            mean_precision = sum(q.precision_at_5 for q in queries_with_relevant) / n_with_relevant
            mean_recall = sum(q.recall_at_5 for q in queries_with_relevant) / n_with_relevant
            mean_mrr = sum(q.mrr for q in queries_with_relevant) / n_with_relevant
            mean_ndcg = sum(q.ndcg_at_5 for q in queries_with_relevant) / n_with_relevant
        else:
            mean_precision = mean_recall = mean_mrr = mean_ndcg = 0.0

        mean_latency = (
            sum(q.latency_ms for q in per_query) / n_queries if n_queries > 0 else 0.0
        )

        return BenchmarkReport(
            strategy_name=strategy_name,
            n_queries=n_queries,
            n_queries_with_relevant=n_with_relevant,
            mean_precision_at_5=mean_precision,
            mean_recall_at_5=mean_recall,
            mrr=mean_mrr,
            mean_ndcg_at_5=mean_ndcg,
            mean_latency_ms=mean_latency,
            index_build_time_seconds=build_time,
            cache_hit_rate=cache_hit_rate,
            per_query_results=per_query,
            errors=errors,
            limitation_notes=[RetrievalBenchmark._LIMITATION_NOTE],
            timestamp=datetime.datetime.now().isoformat(),
        )

    # ── Public benchmark methods ───────────────────────────────────────────────

    def run_tfidf(
        self,
        queries: Optional[List[BenchmarkQuery]] = None,
    ) -> BenchmarkReport:
        """Benchmark the TF-IDF (sparse) retriever."""
        queries = queries or BENCHMARK_QUERIES
        errors: List[str] = []

        # Build index and time it
        retriever = KnowledgeRetriever(kb_dir=self.kb_dir)
        t0 = time.perf_counter()
        retriever.build_index()
        build_time = time.perf_counter() - t0

        if not retriever.is_built:
            errors.append("TF-IDF index failed to build — no documents loaded.")
            return BenchmarkReport(
                strategy_name="tfidf",
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=build_time,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )

        per_query: List[QueryResult] = []
        for bq in queries:
            t_start = time.perf_counter()
            try:
                docs = retriever.retrieve(bq.query_text, top_k=5, min_score=0.0)
            except Exception as exc:
                logger.warning("TF-IDF retrieve failed for %s: %s", bq.query_id, exc)
                errors.append(f"{bq.query_id}: {exc}")
                docs = []
            latency_ms = (time.perf_counter() - t_start) * 1_000
            per_query.append(self._evaluate_query(bq, docs, latency_ms))

        return self._aggregate(
            per_query,
            strategy_name="tfidf",
            build_time=build_time,
            cache_hit_rate=None,
            errors=errors,
        )

    def run_dense(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        queries: Optional[List[BenchmarkQuery]] = None,
    ) -> BenchmarkReport:
        """Benchmark the dense (Ollama embedding) retriever."""
        queries = queries or BENCHMARK_QUERIES
        strategy_name = f"dense:{embedding_model}"
        errors: List[str] = []

        # Late import so benchmark works even if dense module is absent / broken
        try:
            from src.rag.dense_retriever import (  # type: ignore[import]
                ModelNotInstalledError,
                OllamaDenseRetriever,
                OllamaUnavailableError,
            )
        except ImportError as exc:
            errors.append(f"Could not import dense retriever: {exc}")
            return BenchmarkReport(
                strategy_name=strategy_name,
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=0.0,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )

        retriever = OllamaDenseRetriever(
            kb_dir=self.kb_dir,
            embedding_model=embedding_model,
            cache_dir=self.cache_dir,
        )

        t0 = time.perf_counter()
        build_info = {}
        try:
            build_info = retriever.build_index()
        except (OllamaUnavailableError, ModelNotInstalledError) as exc:
            errors.append(f"Dense index build failed: {exc}")
        except Exception as exc:
            errors.append(f"Dense index build failed (unexpected): {exc}")

        build_time = time.perf_counter() - t0

        if not retriever.is_built:
            if not errors:
                err_msg = build_info.get("error") if isinstance(build_info, dict) else "Unknown build failure"
                errors.append(f"Dense index build failed: {err_msg}")
            return BenchmarkReport(
                strategy_name=strategy_name,
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=build_time,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )

        per_query: List[QueryResult] = []
        for bq in queries:
            t_start = time.perf_counter()
            try:
                docs = retriever.retrieve(bq.query_text, top_k=5, min_score=0.0)
            except Exception as exc:
                logger.warning("Dense retrieve failed for %s: %s", bq.query_id, exc)
                errors.append(f"{bq.query_id}: {exc}")
                docs = []
            latency_ms = (time.perf_counter() - t_start) * 1_000
            per_query.append(self._evaluate_query(bq, docs, latency_ms))

        # Cache hit rate (attribute may not exist on all implementations)
        cache_hit_rate: Optional[float] = getattr(retriever, "cache_hit_rate", None)
        if cache_hit_rate is None and hasattr(retriever, "cache_stats"):
            cache_hit_rate = retriever.cache_stats.get("hit_rate")

        return self._aggregate(
            per_query,
            strategy_name=strategy_name,
            build_time=build_time,
            cache_hit_rate=cache_hit_rate,
            errors=errors,
        )

    def run_hybrid(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        queries: Optional[List[BenchmarkQuery]] = None,
    ) -> BenchmarkReport:
        """Benchmark the hybrid (TF-IDF + dense) retriever."""
        queries = queries or BENCHMARK_QUERIES
        strategy_name = f"hybrid:{embedding_model}"
        errors: List[str] = []

        # Late import so benchmark works even if hybrid/dense modules are absent
        try:
            from src.rag.dense_retriever import (  # type: ignore[import]
                ModelNotInstalledError,
                OllamaDenseRetriever,
                OllamaUnavailableError,
            )
            from src.rag.hybrid_retriever import HybridRetriever  # type: ignore[import]
        except ImportError as exc:
            errors.append(f"Could not import hybrid retriever: {exc}")
            return BenchmarkReport(
                strategy_name=strategy_name,
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=0.0,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )

        tfidf_retriever = KnowledgeRetriever(kb_dir=self.kb_dir)
        dense_retriever = OllamaDenseRetriever(
            kb_dir=self.kb_dir,
            embedding_model=embedding_model,
        )

        t0 = time.perf_counter()
        tfidf_retriever.build_index()

        try:
            dense_retriever.build_index()
        except (OllamaUnavailableError, ModelNotInstalledError) as exc:
            errors.append(f"Hybrid: dense index build failed: {exc}")
            build_time = time.perf_counter() - t0
            return BenchmarkReport(
                strategy_name=strategy_name,
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=build_time,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )
        except Exception as exc:
            errors.append(f"Hybrid: dense index build failed (unexpected): {exc}")
            build_time = time.perf_counter() - t0
            return BenchmarkReport(
                strategy_name=strategy_name,
                n_queries=len(queries),
                n_queries_with_relevant=0,
                mean_precision_at_5=0.0,
                mean_recall_at_5=0.0,
                mrr=0.0,
                mean_ndcg_at_5=0.0,
                mean_latency_ms=0.0,
                index_build_time_seconds=build_time,
                cache_hit_rate=None,
                errors=errors,
                limitation_notes=[self._LIMITATION_NOTE],
                timestamp=datetime.datetime.now().isoformat(),
            )

        retriever = HybridRetriever(
            tfidf_retriever=tfidf_retriever,
            dense_retriever=dense_retriever,
        )
        build_time = time.perf_counter() - t0

        per_query: List[QueryResult] = []
        for bq in queries:
            t_start = time.perf_counter()
            try:
                docs = retriever.retrieve(bq.query_text, top_k=5, min_score=0.0)
            except Exception as exc:
                logger.warning("Hybrid retrieve failed for %s: %s", bq.query_id, exc)
                errors.append(f"{bq.query_id}: {exc}")
                docs = []
            latency_ms = (time.perf_counter() - t_start) * 1_000
            per_query.append(self._evaluate_query(bq, docs, latency_ms))

        cache_hit_rate: Optional[float] = getattr(dense_retriever, "cache_hit_rate", None)

        return self._aggregate(
            per_query,
            strategy_name=strategy_name,
            build_time=build_time,
            cache_hit_rate=cache_hit_rate,
            errors=errors,
        )

    def run_all(
        self,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> Dict[str, BenchmarkReport]:
        """
        Run all three strategies and return a mapping of strategy key → report.

        Returns
        -------
        Dict with keys "tfidf", "dense", "hybrid".
        """
        logger.info("Running full retrieval benchmark (tfidf + dense + hybrid)…")
        return {
            "tfidf": self.run_tfidf(),
            "dense": self.run_dense(embedding_model=embedding_model),
            "hybrid": self.run_hybrid(embedding_model=embedding_model),
        }

    # ── Comparison / reporting ─────────────────────────────────────────────────

    @staticmethod
    def compare_reports(reports: Dict[str, BenchmarkReport]) -> str:
        """
        Return a Markdown table comparing all strategies.

        Columns: Strategy, Precision@5, Recall@5, MRR, nDCG@5,
                 Latency(ms), Build Time(s), Cache Hit Rate.
        The best value in each numeric metric is marked with ★.
        """
        if not reports:
            return "_No reports to compare._"

        report_list = list(reports.values())

        # ── Find best values (higher is better for IR metrics; lower for latency/build) ──
        def _best(values: List[float], lower_is_better: bool = False) -> float:
            if not values:
                return float("nan")
            return min(values) if lower_is_better else max(values)

        precisions   = [r.mean_precision_at_5 for r in report_list]
        recalls      = [r.mean_recall_at_5 for r in report_list]
        mrrs         = [r.mrr for r in report_list]
        ndcgs        = [r.mean_ndcg_at_5 for r in report_list]
        latencies    = [r.mean_latency_ms for r in report_list]
        build_times  = [r.index_build_time_seconds for r in report_list]

        best_precision  = _best(precisions)
        best_recall     = _best(recalls)
        best_mrr        = _best(mrrs)
        best_ndcg       = _best(ndcgs)
        best_latency    = _best(latencies, lower_is_better=True)
        best_build      = _best(build_times, lower_is_better=True)

        def _fmt(value: float, best: float, lower_is_better: bool = False, pct: bool = False) -> str:
            """Format metric value, appending ★ if this is the best."""
            formatted = f"{value:.4f}" if not pct else f"{value:.2%}"
            is_best = (value == best) if not lower_is_better else (value == best)
            return f"{formatted} ★" if is_best else formatted

        def _fmt_cache(value: Optional[float]) -> str:
            if value is None:
                return "N/A"
            return f"{value:.2%}"

        # ── Header ────────────────────────────────────────────────────────────
        header = (
            "| Strategy | Precision@5 | Recall@5 | MRR | nDCG@5 | "
            "Latency (ms) | Build Time (s) | Cache Hit Rate |"
        )
        separator = (
            "|----------|-------------|----------|-----|--------|"
            "--------------|----------------|----------------|"
        )

        rows = [header, separator]

        for r in report_list:
            p_str   = _fmt(r.mean_precision_at_5, best_precision)
            rc_str  = _fmt(r.mean_recall_at_5, best_recall)
            mrr_str = _fmt(r.mrr, best_mrr)
            nd_str  = _fmt(r.mean_ndcg_at_5, best_ndcg)
            lat_str = _fmt(r.mean_latency_ms, best_latency, lower_is_better=True)
            bt_str  = _fmt(r.index_build_time_seconds, best_build, lower_is_better=True)
            ch_str  = _fmt_cache(r.cache_hit_rate)

            rows.append(
                f"| {r.strategy_name} | {p_str} | {rc_str} | {mrr_str} | "
                f"{nd_str} | {lat_str} | {bt_str} | {ch_str} |"
            )

        # ── Footer note ───────────────────────────────────────────────────────
        rows.append("")
        rows.append(
            "> **Note:** Ground truth relevance labels are human-defined for 12 queries "
            "covering 12 topics. Completeness is limited by knowledge base size. "
            "★ marks the best value in each column."
        )

        # Report any errors inline
        all_errors = [
            f"- **{r.strategy_name}**: {err}"
            for r in report_list
            for err in r.errors
        ]
        if all_errors:
            rows.append("")
            rows.append("### Errors / Warnings")
            rows.extend(all_errors)

        return "\n".join(rows)


def compare_reports(reports: Dict[str, BenchmarkReport]) -> str:
    """Convenience module-level alias for RetrievalBenchmark.compare_reports."""
    return RetrievalBenchmark.compare_reports(reports)

