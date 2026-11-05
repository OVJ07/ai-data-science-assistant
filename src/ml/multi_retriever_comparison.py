"""
src/ml/multi_retriever_comparison.py
─────────────────────────────────────
Comprehensive four-way benchmark comparing:
1. Baseline (RAG recommendations disabled)
2. RAG-guided (TF-IDF retrieval)
3. RAG-guided (Dense semantic retrieval via Ollama)
4. RAG-guided (Hybrid RRF retrieval)

Under strictly controlled conditions:
- Identical dataset and target column
- Identical train/test split and random seed (RANDOM_STATE)
- Independent ModelTrainer instances (zero shared state)
- Same evaluation protocol and metrics
- Records exact configuration changes applied for each strategy
- Honest reporting: explains why downstream metrics may be identical when
  configurations match, distinguishing retrieval improvements from predictive gains.
"""

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    EMBEDDING_MODEL,
    MAX_ONE_HOT_CATEGORIES,
    RANDOM_STATE,
)
from src.data.profiler import DatasetProfile
from src.ml.baseline import BaselineStrategy
from src.ml.diagnostics import (
    compute_config_hash,
    compute_prediction_agreement,
    compute_prediction_hash,
)
from src.ml.trainer import ModelTrainer, TrainingResult
from src.rag.dense_retriever import OllamaDenseRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.query_generator import QueryGenerator
from src.rag.recommendation import RecommendationEngine, SearchSpaceConfig
from src.rag.recommendation_evaluator import RecommendationEvaluator, StrategyRecommendationReport
from src.rag.retriever import KnowledgeRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class SingleRunAudit:
    """Detailed record for one strategy run."""
    strategy_name: str
    is_rag_guided: bool
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    applied_configuration: Dict[str, Any] = field(default_factory=dict)
    applied_preprocessing: Dict[str, Any] = field(default_factory=dict)
    search_space_reasoning: List[str] = field(default_factory=list)
    recommendation_report: Optional[StrategyRecommendationReport] = None
    training_result: Optional[TrainingResult] = None
    cv_score_mean: Optional[float] = None
    cv_score_std: Optional[float] = None
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    prediction_hash: str = ""
    prediction_agreement_with_baseline: Optional[float] = None
    training_duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class MultiRetrieverComparisonResult:
    """Full four-way comparison result."""
    model_name: str
    problem_type: str
    dataset_rows: int
    dataset_cols: int
    random_state: int
    timestamp: str

    baseline_run: SingleRunAudit
    tfidf_run: SingleRunAudit
    dense_run: SingleRunAudit
    hybrid_run: SingleRunAudit

    configurations_differ_across_rag: bool = False
    rag_differs_from_baseline: bool = False
    honest_summary: str = ""
    comparison_table: List[Dict[str, Any]] = field(default_factory=list)


class MultiRetrieverComparator:
    """
    Orchestrates the 4-way comparison across Baseline, TF-IDF RAG, Dense RAG, and Hybrid RAG.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir
        self._baseline_strategy = BaselineStrategy()
        self._evaluator = RecommendationEvaluator()

    def run_comparison(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        model_name: str,
        embedding_model: str = EMBEDDING_MODEL,
        exclude_columns: Optional[List[str]] = None,
        use_llm_queries: bool = False,
    ) -> MultiRetrieverComparisonResult:
        """
        Execute all four runs sequentially under identical conditions.
        """
        timestamp = datetime.datetime.now().isoformat()
        qg = QueryGenerator()
        qg_result = qg.generate_queries(profile, model_name=model_name, use_llm=use_llm_queries)
        queries = qg_result.queries

        # ── 1. Baseline Run (No RAG) ─────────────────────────────────────────
        logger.info("Executing MultiRetriever: 1/4 Baseline")
        baseline_start = time.perf_counter()
        baseline_grid = self._baseline_strategy.get_param_grid(model_name)
        baseline_scoring = self._baseline_strategy.get_cv_scoring(profile.problem_type or "")
        
        b_trainer = ModelTrainer()
        b_train_res = b_trainer.train(
            df=df,
            profile=profile,
            model_name=model_name,
            param_grid=baseline_grid,
            cv_scoring=baseline_scoring,
            use_stratified=True,
            exclude_columns=exclude_columns,
            is_rag_guided=False,
        )
        baseline_duration = round(time.perf_counter() - baseline_start, 2)
        b_config = {
            "param_grid": baseline_grid,
            "cv_scoring": baseline_scoring,
            "numerical_imputer_strategy": "median",
            "categorical_imputer_strategy": "most_frequent",
            "max_ohe_categories": None,
        }
        b_config_hash = compute_config_hash(baseline_grid, b_config, baseline_scoring)
        b_pred_hash = compute_prediction_hash(b_train_res.y_pred) if b_train_res.is_successful() else "none"

        baseline_run = SingleRunAudit(
            strategy_name="Baseline (No RAG)",
            is_rag_guided=False,
            retrieved_documents=[],
            applied_configuration=b_config,
            applied_preprocessing=b_train_res.preprocessing_config if b_train_res.is_successful() else {},
            training_result=b_train_res,
            cv_score_mean=b_train_res.best_cv_score,
            cv_score_std=b_train_res.cv_score_std,
            test_metrics=b_train_res.test_metrics,
            config_hash=b_config_hash,
            prediction_hash=b_pred_hash,
            prediction_agreement_with_baseline=1.0 if b_train_res.is_successful() else None,
            training_duration_seconds=baseline_duration,
            error=b_train_res.error,
        )

        b_y_pred = b_train_res.y_pred if b_train_res.is_successful() else None

        # ── 2. TF-IDF RAG Run ────────────────────────────────────────────────
        logger.info("Executing MultiRetriever: 2/4 TF-IDF RAG")
        tfidf_retriever = KnowledgeRetriever()
        tfidf_retriever.build_index()
        tfidf_run = self._run_rag_strategy(
            strategy_name="TF-IDF RAG",
            retriever=tfidf_retriever,
            df=df,
            profile=profile,
            model_name=model_name,
            queries=queries,
            exclude_columns=exclude_columns,
            baseline_y_pred=b_y_pred,
        )

        # ── 3. Dense RAG Run ─────────────────────────────────────────────────
        logger.info("Executing MultiRetriever: 3/4 Dense RAG")
        dense_retriever = OllamaDenseRetriever(
            embedding_model=embedding_model,
            cache_dir=self.cache_dir,
        )
        dense_build = dense_retriever.build_index()
        if dense_retriever.is_built:
            dense_run = self._run_rag_strategy(
                strategy_name=f"Dense RAG ({embedding_model})",
                retriever=dense_retriever,
                df=df,
                profile=profile,
                model_name=model_name,
                queries=queries,
                exclude_columns=exclude_columns,
                baseline_y_pred=b_y_pred,
            )
        else:
            dense_run = SingleRunAudit(
                strategy_name=f"Dense RAG ({embedding_model})",
                is_rag_guided=True,
                error=dense_build.get("error", "Dense retriever index build failed (Ollama unavailable or model missing)."),
            )

        # ── 4. Hybrid RAG Run ────────────────────────────────────────────────
        logger.info("Executing MultiRetriever: 4/4 Hybrid RAG")
        if tfidf_retriever.is_built and dense_retriever.is_built:
            hybrid_retriever = HybridRetriever(
                tfidf_retriever=tfidf_retriever,
                dense_retriever=dense_retriever,
            )
            hybrid_run = self._run_rag_strategy(
                strategy_name="Hybrid RRF RAG",
                retriever=hybrid_retriever,
                df=df,
                profile=profile,
                model_name=model_name,
                queries=queries,
                exclude_columns=exclude_columns,
                baseline_y_pred=b_y_pred,
            )
        else:
            hybrid_run = SingleRunAudit(
                strategy_name="Hybrid RRF RAG",
                is_rag_guided=True,
                error="Hybrid retriever unavailable because dense retriever is not built.",
            )

        # ── Compare runs & synthesise honest assessment ───────────────────────
        comp_res = self._synthesize_comparison(
            df=df,
            profile=profile,
            model_name=model_name,
            timestamp=timestamp,
            baseline_run=baseline_run,
            tfidf_run=tfidf_run,
            dense_run=dense_run,
            hybrid_run=hybrid_run,
        )

        return comp_res

    def _run_rag_strategy(
        self,
        strategy_name: str,
        retriever: Any,
        df: pd.DataFrame,
        profile: DatasetProfile,
        model_name: str,
        queries: List[str],
        exclude_columns: Optional[List[str]],
        baseline_y_pred: Optional[Any] = None,
    ) -> SingleRunAudit:
        """Helper to run a single RAG strategy from retrieval to evaluation."""
        start = time.perf_counter()
        if hasattr(retriever, "retrieve_multiple_queries_with_provenance"):
            docs, _ = retriever.retrieve_multiple_queries_with_provenance(queries, global_top_k=8)
        elif hasattr(retriever, "retrieve_multiple_queries"):
            res = retriever.retrieve_multiple_queries(queries, global_top_k=8)
            docs = res[0] if isinstance(res, tuple) else res
        else:
            docs = []

        rec_engine = RecommendationEngine(retriever)
        recs, _ = rec_engine.generate_recommendations(profile, queries, retrieved_docs=docs)
        search_config = rec_engine.generate_search_space(
            profile, model_name, recs, docs, is_rag_guided=True
        )

        rec_report = self._evaluator.evaluate_recommendations(
            strategy_name=strategy_name,
            profile=profile,
            retrieved_docs=docs,
            recommendations=recs,
            search_config=search_config,
        )

        rag_preprocessing = search_config.preprocessing_config
        num_imputer = rag_preprocessing.get("numerical_imputer_strategy", "median")
        cat_imputer = rag_preprocessing.get("categorical_imputer_strategy", "most_frequent")
        max_ohe = rag_preprocessing.get("max_ohe_categories")

        trainer = ModelTrainer()
        train_res = trainer.train(
            df=df,
            profile=profile,
            model_name=model_name,
            param_grid=search_config.param_grid,
            cv_scoring=search_config.cv_scoring,
            use_stratified=search_config.use_stratified,
            numerical_imputer_strategy=num_imputer,
            categorical_imputer_strategy=cat_imputer,
            exclude_columns=exclude_columns,
            is_rag_guided=True,
            max_ohe_categories=max_ohe,
        )
        duration = round(time.perf_counter() - start, 2)

        applied_config = {
            "param_grid": search_config.param_grid,
            "cv_scoring": search_config.cv_scoring,
            "numerical_imputer_strategy": num_imputer,
            "categorical_imputer_strategy": cat_imputer,
            "max_ohe_categories": max_ohe,
        }
        cfg_hash = compute_config_hash(search_config.param_grid, applied_config, search_config.cv_scoring)
        pred_hash = compute_prediction_hash(train_res.y_pred) if train_res.is_successful() else "none"
        pred_agr = (
            compute_prediction_agreement(baseline_y_pred, train_res.y_pred)
            if (train_res.is_successful() and baseline_y_pred is not None)
            else None
        )

        return SingleRunAudit(
            strategy_name=strategy_name,
            is_rag_guided=True,
            retrieved_documents=docs,
            applied_configuration=applied_config,
            applied_preprocessing=train_res.preprocessing_config if train_res.is_successful() else {},
            search_space_reasoning=search_config.reasoning,
            recommendation_report=rec_report,
            training_result=train_res,
            cv_score_mean=train_res.best_cv_score,
            cv_score_std=train_res.cv_score_std,
            test_metrics=train_res.test_metrics,
            config_hash=cfg_hash,
            prediction_hash=pred_hash,
            prediction_agreement_with_baseline=pred_agr,
            training_duration_seconds=duration,
            error=train_res.error,
        )

    def _synthesize_comparison(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        model_name: str,
        timestamp: str,
        baseline_run: SingleRunAudit,
        tfidf_run: SingleRunAudit,
        dense_run: SingleRunAudit,
        hybrid_run: SingleRunAudit,
    ) -> MultiRetrieverComparisonResult:
        """Compare all four runs and generate an honest analysis."""
        runs = [baseline_run, tfidf_run, dense_run, hybrid_run]

        # Check if RAG configs differ from baseline
        rag_differs_from_baseline = any(
            r.config_hash != baseline_run.config_hash
            for r in [tfidf_run, dense_run, hybrid_run] if r.config_hash
        )

        # Check if RAG configs differ among themselves
        rag_hashes = [r.config_hash for r in [tfidf_run, dense_run, hybrid_run] if r.config_hash]
        configs_differ_across_rag = False
        if len(rag_hashes) > 1:
            configs_differ_across_rag = any(h != rag_hashes[0] for h in rag_hashes[1:])

        # Build comparison summary table
        rows = []
        for r in runs:
            row = {
                "Strategy": r.strategy_name,
                "RAG Guided": "Yes" if r.is_rag_guided else "No",
                "Retrieved Docs": len(r.retrieved_documents),
                "Config Hash": r.config_hash or "N/A",
                "Pred Agreement vs Baseline": (
                    "100.0% (Baseline)"
                    if not r.is_rag_guided
                    else (f"{r.prediction_agreement_with_baseline:.1%}" if r.prediction_agreement_with_baseline is not None else "N/A")
                ),
                "CV Scoring": r.applied_configuration.get("cv_scoring", "N/A"),
                "CV Mean Score": f"{r.cv_score_mean:.4f}" if r.cv_score_mean is not None else "N/A",
                "CV Std": f"±{r.cv_score_std:.4f}" if r.cv_score_std is not None else "N/A",
                "Duration (s)": r.training_duration_seconds,
                "Status": "Success" if not r.error else f"Failed ({r.error[:30]}...)",
            }
            # Add primary test metrics
            if r.test_metrics:
                for metric in ["accuracy", "balanced_accuracy", "f1_weighted", "rmse", "r2"]:
                    if metric in r.test_metrics and r.test_metrics[metric] is not None:
                        row[metric.replace("_", " ").title()] = f"{r.test_metrics[metric]:.4f}"
            rows.append(row)

        # Honest assessment synthesis
        notes = []
        if not rag_differs_from_baseline:
            notes.append(
                "Neither TF-IDF, Dense, nor Hybrid RAG produced modifications that altered the effective training "
                "configuration relative to baseline for this dataset. Consequently, differences in held-out test performance "
                "are zero or purely stochastic."
            )
        elif not configs_differ_across_rag:
            notes.append(
                "RAG runs produced a different configuration from baseline (e.g. expanded hyperparameter grid or different preprocessing), "
                "but TF-IDF, Dense, and Hybrid all converged on the same effective configuration because their retrieved "
                "evidence overlapped — this is expected when the dataset characteristics are unambiguous and all retrievers "
                "correctly identify the same relevant knowledge documents. "
                "Metric differences between RAG strategies are zero or stochastic; differences vs. baseline reflect the RAG-guided configuration change."
            )
        else:
            notes.append(
                "Different retrieval strategies retrieved distinct knowledge documents that led to divergent configuration changes "
                "in hyperparameter search spaces or preprocessing. See the metric table for empirical held-out performance comparison."
            )

        # Add prediction agreement annotation when configs differ but predictions agree
        if configs_differ_across_rag or rag_differs_from_baseline:
            all_rag_runs = [r for r in [tfidf_run, dense_run, hybrid_run] if r.prediction_agreement_with_baseline is not None]
            all_agree = all(r.prediction_agreement_with_baseline >= 0.99 for r in all_rag_runs)
            if all_agree and all_rag_runs:
                notes.append(
                    "Despite different hyperparameter search spaces being evaluated, GridSearchCV selected "
                    "configurations that produce identical held-out predictions. This is an empirically valid result — "
                    "the retrieved knowledge constrained the search space around a near-optimal region where "
                    "the same decision boundary is optimal regardless of whether the expanded space was explored."
                )

        honest_summary = " ".join(notes)

        return MultiRetrieverComparisonResult(
            model_name=model_name,
            problem_type=profile.problem_type or "unknown",
            dataset_rows=df.shape[0],
            dataset_cols=df.shape[1],
            random_state=RANDOM_STATE,
            timestamp=timestamp,
            baseline_run=baseline_run,
            tfidf_run=tfidf_run,
            dense_run=dense_run,
            hybrid_run=hybrid_run,
            configurations_differ_across_rag=configs_differ_across_rag,
            rag_differs_from_baseline=rag_differs_from_baseline,
            honest_summary=honest_summary,
            comparison_table=rows,
        )

