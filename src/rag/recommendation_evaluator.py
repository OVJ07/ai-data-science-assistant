"""
src/rag/recommendation_evaluator.py
───────────────────────────────────
Evaluates the quality of recommendations produced under different retrieval
strategies (TF-IDF, Dense, Hybrid).

Evaluates:
1. Grounding / Support: Whether recommendations cite documents that were
   actually retrieved.
2. Technical Validity: Whether proposed parameters have valid types and values.
3. Relevance: Whether recommendations match the actual dataset profile
   (e.g., class_weight only for imbalanced datasets, imputation only for missing).
4. Pipeline Compatibility: Whether proposed configurations can be safely
   consumed by PreprocessingBuilder and ModelTrainer.
5. Application Status: Whether the recommendations were successfully applied
   or marked for manual review.
6. Diagnostics: LLM latency, parse failures, and fallback frequency if LLM was used.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.data.profiler import DatasetProfile
from src.rag.recommendation import Recommendation, RecommendationEngine, SearchSpaceConfig
from src.rag.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

# Valid keys that the pipeline knows how to consume
SUPPORTED_PIPELINE_KEYS = {
    "numerical_imputer_strategy": {"mean", "median", "most_frequent", "constant"},
    "categorical_imputer_strategy": {"most_frequent", "constant"},
    "max_categories": int,
    "class_weight": list,
    "cv_scoring": str,
    "exclude_columns": list,
    "exclude_or_engineer": list,
    "max_depth": list,
    "C": list,
}


@dataclass
class RecommendationEvaluationItem:
    """Evaluation of a single recommendation."""
    technique: str
    supporting_doc: str
    is_supported_by_retrieval: bool
    is_technically_valid: bool
    is_dataset_relevant: bool
    is_pipeline_compatible: bool
    application_status: str       # "applied" | "skipped" | "manual review" | "unsupported"
    notes: List[str] = field(default_factory=list)


@dataclass
class StrategyRecommendationReport:
    """Evaluation report for recommendations generated under one retriever strategy."""
    strategy_name: str
    retrieved_doc_ids: List[str]
    total_recommendations: int
    supported_count: int
    valid_count: int
    relevant_count: int
    compatible_count: int
    applied_count: int
    support_rate: float
    validity_rate: float
    relevance_rate: float
    compatibility_rate: float
    application_rate: float
    search_space_reasoning: List[str] = field(default_factory=list)
    applied_preprocessing: Dict[str, Any] = field(default_factory=dict)
    applied_param_grid_keys: List[str] = field(default_factory=list)
    evaluations: List[RecommendationEvaluationItem] = field(default_factory=list)
    llm_diagnostics: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_seconds: float = 0.0


class RecommendationEvaluator:
    """
    Evaluates recommendation quality across retrieval strategies on fixed
    dataset profiles and retrieved evidence.
    """

    def evaluate_recommendations(
        self,
        strategy_name: str,
        profile: DatasetProfile,
        retrieved_docs: List[RetrievedDocument],
        recommendations: List[Recommendation],
        search_config: Optional[SearchSpaceConfig] = None,
        llm_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> StrategyRecommendationReport:
        """
        Evaluate a set of recommendations generated for a given profile and retrieved documents.
        """
        start = time.perf_counter()
        retrieved_ids = {doc.doc_id for doc in retrieved_docs}
        eval_items: List[RecommendationEvaluationItem] = []

        supported_count = 0
        valid_count = 0
        relevant_count = 0
        compatible_count = 0
        applied_count = 0

        # Build application status lookup from search_config if available
        outcome_lookup = {}
        if search_config and search_config.recommendation_outcomes:
            for outcome in search_config.recommendation_outcomes:
                outcome_lookup[outcome["recommendation"]] = outcome["status"]

        for rec in recommendations:
            notes = []

            # 1. Supported by retrieved documents
            is_supported = rec.supporting_doc in retrieved_ids
            if is_supported:
                supported_count += 1
            else:
                notes.append(f"Supporting doc '{rec.supporting_doc}' was not in retrieved docs ({list(retrieved_ids)}). Fallback rule-based.")

            # 2. Technical validity
            is_valid = self._check_validity(rec, notes)
            if is_valid:
                valid_count += 1

            # 3. Relevance to dataset profile
            is_relevant = self._check_relevance(rec, profile, notes)
            if is_relevant:
                relevant_count += 1

            # 4. Pipeline compatibility
            is_compat = self._check_pipeline_compatibility(rec, notes)
            if is_compat:
                compatible_count += 1

            # 5. Application status
            app_status = outcome_lookup.get(rec.technique, "manual review")
            if app_status == "applied":
                applied_count += 1

            eval_items.append(RecommendationEvaluationItem(
                technique=rec.technique,
                supporting_doc=rec.supporting_doc,
                is_supported_by_retrieval=is_supported,
                is_technically_valid=is_valid,
                is_dataset_relevant=is_relevant,
                is_pipeline_compatible=is_compat,
                application_status=app_status,
                notes=notes,
            ))

        n_recs = len(recommendations)
        total_recs = max(n_recs, 1)

        applied_prep = search_config.preprocessing_config if search_config else {}
        applied_keys = list(search_config.param_grid.keys()) if search_config else []
        reasoning = search_config.reasoning if search_config else []

        duration = round(time.perf_counter() - start, 4)

        return StrategyRecommendationReport(
            strategy_name=strategy_name,
            retrieved_doc_ids=list(retrieved_ids),
            total_recommendations=n_recs,
            supported_count=supported_count,
            valid_count=valid_count,
            relevant_count=relevant_count,
            compatible_count=compatible_count,
            applied_count=applied_count,
            support_rate=round(supported_count / total_recs, 4),
            validity_rate=round(valid_count / total_recs, 4),
            relevance_rate=round(relevant_count / total_recs, 4),
            compatibility_rate=round(compatible_count / total_recs, 4),
            application_rate=round(applied_count / total_recs, 4),
            search_space_reasoning=reasoning,
            applied_preprocessing=applied_prep,
            applied_param_grid_keys=applied_keys,
            evaluations=eval_items,
            llm_diagnostics=llm_diagnostics or {},
            evaluation_time_seconds=duration,
        )

    def _check_validity(self, rec: Recommendation, notes: List[str]) -> bool:
        """Check whether recommendation config is technically well-formed."""
        if not isinstance(rec.config, dict):
            notes.append("Config is not a dictionary.")
            return False

        for k, v in rec.config.items():
            if k == "numerical_imputer_strategy":
                if v not in {"mean", "median", "most_frequent", "constant"}:
                    notes.append(f"Invalid numerical imputer strategy '{v}'.")
                    return False
            elif k == "categorical_imputer_strategy":
                if v not in {"most_frequent", "constant"}:
                    notes.append(f"Invalid categorical imputer strategy '{v}'.")
                    return False
            elif k == "max_categories":
                if not isinstance(v, int) or v <= 0:
                    notes.append(f"max_categories must be positive int, got {v}.")
                    return False
            elif k == "class_weight":
                if not isinstance(v, list) or not all(x in [None, "balanced"] for x in v):
                    notes.append(f"Invalid class_weight candidate list {v}.")
                    return False
        return True

    def _check_relevance(
        self,
        rec: Recommendation,
        profile: DatasetProfile,
        notes: List[str],
    ) -> bool:
        """Check whether recommendation is genuinely triggered by dataset traits."""
        cfg = rec.config
        if "numerical_imputer_strategy" in cfg or "categorical_imputer_strategy" in cfg:
            if not profile.has_missing:
                notes.append("Imputation suggested but dataset has no missing values.")
                return False
        if "class_weight" in cfg or (cfg.get("cv_scoring") == "balanced_accuracy"):
            if not profile.is_imbalanced and "classification" in (profile.problem_type or ""):
                notes.append("Class imbalance technique suggested but dataset is balanced.")
                return False
        if "max_categories" in cfg:
            if not profile.high_cardinality_features:
                notes.append("High cardinality category truncation suggested but no high-cardinality features detected.")
                return False
        if "exclude_columns" in cfg:
            if not profile.id_like_features and not profile.constant_features:
                notes.append("Column exclusion suggested without detected ID or constant features.")
                return False
        return True

    def _check_pipeline_compatibility(
        self,
        rec: Recommendation,
        notes: List[str],
    ) -> bool:
        """Check whether proposed configs can be consumed by the training pipeline."""
        cfg = rec.config
        if not cfg:
            return True
        for k in cfg:
            if k not in SUPPORTED_PIPELINE_KEYS:
                notes.append(f"Parameter '{k}' is advisory; not directly consumed by pipeline.")
                return False
        return True

