"""
src/rag/recommendation.py
─────────────────────────
Structured preprocessing and hyperparameter recommendations
driven by retrieved knowledge and explicit dataset-characteristic rules.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data.profiler import DatasetProfile
from src.rag.retriever import KnowledgeRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single structured recommendation with evidence."""
    technique: str
    trigger: str                    # Dataset characteristic that triggered this
    reason: str
    supporting_doc: str             # doc_id of supporting knowledge-base document
    config: Dict[str, Any] = field(default_factory=dict)   # Proposed config
    limitations: str = ""
    priority: int = 5               # 1=high, 10=low
    evidence_status: str = "rule-based"  # retrieved | rule-based; surfaced for auditability


@dataclass
class SearchSpaceConfig:
    """A generated hyperparameter search space for a model."""
    model_name: str
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    cv_scoring: str = "accuracy"
    use_stratified: bool = True
    reasoning: List[str] = field(default_factory=list)
    retrieved_docs: List[str] = field(default_factory=list)
    is_rag_guided: bool = False
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    application_notes: List[str] = field(default_factory=list)
    recommendation_outcomes: List[Dict[str, str]] = field(default_factory=list)


class RecommendationEngine:
    """
    Combines rule-based logic with retrieved knowledge to produce
    structured preprocessing and hyperparameter recommendations.
    """

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever

    def generate_recommendations(
        self,
        profile: DatasetProfile,
        queries: List[str],
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
        strict_grounding: bool = True,
    ) -> tuple:
        """
        Generate preprocessing recommendations from dataset profile + retrieval.

        When strict_grounding=True (default), recommendations are strictly generated
        only when supported by retrieved knowledge-base documents, ensuring downstream
        pipeline behavior genuinely reflects retrieval strategy quality.

        Returns
        -------
        (recommendations, retrieved_documents)
        """
        # Retrieve relevant knowledge
        raw_docs = retrieved_docs if retrieved_docs is not None else self.retriever.retrieve_multiple_queries(
            queries, top_k_per_query=3, global_top_k=8
        )
        docs = raw_docs[0] if isinstance(raw_docs, tuple) else (raw_docs or [])
        retrieved_ids = {doc.doc_id for doc in docs}

        recommendations: List[Recommendation] = []

        # ── Missing value recommendations ─────────────────────────────────────
        if profile.has_missing and ("missing_values" in retrieved_ids or not strict_grounding):
            rate = profile.missing_rate_overall
            strategy = "median" if rate > 0.05 else "mean"
            recommendations.append(Recommendation(
                technique=f"Numerical Imputation (strategy='{strategy}')",
                trigger=f"Missing values detected ({rate*100:.1f}% overall missing rate)",
                reason=(
                    f"The dataset has {rate*100:.1f}% missing values overall. "
                    f"{'Median imputation is more robust to outliers with higher missing rates.' if rate > 0.05 else 'Mean imputation is appropriate for low missing rates.'}"
                ),
                supporting_doc="missing_values",
                config={"numerical_imputer_strategy": strategy},
                limitations="Imputation must be fit on training data only to prevent data leakage.",
                priority=1,
                evidence_status="retrieved" if "missing_values" in retrieved_ids else "rule-based",
            ))
            recommendations.append(Recommendation(
                technique="Categorical Imputation (strategy='most_frequent')",
                trigger="Missing values in categorical columns",
                reason=(
                    "Categorical columns with missing values should use mode imputation "
                    "or a dedicated 'Unknown' category to preserve missingness information."
                ),
                supporting_doc="missing_values",
                config={"categorical_imputer_strategy": "most_frequent"},
                limitations="Mode imputation may introduce bias if the most frequent category is not representative.",
                priority=1,
                evidence_status="retrieved" if "missing_values" in retrieved_ids else "rule-based",
            ))

        # ── Class imbalance recommendations ───────────────────────────────────
        if profile.is_imbalanced and ("class_imbalance" in retrieved_ids or not strict_grounding):
            recommendations.append(Recommendation(
                technique="class_weight='balanced'",
                trigger=(
                    f"Class imbalance detected: minority/majority ratio = {profile.imbalance_ratio:.3f}"
                ),
                reason=(
                    "The dataset has significant class imbalance. "
                    "Setting class_weight='balanced' adjusts the loss function to give "
                    "more importance to the minority class without resampling."
                ),
                supporting_doc="class_imbalance",
                config={"class_weight": ["balanced", None]},
                limitations=(
                    "class_weight='balanced' does not create new samples; it only reweights. "
                    "May trade precision for recall."
                ),
                priority=1,
                evidence_status="retrieved" if "class_imbalance" in retrieved_ids else "rule-based",
            ))
            recommendations.append(Recommendation(
                technique="Evaluation Metric: Balanced Accuracy / F1-weighted",
                trigger="Class imbalance",
                reason=(
                    "Standard accuracy is misleading for imbalanced datasets. "
                    "Use balanced_accuracy or f1_weighted for cross-validation scoring."
                ),
                supporting_doc="evaluation_metrics" if "evaluation_metrics" in retrieved_ids else "class_imbalance",
                config={"cv_scoring": "balanced_accuracy"},
                limitations="Metric choice affects which model configuration is selected by GridSearchCV.",
                priority=2,
                evidence_status="retrieved" if ("evaluation_metrics" in retrieved_ids or "class_imbalance" in retrieved_ids) else "rule-based",
            ))

        # ── High-cardinality categorical recommendations ───────────────────────
        if profile.high_cardinality_features and (
            "high_cardinality" in retrieved_ids or "categorical_encoding" in retrieved_ids or not strict_grounding
        ):
            recommendations.append(Recommendation(
                technique="Limited One-Hot Encoding (top-K categories)",
                trigger=f"High-cardinality features: {profile.high_cardinality_features}",
                reason=(
                    "High-cardinality categorical features create many OHE columns, "
                    "increasing dimensionality and risk of overfitting. "
                    "Limiting to the top-K most frequent categories reduces this risk."
                ),
                supporting_doc="high_cardinality" if "high_cardinality" in retrieved_ids else "categorical_encoding",
                config={"max_categories": 15, "handle_unknown": "infrequent_if_exist"},
                limitations="Categories outside top-K are grouped into 'infrequent'. Some information is lost.",
                priority=2,
                evidence_status="retrieved" if ("high_cardinality" in retrieved_ids or "categorical_encoding" in retrieved_ids) else "rule-based",
            ))

        # ── Text/name field recommendations ───────────────────────────────────
        if profile.text_like_features and (
            "high_cardinality" in retrieved_ids or "categorical_encoding" in retrieved_ids or not strict_grounding
        ):
            recommendations.append(Recommendation(
                technique="Feature engineering for text/name columns",
                trigger=f"Text/name-like features detected: {profile.text_like_features}",
                reason=(
                    "Text or name fields may contain useful extractable information "
                    "(e.g., titles in passenger names like 'Mr.', 'Mrs.', 'Dr.'). "
                    "Consider extracting structured features rather than encoding raw text."
                ),
                supporting_doc="high_cardinality" if "high_cardinality" in retrieved_ids else "categorical_encoding",
                config={"exclude_or_engineer": profile.text_like_features},
                limitations="Text feature engineering requires domain knowledge.",
                priority=3,
                evidence_status="retrieved" if ("high_cardinality" in retrieved_ids or "categorical_encoding" in retrieved_ids) else "rule-based",
            ))

        # ── ID-like column recommendations ────────────────────────────────────
        if profile.id_like_features and ("high_cardinality" in retrieved_ids or not strict_grounding):
            recommendations.append(Recommendation(
                technique=f"Exclude ID columns: {profile.id_like_features}",
                trigger="ID-like columns detected with unique ratio close to 1.0",
                reason=(
                    "Columns that are unique per row (database ID columns) carry no generalizable signal "
                    "and will cause severe overfitting if included."
                ),
                supporting_doc="high_cardinality",
                config={"exclude_columns": profile.id_like_features},
                limitations="Verify these are truly ID columns before excluding.",
                priority=1,
                evidence_status="retrieved" if "high_cardinality" in retrieved_ids else "rule-based",
            ))

        # ── Dataset size recommendations ──────────────────────────────────────
        if profile.n_rows < 200 and (
            "cross_validation" in retrieved_ids or "hyperparameter_tuning" in retrieved_ids or not strict_grounding
        ):
            recommendations.append(Recommendation(
                technique="Reduce model complexity",
                trigger=f"Small dataset ({profile.n_rows} rows)",
                reason=(
                    "With fewer than 200 samples, complex models are prone to overfitting. "
                    "Prefer simpler models (Logistic Regression, shallow Decision Tree) "
                    "or heavily regularized versions."
                ),
                supporting_doc="cross_validation" if "cross_validation" in retrieved_ids else "hyperparameter_tuning",
                config={"max_depth": [3, 5], "C": [0.01, 0.1, 1]},
                limitations="Small dataset estimates are high-variance regardless of model choice.",
                priority=2,
                evidence_status="retrieved" if ("cross_validation" in retrieved_ids or "hyperparameter_tuning" in retrieved_ids) else "rule-based",
            ))

        # ── Evaluation metrics recommendations (general) ──────────────────────
        if "evaluation_metrics" in retrieved_ids:
            is_class = "classification" in (profile.problem_type or "")
            metric_desc = "F1-weighted and ROC-AUC" if is_class else "RMSE and MAE"
            recommendations.append(Recommendation(
                technique=f"Evaluation Metrics: Multi-Metric Assessment ({metric_desc})",
                trigger=f"Problem type: {profile.problem_type or 'classification'}",
                reason=(
                    "Evaluation metrics knowledge base recommends assessing multiple complementary metrics "
                    "rather than relying on a single summary metric."
                ),
                supporting_doc="evaluation_metrics",
                config={"primary_metric": "f1_weighted" if is_class else "neg_root_mean_squared_error"},
                limitations="Multi-metric evaluation requires balancing trade-offs between precision and recall.",
                priority=3,
                evidence_status="retrieved",
            ))

        # ── Model-specific knowledge recommendations ─────────────────────────
        if "decision_tree" in retrieved_ids:
            recommendations.append(Recommendation(
                technique="Decision Tree Tuning: Criterion & Split Search",
                trigger="Decision Tree knowledge retrieved",
                reason=(
                    "Decision tree guidance recommends exploring both Gini impurity and Entropy "
                    "criteria, along with minimum leaf sample constraints."
                ),
                supporting_doc="decision_tree",
                config={"criterion": ["gini", "entropy"]},
                limitations="Entropy calculation is slightly slower due to logarithmic computation.",
                priority=3,
                evidence_status="retrieved",
            ))

        if "random_forest" in retrieved_ids:
            recommendations.append(Recommendation(
                technique="Random Forest Tuning: Feature Subsampling Diversity",
                trigger="Random Forest knowledge retrieved",
                reason=(
                    "Random forest guidance recommends evaluating feature subsampling ('sqrt' vs 'log2') "
                    "to enhance tree decorrelation and ensemble performance."
                ),
                supporting_doc="random_forest",
                config={"max_features": ["sqrt", "log2"]},
                limitations="Searching max_features expands GridSearchCV search time.",
                priority=3,
                evidence_status="retrieved",
            ))

        if "logistic_regression" in retrieved_ids:
            recommendations.append(Recommendation(
                technique="Logistic Regression Tuning: Regularization Spectrum",
                trigger="Logistic Regression knowledge retrieved",
                reason=(
                    "Logistic regression guidance recommends evaluating a wide logarithmic C range "
                    "from 0.01 to 100.0 to identify optimal regularization strength."
                ),
                supporting_doc="logistic_regression",
                config={"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
                limitations="Extreme C values may take longer to converge.",
                priority=3,
                evidence_status="retrieved",
            ))

        logger.info(
            "Generated %d recommendations from %d retrieved documents (retrieved IDs: %s).",
            len(recommendations),
            len(docs),
            sorted(list(retrieved_ids)),
        )
        return recommendations, docs

    def generate_search_space(
        self,
        profile: DatasetProfile,
        model_name: str,
        recommendations: List[Recommendation],
        retrieved_docs: List[RetrievedDocument],
        is_rag_guided: bool = True,
    ) -> SearchSpaceConfig:
        """
        Build a hyperparameter search space for the given model and profile.

        Uses BASELINE_GRIDS from model_factory as the single source of truth,
        then applies RAG-guided modifications when is_rag_guided=True, strictly
        conditioned on retrieved knowledge.
        """
        from src.ml.model_factory import BASELINE_GRIDS

        config = SearchSpaceConfig(
            model_name=model_name,
            retrieved_docs=[d.doc_id for d in retrieved_docs],
            is_rag_guided=is_rag_guided,
        )

        retrieved_ids = {d.doc_id for d in retrieved_docs}
        has_imbalance_retrieval = "class_imbalance" in retrieved_ids or any(
            "class_weight" in r.config for r in recommendations
        )

        # ── Determine CV scoring ──────────────────────────────────────────────
        if "classification" in (profile.problem_type or ""):
            if profile.is_imbalanced and is_rag_guided and has_imbalance_retrieval:
                config.cv_scoring = "balanced_accuracy"
                config.reasoning.append(
                    "Using balanced_accuracy scoring due to detected class imbalance and retrieved guidance."
                )
            else:
                config.cv_scoring = "f1_weighted"
            config.use_stratified = True
        else:
            config.cv_scoring = "neg_root_mean_squared_error"
            config.use_stratified = False

        # ── Base search space from BASELINE_GRIDS (single source of truth) ────
        config.param_grid = dict(BASELINE_GRIDS.get(model_name, {}))

        if not config.param_grid:
            logger.debug("No baseline param grid found for model '%s'", model_name)

        # ── RAG-guided modifications ──────────────────────────────────────────
        if is_rag_guided:
            self._apply_rag_modifications(config, profile, recommendations, retrieved_docs)
            # Propagate only settings consumed by ModelTrainer. Other advice
            # remains a visible suggestion rather than a silent action.
            for rec in recommendations:
                for source_key, target_key in (
                    ("numerical_imputer_strategy", "numerical_imputer_strategy"),
                    ("categorical_imputer_strategy", "categorical_imputer_strategy"),
                    ("max_categories", "max_ohe_categories"),
                ):
                    if source_key in rec.config:
                        config.preprocessing_config[target_key] = rec.config[source_key]
            for key, value in config.preprocessing_config.items():
                config.application_notes.append(f"Applied RAG recommendation: {key}={value!r}.")
            for rec in recommendations:
                applied_key = next((key for key in rec.config if key in {
                    "numerical_imputer_strategy", "categorical_imputer_strategy", "max_categories"
                }), None)
                if applied_key:
                    config.recommendation_outcomes.append({
                        "recommendation": rec.technique,
                        "status": "applied",
                        "reason": f"Mapped {applied_key} to the training preprocessor.",
                    })
                elif "class_weight" in rec.config:
                    applied = any("class_weight" in key for key in config.param_grid)
                    config.recommendation_outcomes.append({
                        "recommendation": rec.technique,
                        "status": "applied" if applied else "skipped",
                        "reason": "Added to the estimator search space." if applied else "Selected estimator does not support class_weight.",
                    })
                elif "cv_scoring" in rec.config:
                    config.recommendation_outcomes.append({
                        "recommendation": rec.technique,
                        "status": "applied" if config.cv_scoring == rec.config["cv_scoring"] else "skipped",
                        "reason": f"Effective CV scoring is {config.cv_scoring}.",
                    })
                elif "criterion" in rec.config or "max_features" in rec.config or "C" in rec.config:
                    applied = any(k in key for key in config.param_grid for k in ["criterion", "max_features", "classifier__C"])
                    config.recommendation_outcomes.append({
                        "recommendation": rec.technique,
                        "status": "applied" if applied else "skipped",
                        "reason": "Added to hyperparameter search space from model knowledge." if applied else "Not applicable to model.",
                    })
                else:
                    config.recommendation_outcomes.append({
                        "recommendation": rec.technique,
                        "status": "manual review",
                        "reason": "Not safely automatable by the current preprocessing/training pipeline.",
                    })
            if any("exclude_columns" in rec.config or "exclude_or_engineer" in rec.config for rec in recommendations):
                config.application_notes.append("Feature exclusion/text engineering requires user confirmation and is not applied silently.")

        return config

    def _apply_rag_modifications(
        self,
        config: SearchSpaceConfig,
        profile: DatasetProfile,
        recommendations: List[Recommendation],
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
    ) -> None:
        """Apply dataset-aware modifications to the search space strictly backed by retrieved knowledge."""
        model_lower = config.model_name.lower()
        is_classification = "classification" in (profile.problem_type or "")
        normalised_model = model_lower.replace("_", " ").replace("-", " ")
        retrieved_ids = {d.doc_id for d in retrieved_docs} if retrieved_docs else set()
        for r in recommendations:
            if r.supporting_doc:
                retrieved_ids.add(r.supporting_doc)

        # class_weight for imbalanced classification datasets
        has_imbalance_guidance = (
            "class_imbalance" in retrieved_ids
            or any("class_weight" in r.config for r in recommendations)
        )
        if profile.is_imbalanced and is_classification and has_imbalance_guidance:
            prefix = "classifier"
            cw_key = f"{prefix}__class_weight"
            models_with_cw = [
                "logistic", "decision tree", "random forest",
                "extra trees", "hist gradient",
            ]
            if any(m in normalised_model for m in models_with_cw):
                if cw_key not in config.param_grid:
                    config.param_grid[cw_key] = [None, "balanced"]
                    config.reasoning.append(
                        f"Added class_weight candidates [None, 'balanced'] due to class imbalance "
                        f"(ratio={profile.imbalance_ratio:.3f}) and retrieved class_imbalance guidance."
                    )

        # Model-specific tuning from retrieved model knowledge
        if ("decision_tree" in retrieved_ids or any(r.supporting_doc == "decision_tree" for r in recommendations)) and "decision tree" in normalised_model:
            prefix = "classifier" if is_classification else "regressor"
            crit_key = f"{prefix}__criterion"
            if crit_key not in config.param_grid:
                config.param_grid[crit_key] = ["gini", "entropy"] if is_classification else ["squared_error", "absolute_error"]
                config.reasoning.append(f"Applied RAG guidance from decision_tree: added {crit_key} candidates {config.param_grid[crit_key]}.")

            depth_key = f"{prefix}__max_depth"
            config.param_grid[depth_key] = [5, 10, 15, None]
            config.reasoning.append(f"Applied RAG guidance from decision_tree: refined {depth_key} to [5, 10, 15, None].")

            split_key = f"{prefix}__min_samples_split"
            config.param_grid[split_key] = [2, 5, 10]
            config.reasoning.append(f"Applied RAG guidance from decision_tree: extended {split_key} to [2, 5, 10].")

            leaf_key = f"{prefix}__min_samples_leaf"
            config.param_grid[leaf_key] = [1, 2, 5]
            config.reasoning.append(f"Applied RAG guidance from decision_tree: extended {leaf_key} to [1, 2, 5].")

        if ("random_forest" in retrieved_ids or any(r.supporting_doc == "random_forest" for r in recommendations)) and any(m in normalised_model for m in ["random forest", "extra trees"]):
            prefix = "classifier" if is_classification else "regressor"
            feat_key = f"{prefix}__max_features"
            config.param_grid[feat_key] = ["sqrt", "log2", None] if is_classification else [1.0, "sqrt"]
            config.reasoning.append(f"Applied RAG guidance from random_forest: expanded {feat_key} candidates {config.param_grid[feat_key]}.")

            est_key = f"{prefix}__n_estimators"
            config.param_grid[est_key] = [50, 100, 200]
            config.reasoning.append(f"Applied RAG guidance from random_forest: expanded {est_key} to [50, 100, 200].")

            split_key = f"{prefix}__min_samples_split"
            config.param_grid[split_key] = [2, 5, 10]
            config.reasoning.append(f"Applied RAG guidance from random_forest: extended {split_key} to [2, 5, 10].")

            leaf_key = f"{prefix}__min_samples_leaf"
            config.param_grid[leaf_key] = [1, 2, 4]
            config.reasoning.append(f"Applied RAG guidance from random_forest: added {leaf_key} candidates [1, 2, 4].")

        if ("logistic_regression" in retrieved_ids or any(r.supporting_doc == "logistic_regression" for r in recommendations)) and "logistic" in normalised_model:
            if "classifier__C" in config.param_grid and len(config.param_grid["classifier__C"]) <= 3:
                config.param_grid["classifier__C"] = [0.01, 0.1, 1.0, 10.0, 100.0]
                config.reasoning.append("Applied RAG guidance from logistic_regression: expanded C search space [0.01, 0.1, 1.0, 10.0, 100.0].")

        # Reduce complexity for small datasets (when cross_validation or hyperparameter_tuning is retrieved)
        has_cv_guidance = (
            "cross_validation" in retrieved_ids
            or "hyperparameter_tuning" in retrieved_ids
            or any(r.supporting_doc in ("cross_validation", "hyperparameter_tuning") for r in recommendations)
        )
        if profile.n_rows < 200 and has_cv_guidance:
            if "classifier__C" in config.param_grid:
                config.param_grid["classifier__C"] = [0.01, 0.1, 1]
                config.reasoning.append("Reduced C range for small dataset to prevent overfitting.")
            if "regressor__alpha" in config.param_grid:
                config.param_grid["regressor__alpha"] = [1.0, 10.0, 100.0]
                config.reasoning.append("Increased regularization (alpha) for small dataset.")

            depth_keys = [k for k in config.param_grid if "max_depth" in k]
            for dk in depth_keys:
                current = list(config.param_grid[dk])
                if profile.n_rows < 100 and None in current:
                    current = [v for v in current if v is not None]
                    config.param_grid[dk] = current
                    config.reasoning.append(
                        f"Removed unlimited depth (None) from {dk} for small dataset ({profile.n_rows} rows)."
                    )

        # Extend n_estimators for larger datasets when hyperparameter_tuning is retrieved
        if has_cv_guidance:
            for key in ("classifier__n_estimators", "regressor__n_estimators"):
                if profile.n_rows > 500 and key in config.param_grid:
                    current = list(config.param_grid[key])
                    if 300 not in current:
                        config.param_grid[key] = sorted(set(current + [300]))
                        config.reasoning.append(
                            f"Extended n_estimators to 300 for larger dataset ({profile.n_rows} rows)."
                        )

            # Hist Gradient Boosting: more learning rates for larger datasets
            if "hist" in model_lower and profile.n_rows > 300:
                for key in ("classifier__learning_rate", "regressor__learning_rate"):
                    if key in config.param_grid and len(config.param_grid[key]) < 2:
                        config.param_grid[key] = [0.05, 0.1, 0.2]
                        config.reasoning.append(
                            "Extended learning_rate candidates for gradient boosting on larger dataset."
                        )
