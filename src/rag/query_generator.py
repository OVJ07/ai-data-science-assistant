"""
src/rag/query_generator.py
──────────────────────────
Generates dataset-aware retrieval queries from a DatasetProfile.

Strategy: Deterministic rule-based generation is always available and
reliable. An LLM-enhanced path can be enabled when a configured provider
is available, but the deterministic path is the default and fallback.

Design:
- Deterministic queries cover all known dataset characteristics.
- LLM-enhanced queries supplement (not replace) the deterministic ones.
- All generated queries are validated and deduplicated.
- The selection strategy, provider, and any fallbacks are recorded.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.data.profiler import DatasetProfile

logger = logging.getLogger(__name__)

MAX_LLM_QUERIES = 8


@dataclass
class QueryGenerationResult:
    """Structured result from query generation."""
    queries: List[str] = field(default_factory=list)
    strategy: str = "deterministic"          # 'deterministic' | 'llm_enhanced' | 'llm_fallback'
    provider_name: str = "none"
    model_name: str = ""
    selection_reason: str = ""
    llm_queries_generated: int = 0
    deterministic_queries_generated: int = 0
    fallback_occurred: bool = False
    fallback_reason: str = ""
    total_queries: int = 0
    deterministic_queries: List[str] = field(default_factory=list)
    llm_queries: List[str] = field(default_factory=list)
    llm_diagnostics: Dict[str, object] = field(default_factory=dict)
    generation_duration_seconds: float = 0.0


class QueryGenerator:
    """
    Translates dataset characteristics into targeted retrieval queries.

    Default: deterministic rule-based generation (always works, no API key).
    Optional: LLM-enhanced generation when a provider is configured.

    The LLM path is additive — it supplements deterministic queries.
    If LLM generation fails, the deterministic queries are used unchanged.
    """

    def generate_queries(
        self,
        profile: DatasetProfile,
        model_name: Optional[str] = None,
        use_llm: Optional[bool] = None,
    ) -> QueryGenerationResult:
        """
        Generate retrieval queries for the given dataset profile.

        Parameters
        ----------
        profile : DatasetProfile
        model_name : str, optional — adds model-specific queries
        use_llm : bool | None — True forces an eligible enhancement attempt,
            False disables it, and None (default) selects automatically.

        Returns
        -------
        QueryGenerationResult with queries and strategy metadata
        """
        started = time.perf_counter()
        result = QueryGenerationResult()

        # Always generate deterministic queries first
        det_queries = self._generate_deterministic(profile, model_name)
        result.deterministic_queries_generated = len(det_queries)
        result.deterministic_queries = list(det_queries)
        result.strategy = "deterministic"
        result.provider_name = "none"
        result.selection_reason = "Deterministic rule-based generation (reliable, no external API required)"

        queries = det_queries

        # LLM enhancement is additive and automatically selected only when a
        # configured model is eligible.  Deterministic queries remain the
        # reliable baseline and never require credentials or network access.
        attempt_llm, selection_info = self._should_attempt_llm(use_llm)
        if attempt_llm:
            llm_queries, provider_info = self._try_llm_enhancement(profile, model_name)
            result.llm_diagnostics = provider_info.get("diagnostics", {})
            if llm_queries:
                # Merge: deduplicated union of deterministic + LLM queries
                before = len(queries)
                queries = _deduplicate(queries + llm_queries)
                added = len(queries) - before
                result.llm_queries_generated = len(llm_queries)
                result.llm_queries = list(llm_queries)
                result.strategy = "llm_enhanced"
                result.provider_name = provider_info.get("provider", "unknown")
                result.model_name = provider_info.get("model", "")
                result.selection_reason = (
                    f"Deterministic queries supplemented with {added} unique LLM-generated queries "
                    f"via {result.provider_name}"
                )
            else:
                result.fallback_occurred = True
                result.fallback_reason = provider_info.get(
                    "error", "LLM not available or produced no valid queries"
                )
                result.strategy = "llm_fallback"
                result.selection_reason = (
                    f"LLM enhancement attempted but failed ({result.fallback_reason}). "
                    "Using deterministic queries."
                )
        elif use_llm is not False:
            result.selection_reason = selection_info.get("reason", result.selection_reason)

        # Final validation and deduplication
        queries = _validate_and_deduplicate(queries)
        result.queries = queries
        result.total_queries = len(queries)
        result.generation_duration_seconds = round(time.perf_counter() - started, 3)

        logger.info(
            "QueryGenerator: strategy='%s', %d queries (det=%d, llm=%d)",
            result.strategy,
            result.total_queries,
            result.deterministic_queries_generated,
            result.llm_queries_generated,
        )
        return result

    @staticmethod
    def _should_attempt_llm(use_llm: Optional[bool]) -> tuple:
        """Select a safe query-generation strategy without contacting a model."""
        if use_llm is False:
            return False, {"reason": "Deterministic generation selected by configuration."}
        try:
            from config.settings import LLM_ALLOW_REMOTE_DATA, LLM_QUERY_GENERATION_ENABLED
            from src.llm.provider import LLMExplanationService

            if not LLM_QUERY_GENERATION_ENABLED:
                return False, {"reason": "LLM query generation is disabled by configuration; using deterministic queries."}
            candidate = LLMExplanationService().query_generation_candidate()
            if not candidate.get("eligible"):
                return False, {"reason": f"{candidate.get('reason')}; using deterministic queries."}
            if not candidate.get("is_local") and not LLM_ALLOW_REMOTE_DATA:
                return False, {"reason": "Remote LLM is configured but dataset transmission is not opted in; using deterministic queries."}
            return True, candidate
        except Exception as exc:
            logger.info("LLM capability check failed: %s", exc)
            return False, {"reason": "LLM capability check failed; using deterministic queries."}

    def generate_xai_queries(
        self,
        profile: DatasetProfile,
        top_features: List[str],
        model_name: str = "",
    ) -> List[str]:
        """
        Generate queries for the XAI explanation retrieval step.
        Used to retrieve knowledge relevant to interpreting SHAP/LIME outputs.
        """
        queries = [
            "SHAP values feature contributions local explanation interpretability",
            f"feature importance interpretation {model_name} model prediction explanation",
        ]
        if profile.is_imbalanced:
            queries.append("SHAP values class imbalance interpretation minority class bias")
        if profile.has_missing:
            queries.append("SHAP feature importance missing values imputed features interpretation")
        if top_features:
            feature_str = " ".join(top_features[:5])
            queries.append(f"feature contribution direction magnitude {feature_str}")
        if profile.problem_type == "regression":
            queries.append("SHAP regression continuous target feature contribution interpretation")
        else:
            queries.append("SHAP classification probability contribution association causation")
        return queries

    # ── Private: deterministic generation ────────────────────────────────────

    def _generate_deterministic(
        self,
        profile: DatasetProfile,
        model_name: Optional[str],
    ) -> List[str]:
        """Generate rule-based queries covering all detected dataset characteristics."""
        queries: List[str] = []

        # Problem type
        if profile.problem_type == "binary_classification":
            queries.append("binary classification evaluation metrics accuracy F1 ROC-AUC AUC-PR")
        elif profile.problem_type == "multiclass_classification":
            queries.append("multiclass classification evaluation metrics macro F1 balanced accuracy")
        elif profile.problem_type == "regression":
            queries.append("regression evaluation metrics MAE RMSE R-squared mean absolute error")

        # Missing values
        if profile.has_missing:
            rate = profile.missing_rate_overall
            if rate > 0.1:
                queries.append("missing values high rate imputation median KNN strategy")
            else:
                queries.append("missing values imputation mean median simple strategy")

        # Class imbalance
        if profile.is_imbalanced:
            queries.append(
                "class imbalance handling class_weight balanced stratified cross-validation "
                f"minority class ratio {profile.imbalance_ratio:.2f}"
            )

        # Categorical features
        if profile.categorical_features:
            if profile.high_cardinality_features:
                queries.append(
                    "high cardinality categorical features one-hot encoding alternative "
                    "frequency encoding top-k categories dimensionality"
                )
            else:
                queries.append("categorical features one-hot encoding ordinal encoding preprocessing")

        # Text/name fields
        if profile.text_like_features:
            queries.append(
                "text name fields feature engineering title extraction high cardinality preprocessing"
            )

        # ID-like features
        if profile.id_like_features:
            queries.append(
                "ID columns identifier features should be dropped before training overfitting leakage"
            )

        # Numerical scaling
        if profile.numerical_features:
            queries.append("numerical feature scaling StandardScaler MinMaxScaler normalization")

        # Model-specific
        if model_name:
            model_queries = self._model_specific_queries(model_name, profile)
            queries.extend(model_queries)

        # Cross-validation strategy
        if profile.n_rows < 200:
            queries.append(
                "cross-validation small dataset stratified k-fold hyperparameter tuning"
            )
        else:
            queries.append("hyperparameter tuning cross-validation GridSearchCV search space")

        # XAI (always relevant)
        queries.append("SHAP feature importance interpretable machine learning XAI explanation")

        return queries

    def _model_specific_queries(
        self,
        model_name: str,
        profile: DatasetProfile,
    ) -> List[str]:
        """Return model-specific retrieval queries."""
        model_lower = model_name.lower()
        imb = profile.is_imbalanced
        queries = []

        if "logistic" in model_lower:
            q = "logistic regression hyperparameters C penalty regularization scaling"
            if imb:
                q += " class_weight balanced"
            queries.append(q)
        elif "extra_trees" in model_lower or "extra trees" in model_lower:
            q = "extra trees classifier ensemble feature importance n_estimators"
            if imb:
                q += " class_weight balanced"
            queries.append(q)
        elif "hist_gradient" in model_lower or "hist gradient" in model_lower:
            q = "histogram gradient boosting hyperparameters learning_rate max_iter early_stopping"
            if imb:
                q += " class_weight"
            queries.append(q)
        elif "random_forest" in model_lower or "random forest" in model_lower:
            q = "random forest hyperparameters n_estimators max_depth max_features min_samples"
            if imb:
                q += " class_weight balanced_subsample"
            queries.append(q)
        elif "decision_tree" in model_lower or "decision tree" in model_lower:
            # Two queries: one keyword-specific for TF-IDF (matches decision_tree.md title
            # and section headers), one semantic for dense retrieval.
            queries.append("decision tree DecisionTreeClassifier classifier gini entropy criterion")
            q = "decision tree hyperparameters max_depth min_samples_split min_samples_leaf criterion"
            if imb:
                q += " class_weight balanced"
            queries.append(q)
        elif "ridge" in model_lower:
            queries.append("ridge regression regularization alpha hyperparameter tuning")
        elif "linear_regression" in model_lower or "linear regression" in model_lower:
            queries.append("linear regression hyperparameters ridge lasso regularization scaling")
        else:
            queries.append("regression hyperparameter tuning cross-validation RMSE R2")

        return queries

    # ── Private: LLM enhancement ──────────────────────────────────────────────

    def _try_llm_enhancement(
        self,
        profile: DatasetProfile,
        model_name: Optional[str],
    ) -> tuple:
        """
        Attempt to generate additional queries using LLM.

        Returns
        -------
        (additional_queries, provider_info_dict)
        If LLM is unavailable or fails, returns ([], {error: ...}).
        """
        try:
            from src.llm.provider import LLMExplanationService
            service = LLMExplanationService()

            if not service.is_llm_available:
                return [], {"error": "LLM provider not available"}

            prompt = _build_query_generation_prompt(profile, model_name)
            raw, diagnostics = service.generate_query_completion(prompt)
            if not raw:
                error = diagnostics.get("error")
                if not error and diagnostics.get("success"):
                    error = "LLM returned an empty message.content"
                logger.info("LLM query completion had no parseable content: %s", error or "request failed")
                return [], {"error": error or "LLM request failed", "diagnostics": diagnostics}
            llm_queries, parse_diagnostics = _parse_llm_queries_with_diagnostics(raw)

            if not llm_queries:
                return [], {
                    "error": "LLM returned no valid queries",
                    "diagnostics": {**diagnostics, **parse_diagnostics, "parse_success": False},
                }

            return llm_queries, {
                "provider": service.provider_name,
                # Prefer the model echoed by the server; compatible endpoints
                # can resolve aliases to a different concrete identifier.
                "model": diagnostics.get("response_model") or service.query_generation_candidate().get("model", ""),
                "diagnostics": {
                    **diagnostics,
                    **parse_diagnostics,
                    "parse_success": True,
                    "parsed_query_count": len(llm_queries),
                },
            }

        except Exception as exc:
            logger.warning("LLM query enhancement failed: %s", exc)
            return [], {"error": str(exc)}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _build_query_generation_prompt(
    profile: DatasetProfile,
    model_name: Optional[str],
) -> str:
    """Build a prompt for LLM-enhanced query generation."""
    chars = []
    if profile.problem_type:
        chars.append(f"Problem type: {profile.problem_type}")
    if profile.n_rows:
        chars.append(f"Dataset size: {profile.n_rows} rows, {profile.n_cols} columns")
    if profile.has_missing:
        chars.append(f"Missing values: {profile.missing_rate_overall*100:.1f}% overall")
    if profile.is_imbalanced:
        chars.append(f"Class imbalance: ratio={profile.imbalance_ratio:.3f}")
    if profile.high_cardinality_features:
        chars.append(f"High-cardinality features: {profile.high_cardinality_features[:3]}")
    if profile.numerical_features:
        chars.append(f"Numerical features: {len(profile.numerical_features)}")
    if profile.categorical_features:
        chars.append(f"Categorical features: {len(profile.categorical_features)}")
    if model_name:
        chars.append(f"Model: {model_name}")

    char_str = "\n".join(f"- {c}" for c in chars)

    return f"""Generate up to 5 concise machine-learning retrieval queries.

Dataset characteristics:
{char_str}

Cover relevant preprocessing, model tuning, evaluation, and visible data
challenges. Do not include column names or dataset values. Return only a JSON
array of non-empty strings; no Markdown, explanation, or reasoning.
Example: ["high cardinality categorical encoding", "stratified cross validation"]
"""


def _parse_llm_queries(raw: str) -> List[str]:
    """Parse LLM output while retaining the legacy line-oriented fallback."""
    queries, _ = _parse_llm_queries_with_diagnostics(raw)
    return queries


def _parse_llm_queries_with_diagnostics(raw: str) -> tuple[List[str], Dict[str, object]]:
    """Parse JSON or legacy line output without retaining the response text.

    Diagnostics intentionally describe only the parsing outcome.  They never
    include model output, prompts, credentials, or dataset-derived content.
    """
    if not isinstance(raw, str) or not raw.strip():
        logger.info("LLM query parsing rejected an empty or non-string response.")
        return [], {"parse_format": "empty", "candidate_count": 0}

    text = _strip_json_fence(raw)
    looks_like_json = text.startswith(("[", "{"))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        if looks_like_json:
            logger.info("LLM query JSON parsing failed: %s.", exc.msg)
            return [], {"parse_format": "json_invalid", "candidate_count": 0}
        candidates = [re.sub(r"^[\d\.\-\*\s]+", "", line).strip() for line in text.splitlines()]
        return _validate_llm_query_candidates(candidates, "lines")

    if isinstance(parsed, list):
        return _validate_llm_query_candidates(parsed, "json_array")
    if isinstance(parsed, dict):
        for key in ("queries", "query_list", "search_queries"):
            if key in parsed:
                if isinstance(parsed[key], list):
                    return _validate_llm_query_candidates(parsed[key], f"json_object:{key}")
                logger.info("LLM query JSON object field '%s' is not an array.", key)
                return [], {"parse_format": f"json_object:{key}", "candidate_count": 0}
        logger.info("LLM query JSON object has no supported query-list field.")
        return [], {"parse_format": "json_object_unknown", "candidate_count": 0}

    logger.info("LLM query JSON response is neither an array nor an object.")
    return [], {"parse_format": "json_scalar", "candidate_count": 0}


def _strip_json_fence(raw: str) -> str:
    """Remove one optional fenced-code wrapper around otherwise JSON output."""
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _validate_llm_query_candidates(candidates: List[object], parse_format: str) -> tuple[List[str], Dict[str, object]]:
    """Keep unique, non-empty string queries only, capped for retrieval cost."""
    valid: List[str] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        query = candidate.strip()
        if not query:
            continue
        # Prevent pathological payloads from inflating the retrieval request.
        if len(query) > 300:
            query = query[:300].rstrip()
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            valid.append(query)
        if len(valid) == MAX_LLM_QUERIES:
            break

    if not valid:
        logger.info("LLM query parsing produced no non-empty string queries (format=%s, candidates=%d).", parse_format, len(candidates))
    return valid, {"parse_format": parse_format, "candidate_count": len(candidates)}


def _deduplicate(queries: List[str]) -> List[str]:
    """Remove exact duplicates while preserving order."""
    seen = set()
    result = []
    for q in queries:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(q)
    return result


def _validate_and_deduplicate(queries: List[str]) -> List[str]:
    """Validate queries and remove duplicates."""
    valid = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        # Must have at least 2 words
        if len(q.split()) < 2:
            continue
        # Remove excessively long queries
        if len(q) > 300:
            q = q[:300]
        valid.append(q)
    return _deduplicate(valid)
