"""
src/rag/explanation.py
──────────────────────
Grounds XAI outputs (SHAP/LIME) in retrieved RAG knowledge
before passing them to the LLM. This module enforces that
actual feature-contribution values are always present in the
explanation context — the LLM cannot invent them.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.retriever import KnowledgeRetriever, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class GroundedExplanationContext:
    """
    The complete context passed to the LLM for grounded explanation generation.
    All numerical evidence is populated from actual SHAP/LIME outputs.
    """
    # Observation metadata
    observation_index: int = -1
    prediction: Any = None
    prediction_probability: Optional[float] = None
    predicted_class_label: Optional[str] = None
    explained_output_label: Optional[str] = None
    explained_output_probability: Optional[float] = None
    output_space: str = ""
    additivity_error: Optional[float] = None
    additivity_verified: Optional[bool] = None

    # Actual XAI evidence (must be populated from SHAP/LIME)
    shap_values: Dict[str, float] = field(default_factory=dict)     # feature -> SHAP value
    lime_values: Dict[str, float] = field(default_factory=dict)     # feature -> LIME contribution
    base_value: Optional[float] = None                              # SHAP base (expected) value
    top_positive_features: List[Dict[str, Any]] = field(default_factory=list)
    top_negative_features: List[Dict[str, Any]] = field(default_factory=list)
    global_importance: Dict[str, float] = field(default_factory=dict)

    # Retrieved knowledge
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)

    # Model metadata
    model_name: str = ""
    problem_type: str = ""
    target_column: str = ""
    feature_names: List[str] = field(default_factory=list)

    # Limitations
    xai_method: str = "SHAP"
    causation_disclaimer: str = (
        "IMPORTANT: Feature contributions reflect statistical associations learned by the model. "
        "They do NOT establish causation. A high SHAP value does not mean the feature caused the outcome."
    )

    def has_xai_evidence(self) -> bool:
        return bool(self.shap_values or self.lime_values)

    def format_shap_evidence(self, top_n: int = 10) -> str:
        """Format SHAP values as a structured string for LLM prompt injection."""
        if not self.shap_values:
            return "No SHAP values available."

        sorted_shap = sorted(self.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        lines = [
            f"SHAP Feature Contributions for Observation #{self.observation_index}:",
            f"Base Value (expected model output): {self.base_value:.4f}" if self.base_value is not None else "",
            f"Prediction: {self.prediction}",
            f"Predicted-class probability: {self.prediction_probability:.4f}" if self.prediction_probability is not None else "",
            f"SHAP output: class {self.explained_output_label} ({self.explained_output_probability:.4f})" if self.explained_output_probability is not None else "",
            f"Output space: {self.output_space}" if self.output_space else "",
            f"Additivity error: {self.additivity_error:.6g}" if self.additivity_error is not None else "",
            "",
            "Feature | SHAP Value | Direction",
            "-" * 50,
        ]
        for feat, val in sorted_shap:
            target = f"toward class {self.explained_output_label}" if "classification" in self.problem_type else "predicted value"
            direction = f"↑ moves {target} higher" if val > 0 else f"↓ moves {target} lower"
            lines.append(f"{feat:30s} | {val:+.4f}     | {direction}")
        return "\n".join(line for line in lines if line is not None)

    def format_retrieved_knowledge(self) -> str:
        """Format retrieved documents as structured evidence for the LLM prompt."""
        if not self.retrieved_docs:
            return "No relevant knowledge retrieved."
        lines = ["Retrieved Knowledge Documents (for context only, not proof of causality):"]
        for i, doc in enumerate(self.retrieved_docs[:3], 1):
            lines.append(f"\n[Document {i}: {doc.title} (relevance={doc.relevance_score:.3f})]")
            lines.append(doc.snippet[:400])
        return "\n".join(lines)


class ExplanationGrounder:
    """
    Constructs a GroundedExplanationContext by combining XAI outputs
    with retrieved knowledge.
    """

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self.retriever = retriever

    def build_context(
        self,
        shap_local: Optional[Dict[str, float]],
        shap_global: Optional[Dict[str, float]],
        lime_local: Optional[Dict[str, float]],
        observation_index: int,
        prediction: Any,
        prediction_probability: Optional[float],
        predicted_class_label: Optional[str],
        explained_output_label: Optional[str],
        explained_output_probability: Optional[float],
        output_space: str,
        additivity_error: Optional[float],
        additivity_verified: Optional[bool],
        base_value: Optional[float],
        model_name: str,
        problem_type: str,
        target_column: str,
        feature_names: List[str],
        xai_queries: List[str],
        top_k_docs: int = 3,
    ) -> GroundedExplanationContext:
        """
        Build a fully grounded explanation context from XAI evidence.

        Parameters
        ----------
        shap_local : dict mapping feature_name -> SHAP value for this observation
        shap_global : dict mapping feature_name -> mean absolute SHAP across all samples
        lime_local : dict mapping feature_name -> LIME contribution
        ... (see dataclass fields)
        xai_queries : list of retrieval queries for XAI knowledge
        """
        ctx = GroundedExplanationContext(
            observation_index=observation_index,
            prediction=prediction,
            prediction_probability=prediction_probability,
            predicted_class_label=predicted_class_label,
            explained_output_label=explained_output_label,
            explained_output_probability=explained_output_probability,
            output_space=output_space,
            additivity_error=additivity_error,
            additivity_verified=additivity_verified,
            model_name=model_name,
            problem_type=problem_type,
            target_column=target_column,
            feature_names=feature_names,
            base_value=base_value,
        )

        # Populate actual XAI evidence
        if shap_local:
            ctx.shap_values = {k: float(v) for k, v in shap_local.items()}
            ctx.xai_method = "SHAP"

            # Separate positive and negative contributors
            sorted_by_val = sorted(ctx.shap_values.items(), key=lambda x: x[1], reverse=True)
            ctx.top_positive_features = [
                {"feature": f, "shap_value": v, "direction": "increases"}
                for f, v in sorted_by_val if v > 0
            ][:5]
            ctx.top_negative_features = [
                {"feature": f, "shap_value": v, "direction": "decreases"}
                for f, v in sorted_by_val if v < 0
            ][-5:]

        if lime_local:
            ctx.lime_values = {k: float(v) for k, v in lime_local.items()}
            if not shap_local:
                ctx.xai_method = "LIME"

        if shap_global:
            ctx.global_importance = {k: float(v) for k, v in shap_global.items()}

        # Retrieve relevant knowledge
        if self.retriever.is_built and xai_queries:
            raw_docs = self.retriever.retrieve_multiple_queries(
                xai_queries, top_k_per_query=2, global_top_k=top_k_docs
            )
            ctx.retrieved_docs = raw_docs[0] if isinstance(raw_docs, tuple) else (raw_docs or [])

        logger.info(
            "Built grounded context: obs=%d, shap_features=%d, docs=%d",
            observation_index,
            len(ctx.shap_values),
            len(ctx.retrieved_docs),
        )
        return ctx
