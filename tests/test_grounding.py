"""Tests for evidence-preserving XAI/RAG explanation generation."""

from src.llm.provider import DeterministicProvider, LLMExplanationService
from src.llm.prompts import build_local_explanation_prompt
from src.rag.explanation import GroundedExplanationContext
from src.rag.retriever import RetrievedDocument


def _context():
    return GroundedExplanationContext(
        observation_index=7,
        prediction=1,
        prediction_probability=0.91,
        predicted_class_label="yes",
        explained_output_label="yes",
        explained_output_probability=0.91,
        output_space="probability",
        base_value=0.4,
        shap_values={"numerical__income": 0.1234, "categorical__region_north": -0.02},
        model_name="Logistic Regression",
        problem_type="binary_classification",
        target_column="target",
        retrieved_docs=[RetrievedDocument("metrics", "Metrics", "Precision context", 0.8, "metrics.md", "Precision context")],
    )


def test_prompt_contains_actual_xai_and_retrieved_evidence():
    prompt = build_local_explanation_prompt(_context())
    assert "0.1234" in prompt
    assert "Precision context" in prompt
    assert "class/value yes" in prompt


def test_deterministic_fallback_uses_supplied_values_and_identifies_itself():
    text = LLMExplanationService(provider=DeterministicProvider()).explain_local(_context())
    assert "0.1234" in text
    assert "Deterministic" not in text  # service output is the grounded fallback, not a fake provider response
    assert "LOCAL explanation" in text


def test_default_service_reports_deterministic_mode_without_credentials():
    service = LLMExplanationService()
    if not service.is_llm_available:
        assert service.provider_name == "Deterministic Fallback (no LLM)"
