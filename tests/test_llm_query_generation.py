"""Focused tests for LLM-enhanced RAG query response handling."""

import requests
import pytest

import src.llm.provider as provider_module
from src.data.profiler import DatasetProfile, DatasetProfiler
from src.llm.provider import LLMExplanationService, OpenAIProvider
from src.rag.query_generator import QueryGenerator, _parse_llm_queries


def test_parse_json_array_strips_whitespace_deduplicates_and_caps():
    raw = '  [" first query ", "second query", "FIRST QUERY", "third query"]  '
    assert _parse_llm_queries(raw) == ["first query", "second query", "third query"]


def test_parse_markdown_fenced_json_array():
    raw = "```json\n[\"encoding recommendation\", \"cross validation guidance\"]\n```"
    assert _parse_llm_queries(raw) == ["encoding recommendation", "cross validation guidance"]


def test_parse_json_object_query_list():
    raw = '{"queries": ["categorical preprocessing", "classification calibration"]}'
    assert _parse_llm_queries(raw) == ["categorical preprocessing", "classification calibration"]


@pytest.mark.parametrize(
    "raw",
    [
        '["unterminated"',
        "[]",
        "[12, null, {}]",
        '{"queries": "not an array"}',
    ],
)
def test_parse_rejects_malformed_empty_or_invalid_query_payloads(raw):
    assert _parse_llm_queries(raw) == []


def test_openai_compatible_provider_uses_content_not_reasoning(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "qwen3.5-9b-fast:latest",
                "choices": [{"message": {
                    "content": '["retrieval query one", "retrieval query two"]',
                    "reasoning": "This must never be parsed as a retrieval query.",
                }}],
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())
    provider = OpenAIProvider(api_key="test-key", model="configured-model", base_url="http://localhost:11434/v1")
    text = provider.complete("safe test prompt")

    assert _parse_llm_queries(text) == ["retrieval query one", "retrieval query two"]
    assert "reasoning" not in text
    assert provider.last_completion_diagnostics["content_extracted"] is True
    assert provider.last_completion_diagnostics["reasoning_field_present"] is True


def test_query_generation_uses_dedicated_completion_budget(monkeypatch):
    captured = {}

    class FakeProvider:
        def complete(self, prompt, max_tokens=None):
            captured["max_tokens"] = max_tokens
            return '["valid query"]'

        @property
        def last_completion_diagnostics(self):
            return {"success": True}

    service = LLMExplanationService(provider=FakeProvider())
    # The service's availability check intentionally recognises only the real
    # compatible provider, so exercise its request method with that predicate.
    monkeypatch.setattr(LLMExplanationService, "is_llm_available", property(lambda _self: True))
    text, _ = service.generate_query_completion("safe test prompt")
    assert text == '["valid query"]'
    assert captured["max_tokens"] == provider_module.LLM_QUERY_MAX_TOKENS


def test_api_error_is_reported_as_failed_completion(monkeypatch):
    class FailingResponse:
        def raise_for_status(self):
            raise requests.HTTPError("503 Service Unavailable")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FailingResponse())
    service = LLMExplanationService(
        provider=OpenAIProvider(api_key="test-key", model="test", base_url="http://localhost:11434/v1")
    )
    text, diagnostics = service.generate_query_completion("safe test prompt")

    assert text == ""
    assert diagnostics["success"] is False
    assert "503" in diagnostics["error"]


def test_query_generator_uses_valid_json_content_for_llm_queries(
    binary_classification_df, monkeypatch
):
    class FakeService:
        is_llm_available = True
        provider_name = "Local OpenAI-compatible (qwen3.5-9b-fast:latest)"

        def generate_query_completion(self, _prompt):
            return '["distinctive query alpha", "distinctive query beta"]', {
                "success": True,
                "response_model": "qwen3.5-9b-fast:latest",
                "content_extracted": True,
            }

        def query_generation_candidate(self):
            return {"model": "qwen3.5-9b-fast:latest"}

    monkeypatch.setattr("src.llm.provider.LLMExplanationService", FakeService)
    monkeypatch.setattr(QueryGenerator, "_should_attempt_llm", staticmethod(lambda _requested: (True, {})))
    profile = DatasetProfiler().profile(binary_classification_df, target_column="target")

    result = QueryGenerator().generate_queries(profile, use_llm=True)

    assert result.strategy == "llm_enhanced"
    assert result.llm_queries == ["distinctive query alpha", "distinctive query beta"]
    assert result.llm_diagnostics["parse_format"] == "json_array"


def test_query_generator_reports_successful_but_empty_content_accurately(monkeypatch):
    class FakeService:
        is_llm_available = True

        def generate_query_completion(self, _prompt):
            return "", {"success": True, "response_characters": 0}

    monkeypatch.setattr("src.llm.provider.LLMExplanationService", FakeService)
    queries, info = QueryGenerator()._try_llm_enhancement(
        DatasetProfile(problem_type="binary_classification", n_rows=2, n_cols=2),
        None,
    )

    assert queries == []
    assert info["error"] == "LLM returned an empty message.content"
