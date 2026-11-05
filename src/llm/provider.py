"""
src/llm/provider.py
────────────────────
LLM provider abstraction with deterministic fallback.
The application works completely without an LLM API key.
"""

import logging
import os
import time
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_QUERY_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from src.llm.prompts import (
    build_fallback_global_explanation,
    build_fallback_local_explanation,
    build_global_explanation_prompt,
    build_local_explanation_prompt,
)
from src.rag.explanation import GroundedExplanationContext

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Send a prompt and return the completion text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI or OpenAI-compatible endpoint provider."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> None:
        self._api_key = api_key or OPENAI_API_KEY
        self._model = model or OPENAI_MODEL
        self._base_url = base_url or OPENAI_BASE_URL
        self._available: Optional[bool] = None
        self._last_completion_diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        label = "Local OpenAI-compatible" if self.is_local else "OpenAI-compatible"
        return f"{label} ({self._model})"

    @property
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        self._available = bool(self._api_key)
        return self._available

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_local(self) -> bool:
        """Whether this configured compatible endpoint is loopback-only."""
        host = (urlparse(self._base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    @property
    def endpoint_host(self) -> str:
        return urlparse(self._base_url).netloc or "unknown"

    @property
    def last_completion_diagnostics(self) -> Dict[str, Any]:
        """Safe request metadata; deliberately excludes credentials and prompts."""
        return dict(self._last_completion_diagnostics)

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self.is_available:
            raise RuntimeError("OpenAI API key not configured.")
        started = time.perf_counter()
        diagnostics: Dict[str, Any] = {
            "provider": "openai_compatible",
            "requested_model": self._model,
            "endpoint_host": self.endpoint_host,
            "success": False,
        }
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            }
            response = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("LLM response contains no choices")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if not isinstance(message, dict):
                raise ValueError("LLM response choice contains no message object")

            # OpenAI-compatible reasoning models may provide a separate
            # ``reasoning`` field.  Query generation must use only content;
            # reasoning is deliberately neither returned nor parsed.
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("LLM response message.content is not a string")
            text = content.strip()
            # If content is empty but reasoning was emitted (e.g. reasoning model token exhaustion)
            if not text and message.get("reasoning"):
                reasoning_text = str(message["reasoning"]).strip()
                if reasoning_text:
                    logger.warning("LLM returned empty content; using reasoning field as fallback (%d chars)", len(reasoning_text))
                    text = reasoning_text
            diagnostics.update({
                "success": True,
                # Compatible servers may echo a resolved model identifier.
                "response_model": str(data.get("model") or self._model),
                "response_characters": len(text),
                "max_tokens": max_tokens or LLM_MAX_TOKENS,
                "content_extracted": True,
                "reasoning_field_present": "reasoning" in message,
            })
            logger.info("LLM completion successful (%d chars).", len(text))
            return text
        except Exception as exc:
            diagnostics["error"] = str(exc)
            logger.error("LLM completion failed: %s", exc)
            raise
        finally:
            diagnostics["duration_seconds"] = round(time.perf_counter() - started, 3)
            self._last_completion_diagnostics = diagnostics


class DeterministicProvider(BaseLLMProvider):
    """
    Deterministic fallback provider.
    Returns structured explanations from actual XAI data without any LLM call.
    Always available; used when no LLM is configured.
    """

    @property
    def name(self) -> str:
        return "Deterministic Fallback (no LLM)"

    @property
    def is_available(self) -> bool:
        return True

    def complete(self, prompt: str) -> str:
        """Return a polite message indicating deterministic mode."""
        return (
            "LLM provider is not configured. "
            "The explanation below is generated deterministically from actual XAI values. "
            "No LLM-generated text is used.\n\n"
            "[Deterministic explanation follows from the grounded context]"
        )


class LLMExplanationService:
    """
    High-level service for generating grounded explanations.
    Selects the appropriate provider and falls back gracefully.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        if provider is not None:
            self._provider = provider
        else:
            self._provider = self._create_default_provider()

    def _create_default_provider(self) -> BaseLLMProvider:
        """Create provider based on environment settings."""
        if LLM_PROVIDER in ("openai", "openai_compatible") and OPENAI_API_KEY:
            logger.info("Using OpenAI LLM provider.")
            return OpenAIProvider()
        logger.info("Using deterministic fallback (no LLM configured).")
        return DeterministicProvider()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def is_llm_available(self) -> bool:
        return isinstance(self._provider, OpenAIProvider) and self._provider.is_available

    def query_generation_candidate(self) -> Dict[str, Any]:
        """Describe an eligible configured model without probing or charging it.

        This deliberately performs no model download, remote model listing, or
        trial generation.  The configured model is an explicit deployment
        choice; actual generation remains validated by the query generator.
        """
        if not self.is_llm_available:
            return {"eligible": False, "reason": "No configured LLM provider is available"}
        provider = self._provider
        return {
            "eligible": True,
            "provider": provider.name,
            "model": provider.model_name,
            "is_local": provider.is_local,
            "reason": "Configured local compatible model" if provider.is_local else "Configured remote model",
        }

    def generate_query_completion(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        """Run one query-generation completion with safe observability metadata."""
        if not self.is_llm_available:
            return "", {"success": False, "error": "LLM provider not available"}
        try:
            text = self._provider.complete(prompt, max_tokens=LLM_QUERY_MAX_TOKENS)
            diagnostics = getattr(self._provider, "last_completion_diagnostics", {})
            return text, diagnostics
        except Exception as exc:
            diagnostics = getattr(self._provider, "last_completion_diagnostics", {})
            return "", {**diagnostics, "success": False, "error": str(exc)}

    def explain_local(self, ctx: GroundedExplanationContext) -> str:
        """Generate a grounded local prediction explanation."""
        if not ctx.has_xai_evidence():
            return build_fallback_local_explanation(ctx)

        if self.is_llm_available:
            try:
                prompt = build_local_explanation_prompt(ctx)
                res = self._provider.complete(prompt)
                if res and res.strip():
                    return res.strip()
                logger.warning("LLM local explanation returned empty text, using fallback.")
            except Exception as exc:
                logger.warning("LLM local explanation failed, using fallback: %s", exc)

        return build_fallback_local_explanation(ctx)

    def explain_global(
        self,
        global_importance: dict,
        model_name: str,
        problem_type: str,
        target_column: str,
        retrieved_knowledge: str = "",
    ) -> str:
        """Generate a grounded global feature importance explanation."""
        if not global_importance:
            return "No global feature importance data available."

        if self.is_llm_available:
            try:
                prompt = build_global_explanation_prompt(
                    global_importance, model_name, problem_type,
                    target_column, retrieved_knowledge
                )
                res = self._provider.complete(prompt)
                if res and res.strip():
                    return res.strip()
                logger.warning("LLM global explanation returned empty text, using fallback.")
            except Exception as exc:
                logger.warning("LLM global explanation failed, using fallback: %s", exc)

        return build_fallback_global_explanation(
            global_importance, model_name, problem_type, target_column
        )


def create_llm_service() -> LLMExplanationService:
    """Factory function to create the LLM service from environment config."""
    return LLMExplanationService()
