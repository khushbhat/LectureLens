"""LLM providers behind one interface; `build_provider` constructs them from config."""

from __future__ import annotations

from .base import (FatalRequestError, LLMProvider, LLMResult, ProviderUnavailable, RateLimitError,
                   TransientError)

__all__ = ["LLMProvider", "LLMResult", "ProviderUnavailable", "RateLimitError", "TransientError",
           "FatalRequestError", "build_provider"]


def build_provider(name: str, cfg) -> LLMProvider:
    sect = cfg.get(f"llm.{name}", {}) or {}
    timeout = cfg.get("llm.request_timeout_s", 180)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(sect["model"], cfg.api_key("GEMINI_API_KEY"), sect.get("max_output_tokens", 16384),
                              timeout, sect.get("use_json_mode", True), sect.get("thinking_budget"),
                              sect.get("thinking_level"))
    if name == "groq":
        from .groq_llm import GroqProvider

        return GroqProvider(sect["model"], cfg.api_key("GROQ_API_KEY"), sect.get("max_output_tokens", 4096), timeout,
                            sect.get("reasoning_effort"))
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(sect["model"], sect.get("host", "http://localhost:11434"),
                              sect.get("max_output_tokens", 4096), max(timeout, 600), sect.get("num_ctx", 16384))
    if name == "mock":
        from .mock import MockProvider

        return MockProvider(sect.get("model", "mock-1"))
    raise ValueError(f"unknown provider '{name}'")
