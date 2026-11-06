"""Google Gemini (free tier) via the official `google-genai` SDK. Text-only:
LectureLens never sends frames/images to Gemini (slides are read with local OCR)."""

from __future__ import annotations

import logging

from .base import (FatalRequestError, LLMProvider, LLMResult, ProviderUnavailable, RateLimitError,
                   TransientError, looks_daily, parse_retry_after)

log = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None, max_output_tokens: int = 16384,
                 timeout_s: float = 180, use_json_mode: bool = True, thinking_budget: int | None = None,
                 thinking_level: str | None = None):
        super().__init__(model, max_output_tokens, timeout_s)
        self.api_key = api_key
        self.use_json_mode = use_json_mode
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise ProviderUnavailable("GEMINI_API_KEY not set in .env")
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise ProviderUnavailable("google-genai not installed (pip install google-genai)") from e
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            )
        return self._client

    def list_models(self) -> list[str]:
        client = self._get_client()
        names = []
        for m in client.models.list():
            names.append((m.name or "").removeprefix("models/"))
        return names

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GEMINI_API_KEY not set"
        try:
            models = self.list_models()
        except ProviderUnavailable as e:
            return False, str(e)
        except Exception as e:  # network/auth problems
            return False, f"could not list Gemini models: {str(e)[:200]}"
        if self.model not in models:
            flash = sorted(m for m in models if "flash" in m)
            return False, (f"configured Gemini model '{self.model}' not found for this key. "
                           f"Available Flash models: {', '.join(flash[:12]) or 'none'}. Edit llm.gemini.model in config.yaml.")
        return True, "ok"

    def complete(self, prompt, system=None, json_schema=None, temperature=0.2, max_tokens=None) -> LLMResult:
        from google.genai import errors, types

        client = self._get_client()
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens or self.max_output_tokens,
        )
        if system:
            cfg_kwargs["system_instruction"] = system
        if json_schema is not None and self.use_json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
        if self.thinking_level:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        elif self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
        try:
            resp = client.models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(**cfg_kwargs)
            )
        except errors.APIError as e:
            msg = f"{getattr(e, 'code', '')} {e}"
            code = getattr(e, "code", 0) or 0
            if code == 429 or "RESOURCE_EXHAUSTED" in msg:
                raise RateLimitError(msg, parse_retry_after(msg), looks_daily(msg)) from e
            if code >= 500 or code in (408,):
                raise TransientError(msg) from e
            if code in (401, 403):
                raise ProviderUnavailable(f"Gemini auth error: {msg[:300]}") from e
            if code == 404:  # e.g. "no longer available to new users ... use models/<newer>"
                raise ProviderUnavailable(f"Gemini model '{self.model}' unusable: {msg[:300]}") from e
            raise FatalRequestError(msg) from e
        except (TimeoutError, ConnectionError, OSError) as e:
            raise TransientError(str(e)) from e
        except Exception as e:  # httpx timeouts etc.
            if "timeout" in type(e).__name__.lower() or "connect" in type(e).__name__.lower():
                raise TransientError(str(e)) from e
            raise

        text = resp.text or ""
        if not text:
            reason = ""
            try:
                reason = str(resp.candidates[0].finish_reason)
            except Exception:
                pass
            raise FatalRequestError(f"Gemini returned empty text (finish_reason={reason})")
        um = getattr(resp, "usage_metadata", None)
        return LLMResult(
            text=text, provider=self.name, model=self.model,
            input_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
        )
