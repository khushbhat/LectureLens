"""Groq (free tier): Llama chat completions + Whisper transcription."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import (FatalRequestError, LLMProvider, LLMResult, ProviderUnavailable, RateLimitError,
                   TransientError, looks_daily, parse_retry_after)

log = logging.getLogger(__name__)


def _client(api_key: str | None, timeout_s: float):
    if not api_key:
        raise ProviderUnavailable("GROQ_API_KEY not set in .env")
    try:
        from groq import Groq
    except ImportError as e:
        raise ProviderUnavailable("groq package not installed (pip install groq)") from e
    return Groq(api_key=api_key, timeout=timeout_s, max_retries=0)  # our quota layer retries


def _map_error(e: Exception) -> Exception:
    import groq

    msg = str(e)
    if isinstance(e, groq.RateLimitError):
        retry = None
        try:
            retry = float(e.response.headers.get("retry-after"))
        except Exception:
            retry = parse_retry_after(msg)
        return RateLimitError(msg, retry, looks_daily(msg))
    if isinstance(e, groq.AuthenticationError):
        return ProviderUnavailable(f"Groq auth error: {msg[:200]}")
    if isinstance(e, groq.NotFoundError):
        return ProviderUnavailable(f"Groq model not found: {msg[:200]}")
    if isinstance(e, (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError)):
        return TransientError(msg)
    if isinstance(e, groq.APIStatusError):
        if e.status_code == 413:
            return FatalRequestError(f"request too large for Groq: {msg[:200]}")
        if e.status_code >= 500:
            return TransientError(msg)
        return FatalRequestError(msg)
    return e


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, model: str, api_key: str | None, max_output_tokens: int = 4096, timeout_s: float = 180,
                 reasoning_effort: str | None = None):
        super().__init__(model, max_output_tokens, timeout_s)
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self._c = None

    def _get(self):
        if self._c is None:
            self._c = _client(self.api_key, self.timeout_s)
        return self._c

    def list_models(self) -> list[str]:
        return [m.id for m in self._get().models.list().data]

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GROQ_API_KEY not set"
        try:
            models = self.list_models()
        except ProviderUnavailable as e:
            return False, str(e)
        except Exception as e:
            return False, f"could not list Groq models: {str(e)[:200]}"
        if self.model not in models:
            return False, f"configured Groq model '{self.model}' not available. Available: {', '.join(sorted(models)[:15])}"
        return True, "ok"

    def complete(self, prompt, system=None, json_schema=None, temperature=0.2, max_tokens=None) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens or self.max_output_tokens)
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self._get().chat.completions.create(**kwargs)
        except ProviderUnavailable:
            raise
        except Exception as e:
            raise _map_error(e) from e
        u = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "", provider=self.name, model=self.model,
            input_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
        )


class GroqWhisper:
    """Groq speech-to-text. Returns segments [{start, end, text}] relative to the file."""

    name = "groq"

    def __init__(self, model: str, api_key: str | None, timeout_s: float = 300, language: str | None = "en"):
        self.model, self.api_key, self.timeout_s, self.language = model, api_key, timeout_s, language

    def transcribe(self, audio_path: str | Path) -> list[dict]:
        client = _client(self.api_key, self.timeout_s)
        p = Path(audio_path)
        kwargs = dict(file=(p.name, p.read_bytes()), model=self.model, response_format="verbose_json",
                      temperature=0.0)
        if self.language:
            kwargs["language"] = self.language
        try:
            resp = client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            raise _map_error(e) from e
        segs = getattr(resp, "segments", None)
        if segs is None and isinstance(resp, dict):
            segs = resp.get("segments")
        out = []
        for s in segs or []:
            g = s.get if isinstance(s, dict) else (lambda k, _s=s: getattr(_s, k, None))
            out.append({"start": float(g("start")), "end": float(g("end")), "text": (g("text") or "").strip()})
        if not out:
            text = getattr(resp, "text", "") or ""
            if text.strip():
                out.append({"start": 0.0, "end": 0.0, "text": text.strip()})
        return out
