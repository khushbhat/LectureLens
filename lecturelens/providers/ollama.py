"""Ollama (local, free, no key) via its HTTP API."""

from __future__ import annotations

import requests

from .base import FatalRequestError, LLMProvider, LLMResult, ProviderUnavailable, TransientError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", max_output_tokens: int = 4096,
                 timeout_s: float = 600, num_ctx: int = 16384):
        super().__init__(model, max_output_tokens, timeout_s)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ProviderUnavailable(f"Ollama not running at {self.host}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def available(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except ProviderUnavailable as e:
            return False, str(e)
        base = self.model if ":" in self.model else f"{self.model}:latest"
        if self.model not in models and base not in models:
            return False, f"Ollama model '{self.model}' not pulled (run: ollama pull {self.model})"
        return True, "ok"

    def complete(self, prompt, system=None, json_schema=None, temperature=0.2, max_tokens=None) -> LLMResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens or self.max_output_tokens},
        }
        if json_schema is not None:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
        except requests.Timeout as e:
            raise TransientError(f"Ollama timeout: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderUnavailable(f"Ollama not reachable at {self.host}") from e
        if r.status_code >= 500:
            raise TransientError(r.text[:300])
        if r.status_code == 404:
            raise ProviderUnavailable(f"Ollama model '{self.model}' not found")
        if r.status_code >= 400:
            raise FatalRequestError(r.text[:300])
        data = r.json()
        return LLMResult(
            text=data.get("message", {}).get("content", ""), provider=self.name, model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
        )
