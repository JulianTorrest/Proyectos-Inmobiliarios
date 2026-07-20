from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .providers import LLMProviderConfig, resolve_api_key


@dataclass
class LLMResult:
    ok: bool
    content: str
    provider: str
    model: str
    error: Optional[str] = None


class MultiProviderLLM:
    def __init__(self, secrets: Optional[dict[str, Any]] = None):
        self._secrets = secrets or {}

    def is_configured(self, provider: LLMProviderConfig) -> bool:
        return bool(resolve_api_key(provider, self._secrets))

    def chat(
        self,
        provider: LLMProviderConfig,
        model: Optional[str],
        system: str,
        user: str = "",
        messages: Optional[list[dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResult:
        api_key = resolve_api_key(provider, self._secrets)
        chosen_model = model or provider.default_model
        if not api_key:
            return LLMResult(
                ok=False,
                content="",
                provider=provider.name,
                model=chosen_model,
                error=f"Missing API key: {provider.api_key_env}",
            )

        if messages is None:
            messages_payload = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            messages_payload = [{"role": "system", "content": system}] + messages

        try:
            client = OpenAI(api_key=api_key, base_url=provider.base_url)
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=messages_payload,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            return LLMResult(ok=True, content=content, provider=provider.name, model=chosen_model)
        except Exception as e:  # pragma: no cover
            return LLMResult(
                ok=False,
                content="",
                provider=provider.name,
                model=chosen_model,
                error=str(e),
            )
