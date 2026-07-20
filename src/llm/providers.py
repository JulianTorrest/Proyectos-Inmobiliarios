from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    api_key_env: str
    base_url: Optional[str]
    default_model: str


PROVIDERS: dict[str, LLMProviderConfig] = {
    "OpenAI": LLMProviderConfig(
        name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        default_model="gpt-4o-mini",
    ),
    "Grok": LLMProviderConfig(
        name="Grok",
        api_key_env="GROK_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-2-mini",
    ),
    "Mistral": LLMProviderConfig(
        name="Mistral",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
    ),
    "DeepSeek": LLMProviderConfig(
        name="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
    ),
}


def resolve_api_key(provider: LLMProviderConfig, secrets: Optional[dict[str, Any]] = None) -> Optional[str]:
    if secrets and provider.api_key_env in secrets:
        v = secrets.get(provider.api_key_env)
        return str(v) if v else None

    return os.getenv(provider.api_key_env)
