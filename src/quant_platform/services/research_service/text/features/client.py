"""LLM client construction for text feature extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol, cast

from quant_platform.services.research_service.text.features.errors import (
    FeatureExtractionError,
)


class AnthropicMessage(Protocol):
    content: object


class AnthropicMessagesClient(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> AnthropicMessage: ...


class AnthropicClient(Protocol):
    messages: AnthropicMessagesClient


def llm_api_key_env_name(provider: Literal["anthropic", "deepseek"]) -> str:
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    return "ANTHROPIC_API_KEY"


def resolve_llm_api_key(provider: Literal["anthropic", "deepseek"]) -> str:
    env_name = llm_api_key_env_name(provider)
    api_key = os.environ.get(env_name, "").strip()
    if api_key:
        return api_key
    return _dotenv_value(env_name).strip()


def build_llm_client(
    *,
    provider: Literal["anthropic", "deepseek"],
    timeout_seconds: float,
    deepseek_base_url: str,
) -> AnthropicClient:
    try:
        import anthropic
    except ImportError as exc:
        raise FeatureExtractionError(
            "anthropic package is not installed. Run: pip install 'anthropic>=0.40'"
        ) from exc
    if provider == "deepseek":
        api_key = resolve_llm_api_key(provider)
        if not api_key:
            raise FeatureExtractionError("DEEPSEEK_API_KEY must be set when provider='deepseek'")
        return cast(
            "AnthropicClient",
            anthropic.Anthropic(
                api_key=api_key,
                base_url=deepseek_base_url,
                timeout=timeout_seconds,
            ),
        )
    api_key = resolve_llm_api_key(provider)
    if not api_key:
        raise FeatureExtractionError("ANTHROPIC_API_KEY must be set when provider='anthropic'")
    return cast("AnthropicClient", anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds))


def _dotenv_value(env_name: str, env_file: Path = Path(".env")) -> str:
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if not separator or name.strip() != env_name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


__all__ = [
    "AnthropicClient",
    "AnthropicMessage",
    "build_llm_client",
    "llm_api_key_env_name",
    "resolve_llm_api_key",
]
