"""LLM client adapters for text-event extraction.

The pipeline talks to an LLM through the :class:`LLMClient` Protocol
so the extraction loop is testable without a live API call (and so a
future swap from Anthropic to a different provider is a one-class
change, not a rewrite).

Two implementations ship today:

* :class:`MockLLMClient` — deterministic, takes a callable mapping
  ``SourceDocument -> dict`` and returns that. The default test
  fixture; also useful for replay / dry-run pipelines.
* :class:`AnthropicLLMClient` — thin wrapper over the Anthropic SDK's
  Messages API with tool_use for structured output. Disabled by
  default in CI; instantiate only when ``anthropic`` is installed
  and ``ANTHROPIC_API_KEY`` is set.

Both clients return :class:`LLMResponse` so the extraction pipeline
treats them uniformly. The pipeline is responsible for validating
the response against the schema; the client only handles transport.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.research.features.text.prompts import PromptTemplate
    from quant_platform.research.features.text.schemas import SourceDocument


@dataclass(frozen=True)
class LLMResponse:
    """One LLM call's structured output.

    Attributes
    ----------
    payload:
        Parsed JSON object the LLM emitted. The pipeline still
        validates this against the schema; the client only
        guarantees it parsed as JSON.
    raw_text:
        Raw text the LLM returned, before JSON parsing. Kept for
        hashing into :attr:`ExtractionProvenance.raw_response_hash`
        — never persisted in full, per the brief's "no LLM prose
        in training data" rule.
    confidence:
        LLM's self-reported confidence in ``[0, 1]``, or ``1.0``
        when the model doesn't return one.
    model_version:
        Identifier of the model that produced the response.
    """

    payload: dict[str, Any]
    raw_text: str
    confidence: float = 1.0
    model_version: str = "unknown"


class LLMClient(Protocol):
    """Protocol every extraction client implements.

    The pipeline calls ``client.extract(prompt, document)`` and
    receives an :class:`LLMResponse`. Anything that can satisfy that
    signature counts; the Protocol lets us swap mocks for production
    clients without an abstract base class hierarchy.
    """

    def extract(
        self,
        prompt: PromptTemplate,
        document: SourceDocument,
    ) -> LLMResponse:
        """Run one extraction against one document. Raises on transport error."""
        ...

    @property
    def model_version(self) -> str:
        """Stable identifier of the model this client is talking to."""
        ...


# ---------------------------------------------------------------------------
# Mock client — the default for tests and dry-runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockLLMClient:
    """Deterministic stand-in for a live LLM client.

    Takes a callable mapping each :class:`SourceDocument` to a
    dict (the extracted payload). Tests typically pass a closure
    that inspects the document's text and returns canned scores.

    Attributes
    ----------
    responder:
        ``(prompt, document) -> dict`` callable. The returned dict
        must match the prompt's ``output_schema``; the pipeline
        validates downstream.
    model_label:
        String pinned into the returned :class:`LLMResponse`'s
        ``model_version`` so persisted provenance records identify
        the mock.
    confidence:
        Default self-reported confidence; defaults to ``1.0``.
    """

    responder: Callable[[PromptTemplate, SourceDocument], dict[str, Any]]
    model_label: str = "mock-llm-v1"
    confidence: float = 1.0

    def extract(
        self,
        prompt: PromptTemplate,
        document: SourceDocument,
    ) -> LLMResponse:
        payload = self.responder(prompt, document)
        raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return LLMResponse(
            payload=payload,
            raw_text=raw_text,
            confidence=self.confidence,
            model_version=self.model_label,
        )

    @property
    def model_version(self) -> str:
        return self.model_label


# ---------------------------------------------------------------------------
# Anthropic adapter — production client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicLLMClient:
    """Anthropic SDK adapter using tool_use for structured output.

    The :meth:`extract` call:

    1. Builds an Anthropic Messages request from the
       :class:`PromptTemplate`.
    2. Declares a single ``emit_extraction`` tool whose
       ``input_schema`` is the prompt's ``output_schema`` (forces
       the model to emit a JSON object matching the schema).
    3. Pulls the tool_input from the response — that's the
       structured payload.
    4. Returns the parsed dict wrapped in :class:`LLMResponse`.

    The adapter is intentionally thin — no retry logic, no rate
    limiting, no batching. The extraction pipeline
    (:mod:`.extraction`) handles those concerns, so the client can
    stay testable.

    Attributes
    ----------
    api_key:
        Anthropic API key. Required.
    model:
        Anthropic model identifier (e.g. ``"claude-sonnet-4-5"``).
        Pinned into provenance so cross-model comparisons stay
        honest.
    max_tokens:
        Response token cap. Defaults to ``1024`` — extraction
        payloads are small; if the model needs more than that, the
        prompt is probably wrong.
    timeout:
        Request timeout in seconds. Defaults to 60.
    """

    api_key: str
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 1024
    timeout: float = 60.0

    @property
    def model_version(self) -> str:
        return self.model

    def extract(
        self,
        prompt: PromptTemplate,
        document: SourceDocument,
    ) -> LLMResponse:
        # Import locally so this file is importable without the
        # ``anthropic`` package present — the test suite only
        # exercises the mock client.
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover — exercised only in prod
            raise ImportError(
                "AnthropicLLMClient requires the 'anthropic' package; "
                "install with `pip install quant-platform[ml]` or similar."
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        tool_definition = {
            "name": "emit_extraction",
            "description": "Emit the structured extraction payload.",
            "input_schema": prompt.output_schema,
        }
        # The Anthropic SDK's ``create`` overloads expect ``MessageParam``
        # / ``ToolParam`` TypedDicts; we pass plain dicts whose runtime
        # shape matches. Threading the TypedDicts through every call
        # site is not worth the ergonomics cost, so suppress mypy here
        # while the runtime ``ImportError`` guard above still catches the
        # "SDK missing" case.
        message = client.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompt.system,
            tools=[tool_definition],
            tool_choice={"type": "tool", "name": "emit_extraction"},
            messages=[
                {
                    "role": "user",
                    "content": prompt.render_user_message(document),
                }
            ],
        )
        # The model's tool_use block carries the structured payload.
        payload: dict[str, Any] | None = None
        raw_chunks: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                payload = dict(block.input)  # type: ignore[arg-type]
                raw_chunks.append(json.dumps(payload, sort_keys=True))
            elif getattr(block, "type", None) == "text":
                raw_chunks.append(getattr(block, "text", ""))
        if payload is None:
            raise ValueError(
                f"AnthropicLLMClient: model did not emit the emit_extraction tool. "
                f"raw response: {''.join(raw_chunks)!r}"
            )
        return LLMResponse(
            payload=payload,
            raw_text="\n".join(raw_chunks),
            confidence=1.0,
            model_version=self.model,
        )


def hash_raw_response(raw_text: str) -> str:
    """Short stable hash of the raw LLM response.

    Used for :attr:`ExtractionProvenance.raw_response_hash` so an
    auditor can detect "the same prompt + model returned different
    text on a re-run" without storing the prose itself.
    """
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return digest[:16]


__all__ = [
    "AnthropicLLMClient",
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "hash_raw_response",
]
