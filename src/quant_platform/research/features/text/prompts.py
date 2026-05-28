"""Versioned prompt library for text-event extraction.

A prompt is a *contract* between the operator and the LLM. The brief
makes two demands of that contract:

1. **Structured output only** — the system prompt instructs the
   model to emit a JSON object matching the named schema, no prose,
   no markdown, no preamble.
2. **Versioned + immutable** — once an extraction has been
   persisted under ``prompt_version="news-prompt-v1"``, the
   string that produced it must never change. To revise the
   wording, ship ``news-prompt-v2`` and bump the version pin at
   the call site.

This module composes the prompt's user content from
:meth:`NewsExtraction.field_descriptions` so the LLM sees the same
field-meaning catalog the schema validator enforces. Source of
truth is the dataclass; the prompt is a presentation of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_platform.research.features.text.schemas import (
        SourceDocument,
    )


#: Version pin for the news-extraction prompt. Bumped when the
#: system/user wording changes in a way that could alter LLM output
#: meaning.
NEWS_PROMPT_VERSION: str = "news-prompt-v1"

#: Version pin for the SEC-filing extraction prompt.
FILING_PROMPT_VERSION: str = "filing-prompt-v1"

#: Version pin for the earnings-call transcript extraction prompt.
EARNINGS_CALL_PROMPT_VERSION: str = "earnings-call-prompt-v1"


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned (system, user) prompt pair.

    Attributes
    ----------
    version:
        Stable version string persisted as
        :attr:`ExtractionProvenance.prompt_version`.
    system:
        System prompt that instructs the model on what to do and
        what to emit. Should be self-contained — operators don't
        re-paste schema descriptions into every user message.
    user_template:
        User-message template. ``{instrument_id}`` and ``{text}``
        placeholders are filled per document at call time. Any
        other placeholder is rejected at format time so the
        template doesn't silently leak un-substituted braces into
        the LLM input.
    output_schema:
        JSON object describing the expected response shape. Passed
        to the LLM's structured-output mechanism (Anthropic
        tool_use, OpenAI response_format, etc.) so the model is
        constrained to emit valid JSON.
    """

    version: str
    system: str
    user_template: str
    output_schema: dict[str, object]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("PromptTemplate.version must be non-empty")
        if not self.system.strip():
            raise ValueError("PromptTemplate.system must be non-empty")
        if not self.user_template.strip():
            raise ValueError("PromptTemplate.user_template must be non-empty")
        if "{instrument_id}" not in self.user_template:
            raise ValueError(
                "PromptTemplate.user_template must include the {instrument_id} placeholder"
            )
        if "{text}" not in self.user_template:
            raise ValueError("PromptTemplate.user_template must include the {text} placeholder")

    def render_user_message(self, document: SourceDocument) -> str:
        """Fill the template's placeholders with the document's fields."""
        return self.user_template.format(
            instrument_id=document.instrument_id,
            text=document.text,
        )


# ---------------------------------------------------------------------------
# News prompt v1
# ---------------------------------------------------------------------------
#
# The user message instructs the model to (a) emit JSON only, (b)
# constrain every score to the documented range, (c) emit zero — not
# omit the field — when no signal is present. The system prompt
# repeats those constraints because empirically Anthropic models
# follow system-level constraints more reliably than user-level ones.

_NEWS_SCHEMA_FIELDS: dict[str, str] = {}  # populated lazily below

_NEWS_SYSTEM_PROMPT = """\
You are a financial-news structured extractor.

Your job is to read a news article about a specific instrument and emit a JSON
object scoring named dimensions of the article's content. You DO NOT produce
buy/sell recommendations, price targets, or free-form commentary. You DO NOT
include any text outside the JSON object.

Rules:
1. Emit a single JSON object — no markdown fences, no preamble, no trailing
   commentary.
2. Every score must lie strictly within the documented range. A score outside
   the range will cause your output to be rejected.
3. "No signal" or "not mentioned" is ``0.0`` for signed fields and ``0.5``
   for novelty. Do not omit fields.
4. Sentiment is *toward the named instrument*, not toward "the market" or
   "the company's competitors".
5. ``materiality`` is how much of the article focuses on the instrument, not
   how impactful the news is. A devastating SEC charge with one mention in a
   sector roundup is materiality 0.1, sentiment -1.
"""


def _build_news_user_template() -> str:
    """Lazy-import the schema so prompts.py doesn't pull dataclass
    machinery at module-import time."""
    from quant_platform.research.features.text.schemas import NewsExtraction

    descriptions = NewsExtraction.field_descriptions()
    field_lines = "\n".join(
        f"  - {name}: {description}" for name, description in descriptions.items()
    )
    return f"""\
Extract the news-event features for instrument {{instrument_id}}.

Article:
{{text}}

Emit a JSON object with the following fields:
{field_lines}

Return JSON only.
"""


def _build_news_output_schema() -> dict[str, object]:
    """JSON schema for the news extraction output.

    Built from the dataclass field list so the LLM's
    structured-output constraint stays in sync with the validator
    in :class:`NewsExtraction.__post_init__`.
    """
    from quant_platform.research.features.text.schemas import NewsExtraction

    properties: dict[str, object] = {}
    for name in NewsExtraction.SIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": NewsExtraction.field_descriptions()[name],
        }
    for name in NewsExtraction.UNSIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": NewsExtraction.field_descriptions()[name],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": [*NewsExtraction.SIGNED_FIELDS, *NewsExtraction.UNSIGNED_FIELDS],
        "additionalProperties": False,
    }


def get_news_prompt() -> PromptTemplate:
    """Return the v1 news-extraction prompt template.

    Module-level lazy load so importing :mod:`prompts` doesn't drag
    in the schemas module unless someone actually asks for a
    prompt.
    """
    return PromptTemplate(
        version=NEWS_PROMPT_VERSION,
        system=_NEWS_SYSTEM_PROMPT,
        user_template=_build_news_user_template(),
        output_schema=_build_news_output_schema(),
    )


# ---------------------------------------------------------------------------
# Filing prompt v1
# ---------------------------------------------------------------------------


_FILING_SYSTEM_PROMPT = """\
You are a SEC-filing structured extractor.

Your job is to read an SEC filing (10-K, 10-Q, or 8-K) about a specific
instrument and emit a JSON object scoring named dimensions of the
filing's content. You DO NOT produce buy/sell recommendations, price
targets, or free-form commentary. You DO NOT include any text outside
the JSON object.

Rules:
1. Emit a single JSON object — no markdown fences, no preamble, no trailing
   commentary.
2. Every score must lie strictly within the documented range. A score outside
   the range will cause your output to be rejected.
3. "No signal" or "not mentioned" is ``0.0`` for signed fields and ``0.0``
   for ``uncertainty_score`` (the only unsigned field). Do not omit fields.
4. Scores describe the *filing's content*, not your own analysis. If the
   filing reads like a positive 10-K, score the tone positive — even if
   you personally think the company is mispriced.
5. Reverse-coded fields (``litigation_risk``, ``supply_chain_risk``,
   ``inventory_risk``, ``margin_pressure``, ``demand_weakness``,
   ``financing_stress``) follow the convention ``+1`` = healthy /
   low risk, ``-1`` = stressed / elevated risk. The field name names
   the *risk*, but the score names how *much* of it is present —
   sign-flipped so positive is always "good for forward return".
"""


def _build_filing_user_template() -> str:
    from quant_platform.research.features.text.schemas import FilingExtraction

    descriptions = FilingExtraction.field_descriptions()
    field_lines = "\n".join(
        f"  - {name}: {description}" for name, description in descriptions.items()
    )
    return f"""\
Extract the SEC-filing features for instrument {{instrument_id}}.

Filing text:
{{text}}

Emit a JSON object with the following fields:
{field_lines}

Return JSON only.
"""


def _build_filing_output_schema() -> dict[str, object]:
    from quant_platform.research.features.text.schemas import FilingExtraction

    properties: dict[str, object] = {}
    for name in FilingExtraction.SIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": FilingExtraction.field_descriptions()[name],
        }
    for name in FilingExtraction.UNSIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": FilingExtraction.field_descriptions()[name],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": [*FilingExtraction.SIGNED_FIELDS, *FilingExtraction.UNSIGNED_FIELDS],
        "additionalProperties": False,
    }


def get_filing_prompt() -> PromptTemplate:
    """Return the v1 SEC-filing extraction prompt template."""
    return PromptTemplate(
        version=FILING_PROMPT_VERSION,
        system=_FILING_SYSTEM_PROMPT,
        user_template=_build_filing_user_template(),
        output_schema=_build_filing_output_schema(),
    )


# ---------------------------------------------------------------------------
# Earnings-call prompt v1
# ---------------------------------------------------------------------------


_EARNINGS_CALL_SYSTEM_PROMPT = """\
You are an earnings-call transcript structured extractor.

Your job is to read an earnings-call transcript about a specific instrument
and emit a JSON object scoring named dimensions of the call. You DO NOT
produce buy/sell recommendations, price targets, or free-form commentary.
You DO NOT include any text outside the JSON object.

Rules:
1. Emit a single JSON object — no markdown fences, no preamble, no trailing
   commentary.
2. Every score must lie strictly within the documented range. A score outside
   the range will cause your output to be rejected.
3. "No signal" or "not discussed" is ``0.0`` for signed fields and ``0.0``
   for ``analyst_pushback``. Do not omit fields.
4. Earnings calls have two phases: prepared remarks (managed tone) and
   Q&A (analyst-driven). Score ``management_confidence`` from prepared
   remarks; score ``analyst_pushback`` from Q&A; the rest combine both.
5. Reverse-coded fields (``margin_pressure``) follow the news-prompt
   convention: ``+1`` = expanding, ``-1`` = pressure rising.
"""


def _build_earnings_call_user_template() -> str:
    from quant_platform.research.features.text.schemas import EarningsCallExtraction

    descriptions = EarningsCallExtraction.field_descriptions()
    field_lines = "\n".join(
        f"  - {name}: {description}" for name, description in descriptions.items()
    )
    return f"""\
Extract the earnings-call features for instrument {{instrument_id}}.

Transcript:
{{text}}

Emit a JSON object with the following fields:
{field_lines}

Return JSON only.
"""


def _build_earnings_call_output_schema() -> dict[str, object]:
    from quant_platform.research.features.text.schemas import EarningsCallExtraction

    properties: dict[str, object] = {}
    for name in EarningsCallExtraction.SIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": EarningsCallExtraction.field_descriptions()[name],
        }
    for name in EarningsCallExtraction.UNSIGNED_FIELDS:
        properties[name] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": EarningsCallExtraction.field_descriptions()[name],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": [
            *EarningsCallExtraction.SIGNED_FIELDS,
            *EarningsCallExtraction.UNSIGNED_FIELDS,
        ],
        "additionalProperties": False,
    }


def get_earnings_call_prompt() -> PromptTemplate:
    """Return the v1 earnings-call extraction prompt template."""
    return PromptTemplate(
        version=EARNINGS_CALL_PROMPT_VERSION,
        system=_EARNINGS_CALL_SYSTEM_PROMPT,
        user_template=_build_earnings_call_user_template(),
        output_schema=_build_earnings_call_output_schema(),
    )


def get_prompt_for_kind(source_kind: str) -> PromptTemplate:
    """Dispatch on ``ExtractionProvenance.source_kind`` to the right
    prompt template. Only whitelisted kinds (see
    :data:`~quant_platform.research.features.text.schemas.KNOWN_SOURCE_KINDS`)
    are accepted — ``filing-10k`` / ``filing-10q`` / ``filing-8k`` all
    share :func:`get_filing_prompt`; an unknown kind raises rather
    than silently routing to a wrong prompt."""
    from quant_platform.research.features.text.schemas import (
        KNOWN_FILING_KINDS,
        KNOWN_SOURCE_KINDS,
    )

    if source_kind == "news":
        return get_news_prompt()
    if source_kind in KNOWN_FILING_KINDS:
        return get_filing_prompt()
    if source_kind == "earnings-call":
        return get_earnings_call_prompt()
    raise ValueError(
        f"Unsupported source_kind for prompt dispatch: {source_kind!r}. "
        f"Known kinds: {KNOWN_SOURCE_KINDS!r}"
    )


def render_user_message(prompt: PromptTemplate, document: SourceDocument) -> str:
    """Alias of :meth:`PromptTemplate.render_user_message` for symmetry
    with operator scripts that import the function directly."""
    return prompt.render_user_message(document)


def serialise_output_schema(prompt: PromptTemplate) -> str:
    """Return ``prompt.output_schema`` as a stable JSON string.

    Used by Anthropic tool_use definitions, which want the schema as
    a JSON string. Sorted keys + no extraneous whitespace so the
    serialisation is reproducible across runs.
    """
    return json.dumps(prompt.output_schema, sort_keys=True, separators=(",", ":"))


__all__ = [
    "EARNINGS_CALL_PROMPT_VERSION",
    "FILING_PROMPT_VERSION",
    "NEWS_PROMPT_VERSION",
    "PromptTemplate",
    "get_earnings_call_prompt",
    "get_filing_prompt",
    "get_news_prompt",
    "get_prompt_for_kind",
    "render_user_message",
    "serialise_output_schema",
]
