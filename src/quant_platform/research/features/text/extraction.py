"""Extraction pipeline.

Drives an :class:`LLMClient` over a list of :class:`SourceDocument`
instances, validates each response against the prompt's schema, and
returns one :class:`ExtractedRecord` per input document — successes
and failures alike. Order is preserved so callers can index by input
position.

Two non-obvious behaviours worth knowing:

1. **Failures are persisted, not dropped.** When the LLM returns
   malformed JSON, out-of-range scores, or any other unrecoverable
   error after the retry budget, the pipeline emits a
   :class:`FailedExtraction` record. The downstream storage layer
   is the source of truth for "every document we tried to process";
   silently dropping failed extractions would let coverage stats
   hallucinate success.

2. **Retries are bounded and exponential.** The default budget is
   3 attempts with delays 0.5s, 1.5s, 4.5s. Transient transport
   errors (connection reset, rate-limit-with-retry-hint, server
   5xx) consume the budget; semantic errors (validation failure,
   payload key mismatch) do *not* — re-asking the same model the
   same question won't fix a schema mismatch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.research.features.text.client import hash_raw_response
from quant_platform.research.features.text.schemas import (
    ExtractedRecord,
    ExtractionProvenance,
    FailedExtraction,
    extraction_from_dict_for_kind,
    utc_now_iso,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.research.features.text.client import LLMClient
    from quant_platform.research.features.text.prompts import PromptTemplate
    from quant_platform.research.features.text.schemas import SourceDocument

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionConfig:
    """Tunable extraction-loop behaviour.

    Attributes
    ----------
    max_attempts:
        Total number of attempts per document, including the first.
        ``1`` disables retries.
    initial_delay_seconds:
        Base for the exponential backoff. The Nth retry waits
        ``initial_delay_seconds * backoff_factor ** (N-1)`` seconds.
    backoff_factor:
        Exponential backoff multiplier.
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    backoff_factor: float = 3.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("ExtractionConfig.max_attempts must be >= 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("ExtractionConfig.initial_delay_seconds must be >= 0")
        if self.backoff_factor < 1.0:
            raise ValueError("ExtractionConfig.backoff_factor must be >= 1.0")


DEFAULT_CONFIG: ExtractionConfig = ExtractionConfig()


# ---------------------------------------------------------------------------
# Internal error classification
# ---------------------------------------------------------------------------
#
# Transient errors retry; semantic errors fail fast. We can't know
# the full taxonomy of every provider, so the heuristic is:
#
#   * ``LLMTransientError`` / network exceptions / generic
#     ``ConnectionError`` / ``TimeoutError`` → retry
#   * Everything else (schema validation, payload mismatch,
#     application-level provider errors) → fail fast


class LLMTransientError(Exception):
    """Marker exception clients can raise to signal "retry me".

    The default :class:`MockLLMClient` never raises this; an
    Anthropic adapter can wrap ``anthropic.APITimeoutError`` /
    ``anthropic.APIConnectionError`` / 5xx responses in it.
    """


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (LLMTransientError, ConnectionError, TimeoutError))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_documents(
    *,
    client: LLMClient,
    prompt: PromptTemplate,
    documents: Sequence[SourceDocument],
    config: ExtractionConfig = DEFAULT_CONFIG,
    sleep: object = time.sleep,
) -> list[ExtractedRecord]:
    """Run ``client`` over ``documents`` and return one record per input.

    Parameters
    ----------
    client:
        Any :class:`LLMClient` (Anthropic, mock, future providers).
    prompt:
        Prompt template to feed every document.
    documents:
        Iterable of :class:`SourceDocument` instances.
    config:
        Retry / backoff config.
    sleep:
        Sleep function used between retries. Tests pass a no-op so
        the retry loop doesn't block.

    Returns
    -------
    list[ExtractedRecord]
        Same order as ``documents``. Each record is either a
        successful extraction with full provenance OR a
        :class:`FailedExtraction` sentinel — the storage layer
        persists both shapes so coverage stats stay honest.
    """
    records: list[ExtractedRecord] = []
    for document in documents:
        record = _extract_one(
            client=client,
            prompt=prompt,
            document=document,
            config=config,
            sleep=sleep,
        )
        records.append(record)
    return records


def _extract_one(
    *,
    client: LLMClient,
    prompt: PromptTemplate,
    document: SourceDocument,
    config: ExtractionConfig,
    sleep: object,
) -> ExtractedRecord:
    last_error: str = ""
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = client.extract(prompt, document)
        except BaseException as exc:  # noqa: BLE001 — classify before swallowing
            if not _is_transient(exc) or attempt == config.max_attempts:
                LOG.warning(
                    "extraction.transport_error",
                    extra={"doc_id": document.doc_id, "error": str(exc)},
                )
                return _failure_record(
                    document=document,
                    prompt=prompt,
                    client=client,
                    reason="client_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            last_error = f"{type(exc).__name__}: {exc}"
            delay = config.initial_delay_seconds * (config.backoff_factor ** (attempt - 1))
            LOG.info(
                "extraction.retry",
                extra={"doc_id": document.doc_id, "attempt": attempt, "delay": delay},
            )
            sleep(delay)  # type: ignore[operator]
            continue
        # Validate the payload against the schema. Schema errors are
        # semantic — no retry — but we still capture the detail so an
        # operator can debug the prompt.
        try:
            extraction = extraction_from_dict_for_kind(document.kind, response.payload)
        except (KeyError, ValueError, TypeError) as exc:
            return _failure_record(
                document=document,
                prompt=prompt,
                client=client,
                reason="score_out_of_range" if isinstance(exc, ValueError) else "malformed_payload",
                detail=f"{type(exc).__name__}: {exc}",
            )
        provenance = ExtractionProvenance(
            prompt_version=prompt.version,
            model_version=response.model_version or client.model_version,
            source_kind=document.kind,
            source_doc_id=document.doc_id,
            extracted_at=utc_now_iso(),
            confidence=float(response.confidence),
            raw_response_hash=hash_raw_response(response.raw_text),
        )
        return ExtractedRecord(
            instrument_id=document.instrument_id,
            extraction=extraction,
            provenance=provenance,
        )

    # Loop exhausted retries on transient errors.
    return _failure_record(
        document=document,
        prompt=prompt,
        client=client,
        reason="client_error",
        detail=f"retry budget exhausted; last error: {last_error}",
    )


def _failure_record(
    *,
    document: SourceDocument,
    prompt: PromptTemplate,
    client: LLMClient,
    reason: str,
    detail: str,
) -> ExtractedRecord:
    failure = FailedExtraction(
        source_doc_id=document.doc_id,
        source_kind=document.kind,
        failed_at=utc_now_iso(),
        reason=reason,
        detail=detail,
        prompt_version=prompt.version,
        model_version=client.model_version,
    )
    return ExtractedRecord(instrument_id=document.instrument_id, failure=failure)


__all__ = [
    "DEFAULT_CONFIG",
    "ExtractionConfig",
    "LLMTransientError",
    "extract_documents",
]
