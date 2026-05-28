"""Historical text-event feature extraction helpers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from quant_platform.services.research_service.text.extraction.text_event_extraction_helpers import (
    events_for_targets,
    failure_detail,
    matches_document_role,
    read_event_content,
    target_for_event,
)
from quant_platform.services.research_service.text.extraction.text_event_extraction_progress import (  # noqa: E501
    heartbeat_loop,
    write_status,
)
from quant_platform.services.research_service.text.features.errors import (
    FeatureExtractionError,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence
    from pathlib import Path

    from quant_platform.core.contracts import FeatureRepository, TextEventProvider
    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.services.research_service.text.features import LLMTextFeatureExtractor

log = structlog.get_logger(__name__)

DEFAULT_PER_CALL_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class TextEventExtractionTarget:
    """One source-manifest event targeted for governed extraction."""

    event_id: uuid.UUID
    symbol: str = ""
    instrument_id: uuid.UUID | None = None
    occurred_at: datetime | None = None
    source_uri: str = ""
    artifact_uri: str = ""
    accession_number: str = ""
    form_type: str = ""
    document_type: str = ""
    document_name: str = ""
    document_description: str = ""
    is_primary_document: bool | None = None
    content_hash: str = ""


@dataclass(frozen=True)
class TextEventExtractionResult:
    """Summary of one historical text extraction pass."""

    events_seen: int
    vectors_stored: int
    skipped_macro_events: int
    skipped_missing_content: int
    skipped_document_role: int
    skipped_duplicate_vectors: int
    failed_events: int
    failures: tuple[str, ...]
    failed_event_details: tuple[dict[str, object], ...]

    @property
    def passed(self) -> bool:
        return self.failed_events == 0 and (
            self.vectors_stored > 0 or self.skipped_duplicate_vectors > 0
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "events_seen": self.events_seen,
            "vectors_stored": self.vectors_stored,
            "skipped_macro_events": self.skipped_macro_events,
            "skipped_missing_content": self.skipped_missing_content,
            "skipped_document_role": self.skipped_document_role,
            "skipped_duplicate_vectors": self.skipped_duplicate_vectors,
            "failed_events": self.failed_events,
            "failures": list(self.failures),
            "failed_event_details": list(self.failed_event_details),
            "passed": self.passed,
        }


async def extract_text_event_features(
    *,
    text_event_store: TextEventProvider,
    feature_repo: FeatureRepository,
    extractor: LLMTextFeatureExtractor,
    strategy_run_id: uuid.UUID,
    start: datetime,
    end: datetime,
    document_role: str = "all",
    source_targets: Sequence[TextEventExtractionTarget] | None = None,
    concurrency: int = 1,
    per_call_timeout_seconds: float = DEFAULT_PER_CALL_TIMEOUT_SECONDS,
    status_file: Path | None = None,
) -> TextEventExtractionResult:
    """Extract event-level text feature vectors and persist them.

    ``concurrency`` runs up to N extractions in parallel via ``asyncio.to_thread``
    (the underlying LLM call is blocking httpx). ``per_call_timeout_seconds``
    caps any single call so a hung extraction can't stall the whole run.
    When ``status_file`` is given, a heartbeat task writes JSON progress every
    ~10 seconds so the operator can monitor a long-running backfill.
    """
    if document_role not in {"all", "primary", "exhibit"}:
        raise ValueError("document_role must be one of: all, primary, exhibit")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    all_events = await text_event_store.get_events(start, end)
    events, missing_targets = events_for_targets(all_events, source_targets)

    state: dict[str, int | float | str] = {
        "started_at": datetime.now(tz=UTC).isoformat(),
        "total_events": len(events),
        "extracted": 0,
        "skipped_macro": 0,
        "skipped_missing": 0,
        "skipped_document_role": 0,
        "skipped_duplicates": 0,
        "failed": 0,
        "in_flight": 0,
    }
    failures: list[str] = []
    failure_details: list[dict[str, object]] = []

    for missing_target in missing_targets:
        reason = "manifest event not found in durable text_events"
        failures.append(f"{missing_target.event_id}: {reason}")
        failure_details.append(
            failure_detail(
                event=None,
                target=missing_target,
                reason=reason,
                error_class="MissingDurableTextEvent",
            )
        )
    state["failed"] = len(failures)

    started_monotonic = time.monotonic()
    stop_heartbeat = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None
    if status_file is not None:
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(status_file, state, started_monotonic, stop_heartbeat)
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(event: TextEvent) -> None:
        nonlocal failures, failure_details
        async with semaphore:
            state["in_flight"] = int(state["in_flight"]) + 1
            try:
                if event.instrument_id is None:
                    state["skipped_macro"] = int(state["skipped_macro"]) + 1
                    return
                if not matches_document_role(event, document_role):
                    state["skipped_document_role"] = int(state["skipped_document_role"]) + 1
                    return
                content = read_event_content(event.artifact_uri)
                if not content.strip():
                    state["skipped_missing"] = int(state["skipped_missing"]) + 1
                    if source_targets is not None:
                        target = target_for_event(source_targets, event.event_id)
                        reason = "manifest event artifact has no readable content"
                        failures.append(f"{event.event_id}: {reason}")
                        failure_details.append(
                            failure_detail(
                                event=event,
                                target=target,
                                reason=reason,
                                error_class="MissingContent",
                            )
                        )
                        state["failed"] = len(failures)
                    return
                try:
                    vector = await asyncio.wait_for(
                        asyncio.to_thread(
                            extractor.extract,
                            event,
                            content,
                            strategy_run_id,
                            as_of=event.occurred_at,
                        ),
                        timeout=per_call_timeout_seconds,
                    )
                    await feature_repo.store_vector(vector)
                    state["extracted"] = int(state["extracted"]) + 1
                except ValueError as exc:
                    if "Duplicate FeatureVector" in str(exc):
                        state["skipped_duplicates"] = int(state["skipped_duplicates"]) + 1
                        return
                    failures.append(f"{event.event_id}: {exc}")
                    failure_details.append(
                        failure_detail(
                            event=event,
                            target=target_for_event(source_targets, event.event_id),
                            reason=str(exc),
                            error_class=exc.__class__.__name__,
                        )
                    )
                    state["failed"] = len(failures)
                    log.warning(
                        "text_extractor.failure",
                        event_id=str(event.event_id),
                        error_class=exc.__class__.__name__,
                        reason=str(exc)[:300],
                    )
                except TimeoutError as exc:
                    reason = f"extraction timed out after {per_call_timeout_seconds}s"
                    failures.append(f"{event.event_id}: {reason}")
                    failure_details.append(
                        failure_detail(
                            event=event,
                            target=target_for_event(source_targets, event.event_id),
                            reason=reason,
                            error_class="ExtractionTimeout",
                        )
                    )
                    state["failed"] = len(failures)
                    log.warning("text_extractor.timeout", event_id=str(event.event_id))
                    del exc
                except (FeatureExtractionError, OSError) as exc:
                    failures.append(f"{event.event_id}: {exc}")
                    failure_details.append(
                        failure_detail(
                            event=event,
                            target=target_for_event(source_targets, event.event_id),
                            reason=str(exc),
                            error_class=exc.__class__.__name__,
                        )
                    )
                    state["failed"] = len(failures)
                    log.warning(
                        "text_extractor.failure",
                        event_id=str(event.event_id),
                        error_class=exc.__class__.__name__,
                        reason=str(exc)[:300],
                    )
            finally:
                state["in_flight"] = int(state["in_flight"]) - 1

    try:
        await asyncio.gather(*(_one(event) for event in events))
    finally:
        stop_heartbeat.set()
        if heartbeat_task is not None:
            await heartbeat_task
            # Final flush so the status file reflects the terminal state.
            write_status(status_file, state, started_monotonic, terminal=True)

    return TextEventExtractionResult(
        events_seen=len(source_targets) if source_targets is not None else len(events),
        vectors_stored=int(state["extracted"]),
        skipped_macro_events=int(state["skipped_macro"]),
        skipped_missing_content=int(state["skipped_missing"]),
        skipped_document_role=int(state["skipped_document_role"]),
        skipped_duplicate_vectors=int(state["skipped_duplicates"]),
        failed_events=len(failures),
        failures=tuple(failures),
        failed_event_details=tuple(failure_details),
    )


__all__ = [
    "DEFAULT_PER_CALL_TIMEOUT_SECONDS",
    "TextEventExtractionResult",
    "TextEventExtractionTarget",
    "extract_text_event_features",
]
