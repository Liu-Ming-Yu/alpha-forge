"""CLI operations for governed text-event ingestion and extraction."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.research.common import _load_instrument_contracts
from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
from quant_platform.bootstrap.persistence.runtime_repositories import build_runtime_repositories
from quant_platform.bootstrap.text_events.manifests import (
    event_manifest_records,
    load_manifest_extraction_targets,
)
from quant_platform.bootstrap.text_events.reports import (
    document_type_counts,
    events_by_symbol,
    instrument_coverage,
)
from quant_platform.bootstrap.text_events.support import (
    blocked_text_payload,
    llm_credential_blocker,
    missing_cik_symbols,
    require_text_durable,
    text_artifact_root,
    text_event_slug,
    write_text_manifest,
)
from quant_platform.services.data_service.text.text_providers import (
    SECTextFilingProvider,
    download_sec_filing_records,
    load_sec_cik_map,
)
from quant_platform.services.research_service.text.extraction.text_event_extraction import (
    extract_text_event_features,
)

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.application.operator.requests import TextEventsRequest
    from quant_platform.config import PlatformSettings


async def text_events_command(
    settings: PlatformSettings,
    *,
    request: TextEventsRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.operator.requests import (
        ExtractTextFeaturesRequest,
        IngestNewsTextEventsRequest,
        IngestSecTextEventsRequest,
    )

    if isinstance(request, IngestSecTextEventsRequest):
        payload = await ingest_sec_text_events(
            settings,
            contracts_file=request.contracts_file,
            start=request.start,
            end=request.end,
            cik_map_file=request.cik_map_file,
            sec_user_agent=request.sec_user_agent,
            forms=request.forms,
            timeout_seconds=request.timeout_seconds,
            limit_per_symbol=request.limit_per_symbol,
            include_exhibits=request.include_exhibits,
            artifact_root=request.artifact_root,
        )
    elif isinstance(request, IngestNewsTextEventsRequest):
        from quant_platform.bootstrap.text_events.news import ingest_news_text_events

        payload = await ingest_news_text_events(
            settings,
            vendor=request.vendor,
            contracts_file=request.contracts_file,
            start=request.start,
            end=request.end,
            provider_codes=request.provider_codes,
            total_results_per_symbol=request.total_results_per_symbol,
            include_article_text=request.include_article_text,
            artifact_root=request.artifact_root,
        )
    elif isinstance(request, ExtractTextFeaturesRequest):
        payload = await extract_text_features(
            settings,
            start=request.start,
            end=request.end,
            prompt_version=request.prompt_version,
            document_role=request.document_role,
            source_data_manifest=request.source_data_manifest,
            artifact_root=request.artifact_root,
            concurrency=request.concurrency,
            status_file=request.status_file,
            per_call_timeout_seconds=request.per_call_timeout_seconds,
        )
    else:
        raise OperatorUsageError(f"unknown text-events request: {type(request).__name__}")
    passed = bool(payload.get("passed", False))
    return UseCaseResult(
        status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
        payload=payload,
        exit_code=0 if passed else 2,
        presentation=ResultPresentation.JSON,
    )


async def ingest_sec_text_events(
    settings: PlatformSettings,
    *,
    contracts_file: str,
    start: datetime,
    end: datetime,
    cik_map_file: Path,
    sec_user_agent: str,
    forms: tuple[str, ...],
    timeout_seconds: float,
    limit_per_symbol: int | None,
    include_exhibits: bool,
    artifact_root: Path | None,
) -> dict[str, object]:
    require_text_durable(settings)
    await verify_postgres_schema(settings)
    user_agent = str(sec_user_agent or settings.data_ingest.sec_user_agent).strip()
    slug = text_event_slug("sec", start, end)
    if not user_agent:
        return blocked_text_payload(settings, slug, "SEC user agent is required")

    contracts = _load_instrument_contracts(str(contracts_file))
    try:
        cik_map = load_sec_cik_map(cik_map_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return blocked_text_payload(settings, slug, f"CIK mapping unavailable: {exc}")

    missing = missing_cik_symbols(contracts, cik_map)
    if missing:
        return blocked_text_payload(
            settings,
            slug,
            "CIK mapping is incomplete",
            {"missing_cik_symbols": missing},
        )

    records, download_summary = await download_sec_filing_records(
        contracts=contracts,
        cik_by_symbol=cik_map,
        user_agent=user_agent,
        start=start,
        end=end,
        forms=tuple(str(form).upper() for form in forms),
        timeout_seconds=float(timeout_seconds),
        limit_per_symbol=limit_per_symbol,
        include_exhibits=bool(include_exhibits),
    )
    if not records:
        return blocked_text_payload(
            settings,
            slug,
            "SEC ingestion downloaded zero filing records",
            download_summary.to_payload(),
        )

    repositories = build_runtime_repositories(settings)
    resolved_artifact_root = Path(artifact_root or text_artifact_root(settings) / "sec_filings")
    provider = SECTextFilingProvider(records=records, artifact_root=resolved_artifact_root)
    events = await provider.ingest(repositories.text_event_store)
    manifest_path = write_text_manifest(
        settings,
        slug,
        "source_data_manifest.json",
        {
            "passed": True,
            "source": "sec",
            "download": download_summary.to_payload(),
            "events_ingested": len(events),
            "instrument_coverage": instrument_coverage(events),
            "document_type_counts": document_type_counts(events),
            "primary_events_by_symbol": events_by_symbol(events, primary=True),
            "exhibit_events_by_symbol": events_by_symbol(events, primary=False),
            "events": event_manifest_records(events),
            "content_hashes": sorted(
                str(event.metadata.get("content_hash", "")) for event in events
            ),
            "artifact_root": str(resolved_artifact_root),
        },
    )
    return {
        "passed": True,
        "source_data_manifest": str(manifest_path),
        "events_ingested": len(events),
        "records_downloaded": len(records),
        "artifact_root": str(resolved_artifact_root),
    }


async def extract_text_features(
    settings: PlatformSettings,
    *,
    start: datetime,
    end: datetime,
    prompt_version: str,
    document_role: str,
    source_data_manifest: Path | None,
    artifact_root: Path | None,
    concurrency: int = 1,
    status_file: Path | None = None,
    per_call_timeout_seconds: float = 180.0,
) -> dict[str, object]:
    require_text_durable(settings)
    await verify_postgres_schema(settings)
    slug = text_event_slug("extract", start, end)
    credential_blocker = llm_credential_blocker(settings)
    if credential_blocker:
        return blocked_text_payload(settings, slug, credential_blocker)

    prompt_version = str(prompt_version or settings.llm.text_prompt_version)
    source_targets = None
    document_role = str(document_role)
    manifest_required = (prompt_version in {"v2", "v3"} and document_role == "exhibit") or (
        prompt_version in {"v4", "v5"} and document_role == "primary"
    )
    if manifest_required and source_data_manifest is None:
        return blocked_text_payload(
            settings,
            slug,
            f"text-{prompt_version} {document_role} extraction requires --source-data-manifest",
        )
    if source_data_manifest is not None:
        source_targets, manifest_error = load_manifest_extraction_targets(
            source_data_manifest,
            document_role=document_role,
        )
        if manifest_error:
            return blocked_text_payload(settings, slug, manifest_error)

    from quant_platform.services.governance_service.llm_live_startup import (
        llm_extraction_artifact_root,
    )
    from quant_platform.services.research_service.text.features import LLMTextFeatureExtractor

    repositories = build_runtime_repositories(settings)
    artifact_root = Path(artifact_root or llm_extraction_artifact_root(settings))
    extractor = LLMTextFeatureExtractor(
        provider=settings.llm.provider,
        model=settings.llm.model,
        prompt_version=prompt_version,
        max_tokens=settings.llm.max_tokens,
        timeout_seconds=settings.llm.timeout_seconds,
        deepseek_base_url=settings.llm.deepseek_base_url,
        artifact_root=artifact_root,
        replay_only=settings.llm.live_mode_enabled and settings.llm.replay_only_live,
        max_request_latency_seconds=settings.llm.max_request_latency_seconds,
        max_daily_calls=settings.llm.max_daily_calls,
        max_daily_estimated_cost_usd=settings.llm.max_daily_estimated_cost_usd,
        estimated_cost_per_call_usd=settings.llm.estimated_cost_per_call_usd,
    )
    result = await extract_text_event_features(
        text_event_store=repositories.text_event_store,
        feature_repo=repositories.feature_repo,
        extractor=extractor,
        strategy_run_id=uuid.uuid4(),
        start=start,
        end=end,
        document_role=document_role,
        source_targets=source_targets,
        concurrency=concurrency,
        status_file=status_file,
        per_call_timeout_seconds=per_call_timeout_seconds,
    )
    payload = {
        "passed": result.passed,
        "feature_set_version": f"text-{prompt_version}",
        "prompt_version": prompt_version,
        "document_role": document_role,
        "source_data_manifest": str(source_data_manifest) if source_data_manifest else None,
        "target_manifest_events": len(source_targets) if source_targets is not None else None,
        "artifact_root": str(artifact_root),
        **result.to_payload(),
    }
    manifest_path = write_text_manifest(settings, slug, "text_extraction_manifest.json", payload)
    payload["text_extraction_manifest"] = str(manifest_path)
    if not result.passed:
        blocked = blocked_text_payload(
            settings,
            slug,
            "text feature extraction did not produce passing evidence",
            payload,
        )
        blocked["text_extraction_manifest"] = str(manifest_path)
        return blocked
    return payload


__all__ = ["extract_text_features", "ingest_sec_text_events", "text_events_command"]
