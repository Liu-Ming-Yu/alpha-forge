from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

from quant_platform.bootstrap.text_events.manifests import (
    event_manifest_records,
    load_manifest_extraction_targets,
)
from quant_platform.core.domain.market_data.text_events import TextEventType
from quant_platform.services.data_service.text.text_event_store import InMemoryTextEventStore
from quant_platform.services.data_service.text.text_providers import (
    NewsArticleRecord,
    NewsTextProvider,
    SECFilingRecord,
    SECTextFilingProvider,
    TWSNewsTextProvider,
    clean_sec_document_text,
    download_sec_filing_records,
    load_sec_cik_map,
)


@pytest.mark.asyncio
async def test_sec_provider_writes_content_addressed_artifact_and_event(tmp_path) -> None:
    store = InMemoryTextEventStore()
    instrument_id = uuid.uuid4()
    provider = SECTextFilingProvider(
        records=[
            SECFilingRecord(
                accession_number="0000320193-26-000001",
                cik="320193",
                form_type="10-Q",
                filed_at=datetime(2026, 4, 29, 13, tzinfo=UTC),
                source_uri="https://www.sec.gov/Archives/edgar/data/320193/x.txt",
                text_content="Revenue increased and guidance improved.",
                instrument_id=instrument_id,
            )
        ],
        artifact_root=tmp_path,
    )

    events = await provider.ingest(store)
    stored = await store.get_events(
        datetime(2026, 4, 29, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
    )

    assert len(events) == 1
    assert len(stored) == 1
    assert stored[0].metadata["provider"] == "sec"
    assert stored[0].metadata["ingestion_status"] == "ready"
    assert (tmp_path / "sec").is_dir()
    artifact = tmp_path / "sec" / f"{stored[0].metadata['content_hash']}.txt"
    assert "Revenue increased" in artifact.read_text(encoding="utf-8")

    manifest_records = event_manifest_records(events)
    assert manifest_records == [
        {
            "event_id": str(stored[0].event_id),
            "symbol": "",
            "instrument_id": str(instrument_id),
            "occurred_at": "2026-04-29T13:00:00+00:00",
            "source_uri": "https://www.sec.gov/Archives/edgar/data/320193/x.txt",
            "artifact_uri": str(artifact),
            "accession_number": "0000320193-26-000001",
            "cik": "320193",
            "form_type": "10-Q",
            "document_type": "10-Q",
            "document_name": "",
            "document_description": "",
            "is_primary_document": True,
            "content_hash": stored[0].metadata["content_hash"],
        }
    ]


@pytest.mark.asyncio
async def test_sec_provider_is_deterministic_for_same_filing(tmp_path) -> None:
    store = InMemoryTextEventStore()
    record = SECFilingRecord(
        accession_number="0000000000-26-000001",
        cik="1",
        form_type="8-K",
        filed_at=datetime(2026, 4, 29, tzinfo=UTC),
        source_uri="https://www.sec.gov/filing.txt",
        text_content="same filing",
    )
    provider = SECTextFilingProvider(records=[record], artifact_root=tmp_path)

    first = await provider.ingest(store)
    second = await provider.ingest(store)

    assert first[0].event_id == second[0].event_id
    stored = await store.get_events(
        datetime(2026, 4, 29, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
    )
    assert len(stored) == 1


class _Response:
    def __init__(
        self,
        payload=None,
        text: str = "",
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def get(self, url: str):
        self.urls.append(url)
        if "submissions" in url:
            return _Response(
                {
                    "filings": {
                        "recent": {
                            "form": ["10-Q", "8-K"],
                            "accessionNumber": [
                                "0000320193-26-000001",
                                "0000320193-26-000002",
                            ],
                            "filingDate": ["2026-04-29", "2024-01-01"],
                            "acceptanceDateTime": [
                                "2026-04-29T16:01:02.000Z",
                                "2024-01-01T16:01:02.000Z",
                            ],
                            "primaryDocument": ["aapl-10q.htm", "old-8k.htm"],
                        }
                    }
                }
            )
        if url.endswith("-index.html"):
            return _Response(
                text="""
                <table>
                  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
                  <tr>
                    <td>1</td><td>Primary document</td>
                    <td><a href="/Archives/edgar/data/320193/x/aapl-8k.htm">aapl-8k.htm</a></td>
                    <td>8-K</td>
                  </tr>
                  <tr>
                    <td>2</td><td>Results release</td>
                    <td><a href="/Archives/edgar/data/320193/x/ex991.htm">ex991.htm</a></td>
                    <td>EX-99.1</td>
                  </tr>
                  <tr>
                    <td>3</td><td>Earnings presentation image</td>
                    <td><a href="/Archives/edgar/data/320193/x/earnings001.jpg">earnings001.jpg</a></td>
                    <td>GRAPHIC</td>
                  </tr>
                </table>
                """
            )
        return _Response(text="<html><body>Revenue increased &amp; outlook improved.</body></html>")


@pytest.mark.asyncio
async def test_download_sec_filing_records_filters_forms_and_dates() -> None:
    instrument_id = uuid.uuid4()
    client = _Client()

    records, summary = await download_sec_filing_records(
        contracts={instrument_id: {"symbol": "AAPL"}},
        cik_by_symbol={"AAPL": "320193"},
        user_agent="QuantPlatform test@example.com",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 12, 31, tzinfo=UTC),
        forms=("10-Q", "8-K"),
        client=client,
    )

    assert summary.records_downloaded == 1
    assert records[0].instrument_id == instrument_id
    assert records[0].form_type == "10-Q"
    assert records[0].metadata["symbol"] == "AAPL"
    assert "Revenue increased" in records[0].text_content


@pytest.mark.asyncio
async def test_download_sec_filing_records_includes_preferred_exhibits() -> None:
    instrument_id = uuid.uuid4()
    client = _Client()

    records, summary = await download_sec_filing_records(
        contracts={instrument_id: {"symbol": "AAPL"}},
        cik_by_symbol={"AAPL": "320193"},
        user_agent="QuantPlatform test@example.com",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 12, 31, tzinfo=UTC),
        forms=("10-Q",),
        client=client,
        include_exhibits=True,
    )

    assert len(records) == 2
    assert summary.primary_documents_downloaded == 1
    assert summary.exhibits_downloaded == 1
    exhibit = next(record for record in records if not record.is_primary_document)
    assert exhibit.document_type == "EX-99.1"
    assert exhibit.document_description == "Results release"
    assert exhibit.filed_at > records[0].filed_at


def test_load_sec_cik_map_normalizes_values(tmp_path) -> None:
    path = tmp_path / "cik.json"
    path.write_text('{"aapl": "0000320193"}', encoding="utf-8")

    assert load_sec_cik_map(path) == {"AAPL": "320193"}


def test_clean_sec_document_text_strips_html() -> None:
    assert clean_sec_document_text("<p>Guidance &amp; revenue</p>") == "Guidance & revenue"


@pytest.mark.asyncio
async def test_tws_news_provider_writes_article_artifact_and_event(tmp_path) -> None:
    store = InMemoryTextEventStore()
    instrument_id = uuid.uuid4()
    provider = TWSNewsTextProvider(
        records=[
            NewsArticleRecord(
                vendor="tws",
                provider_code="BRFG",
                article_id="BRFG$123",
                headline="Apple supplier raises outlook",
                published_at=datetime(2026, 4, 29, 14, 30, tzinfo=UTC),
                source_uri="ibkr://news/BRFG/BRFG$123",
                article_text="<p>Revenue outlook improved.</p>",
                instrument_id=instrument_id,
                symbol="AAPL",
                metadata={"con_id": "265598"},
            )
        ],
        artifact_root=tmp_path,
    )

    events = await provider.ingest(store)
    stored = await store.get_events(
        datetime(2026, 4, 29, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
    )

    assert len(events) == 1
    assert stored[0].event_type is TextEventType.NEWS_HEADLINE
    assert stored[0].metadata["provider"] == "tws"
    assert stored[0].metadata["provider_code"] == "BRFG"
    assert stored[0].metadata["article_id"] == "BRFG$123"
    assert stored[0].metadata["ingestion_status"] == "ready"
    artifact = tmp_path / "tws" / f"{stored[0].metadata['content_hash']}.txt"
    artifact_text = artifact.read_text(encoding="utf-8")
    assert "Headline: Apple supplier raises outlook" in artifact_text
    assert "Revenue outlook improved." in artifact_text

    manifest_records = event_manifest_records(events)
    assert manifest_records[0]["provider"] == "tws"
    assert manifest_records[0]["provider_code"] == "BRFG"
    assert manifest_records[0]["article_id"] == "BRFG$123"


@pytest.mark.asyncio
async def test_news_provider_is_deterministic_for_same_article(tmp_path) -> None:
    store = InMemoryTextEventStore()
    record = NewsArticleRecord(
        vendor="tws",
        provider_code="DJNL",
        article_id="DJNL$abc",
        headline="Same headline",
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        source_uri="ibkr://news/DJNL/DJNL$abc",
    )
    provider = TWSNewsTextProvider(records=[record], artifact_root=tmp_path)

    first = await provider.ingest(store)
    second = await provider.ingest(store)

    assert first[0].event_id == second[0].event_id
    stored = await store.get_events(
        datetime(2026, 4, 29, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
    )
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_empty_news_provider_keeps_legacy_noop_behavior() -> None:
    events = await NewsTextProvider().ingest(InMemoryTextEventStore())

    assert events == []


def test_source_manifest_targets_require_per_event_records(tmp_path) -> None:
    path = tmp_path / "source_data_manifest.json"
    path.write_text(json.dumps({"passed": True, "events_ingested": 1}), encoding="utf-8")

    targets, error = load_manifest_extraction_targets(path, document_role="exhibit")

    assert targets == ()
    assert "missing per-event records" in error


def test_source_manifest_targets_filter_exhibits(tmp_path) -> None:
    exhibit_id = uuid.uuid4()
    primary_id = uuid.uuid4()
    path = tmp_path / "source_data_manifest.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": str(primary_id),
                        "symbol": "AAPL",
                        "instrument_id": str(uuid.uuid4()),
                        "occurred_at": "2026-04-01T00:00:00+00:00",
                        "is_primary_document": True,
                    },
                    {
                        "event_id": str(exhibit_id),
                        "symbol": "AAPL",
                        "instrument_id": str(uuid.uuid4()),
                        "occurred_at": "2026-04-01T00:00:00+00:00",
                        "document_type": "EX-99.1",
                        "is_primary_document": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    targets, error = load_manifest_extraction_targets(path, document_role="exhibit")

    assert error == ""
    assert [target.event_id for target in targets] == [exhibit_id]
    assert targets[0].document_type == "EX-99.1"
