"""SEC EDGAR HTTP helpers."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from typing import Any, Protocol

_SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.25
_SEC_MAX_REQUEST_ATTEMPTS = 4
_SEC_DEFAULT_RETRY_AFTER_SECONDS = 2.0


class HTTPResponse(Protocol):
    text: str

    def json(self) -> object: ...

    def raise_for_status(self) -> None: ...


class AsyncHTTPClient(Protocol):
    async def get(self, url: str) -> HTTPResponse: ...


class RateLimitedSECClient:
    """Polite SEC client wrapper for real network calls."""

    def __init__(
        self,
        client: AsyncHTTPClient,
        *,
        min_interval_seconds: float = _SEC_MIN_REQUEST_INTERVAL_SECONDS,
        max_attempts: int = _SEC_MAX_REQUEST_ATTEMPTS,
    ) -> None:
        self._client = client
        self._min_interval_seconds = min_interval_seconds
        self._max_attempts = max_attempts
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def get(self, url: str) -> HTTPResponse:
        response: HTTPResponse | None = None
        for attempt in range(self._max_attempts):
            await self._pace()
            response = await self._client.get(url)
            if getattr(response, "status_code", None) != 429:
                return response
            await asyncio.sleep(_retry_after_seconds(response) * (attempt + 1))
        if response is None:
            raise RuntimeError(f"SEC request failed before response: {url}")
        return response

    async def _pace(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait_seconds = self._min_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_at = time.monotonic()


async def get_json(client: AsyncHTTPClient, url: str) -> dict[str, Any]:
    response = await client.get(url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"SEC response was not a JSON object: {url}")
    return payload


def retry_after_seconds(response: HTTPResponse) -> float:
    return _retry_after_seconds(response)


def _retry_after_seconds(response: HTTPResponse) -> float:
    headers = getattr(response, "headers", {})
    raw_retry_after = ""
    if isinstance(headers, Mapping):
        raw_retry_after = str(headers.get("retry-after", "")).strip()
    if raw_retry_after:
        with contextlib.suppress(ValueError):
            return max(float(raw_retry_after), _SEC_DEFAULT_RETRY_AFTER_SECONDS)
    return _SEC_DEFAULT_RETRY_AFTER_SECONDS


def sec_submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"


def sec_archive_url(cik: str, accession: str, document_name: str) -> str:
    accession_path = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_path}/{document_name}"


def sec_filing_detail_url(cik: str, accession: str) -> str:
    accession_path = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_path}/{accession}-index.html"
    )


def sec_archive_index_json_url(cik: str, accession: str) -> str:
    accession_path = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_path}/index.json"


def absolute_sec_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"https://www.sec.gov{href}"
    return f"https://www.sec.gov/{href.lstrip('/')}"


__all__ = [
    "AsyncHTTPClient",
    "HTTPResponse",
    "RateLimitedSECClient",
    "absolute_sec_url",
    "get_json",
    "retry_after_seconds",
    "sec_archive_index_json_url",
    "sec_archive_url",
    "sec_filing_detail_url",
    "sec_submissions_url",
]
