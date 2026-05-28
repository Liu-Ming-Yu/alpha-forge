"""Error callbacks for the IB wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.core.exceptions import BrokerSubmissionError, BrokerUnavailableError

if TYPE_CHECKING:
    import asyncio
    from _thread import LockType

    HistoricalBarTuple = tuple[str, float, float, float, float, int]
    HistoricalNewsTuple = tuple[str, str, str, str]
    NewsArticleTuple = tuple[int, str]

log = structlog.get_logger(__name__)


class IBErrorCallbackMixin:
    """Broker error callback handling used by ``_IBWrapper``."""

    _cancel_futures: dict[int, asyncio.Future[None]]
    _connect_error: tuple[int, str] | None
    _connect_error_event: Any
    _hist_data: dict[int, list[HistoricalBarTuple]]
    _hist_futures: dict[int, asyncio.Future[list[HistoricalBarTuple]]]
    _historical_news: dict[int, list[HistoricalNewsTuple]]
    _historical_news_futures: dict[int, asyncio.Future[list[HistoricalNewsTuple]]]
    _lifecycle_lock: LockType
    _news_article_futures: dict[int, asyncio.Future[NewsArticleTuple]]
    _order_statuses: dict[int, asyncio.Future[str]]

    def _reject(self, future: asyncio.Future[Any], exc: Exception) -> None:
        raise NotImplementedError

    def _resolve(self, future: asyncio.Future[Any], value: object) -> None:
        raise NotImplementedError

    def error(self, reqId: int, *args: object, **kwargs: object) -> None:
        args = _normalise_keyword_error_args(args, kwargs)
        error_code, error_string, error_time, advanced_order_reject_json = _normalise_error_args(
            args
        )
        if error_code is None:
            log.warning(
                "ib_wrapper.error.unrecognised_signature",
                req_id=reqId,
                args=[str(arg) for arg in args],
            )
            return
        log.warning(
            "ib_wrapper.error",
            req_id=reqId,
            code=error_code,
            message=error_string,
            error_time=error_time,
            advanced_order_reject_json=advanced_order_reject_json,
        )
        if reqId == -1 and error_code == 326:
            self._connect_error = (error_code, error_string)
            self._connect_error_event.set()

        future = self._order_statuses.get(reqId)
        if future and not future.done() and reqId >= 0:
            self._reject(future, BrokerSubmissionError(f"IB error {error_code}: {error_string}"))

        already_cancelled = error_code == 10148 and "state: Cancelled" in error_string
        if error_code in (201, 202) or already_cancelled:
            if future and not future.done():
                self._reject(
                    future, BrokerSubmissionError(f"IB error {error_code}: {error_string}")
                )
            cancel_future = self._cancel_futures.get(reqId)
            if cancel_future and not cancel_future.done():
                self._resolve(cancel_future, None)

        if error_code in (162, 165, 200, 321, 354, 366):
            with self._lifecycle_lock:
                hist_future = self._hist_futures.pop(reqId, None)
                self._hist_data.pop(reqId, None)
            if hist_future is not None:
                self._reject(
                    hist_future,
                    BrokerUnavailableError(
                        f"historicalData request {reqId} failed: code={error_code} {error_string}"
                    ),
                )

        if reqId >= 0:
            with self._lifecycle_lock:
                historical_news_future = self._historical_news_futures.pop(reqId, None)
                self._historical_news.pop(reqId, None)
                news_article_future = self._news_article_futures.pop(reqId, None)
            if historical_news_future is not None:
                self._reject(
                    historical_news_future,
                    BrokerUnavailableError(
                        f"historicalNews request {reqId} failed: code={error_code} {error_string}"
                    ),
                )
            if news_article_future is not None:
                self._reject(
                    news_article_future,
                    BrokerUnavailableError(
                        f"newsArticle request {reqId} failed: code={error_code} {error_string}"
                    ),
                )


def _normalise_keyword_error_args(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[object, ...]:
    if args or not kwargs:
        return args
    if "errorCode" not in kwargs or "errorString" not in kwargs:
        return args
    advanced = kwargs.get("advancedOrderRejectJson", "")
    if "errorTime" in kwargs:
        return (kwargs["errorTime"], kwargs["errorCode"], kwargs["errorString"], advanced)
    return (kwargs["errorCode"], kwargs["errorString"], advanced)


def _normalise_error_args(args: tuple[object, ...]) -> tuple[int | None, str, str, str]:
    """Accept old and protobuf-era ibapi error callback signatures."""
    error_time = ""
    advanced = ""
    if len(args) == 2:
        raw_code, raw_message = args
    elif len(args) == 3:
        first, second, third = args
        if _looks_like_error_code(second) and not _looks_like_error_code(third):
            error_time = str(first)
            raw_code = second
            raw_message = third
        else:
            raw_code = first
            raw_message = second
            advanced = str(third or "")
    elif len(args) >= 4:
        error_time = str(args[0])
        raw_code = args[1]
        raw_message = args[2]
        advanced = str(args[3] or "")
    else:
        return None, "", "", ""

    try:
        code = int(cast("Any", raw_code))
    except (TypeError, ValueError):
        return None, str(raw_message), error_time, advanced
    return code, str(raw_message), error_time, advanced


def _looks_like_error_code(value: object) -> bool:
    try:
        int(cast("Any", value))
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["IBErrorCallbackMixin"]
