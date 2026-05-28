"""Prometheus registry factory and no-op metric shims."""

from __future__ import annotations

from typing import Any, Protocol, cast


class MetricLike(Protocol):
    def labels(self, *args: str, **kwargs: str) -> MetricLike: ...
    def inc(self, amount: float = 1.0) -> None: ...
    def dec(self, amount: float = 1.0) -> None: ...
    def set(self, value: float) -> None: ...
    def observe(self, amount: float) -> None: ...


class CollectorRegistryLike(Protocol):
    """Structural marker for Prometheus collector registries."""


class NoopMetric:
    """No-op stand-in for a Prometheus metric when the client is unavailable."""

    def labels(self, *args: str, **kwargs: str) -> NoopMetric:  # noqa: D401
        return self

    def inc(self, amount: float = 1.0) -> None:  # noqa: D401
        return None

    def dec(self, amount: float = 1.0) -> None:  # noqa: D401
        return None

    def set(self, value: float) -> None:  # noqa: D401
        return None

    def observe(self, amount: float) -> None:  # noqa: D401
        return None


NOOP = NoopMetric()

_prometheus_client: Any | None
try:  # pragma: no cover - the import path is exercised in both directions.
    import prometheus_client as _prometheus_client

    PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _prometheus_client = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    PROM_AVAILABLE = False
else:
    CONTENT_TYPE_LATEST = (
        str(_prometheus_client.CONTENT_TYPE_LATEST)
        if _prometheus_client is not None
        else "text/plain; version=0.0.4; charset=utf-8"
    )


REGISTRY: CollectorRegistryLike | None = None


def get_registry() -> CollectorRegistryLike | None:
    """Return the module-level ``CollectorRegistry`` or ``None`` when disabled."""
    global REGISTRY
    if _prometheus_client is None:
        return None
    if REGISTRY is None:
        REGISTRY = cast("CollectorRegistryLike", _prometheus_client.CollectorRegistry())
    return REGISTRY


def reset_registry() -> None:
    """Reset the module-level registry before rebuilding the metric catalogue."""
    global REGISTRY
    REGISTRY = None


def render_latest() -> tuple[bytes, str]:
    """Produce the textual exposition payload and its content type."""
    if _prometheus_client is None:
        return b"", CONTENT_TYPE_LATEST
    return _prometheus_client.generate_latest(get_registry()), CONTENT_TYPE_LATEST


def counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> MetricLike:
    if _prometheus_client is None:
        return NOOP
    return cast(
        "MetricLike",
        _prometheus_client.Counter(name, doc, list(labels), registry=get_registry()),
    )


def gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> MetricLike:
    if _prometheus_client is None:
        return NOOP
    return cast(
        "MetricLike",
        _prometheus_client.Gauge(name, doc, list(labels), registry=get_registry()),
    )


def histogram(
    name: str,
    doc: str,
    labels: tuple[str, ...] = (),
    buckets: tuple[float, ...] | None = None,
) -> MetricLike:
    if _prometheus_client is None:
        return NOOP
    kwargs: dict[str, Any] = {"registry": get_registry()}
    if buckets is not None:
        kwargs["buckets"] = buckets
    return cast("MetricLike", _prometheus_client.Histogram(name, doc, list(labels), **kwargs))
