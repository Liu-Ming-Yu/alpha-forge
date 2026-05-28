"""Provider error classification for LLM text feature extraction."""

from __future__ import annotations


def is_retryable_provider_error(exc: Exception) -> bool:
    """Return True for transient provider errors worth retrying."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "rate", "overload"))


__all__ = ["is_retryable_provider_error"]
