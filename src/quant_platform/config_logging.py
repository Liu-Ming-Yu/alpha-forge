"""Structured logging bootstrap settings."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    log_level: str = "INFO"
    log_format: str = "json"
    sentry_dsn: str = ""


_SECRET_KEYS: frozenset[str] = frozenset(
    {"password", "dsn", "api_key", "api_token", "token", "secret", "credential"}
)


def _mask_secrets(
    logger: object, method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if any(s in key.lower() for s in _SECRET_KEYS):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Configure ``structlog`` processors and stdlib log level.

    Call once at process start, before any log statements.

    Args:
        settings: Logging sub-model.  Uses defaults when ``None``.
    """
    cfg = settings or LoggingSettings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        force=True,
    )

    if cfg.log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _mask_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, cfg.log_level.upper(), logging.INFO),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if cfg.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=cfg.sentry_dsn,
                traces_sample_rate=0.05,
                environment="production",
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "Sentry DSN is configured but sentry-sdk is not installed. "
                "Install with: pip install sentry-sdk"
            )
