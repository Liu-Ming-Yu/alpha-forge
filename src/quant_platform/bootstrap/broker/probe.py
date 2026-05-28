"""Shared broker-probe configuration helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_platform.config import BrokerSettings, PlatformSettings


def broker_gate_settings(settings: PlatformSettings) -> BrokerSettings:
    """Return the configured broker settings for broker probes.

    Host and port are operator-owned runtime configuration. Docker/WSL
    deployments should set ``QP__DOCKER_BROKER_HOST`` /
    ``QP__DOCKER_BROKER_PORT`` in compose rather than relying on a hidden
    native-Windows rewrite here.
    """

    return settings.broker


def classify_broker_probe_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "socket connection failed" in message or "connection refused" in message:
        return "socket_failure"
    if "nextvalidid" in message or "did not send" in message:
        if "trusted ips" in message or "localhost only" in message:
            return "auth_or_trusted_ip"
        return "handshake_timeout"
    if "trusted ips" in message or "localhost only" in message:
        return "auth_or_trusted_ip"
    return "broker_probe_failure"


__all__ = [
    "broker_gate_settings",
    "classify_broker_probe_failure",
]
