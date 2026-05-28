"""Composition root for CLI-facing application use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.operator_use_cases import (
    register_broker_use_cases,
    register_data_use_cases,
    register_engine_use_cases,
    register_governance_use_cases,
    register_infra_use_cases,
    register_research_use_cases,
    register_runtime_use_cases,
)
from quant_platform.application.use_cases import UseCaseRegistry
from quant_platform.bootstrap.operator_adapters import (
    BrokerAdapters,
    DataAdapters,
    EngineAdapters,
    GovernanceAdapters,
    InfraAdapters,
    RuntimeAdapters,
)
from quant_platform.research.adapters import ResearchAdapters

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def build_cli_use_cases(settings: PlatformSettings) -> UseCaseRegistry:
    """Build the CLI use-case registry from concrete runtime adapters."""
    registry = UseCaseRegistry()
    register_runtime_use_cases(registry, RuntimeAdapters(settings))
    register_broker_use_cases(registry, BrokerAdapters(settings))
    register_engine_use_cases(registry, EngineAdapters(settings))
    register_data_use_cases(registry, DataAdapters(settings))
    register_infra_use_cases(registry, InfraAdapters(settings))
    register_research_use_cases(registry, ResearchAdapters(settings))
    register_governance_use_cases(registry, GovernanceAdapters(settings))
    return registry


__all__ = ["build_cli_use_cases"]
