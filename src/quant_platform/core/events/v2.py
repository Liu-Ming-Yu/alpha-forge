"""V2 orchestration and operator-governance domain events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from quant_platform.core.events.base import DomainEvent


@dataclass(frozen=True)
class EngineProposalGenerated(DomainEvent):
    """An alpha engine has produced a target proposal for the account orchestrator.

    Emitted by EngineRunner.generate_proposal() / run_cycle() in V2 mode.
    The proposal is not yet merged, optimised, or submitted.

    Args:
        proposal_id: FK to EngineTargetProposal.
        engine_name: Emitting engine identifier.
        as_of: UTC timestamp of the proposal.
        weight_count: Number of instruments in the proposed target.
    """

    proposal_id: uuid.UUID
    engine_name: str
    as_of: datetime
    weight_count: int


@dataclass(frozen=True)
class ProposalsMerged(DomainEvent):
    """Multiple engine proposals have been merged into a CombinedPortfolioTarget.

    Emitted by MultiEngineRunner.merge_proposals() after budget-scaling.

    Args:
        target_id: FK to CombinedPortfolioTarget.
        engine_names: Engines whose proposals were merged.
        as_of: UTC timestamp of the combined target.
        instrument_count: Number of instruments in the combined target.
    """

    target_id: uuid.UUID
    engine_names: tuple[str, ...]
    as_of: datetime
    instrument_count: int


@dataclass(frozen=True)
class OrderRouted(DomainEvent):
    """An approved order intent has been assigned a VenueRoute.

    Emitted by AccountExecutionOrchestrator after DefaultExecutionRouter.route().

    Args:
        order_id: FK to OrderIntent.
        venue: Destination venue identifier (e.g. "IBKR_SMART").
        tactic: ExecutionTactic value string.
        urgency: Urgency score as a Decimal string (e.g. "0.50").
    """

    order_id: uuid.UUID
    venue: str
    tactic: str
    urgency: str


@dataclass(frozen=True)
class OperatorActionRecorded(DomainEvent):
    """An operator has recorded a governance action in the audit trail.

    Args:
        action_id: FK to OperatorAction.
        action_type: Controlled vocabulary (e.g. "live_multi_engine_start").
        actor: Identifier of the operator (user name or service account).
    """

    action_id: uuid.UUID
    action_type: str
    actor: str


@dataclass(frozen=True)
class AlertFired(DomainEvent):
    """An operational alert has been raised.

    Args:
        alert_id: FK to AlertEvent.
        severity: "critical" | "error" | "warning" | "info".
        component: Source component (e.g. "account_orchestrator").
        message: Human-readable alert description.
    """

    alert_id: uuid.UUID
    severity: str
    component: str
    message: str


@dataclass(frozen=True)
class AlertResolved(DomainEvent):
    """A previously fired alert has been resolved.

    Args:
        alert_id: FK to AlertEvent.
        resolved_at: UTC timestamp of resolution.
    """

    alert_id: uuid.UUID
    resolved_at: datetime


@dataclass(frozen=True)
class ApiKeyRevoked(DomainEvent):
    """An operator API key has been revoked.

    Args:
        key_id: FK to OperatorApiKey.
    """

    key_id: uuid.UUID
