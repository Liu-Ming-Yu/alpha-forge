"""Operator CLI: promote a latest-stack arm to the model registry — one call.

ADR-004 Action Item 14. Given a per-arm evidence JSON from the latest-stack
backtest (e.g. ``…/backtest_latest_stack_realized_v2/arm_*.json``), this checks
the arm is promotable (eligible production-candidate) and either:

* **dry-run (default)** — prints the exact ``register_model`` payload
  (strategy_name, model_version, feature_set_version, as_of, metadata) for
  review, or
* **``--register``** — performs the live, DSN-backed promotion in one call:
  the adapter derives the model identity + evidence metadata, then governance
  ``alpha_promote`` registers it (and writes the audit heartbeat). The
  eligibility gate is re-enforced — an ineligible arm never reaches the
  registry.

The adapter (research) and ``alpha_promote`` (governance) are different
services; this script is the composition root that wires them, so neither
service depends on the other. A non-promotable arm exits non-zero with the
reason.

Usage::

    # review the payload
    python scripts/promote_latest_stack_arm.py --evidence <run-dir>/arm_*.json

    # live promotion (needs QP__STORAGE__POSTGRES_DSN)
    python scripts/promote_latest_stack_arm.py --evidence <run-dir>/arm_*.json \\
        --register --signal-type xgboost --engine-version engine-v1 \\
        --rollback-target <prev-model-version>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.config_signal_models.alpha import ALPHA_SOURCE_TYPES
from quant_platform.services.research_service.modeling.registry.latest_stack_promotion import (
    NotPromotableError,
    build_registration,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import (
        ModelRegistryRepository,
        OperationalReadinessRepository,
    )
    from quant_platform.services.research_service.modeling.registry.latest_stack_promotion import (
        ModelRegistration,
    )


def registration_to_dict(registration: ModelRegistration) -> dict[str, object]:
    """JSON-serialisable view of the register_model payload."""
    return {
        "strategy_name": registration.strategy_name,
        "model_version": registration.model_version,
        "feature_set_version": registration.feature_set_version,
        "as_of": registration.as_of.isoformat(),
        "metadata": registration.metadata,
    }


async def register_from_evidence(
    settings: PlatformSettings,
    evidence: Mapping[str, object],
    *,
    signal_type: str,
    engine_version: str,
    rollback_target: str,
    artifact_manifest: Path | None,
    as_of: datetime | None,
    model_registry: ModelRegistryRepository | None,
    heartbeat_repository: OperationalReadinessRepository | None,
) -> dict[str, object]:
    """Compose the adapter + governance ``alpha_promote`` into one live call.

    Builds the registration from ``evidence`` (gating on eligibility), then
    registers via ``alpha_promote`` with the adapter's metadata attached.
    Raises :class:`NotPromotableError` (before any registry write) if the arm
    is ineligible. ``model_registry``/``heartbeat_repository`` are forwarded to
    ``alpha_promote`` (inject fakes in tests; pass real repos in production).
    """
    # Governance is a sibling service; import lazily so the dry-run path and the
    # importlib test don't pull it in.
    from quant_platform.services.governance_service.alpha import alpha_promote

    registration = build_registration(evidence, as_of=as_of)
    return await alpha_promote(
        settings,
        signal_name=registration.strategy_name,
        signal_type=signal_type,
        model_version=registration.model_version,
        feature_set_version=registration.feature_set_version,
        engine_version=engine_version,
        artifact_manifest=artifact_manifest,
        rollback_target=rollback_target,
        as_of=registration.as_of,
        evidence_metadata=registration.metadata,
        model_registry=model_registry,
        heartbeat_repository=heartbeat_repository,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Path to a latest-stack per-arm evidence JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="(dry-run) Write the registration payload here. Defaults to stdout.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Perform the live promotion via alpha_promote (needs QP__STORAGE__POSTGRES_DSN).",
    )
    parser.add_argument(
        "--signal-type",
        choices=list(ALPHA_SOURCE_TYPES),
        default=None,
        help="(--register) Alpha-source taxonomy. Use 'classical' for the "
        "latest-stack linear IC-weighted rankers (e.g. Arm G); 'xgboost' for GBDT arms.",
    )
    parser.add_argument(
        "--engine-version",
        default=None,
        help="(--register) Engine version pinned onto the registry record.",
    )
    parser.add_argument(
        "--rollback-target",
        default="",
        help="(--register) Model version to roll back to if this promotion is reverted.",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=None,
        help="(--register) Optional path to a trained-model artifact manifest.",
    )
    return parser


def _run_dry_run(args: argparse.Namespace) -> int:
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    try:
        registration = build_registration(evidence)
    except NotPromotableError as exc:
        print(f"NOT PROMOTABLE: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(registration_to_dict(registration), indent=2, default=str)
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Registration payload written to {args.output}")
    else:
        print(payload)
    return 0


def _run_live_registration(args: argparse.Namespace) -> int:
    import asyncio

    from quant_platform.bootstrap.governance.repositories import (
        build_model_registry,
        build_performance_repository,
    )
    from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema
    from quant_platform.config import PlatformSettings

    # psycopg3 async requires SelectorEventLoop; Windows defaults to ProactorEventLoop.
    # Mirrors quant_platform.cli.registry._configure_windows_event_loop_policy.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if not args.signal_type or not args.engine_version:
        print("--register requires --signal-type and --engine-version", file=sys.stderr)
        return 2
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    # Fail fast on ineligibility before opening a DB connection.
    try:
        build_registration(evidence)
    except NotPromotableError as exc:
        print(f"NOT PROMOTABLE: {exc}", file=sys.stderr)
        return 2

    settings = PlatformSettings()
    dsn = settings.storage.postgres_dsn
    if not dsn:
        print("--register requires QP__STORAGE__POSTGRES_DSN", file=sys.stderr)
        return 2

    async def _go() -> dict[str, object]:
        await verify_postgres_schema(settings)
        return await register_from_evidence(
            settings,
            evidence,
            signal_type=args.signal_type,
            engine_version=args.engine_version,
            rollback_target=args.rollback_target,
            artifact_manifest=args.artifact_manifest,
            as_of=None,
            model_registry=build_model_registry(dsn),
            heartbeat_repository=build_performance_repository(dsn),
        )

    result = asyncio.run(_go())
    print(json.dumps(result, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.register:
        return _run_live_registration(args)
    return _run_dry_run(args)


if __name__ == "__main__":
    sys.exit(main())
