from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import PlatformSettings, V2Settings
from quant_platform.core.domain.portfolio import PortfolioTarget
from quant_platform.core.domain.production import EngineBudget, EngineTargetProposal
from quant_platform.engines.multi_engine import MultiEngineRunner
from quant_platform.infrastructure.repositories.multi_engine_governance import (
    InMemoryMultiEngineGovernanceRepository,
)


def _target(
    *,
    strategy_run_id: uuid.UUID,
    weights: dict[uuid.UUID, Decimal],
) -> PortfolioTarget:
    return PortfolioTarget(
        target_id=uuid.uuid4(),
        strategy_run_id=strategy_run_id,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        regime_id=uuid.uuid4(),
        weights=weights,
        cash_target_weight=Decimal("1") - sum(weights.values(), Decimal("0")),
    )


@pytest.mark.asyncio
async def test_multi_engine_merge_scales_and_nets_overlapping_targets() -> None:
    shared = uuid.uuid4()
    xsec_only = uuid.uuid4()
    etf_only = uuid.uuid4()
    repo = InMemoryMultiEngineGovernanceRepository()
    runner = MultiEngineRunner(
        PlatformSettings(_env_file=None),
        governance_repo=repo,
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("0.20"),
            ),
            EngineBudget(
                "etf_macro_allocator_v1",
                "0.1.0",
                "paper",
                Decimal("0.25"),
                Decimal("0.25"),
                Decimal("0.10"),
            ),
        ),
    )

    # Use high turnover budgets so this test focuses on weight-scaling / merging
    # logic rather than turnover enforcement (which is tested separately).
    repo2 = InMemoryMultiEngineGovernanceRepository()
    runner2 = MultiEngineRunner(
        PlatformSettings(_env_file=None),
        governance_repo=repo2,
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("1.00"),  # unconstrained turnover
            ),
            EngineBudget(
                "etf_macro_allocator_v1",
                "0.1.0",
                "paper",
                Decimal("0.25"),
                Decimal("0.25"),
                Decimal("1.00"),  # unconstrained turnover
            ),
        ),
    )
    merged = await runner2.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={shared: Decimal("0.20"), xsec_only: Decimal("0.10")},
            ),
            "etf_macro_allocator_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={shared: Decimal("0.40"), etf_only: Decimal("0.20")},
            ),
        },
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert merged.weights[shared] == Decimal("0.2400")
    assert merged.weights[xsec_only] == Decimal("0.0700")
    assert merged.weights[etf_only] == Decimal("0.0500")
    assert merged.cash_target_weight == Decimal("0.6400")
    assert len(merged.contributions) == 2
    assert await repo2.list_engine_budgets()
    assert await repo2.list_target_contributions(merged.target_id) == list(merged.contributions)


def test_multi_engine_live_guard_blocks_multiple_live_engines() -> None:
    with pytest.raises(RuntimeError, match="Concurrent multi-engine live"):
        MultiEngineRunner(
            PlatformSettings(_env_file=None),
            budgets=(
                EngineBudget(
                    "cross_sectional_equity_v1",
                    "0.1.0",
                    "live",
                    Decimal("0.70"),
                    Decimal("0.70"),
                    Decimal("0.20"),
                ),
                EngineBudget(
                    "etf_macro_allocator_v1",
                    "0.1.0",
                    "live",
                    Decimal("0.25"),
                    Decimal("0.25"),
                    Decimal("0.10"),
                ),
            ),
        )


@pytest.mark.asyncio
async def test_v2_account_orchestrator_accepts_live_engine_proposals() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    feature_dataset_id = uuid.uuid4()
    model_artifact_id = uuid.uuid4()
    repo = InMemoryMultiEngineGovernanceRepository()
    # Use max_turnover=1.0 so this test focuses on the V2 proposal validation
    # and weight-scaling logic rather than turnover enforcement.
    runner = MultiEngineRunner(
        PlatformSettings(
            _env_file=None,
            v2=V2Settings(
                enabled=True,
                account_orchestrator_enabled=True,
                require_feature_datasets=True,
            ),
        ),
        governance_repo=repo,
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "live",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("1.00"),  # unconstrained for this test
            ),
            EngineBudget(
                "etf_macro_allocator_v1",
                "0.1.0",
                "live",
                Decimal("0.25"),
                Decimal("0.25"),
                Decimal("1.00"),  # unconstrained for this test
            ),
        ),
    )

    merged = await runner.merge_proposals(
        (
            EngineTargetProposal(
                proposal_id=uuid.uuid4(),
                engine_name="cross_sectional_equity_v1",
                engine_version="0.1.0",
                run_mode="live",
                strategy_run_id=uuid.uuid4(),
                as_of=datetime(2026, 1, 2, tzinfo=UTC),
                weights={first: Decimal("0.50")},
                cash_target_weight=Decimal("0.50"),
                promotion_state="live",
                feature_dataset_id=feature_dataset_id,
                model_artifact_id=model_artifact_id,
            ),
            EngineTargetProposal(
                proposal_id=uuid.uuid4(),
                engine_name="etf_macro_allocator_v1",
                engine_version="0.1.0",
                run_mode="live",
                strategy_run_id=uuid.uuid4(),
                as_of=datetime(2026, 1, 2, tzinfo=UTC),
                weights={second: Decimal("0.40")},
                cash_target_weight=Decimal("0.60"),
                promotion_state="live",
                feature_dataset_id=feature_dataset_id,
                model_artifact_id=model_artifact_id,
            ),
        ),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert merged.weights[first] == Decimal("0.3500")
    assert merged.weights[second] == Decimal("0.1000")
    assert merged.cash_target_weight == Decimal("0.5500")
    assert len(merged.contributions) == 2


@pytest.mark.asyncio
async def test_v2_live_proposal_requires_feature_dataset_when_gate_enabled() -> None:
    runner = MultiEngineRunner(
        PlatformSettings(
            _env_file=None,
            v2=V2Settings(
                enabled=True,
                account_orchestrator_enabled=True,
                require_feature_datasets=True,
            ),
        ),
        governance_repo=InMemoryMultiEngineGovernanceRepository(),
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "live",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("0.20"),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="missing feature_dataset_id"):
        await runner.merge_proposals(
            (
                EngineTargetProposal(
                    proposal_id=uuid.uuid4(),
                    engine_name="cross_sectional_equity_v1",
                    engine_version="0.1.0",
                    run_mode="live",
                    strategy_run_id=uuid.uuid4(),
                    as_of=datetime(2026, 1, 2, tzinfo=UTC),
                    weights={uuid.uuid4(): Decimal("0.50")},
                    cash_target_weight=Decimal("0.50"),
                    promotion_state="live",
                    model_artifact_id=uuid.uuid4(),
                ),
            )
        )


@pytest.mark.asyncio
async def test_multi_engine_turnover_enforcement_caps_allocation() -> None:
    """max_turnover is enforced: an engine exceeding its budget is scaled down."""
    instr = uuid.uuid4()
    runner = MultiEngineRunner(
        PlatformSettings(_env_file=None),
        governance_repo=InMemoryMultiEngineGovernanceRepository(),
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("0.05"),  # tight turnover budget
            ),
        ),
    )

    # First call from 0 → 0.70 gross exceeds max_turnover=0.05.
    merged = await runner.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={instr: Decimal("1.00")},  # wants 100% → scaled to 0.70 gross
            ),
        },
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    # Gross cap: 1.0 * 0.70 = 0.70; turnover from 0 = 0.70 > max_turnover=0.05
    # Scale factor: 0.05 / 0.70 ≈ 0.0714
    # Expected weight: 0.70 * (0.05 / 0.70) = 0.05
    weight = merged.weights.get(instr, Decimal("0"))
    assert weight <= Decimal("0.05"), f"expected ≤ 0.05 after turnover cap, got {weight}"
    # Construction notes should mention the cap.
    assert any("turnover" in note.lower() for note in merged.construction_notes)


@pytest.mark.asyncio
async def test_multi_engine_turnover_caps_position_unwind() -> None:
    """Shrinking a held position must observe max_turnover.

    Regression: the prior cap math scaled only the *current* weights, so a
    held position that disappeared from the new target vanished entirely
    even when max_turnover required only a partial unwind. Post-fix the cap
    must scale the *delta*, leaving previous positions winding down at most
    by max_turnover.
    """
    instr = uuid.uuid4()
    runner = MultiEngineRunner(
        PlatformSettings(_env_file=None),
        governance_repo=InMemoryMultiEngineGovernanceRepository(),
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("1.00"),  # unconstrained for the build-up call
            ),
        ),
    )

    # Cycle 1: ramp from 0 to a 0.50 gross position (turnover budget unconstrained).
    await runner.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={instr: Decimal("0.50") / Decimal("0.70")},  # 0.50 / 0.70 → 0.50 gross
            ),
        },
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    # Tighten the budget to max_turnover=0.10 and ask for an empty target
    # (intent to liquidate the held 0.50). With the bug, capped is empty
    # → actual realised turnover = 0.50, violating 0.10. With the fix,
    # capped[instr] should still hold ≈ 0.40 (0.50 - delta * scale, where
    # delta=-0.50, scale=0.10/0.50=0.2, so capped=0.50 + (-0.50)*0.2=0.40).
    runner._budgets = (  # type: ignore[attr-defined]
        EngineBudget(
            "cross_sectional_equity_v1",
            "0.1.0",
            "paper",
            Decimal("0.70"),
            Decimal("0.70"),
            Decimal("0.10"),
        ),
    )
    merged = await runner.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={},  # liquidate
            ),
        },
        as_of=datetime(2026, 1, 3, tzinfo=UTC),
    )

    weight = merged.weights.get(instr, Decimal("0"))
    # Allow a small numerical tolerance for Decimal scaling.
    assert Decimal("0.39") <= weight <= Decimal("0.41"), (
        f"unwind was not capped to max_turnover: weight={weight}, expected ≈0.40"
    )
    assert any("turnover" in note.lower() for note in merged.construction_notes)


@pytest.mark.asyncio
async def test_multi_engine_zero_turnover_locks_position() -> None:
    """max_turnover=0 must keep previous positions and prevent any movement."""
    instr = uuid.uuid4()
    runner = MultiEngineRunner(
        PlatformSettings(_env_file=None),
        governance_repo=InMemoryMultiEngineGovernanceRepository(),
        budgets=(
            EngineBudget(
                "cross_sectional_equity_v1",
                "0.1.0",
                "paper",
                Decimal("0.70"),
                Decimal("0.70"),
                Decimal("1.00"),
            ),
        ),
    )

    await runner.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={instr: Decimal("0.30") / Decimal("0.70")},  # 0.30 gross
            ),
        },
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )

    # Switch to a max_turnover=0 budget and ask for movement; expect lock.
    runner._budgets = (  # type: ignore[attr-defined]
        EngineBudget(
            "cross_sectional_equity_v1",
            "0.1.0",
            "paper",
            Decimal("0.70"),
            Decimal("0.70"),
            Decimal("0"),
        ),
    )
    merged = await runner.merge_targets(
        {
            "cross_sectional_equity_v1": _target(
                strategy_run_id=uuid.uuid4(),
                weights={instr: Decimal("0.50")},  # request a move
            ),
        },
        as_of=datetime(2026, 1, 3, tzinfo=UTC),
    )

    weight = merged.weights.get(instr, Decimal("0"))
    assert Decimal("0.29") <= weight <= Decimal("0.31"), (
        f"max_turnover=0 should hold the prior position (~0.30); got {weight}"
    )
