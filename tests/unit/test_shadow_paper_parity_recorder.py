from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.services.governance_service.shadow_paper_parity import (
    ShadowPaperParityRecorder,
)


@pytest.mark.asyncio
async def test_shadow_paper_parity_recorder_persists_clean_twenty_day_evidence() -> None:
    repo = InMemoryPerformanceRepository()
    recorder = ShadowPaperParityRecorder(repository=repo)
    instrument_id = uuid.uuid4()
    as_of = datetime(2026, 5, 16, tzinfo=UTC)

    for offset in range(20):
        await recorder.record(
            as_of=as_of + timedelta(days=offset),
            shadow_targets={instrument_id: Decimal("0.10000")},
            paper_targets={instrument_id: Decimal("0.10005")},
            shadow_order_plan=[{"instrument_id": instrument_id, "side": "buy"}],
            paper_order_plan=[{"instrument_id": instrument_id, "side": "buy"}],
            instrument_universe=[instrument_id],
            shadow_run_id="shadow-run",
            paper_run_id="paper-run",
            git_commit="abcdef",
            config_hash="config-sha",
            text_model_manifest_sha256="manifest-sha",
            feature_schema_hash="schema-sha",
            source_weights={"classical": 0.99, "text": 0.01},
        )

    status = await repo.shadow_paper_parity_status(
        "text",
        "text",
        as_of=as_of + timedelta(days=19),
        min_trading_days=20,
        max_target_weight_diff_bps=1.0,
    )

    assert status.passed
    assert status.trading_days == 20
    assert status.max_target_weight_diff_bps == pytest.approx(0.5)


def test_shadow_paper_parity_recorder_detects_target_and_order_drift() -> None:
    instrument_id = uuid.uuid4()
    record = ShadowPaperParityRecorder().build_record(
        as_of=datetime(2026, 5, 16, tzinfo=UTC),
        shadow_targets={instrument_id: Decimal("0.10")},
        paper_targets={instrument_id: Decimal("0.0995")},
        shadow_order_plan=[{"instrument_id": instrument_id, "side": "buy"}],
        paper_order_plan=[{"instrument_id": instrument_id, "side": "sell"}],
        instrument_universe=[instrument_id],
    )

    assert record.instruments_compared == 1
    assert record.missing_instruments == 0
    assert record.max_target_weight_diff_bps == pytest.approx(5.0)
    assert record.order_side_mismatches == 1


def test_shadow_paper_parity_recorder_requires_target_artifacts() -> None:
    instrument_id = uuid.uuid4()
    record = ShadowPaperParityRecorder().build_record(
        as_of=datetime(2026, 5, 16, tzinfo=UTC),
        shadow_targets=None,
        paper_targets={instrument_id: Decimal("0")},
        shadow_order_plan=[],
        paper_order_plan=[],
        instrument_universe=[instrument_id],
    )

    assert record.missing_instruments == 1
    assert record.metadata["source_weights"] == {}
