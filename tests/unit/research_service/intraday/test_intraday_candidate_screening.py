from __future__ import annotations

import csv
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.research.campaign.intraday_screen import (
    _filter_samples_by_window,
    _sample_filter_payload,
)
from quant_platform.services.data_service.intraday.intraday_file_loader import (
    load_vendor_bar_batch_from_file,
)
from quant_platform.services.research_service.intraday.candidates.screening import (
    INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION,
    INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION,
    IntradayCandidateScreenThresholds,
    build_intraday_candidate_screen,
    intraday_candidates_for_set,
    write_intraday_candidate_family_artifacts,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from pathlib import Path


def test_intraday_screen_loads_local_file_and_writes_family_after_pass(tmp_path: Path) -> None:
    instruments = tuple(uuid.uuid4() for _ in range(6))
    samples = _samples(instruments)
    contracts = _contracts(instruments)
    intraday_path = tmp_path / "intraday.csv"
    _write_intraday_csv(intraday_path, samples=samples, contracts=contracts)
    batch = load_vendor_bar_batch_from_file(
        intraday_path,
        vendor="polygon",
        instrument_lookup=_lookup(contracts),
        as_of=max(sample.as_of for sample in samples),
    )

    screen = build_intraday_candidate_screen(
        samples=samples,
        intraday_bars=batch.bars,
        sample_build=_sample_build(len(samples)),
        thresholds=IntradayCandidateScreenThresholds(min_passing_candidates=3),
        permutation_count=30,
    )

    assert screen["passed"] is True
    assert len(screen["passing_candidates"]) >= 3
    assert screen["diagnostic_only"] is True
    assert screen["promotion_artifacts_written"] is False

    written = write_intraday_candidate_family_artifacts(
        screen=screen,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path
        / "feature_families"
        / "paper-alpha-intraday-microstructure-v1.json",
    )

    assert written["written"] is True
    assert (tmp_path / "feature_families" / "paper-alpha-intraday-microstructure-v1.json").exists()
    assert len(list((tmp_path / "feature_cards").glob("*.json"))) >= 3


def test_intraday_screen_excludes_future_bars_from_candidate_features(tmp_path: Path) -> None:
    instruments = tuple(uuid.uuid4() for _ in range(6))
    samples = _samples(instruments, days=1)
    contracts = _contracts(instruments)
    intraday_path = tmp_path / "future_intraday.csv"
    _write_intraday_csv(intraday_path, samples=samples, contracts=contracts, future_only=True)
    batch = load_vendor_bar_batch_from_file(
        intraday_path,
        vendor="polygon",
        instrument_lookup=_lookup(contracts),
        as_of=max(sample.as_of for sample in samples),
    )

    screen = build_intraday_candidate_screen(
        samples=samples,
        intraday_bars=batch.bars,
        sample_build=_sample_build(len(samples)),
        permutation_count=20,
    )

    assert screen["passed"] is False
    assert "intraday candidate screen admitted 0 features" in " ".join(screen["blockers"])
    for candidate in screen["candidates"]:
        assert candidate["metrics"]["source_density"] == 0.0


def test_intraday_screen_blocks_sparse_inputs_without_family_artifacts(tmp_path: Path) -> None:
    instruments = tuple(uuid.uuid4() for _ in range(6))
    samples = _samples(instruments, days=4)
    screen = build_intraday_candidate_screen(
        samples=samples,
        intraday_bars=(),
        sample_build=_sample_build(len(samples)),
        permutation_count=10,
    )

    assert screen["passed"] is False
    assert "intraday bar coverage 0" in screen["blockers"]

    written = write_intraday_candidate_family_artifacts(
        screen=screen,
        feature_card_dir=tmp_path / "feature_cards",
        feature_family_file=tmp_path
        / "feature_families"
        / "paper-alpha-intraday-microstructure-v1.json",
    )

    assert written == {"written": False, "reason": "screen did not pass"}
    assert not (tmp_path / "feature_cards").exists()


def test_intraday_candidate_sets_are_deterministic() -> None:
    candidates = intraday_candidates_for_set("seed")

    assert len(candidates) == 6
    assert [candidate.name for candidate in candidates] == [
        "opening_drive_confirmation_1d_decay",
        "opening_drive_reversal_1d_decay",
        "close_pressure_continuation_1d_decay",
        "vwap_accumulation_pressure_1d_decay",
        "intraday_volatility_compression_3d_decay",
        "range_expansion_drift_1d_decay",
    ]
    assert INTRADAY_MICROSTRUCTURE_FEATURE_SET_VERSION == ("paper-alpha-intraday-microstructure-v1")

    v2 = intraday_candidates_for_set("microstructure-v2")
    assert [candidate.name for candidate in v2] == [
        "intraday_v2_signed_vwap_pressure_1d",
        "intraday_v2_signed_close_pressure_3d",
        "intraday_v2_range_volume_share_5d",
        "intraday_v2_open_close_pressure_spread_3d",
        "intraday_v2_volatility_volume_compression_5d",
        "intraday_v2_signed_volume_share_momentum_12m_1d",
        "intraday_v2_opening_drive_reversal_intensity_1d",
        "intraday_v2_volume_share_close_volatility_blend_1d",
        "intraday_v2_volume_share_opening_reversal_blend_1d",
        "intraday_v2_volume_share_range_volatility_blend_1d",
        "intraday_v2_signed_range_expansion_band_2_3_close_pressure_21d",
        "intraday_v2_range_expansion_band_1_5_opening_drive_21d",
        "intraday_v2_signed_opening_drive_band_1_15_opening_drive_21d",
        "intraday_v2_range_expansion_band_1_10_range_21d",
        "intraday_v2_vwap_pressure_band_1_21_close_pressure_21d",
        "intraday_v2_range_opening_band_composite_21d",
        "intraday_v2_range_vwap_band_composite_21d",
        "intraday_v2_short_range_vwap_opening_composite_21d",
        "intraday_v2_range_volume_band_composite_21d",
    ]
    assert INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION == (
        "paper-alpha-intraday-microstructure-v2"
    )


def test_intraday_sample_window_is_inclusive_and_reported() -> None:
    samples = _samples((uuid.uuid4(),), days=4)

    filtered = _filter_samples_by_window(
        samples,
        sample_start=datetime(2026, 1, 4),
        sample_end=datetime(2026, 1, 5, tzinfo=UTC),
    )
    payload = _sample_filter_payload(
        samples,
        filtered,
        type(
            "Args",
            (),
            {
                "sample_start": datetime(2026, 1, 4),
                "sample_end": datetime(2026, 1, 5, tzinfo=UTC),
            },
        )(),
    )

    assert [sample.as_of.date().isoformat() for sample in filtered] == [
        "2026-01-04",
        "2026-01-05",
    ]
    assert payload["sample_start"] == "2026-01-04T00:00:00+00:00"
    assert payload["sample_end"] == "2026-01-05T00:00:00+00:00"
    assert payload["loaded_sample_count"] == 4
    assert payload["screened_sample_count"] == 2


def _samples(
    instruments: tuple[uuid.UUID, ...],
    *,
    days: int = 8,
) -> tuple[SupervisedAlphaSample, ...]:
    start = datetime(2026, 1, 3, tzinfo=UTC)
    samples: list[SupervisedAlphaSample] = []
    for day in range(days):
        as_of = start + timedelta(days=day)
        for rank, instrument_id in enumerate(instruments, start=1):
            samples.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features={},
                    forward_return=float(rank),
                )
            )
    return tuple(samples)


def _contracts(instruments: tuple[uuid.UUID, ...]) -> dict[uuid.UUID, dict[str, object]]:
    return {
        instrument_id: {"symbol": f"S{idx:02d}", "exchange": "SMART", "currency": "USD"}
        for idx, instrument_id in enumerate(instruments, start=1)
    }


def _lookup(contracts: dict[uuid.UUID, dict[str, object]]) -> dict[str, uuid.UUID]:
    return {str(spec["symbol"]).upper(): instrument_id for instrument_id, spec in contracts.items()}


def _write_intraday_csv(
    path: Path,
    *,
    samples: tuple[SupervisedAlphaSample, ...],
    contracts: dict[uuid.UUID, dict[str, object]],
    future_only: bool = False,
) -> None:
    symbols = {instrument_id: str(spec["symbol"]) for instrument_id, spec in contracts.items()}
    ranks = {instrument_id: idx for idx, instrument_id in enumerate(contracts, start=1)}
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"],
        )
        writer.writeheader()
        for sample in samples:
            session_start = (
                sample.as_of + timedelta(hours=14, minutes=30)
                if future_only
                else sample.as_of - timedelta(days=1) + timedelta(hours=14, minutes=30)
            )
            rank = ranks[sample.instrument_id]
            symbol = symbols[sample.instrument_id]
            rows = [
                (session_start, 100.0, 100.0 + rank * 0.20, 100.0 + rank * 0.12),
                (
                    session_start + timedelta(minutes=30),
                    100.0 + rank * 0.20,
                    100.0 + rank * 0.32,
                    100.0 + rank * 0.20,
                ),
                (
                    session_start + timedelta(minutes=389),
                    100.0 + rank * 0.32,
                    100.0 + rank * 0.62,
                    100.0 + rank * 0.25,
                ),
            ]
            for ts, open_price, close_price, vwap in rows:
                high = close_price + rank * 0.08
                low = min(open_price, close_price) - 0.40
                writer.writerow(
                    {
                        "symbol": symbol,
                        "timestamp": ts.isoformat(),
                        "open": f"{open_price:.4f}",
                        "high": f"{high:.4f}",
                        "low": f"{low:.4f}",
                        "close": f"{close_price:.4f}",
                        "volume": 1000 + rank * 10,
                        "vwap": f"{vwap:.4f}",
                    }
                )


def _sample_build(count: int) -> dict[str, object]:
    return {
        "requested_points": count,
        "samples": count,
        "skipped_missing_features": 0,
        "skipped_stale_features": 0,
        "skipped_missing_bars": 0,
        "skipped_invalid_features": 0,
    }
