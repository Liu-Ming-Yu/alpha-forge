"""Tests for the catalyst paper-campaign source-density guard."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from quant_platform.research.campaign.source_density import (
    maybe_block_for_catalyst_source_density,
)
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_V10_ALPHA_FEATURES,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

AS_OF = datetime(2026, 1, 2, tzinfo=UTC)


def _samples(*, dense: bool) -> tuple[SupervisedAlphaSample, ...]:
    rows: list[SupervisedAlphaSample] = []
    for index in range(40):
        value = 1.0 if dense else 0.0
        rows.append(
            SupervisedAlphaSample(
                as_of=AS_OF,
                instrument_id=uuid.uuid4(),
                features={feature: value for feature in TEXT_CATALYST_V10_ALPHA_FEATURES},
                forward_return=0.01,
            )
        )
    return tuple(rows)


def _manifest(tmp_path, *, events: int) -> object:
    symbols = [f"S{i:02d}" for i in range(15)]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "download": {"requested_symbols": symbols},
                "primary_events_by_symbol": {symbol: events for symbol in symbols},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_source_density_guard_ignores_non_catalyst_feature_sets(tmp_path) -> None:
    result = maybe_block_for_catalyst_source_density(
        feature_set_version="1.0.0",
        source_data_manifest=None,
        samples=(),
        output_root=tmp_path,
        sample_slug="run-1",
    )

    assert result is None


def test_source_density_guard_blocks_without_manifest(tmp_path) -> None:
    result = maybe_block_for_catalyst_source_density(
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        source_data_manifest=None,
        samples=_samples(dense=True),
        output_root=tmp_path,
        sample_slug="run-1",
    )

    assert result is not None
    assert result["passed"] is False
    assert any("requires --source-data-manifest" in blocker for blocker in result["blockers"])
    blocked_summary = result["blocked_source_density_summary"]
    assert isinstance(blocked_summary, str)
    assert (tmp_path / "_blocked" / "run-1" / "blocked_source_density_summary.json").exists()


def test_source_density_guard_blocks_sparse_catalyst_features(tmp_path) -> None:
    result = maybe_block_for_catalyst_source_density(
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        source_data_manifest=_manifest(tmp_path, events=3),
        samples=_samples(dense=False),
        output_root=tmp_path,
        sample_slug="run-2",
    )

    assert result is not None
    assert result["passed"] is False
    assert any("nonzero_fraction" in blocker for blocker in result["blockers"])


def test_source_density_guard_passes_with_dense_sources_and_samples(tmp_path) -> None:
    result = maybe_block_for_catalyst_source_density(
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        source_data_manifest=_manifest(tmp_path, events=3),
        samples=_samples(dense=True),
        output_root=tmp_path,
        sample_slug="run-3",
    )

    assert result is None
