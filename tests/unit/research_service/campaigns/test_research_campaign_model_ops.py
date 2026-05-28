from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_platform.research.campaign.model_ops import (
    _feature_versions_from_audits,
    _resolve_model_features,
)


def test_resolve_model_features_uses_all_admitted_features_by_default() -> None:
    args = SimpleNamespace(model_feature=None, min_admitted_features=3)
    admission = SimpleNamespace(admitted_features=("a", "b", "c"))

    assert _resolve_model_features(args, admission) == ("a", "b", "c")


def test_resolve_model_features_accepts_explicit_audited_subset() -> None:
    args = SimpleNamespace(model_feature=["b"], min_admitted_features=3)
    admission = SimpleNamespace(admitted_features=("a", "b", "c", "d"))

    assert _resolve_model_features(args, admission) == ("b",)


def test_resolve_model_features_rejects_missing_or_too_few_admitted_features() -> None:
    admission = SimpleNamespace(admitted_features=("a", "b", "c"))

    with pytest.raises(ValueError, match="must be admitted"):
        _resolve_model_features(
            SimpleNamespace(model_feature=["a", "missing", "b"], min_admitted_features=3),
            admission,
        )

    with pytest.raises(ValueError, match="admitted feature count must meet"):
        _resolve_model_features(
            SimpleNamespace(model_feature=["a"], min_admitted_features=3),
            SimpleNamespace(admitted_features=("a", "b")),
        )


def test_feature_versions_from_audits_uses_source_versions_for_model_features() -> None:
    rows = [
        {
            "feature_name": "text_alpha",
            "feature_set_version": "paper-alpha-composite-v1",
            "feature_version": "paper-alpha-catalyst-v10",
        },
        {
            "feature_name": "event_alpha",
            "feature_set_version": "paper-alpha-composite-v1",
            "feature_version": "paper-alpha-event-reaction-v2",
        },
        {
            "feature_name": "unused",
            "feature_set_version": "paper-alpha-composite-v1",
            "feature_version": "unused-v1",
        },
    ]

    assert _feature_versions_from_audits(rows, ("text_alpha", "event_alpha")) == {
        "text_alpha": "paper-alpha-catalyst-v10",
        "event_alpha": "paper-alpha-event-reaction-v2",
    }
