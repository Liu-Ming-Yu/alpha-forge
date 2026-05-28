"""Campaign feature admission and quarantine helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quant_platform.services.research_service.boosting.artifacts import BoostingSample
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FeatureAdmissionMode = Literal["passing", "all"]


@dataclass(frozen=True)
class CampaignFeatureAdmission:
    """Resolved campaign feature set after paper audit admission."""

    mode: str
    audit_mode: str
    min_admitted_features: int
    audited_features: tuple[str, ...]
    admitted_features: tuple[str, ...]
    quarantined_features: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.blockers

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "audit_mode": self.audit_mode,
            "min_admitted_features": self.min_admitted_features,
            "passed": self.passed,
            "audited_features": list(self.audited_features),
            "admitted_features": list(self.admitted_features),
            "quarantined_features": list(self.quarantined_features),
            "blockers": list(self.blockers),
        }


def campaign_sample_feature_names(samples: Sequence[SupervisedAlphaSample]) -> tuple[str, ...]:
    """Return sorted non-reserved feature names emitted into campaign samples."""
    return tuple(
        sorted(
            {
                str(name)
                for sample in samples
                for name in sample.features
                if not str(name).startswith("_")
            }
        )
    )


def resolve_campaign_feature_admission(
    *,
    samples: Sequence[SupervisedAlphaSample],
    feature_audits: Sequence[Mapping[str, object]],
    audit_mode: str,
    feature_admission: FeatureAdmissionMode,
    min_admitted_features: int,
    candidate_feature_names: Sequence[str] | None = None,
) -> CampaignFeatureAdmission:
    """Resolve admitted and quarantined features for a research campaign."""
    if min_admitted_features <= 0:
        raise ValueError("min_admitted_features must be > 0")
    sample_features = campaign_sample_feature_names(samples)
    candidate_set = (
        {str(value) for value in candidate_feature_names}
        if candidate_feature_names is not None
        else None
    )
    audited_scope = (
        tuple(name for name in sample_features if name in candidate_set)
        if candidate_set is not None
        else sample_features
    )
    if audit_mode != "paper":
        return CampaignFeatureAdmission(
            mode=feature_admission,
            audit_mode=audit_mode,
            min_admitted_features=min_admitted_features,
            audited_features=audited_scope,
            admitted_features=audited_scope,
            quarantined_features=(),
        )

    passing = {
        name
        for row in feature_audits
        if bool(row.get("passed"))
        for name in (_feature_name(row),)
        if name
    }
    if feature_admission == "all":
        failed = tuple(name for name in audited_scope if name not in passing)
        all_mode_blockers = tuple(
            f"feature audit failed or missing for {name!r}" for name in failed
        )
        if not all_mode_blockers and len(audited_scope) < min_admitted_features:
            all_mode_blockers = (
                f"admitted feature count {len(audited_scope)} < {min_admitted_features}",
            )
        admitted = audited_scope if not all_mode_blockers else ()
        return CampaignFeatureAdmission(
            mode=feature_admission,
            audit_mode=audit_mode,
            min_admitted_features=min_admitted_features,
            audited_features=audited_scope,
            admitted_features=admitted,
            quarantined_features=failed,
            blockers=all_mode_blockers,
        )

    admitted = tuple(name for name in audited_scope if name in passing)
    quarantined = tuple(name for name in audited_scope if name not in passing)
    passing_mode_blockers: tuple[str, ...] = ()
    if len(admitted) < min_admitted_features:
        passing_mode_blockers = (
            f"admitted feature count {len(admitted)} < {min_admitted_features}",
        )
    return CampaignFeatureAdmission(
        mode=feature_admission,
        audit_mode=audit_mode,
        min_admitted_features=min_admitted_features,
        audited_features=audited_scope,
        admitted_features=admitted,
        quarantined_features=quarantined,
        blockers=passing_mode_blockers,
    )


def annotate_feature_audits(
    feature_audits: Sequence[Mapping[str, object]],
    admission: CampaignFeatureAdmission,
) -> list[dict[str, object]]:
    """Attach admitted/quarantined status to feature-audit payload rows."""
    admitted = set(admission.admitted_features)
    quarantined = set(admission.quarantined_features)
    annotated: list[dict[str, object]] = []
    for row in feature_audits:
        payload = dict(row)
        name = _feature_name(payload)
        if name in admitted:
            payload["admission"] = "admitted"
        elif name in quarantined:
            payload["admission"] = "quarantined"
        elif name:
            payload["admission"] = "ignored"
        annotated.append(payload)
    return annotated


def filter_supervised_samples_to_features(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str],
) -> list[SupervisedAlphaSample]:
    """Return samples containing only the admitted model-input features."""
    ordered_names = tuple(str(name) for name in feature_names)
    if not ordered_names:
        raise ValueError("feature filtering requires at least one feature")
    filtered: list[SupervisedAlphaSample] = []
    for sample in samples:
        features = {
            name: sample.features[name] for name in ordered_names if name in sample.features
        }
        filtered.append(
            SupervisedAlphaSample(
                as_of=sample.as_of,
                instrument_id=sample.instrument_id,
                features=features,
                forward_return=sample.forward_return,
                metadata=sample.metadata,
            )
        )
    return filtered


def supervised_to_boosting_samples(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str],
) -> list[BoostingSample]:
    """Convert admitted supervised rows into XGBoost ranker samples."""
    return [
        BoostingSample(
            as_of=sample.as_of,
            instrument_id=sample.instrument_id,
            features=dict(sample.features),
            forward_return=sample.forward_return,
        )
        for sample in filter_supervised_samples_to_features(samples, feature_names)
    ]


def _feature_name(row: Mapping[str, object]) -> str:
    raw = row.get("feature_name")
    return str(raw) if raw is not None else ""


__all__ = [
    "CampaignFeatureAdmission",
    "FeatureAdmissionMode",
    "annotate_feature_audits",
    "campaign_sample_feature_names",
    "filter_supervised_samples_to_features",
    "resolve_campaign_feature_admission",
    "supervised_to_boosting_samples",
]
