"""Executable ``FeatureComputer`` / ``FeatureFamilyPlugin`` implementations.

These adapt the platform's feature-bundle builders to the typed
:class:`~quant_platform.application.features.registry.FeatureFamilyRegistry`
extension points. A computer is *pure*: it reads its inputs from
``FeatureRequest.input_context.payloads`` and returns a ``FeatureResult``;
persistence stays with the caller (the data-maintenance scheduler or the
durable feature backfill).

Builders that need to read prior feature vectors (text / catalyst / composite)
receive the repository through a closure captured at registry-build time, so the
``FeatureComputer.compute`` contract itself never grows a repository argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.core.domain.signals.feature_inputs import (
    BARS_EOD_INPUT,
    CLOSE_SERIES_INPUT,
    EVENTS_BY_INSTRUMENT_INPUT,
)
from quant_platform.services.research_service.features.cross_section.cross_section import (
    build_feature_bundle,
)
from quant_platform.services.research_service.features.cross_section.cross_section_factors import (
    STANDARD_FACTOR_SPECS,
)
from quant_platform.services.research_service.features.paper_alpha.composite import (
    PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
    build_paper_alpha_composite_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.event import (
    EVENT_REACTION_V2_ALPHA_FEATURES,
    PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
    build_paper_alpha_event_reaction_v2_feature_bundle,
)
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_V10_ALPHA_FEATURES,
    build_paper_alpha_catalyst_v10_feature_bundle,
)
from quant_platform.services.research_service.features.pipeline.feature_pipeline import (
    FEATURE_SET_VERSION,
)
from quant_platform.services.research_service.features.pipeline.storage import (
    feature_result_from_bundle,
)
from quant_platform.services.research_service.features.pv_formulaic.compute import (
    PV_FORMULAIC_FEATURE_NAMES,
)
from quant_platform.services.research_service.features.pv_formulaic.family import (
    PV_FORMULAIC_FEATURE_SET_VERSION,
    build_pv_formulaic_feature_bundle,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from datetime import datetime

    from quant_platform.core.contracts import (
        FeatureComputer,
        FeatureFamilyPlugin,
        FeatureRepository,
    )
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import FeatureRequest, FeatureResult
    from quant_platform.services.research_service.features.cross_section.cross_section import (
        FeatureBundle,
    )

    #: Uniform builder shape: ``(payloads, as_of) -> FeatureBundle``.
    BundleBuilder = Callable[[Mapping[str, object], datetime], Awaitable[FeatureBundle]]

# Canonical feature family names (the registry keys on ``{name}:{version}``).
CLOSE_FAMILY = "close"
CATALYST_FAMILY = "catalyst"
EVENT_FAMILY = "event"
COMPOSITE_FAMILY = "composite"
PV_FORMULAIC_FAMILY = "pv_formulaic"

_CLOSE_OUTPUT_FEATURES = tuple(spec.name for spec in STANDARD_FACTOR_SPECS if spec.is_alpha)


@dataclass(frozen=True)
class BundleFeatureComputer:
    """A ``FeatureComputer`` that delegates to a ``FeatureBundle`` builder.

    The builder is the existing, unchanged feature math; this class only adapts
    it to the typed registry contract.
    """

    feature_family: str
    feature_set_version: str
    required_inputs: tuple[str, ...]
    output_features: tuple[str, ...]
    builder: BundleBuilder

    @property
    def schema_hash(self) -> str:
        """Return the ordered schema hash for ``output_features``."""
        return ordered_feature_schema_hash(self.output_features)

    async def compute(self, request: FeatureRequest) -> FeatureResult:
        """Compute a typed ``FeatureResult`` from inputs carried in the request."""
        bundle = await self.builder(request.input_context.payloads, request.as_of)
        return feature_result_from_bundle(
            bundle,
            feature_set_version=request.feature_set_version,
            strategy_run_id=request.strategy_run_id,
            as_of=request.as_of,
            artifact_uri=request.artifact_uri,
        )


@dataclass(frozen=True)
class BundleFeatureFamilyPlugin:
    """A ``FeatureFamilyPlugin`` exposing one bundle-backed computer."""

    name: str
    feature_set_version: str
    required_inputs: tuple[str, ...]
    computer: BundleFeatureComputer

    def build_computers(self) -> tuple[FeatureComputer, ...]:
        """Return the single computer for this family/version."""
        return (self.computer,)


def _bars(payloads: Mapping[str, object]) -> Mapping[uuid.UUID, Sequence[MarketBar]]:
    return cast("Mapping[uuid.UUID, Sequence[MarketBar]]", payloads.get(BARS_EOD_INPUT, {}))


def _plugin(
    *,
    family: str,
    version: str,
    required_input: str,
    output_features: tuple[str, ...],
    builder: BundleBuilder,
) -> BundleFeatureFamilyPlugin:
    return BundleFeatureFamilyPlugin(
        name=family,
        feature_set_version=version,
        required_inputs=(required_input,),
        computer=BundleFeatureComputer(
            feature_family=family,
            feature_set_version=version,
            required_inputs=(required_input,),
            output_features=output_features,
            builder=builder,
        ),
    )


def _text_plugin(
    family: str,
    version: str,
    builder: Callable[..., Awaitable[FeatureBundle]],
    output_features: tuple[str, ...],
    feature_repo: FeatureRepository,
) -> BundleFeatureFamilyPlugin:
    async def _build(payloads: Mapping[str, object], as_of: datetime) -> FeatureBundle:
        return await builder(_bars(payloads), text_feature_repo=feature_repo, as_of=as_of)

    return _plugin(
        family=family,
        version=version,
        required_input=BARS_EOD_INPUT,
        output_features=output_features,
        builder=_build,
    )


def build_research_feature_family_plugins(
    feature_repo: FeatureRepository,
) -> tuple[FeatureFamilyPlugin, ...]:
    """Return every executable feature-family plugin.

    ``feature_repo`` is captured by the catalyst / composite builders,
    which read prior governed feature vectors as inputs.
    """

    async def _build_close(payloads: Mapping[str, object], _as_of: datetime) -> FeatureBundle:
        close_data = cast(
            "Mapping[uuid.UUID, Sequence[float]]", payloads.get(CLOSE_SERIES_INPUT, {})
        )
        return build_feature_bundle(close_data)

    async def _build_event(payloads: Mapping[str, object], as_of: datetime) -> FeatureBundle:
        events = cast(
            "Mapping[uuid.UUID, Sequence[datetime]] | None",
            payloads.get(EVENTS_BY_INSTRUMENT_INPUT),
        )
        return build_paper_alpha_event_reaction_v2_feature_bundle(
            _bars(payloads), as_of=as_of, events_by_instrument=events
        )

    async def _build_composite(payloads: Mapping[str, object], as_of: datetime) -> FeatureBundle:
        return await build_paper_alpha_composite_feature_bundle(
            _bars(payloads), source_feature_repo=feature_repo, as_of=as_of
        )

    async def _build_pv_formulaic(payloads: Mapping[str, object], as_of: datetime) -> FeatureBundle:
        # Synchronous kernel compute; wrapped to satisfy the async builder shape.
        return build_pv_formulaic_feature_bundle(_bars(payloads), as_of=as_of)

    return (
        _plugin(
            family=CLOSE_FAMILY,
            version=FEATURE_SET_VERSION,
            required_input=CLOSE_SERIES_INPUT,
            output_features=_CLOSE_OUTPUT_FEATURES,
            builder=_build_close,
        ),
        _text_plugin(
            CATALYST_FAMILY,
            PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
            build_paper_alpha_catalyst_v10_feature_bundle,
            TEXT_CATALYST_V10_ALPHA_FEATURES,
            feature_repo,
        ),
        _plugin(
            family=EVENT_FAMILY,
            version=PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
            required_input=BARS_EOD_INPUT,
            output_features=EVENT_REACTION_V2_ALPHA_FEATURES,
            builder=_build_event,
        ),
        _plugin(
            family=COMPOSITE_FAMILY,
            version=PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
            required_input=BARS_EOD_INPUT,
            # The composite bundle merges admitted source vectors dynamically, so
            # there is no static output-feature contract to declare.
            output_features=(),
            builder=_build_composite,
        ),
        _plugin(
            family=PV_FORMULAIC_FAMILY,
            version=PV_FORMULAIC_FEATURE_SET_VERSION,
            required_input=BARS_EOD_INPUT,
            output_features=PV_FORMULAIC_FEATURE_NAMES,
            builder=_build_pv_formulaic,
        ),
    )


__all__ = [
    "CATALYST_FAMILY",
    "CLOSE_FAMILY",
    "COMPOSITE_FAMILY",
    "EVENT_FAMILY",
    "PV_FORMULAIC_FAMILY",
    "BundleFeatureComputer",
    "BundleFeatureFamilyPlugin",
    "build_research_feature_family_plugins",
]
