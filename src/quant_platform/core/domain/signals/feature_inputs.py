"""Feature request input-context contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping


@dataclass(frozen=True)
class FeatureInputContext:
    """Typed input context describing datasets available to a feature request.

    ``available_inputs`` / ``dataset_ids`` / ``schema_hashes`` / ``vendor_coverage``
    are lightweight, serialisable metadata used for fail-closed admission checks.

    ``payloads`` carries the *live* input data a :class:`FeatureComputer` needs to
    run — for example the per-instrument bar history keyed by an input name such
    as ``"bars_eod"`` or ``"close_series"``. It holds in-memory objects, is never
    serialised, and defaults empty so metadata-only callers are unaffected.
    """

    available_inputs: tuple[str, ...] = ()
    dataset_ids: Mapping[str, uuid.UUID] = field(default_factory=dict)
    schema_hashes: Mapping[str, str] = field(default_factory=dict)
    vendor_coverage: Mapping[str, float] = field(default_factory=dict)
    payloads: Mapping[str, object] = field(default_factory=dict)


FeatureRequestContext = FeatureInputContext

#: Canonical ``payloads`` keys feature computers and their callers share to
#: route inputs. ``bars_eod`` / ``close_series`` are the two bar-history shapes;
#: ``events_by_instrument`` is the optional event-timestamp input.
BARS_EOD_INPUT = "bars_eod"
CLOSE_SERIES_INPUT = "close_series"
EVENTS_BY_INSTRUMENT_INPUT = "events_by_instrument"


def coerce_feature_input_context(context: FeatureRequestContext) -> FeatureInputContext:
    """Return a typed input context."""
    if isinstance(context, FeatureInputContext):
        return context
    raise TypeError("FeatureRequest context must be a FeatureInputContext")


__all__ = [
    "FeatureInputContext",
    "FeatureRequestContext",
    "coerce_feature_input_context",
]
