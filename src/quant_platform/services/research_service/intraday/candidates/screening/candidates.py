"""Deterministic intraday microstructure candidate definitions."""

from __future__ import annotations

from quant_platform.services.research_service.intraday.candidates.features import (
    context_value,
    session_direction,
)
from quant_platform.services.research_service.intraday.candidates.screening.candidates_v2 import (
    INTRADAY_MICROSTRUCTURE_V2_CANDIDATES,
)
from quant_platform.services.research_service.intraday.candidates.screening.types import (
    IntradayCandidateSpec,
)

INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES: tuple[IntradayCandidateSpec, ...] = (
    IntradayCandidateSpec(
        "opening_drive_confirmation_1d_decay",
        lambda ctx: context_value(ctx, "opening_drive"),
        "opening_drive * linear_1d_decay",
        "A strong opening impulse can confirm near-term demand after the prior session.",
        1,
    ),
    IntradayCandidateSpec(
        "opening_drive_reversal_1d_decay",
        lambda ctx: -context_value(ctx, "opening_drive"),
        "-opening_drive * linear_1d_decay",
        "Overextended opening moves can mean-revert when the impulse is not sustained.",
        1,
    ),
    IntradayCandidateSpec(
        "close_pressure_continuation_1d_decay",
        lambda ctx: context_value(ctx, "close_pressure"),
        "close_pressure * linear_1d_decay",
        "Late-session pressure can proxy informed order flow into the next session.",
        1,
    ),
    IntradayCandidateSpec(
        "vwap_accumulation_pressure_1d_decay",
        lambda ctx: context_value(ctx, "vwap_pressure"),
        "(close / session_vwap - 1) * linear_1d_decay",
        "Closing above session VWAP can indicate accumulation pressure.",
        1,
    ),
    IntradayCandidateSpec(
        "intraday_volatility_compression_3d_decay",
        lambda ctx: -context_value(ctx, "intraday_volatility"),
        "-intraday_return_volatility * linear_3d_decay",
        "Compressed intraday volatility may precede cleaner directional follow-through.",
        3,
    ),
    IntradayCandidateSpec(
        "range_expansion_drift_1d_decay",
        lambda ctx: context_value(ctx, "range_expansion") * session_direction(ctx),
        "range_expansion * sign(session_return) * linear_1d_decay",
        "A wide session range aligned with the close can carry short-horizon drift.",
        1,
    ),
)


def intraday_candidates_for_set(candidate_set: str) -> tuple[IntradayCandidateSpec, ...]:
    normalized = candidate_set.strip().lower()
    if normalized == "seed":
        return INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES
    if normalized in {"microstructure-v2", "intraday-microstructure-v2", "v2"}:
        return INTRADAY_MICROSTRUCTURE_V2_CANDIDATES
    raise ValueError(f"unknown intraday candidate set: {candidate_set}")


__all__ = [
    "INTRADAY_MICROSTRUCTURE_SEED_CANDIDATES",
    "INTRADAY_MICROSTRUCTURE_V2_CANDIDATES",
    "intraday_candidates_for_set",
]
