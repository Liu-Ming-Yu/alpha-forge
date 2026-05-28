"""V2 intraday microstructure candidate definitions."""

from __future__ import annotations

from quant_platform.services.research_service.intraday.candidates.features import (
    aggregate_context_band,
    aggregate_context_value,
    sample_context_value,
)
from quant_platform.services.research_service.intraday.candidates.screening.types import (
    IntradayCandidateSpec,
)

INTRADAY_MICROSTRUCTURE_V2_CANDIDATES: tuple[IntradayCandidateSpec, ...] = (
    IntradayCandidateSpec(
        "intraday_v2_signed_vwap_pressure_1d",
        lambda ctx: aggregate_context_value(ctx, "vwap_pressure", 1, signed=True),
        "sign(session_return) * vwap_pressure over 1d",
        "Signed VWAP pressure can identify same-day accumulation or distribution.",
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_signed_close_pressure_3d",
        lambda ctx: aggregate_context_value(ctx, "close_pressure", 3, signed=True),
        "sign(session_return) * close_pressure over 3d",
        "Late-session signed pressure over several sessions can proxy persistent order flow.",
        3,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_volume_share_5d",
        lambda ctx: (
            aggregate_context_value(ctx, "range_expansion", 5, signed=True)
            * aggregate_context_value(ctx, "volume_share", 5)
        ),
        "signed range_expansion * closing_volume_share over 5d",
        "Range expansion confirmed by closing-session volume share can capture informed flow.",
        5,
    ),
    IntradayCandidateSpec(
        "intraday_v2_open_close_pressure_spread_3d",
        lambda ctx: (
            aggregate_context_value(ctx, "close_pressure", 3, signed=True)
            - aggregate_context_value(ctx, "opening_drive", 3, signed=True)
        ),
        "signed close_pressure - signed opening_drive over 3d",
        "A close-over-open pressure spread can separate sustained flow from opening noise.",
        3,
    ),
    IntradayCandidateSpec(
        "intraday_v2_volatility_volume_compression_5d",
        lambda ctx: (
            -aggregate_context_value(ctx, "intraday_volatility", 5)
            * aggregate_context_value(ctx, "volume_share", 5)
        ),
        "-intraday_volatility * closing_volume_share over 5d",
        "Lower realized intraday volatility with closing participation may precede cleaner drift.",
        5,
    ),
    IntradayCandidateSpec(
        "intraday_v2_signed_volume_share_momentum_12m_1d",
        lambda ctx: (
            aggregate_context_value(ctx, "volume_share", 1, signed=True)
            * sample_context_value(ctx, "momentum_12m_1m")
        ),
        "sign(session_return) * closing_volume_share over 1d * momentum_12m_1m",
        "One-day signed closing participation should be most useful when slower momentum agrees.",
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_opening_drive_reversal_intensity_1d",
        lambda ctx: (
            -aggregate_context_value(ctx, "opening_drive", 1, signed=True)
            * aggregate_context_value(ctx, "opening_drive", 1)
        ),
        "-sign(session_return) * opening_drive^2 over 1d",
        "An intense signed opening drive can overextend and reverse after the session completes.",
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_volume_share_close_volatility_blend_1d",
        lambda ctx: (
            aggregate_context_value(ctx, "volume_share", 1, signed=True)
            * sample_context_value(ctx, "momentum_12m_1m")
            + 0.5
            * aggregate_context_value(ctx, "close_pressure", 1, signed=True)
            * aggregate_context_value(ctx, "intraday_volatility", 1)
        ),
        (
            "signed closing_volume_share * momentum_12m_1m + "
            "0.5 * signed close_pressure * intraday_volatility over 1d"
        ),
        (
            "Closing participation with slower momentum can be stabilized "
            "by same-day close volatility."
        ),
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_volume_share_opening_reversal_blend_1d",
        lambda ctx: (
            aggregate_context_value(ctx, "volume_share", 1, signed=True)
            * sample_context_value(ctx, "momentum_12m_1m")
            - aggregate_context_value(ctx, "opening_drive", 1, signed=True)
            * aggregate_context_value(ctx, "opening_drive", 1)
        ),
        (
            "signed closing_volume_share * momentum_12m_1m - "
            "signed opening_drive * opening_drive over 1d"
        ),
        "Closing participation should be cleaner when opening-drive overextension is discounted.",
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_volume_share_range_volatility_blend_1d",
        lambda ctx: (
            aggregate_context_value(ctx, "volume_share", 1, signed=True)
            * sample_context_value(ctx, "momentum_12m_1m")
            - 0.5
            * aggregate_context_value(ctx, "intraday_volatility", 1, signed=True)
            * aggregate_context_value(ctx, "range_expansion", 1)
        ),
        (
            "signed closing_volume_share * momentum_12m_1m - "
            "0.5 * signed intraday_volatility * range_expansion over 1d"
        ),
        (
            "Signed closing participation should be more stable when "
            "range-volatility stress is netted."
        ),
        1,
    ),
    IntradayCandidateSpec(
        "intraday_v2_signed_range_expansion_band_2_3_close_pressure_21d",
        lambda ctx: (
            -aggregate_context_band(ctx, "range_expansion", 2, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 2)
        ),
        "-signed range_expansion band(2d,3d) * close_pressure over 2d",
        (
            "A two-to-three-day signed range expansion change can identify "
            "cleaner close-pressure drift."
        ),
        3,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_expansion_band_1_5_opening_drive_21d",
        lambda ctx: (
            aggregate_context_band(ctx, "range_expansion", 1, 5)
            * aggregate_context_value(ctx, "opening_drive", 1)
        ),
        "range_expansion band(1d,5d) * opening_drive over 1d",
        (
            "Range expansion that persists beyond the latest session should "
            "stabilize opening-drive signals."
        ),
        5,
    ),
    IntradayCandidateSpec(
        "intraday_v2_signed_opening_drive_band_1_15_opening_drive_21d",
        lambda ctx: (
            aggregate_context_band(ctx, "opening_drive", 1, 15, signed=True)
            * aggregate_context_value(ctx, "opening_drive", 1)
        ),
        "signed opening_drive band(1d,15d) * opening_drive over 1d",
        (
            "The change in signed opening pressure over a longer window can "
            "temper one-day opening noise."
        ),
        15,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_expansion_band_1_10_range_21d",
        lambda ctx: (
            aggregate_context_band(ctx, "range_expansion", 1, 10)
            * aggregate_context_value(ctx, "range_expansion", 1)
        ),
        "range_expansion band(1d,10d) * range_expansion over 1d",
        (
            "Range expansion that is broader than the latest day can capture "
            "persistent microstructure stress."
        ),
        10,
    ),
    IntradayCandidateSpec(
        "intraday_v2_vwap_pressure_band_1_21_close_pressure_21d",
        lambda ctx: (
            -aggregate_context_band(ctx, "vwap_pressure", 1, 21)
            * aggregate_context_value(ctx, "close_pressure", 1)
        ),
        "-vwap_pressure band(1d,21d) * close_pressure over 1d",
        "VWAP pressure that diverges from longer context can make close pressure more informative.",
        21,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_opening_band_composite_21d",
        lambda ctx: (
            -3.0
            * aggregate_context_band(ctx, "range_expansion", 2, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 2)
            + aggregate_context_band(ctx, "opening_drive", 1, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 1)
        ),
        (
            "-3 * signed range_expansion band(2d,3d) * close_pressure over 2d + "
            "signed opening_drive band(1d,3d) * close_pressure over 1d"
        ),
        "A signed range reversal component can be strengthened by a short opening-drive change.",
        3,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_vwap_band_composite_21d",
        lambda ctx: (
            -aggregate_context_band(ctx, "range_expansion", 2, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 2)
            - 0.5
            * aggregate_context_band(ctx, "vwap_pressure", 2, 7)
            * aggregate_context_value(ctx, "close_pressure", 2)
        ),
        (
            "-signed range_expansion band(2d,3d) * close_pressure over 2d - "
            "0.5 * vwap_pressure band(2d,7d) * close_pressure over 2d"
        ),
        "Range-change pressure should be more stable when longer VWAP divergence confirms it.",
        7,
    ),
    IntradayCandidateSpec(
        "intraday_v2_short_range_vwap_opening_composite_21d",
        lambda ctx: (
            -aggregate_context_band(ctx, "range_expansion", 1, 2, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 1)
            + 2.0
            * aggregate_context_value(ctx, "vwap_pressure", 1, signed=True)
            * aggregate_context_value(ctx, "opening_drive", 1)
        ),
        (
            "-signed range_expansion band(1d,2d) * close_pressure over 1d + "
            "2 * signed vwap_pressure * opening_drive over 1d"
        ),
        "A very short range-change reversal can be confirmed by signed VWAP and opening pressure.",
        2,
    ),
    IntradayCandidateSpec(
        "intraday_v2_range_volume_band_composite_21d",
        lambda ctx: (
            -aggregate_context_band(ctx, "range_expansion", 2, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 2)
            + aggregate_context_band(ctx, "volume_share", 1, 3, signed=True)
            * aggregate_context_value(ctx, "close_pressure", 1)
        ),
        (
            "-signed range_expansion band(2d,3d) * close_pressure over 2d + "
            "signed volume_share band(1d,3d) * close_pressure over 1d"
        ),
        (
            "Range-change pressure should be stronger when signed closing "
            "participation changes with it."
        ),
        3,
    ),
)


__all__ = ["INTRADAY_MICROSTRUCTURE_V2_CANDIDATES"]
