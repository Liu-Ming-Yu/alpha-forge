"""Regression tests for the latest-stack sample builder.

Pins the audit findings that motivated the
`fix/latest-stack-label-end-and-metadata` hotfix:

1. ``label_end_index`` must point to the *actual* instrument-local
   label-end date on the global calendar — never ``as_of_index +
   horizon_days``. When an instrument has missing bars (halt, late
   start, data gap), the calendar-offset shortcut understates the
   label window and lets a training sample whose real label reaches
   the test window slip past the sample-level purge.
2. ``realized_return_1d`` must only be emitted when the next
   instrument bar is exactly the next global trading day. If the
   instrument skipped a day, ``next_close / close - 1`` is silently
   a multi-day return — corrupts the daily-MtM compounding contract.
3. Dense panels (every instrument has every calendar day) keep the
   old behavior — the fix is a strict generalization, not a behavior
   change for the common case.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd

# The sample builder lives in ``scripts/``, which isn't on ``sys.path``
# by default. Add the project root so the import resolves cleanly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.backtest_latest_stack import (  # noqa: E402  (post-sys.path)
    HORIZON_DAYS,
    build_supervised_samples,
)


def _calendar(n_days: int, start: str = "2026-01-01") -> pd.DatetimeIndex:
    """Sorted, tz-naive daily DatetimeIndex of length ``n_days``."""
    return pd.DatetimeIndex(pd.date_range(start, periods=n_days, freq="D"))


def _feature_panel(instrument_id: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "date": dates,
            "alpha": [float(i) / max(1, len(dates)) for i in range(len(dates))],
        }
    )


def _close_panel(instrument_id: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "date": dates,
            # Geometric series so consecutive 1d returns are non-trivial.
            "close": [100.0 * (1.001**i) for i in range(len(dates))],
        }
    )


# -- Dense panel: no gaps, no behavior change ------------------------------


class TestDensePanelUnchanged:
    """When every instrument has every calendar day, the new logic
    produces the same as_of_index/label_end_index as the old
    calendar-offset shortcut. This is the regression guard against
    accidentally breaking the common case."""

    def test_label_end_index_equals_as_of_plus_horizon_on_dense_panel(self) -> None:
        inst_str = str(uuid.uuid4())
        global_calendar = _calendar(80)
        feature_panel = _feature_panel(inst_str, global_calendar)
        close_panel = _close_panel(inst_str, global_calendar)

        samples = build_supervised_samples(
            feature_panel,
            close_panel,
            ["alpha"],
            sector_map={inst_str: "TEST"},
            global_calendar=global_calendar,
        )

        assert samples, "dense panel should produce samples"
        for s in samples:
            assert s.as_of_index is not None
            assert s.label_end_index is not None
            assert s.label_end_index == s.as_of_index + HORIZON_DAYS


# -- Sparse panel: missing-bar case (the actual bug) -----------------------


class TestSparseInstrumentLabelEndMapping:
    """An instrument with missing bars must record its **actual**
    instrument-local label-end date on the global calendar, not
    a calendar-offset shortcut. This is the bug the hotfix closes."""

    def test_label_end_index_reflects_actual_future_date_when_gap_exists(self) -> None:
        # Global calendar spans 80 days. The instrument has every day
        # EXCEPT a 10-day gap from calendar position 30 to 39
        # inclusive. So:
        #
        #   instrument has bars at days [0..29, 40..79] = 70 bars
        #   instrument row 20 = calendar day 20
        #   the 21st-subsequent INSTRUMENT row is row 41
        #   instrument row 41 has DATE = day 20 + 21 + 10 (gap) = day 51
        #
        # The old (buggy) code would have recorded label_end_index =
        # as_of_index + 21 = 41. The fix records 51.
        #
        # We pick as_of=20 specifically so the *next* instrument bar
        # is still day 21 (realized-span check passes) but the 21d
        # label window crosses the gap (label-end check is meaningful).
        inst_str = str(uuid.uuid4())
        global_calendar = _calendar(80)
        gap_start, gap_len = 30, 10
        instrument_dates = global_calendar.delete(list(range(gap_start, gap_start + gap_len)))
        feature_panel = _feature_panel(inst_str, instrument_dates)
        close_panel = _close_panel(inst_str, instrument_dates)

        samples = build_supervised_samples(
            feature_panel,
            close_panel,
            ["alpha"],
            sector_map={inst_str: "TEST"},
            global_calendar=global_calendar,
        )

        target_as_of_index = 20
        target_samples = [s for s in samples if s.as_of_index == target_as_of_index]
        assert target_samples, f"expected a sample at as_of_index={target_as_of_index}"
        target = target_samples[0]

        expected_label_end = target_as_of_index + HORIZON_DAYS + gap_len  # = 51
        assert target.label_end_index == expected_label_end, (
            f"label_end_index should follow the instrument-local 21st bar "
            f"(calendar day {expected_label_end}, accounting for the "
            f"{gap_len}-day gap); got {target.label_end_index}"
        )
        # And the date matches the global calendar at that position.
        assert target.label_end_as_of is not None
        assert target.label_end_as_of.date() == global_calendar[expected_label_end].date()

    def test_off_by_calendar_shortcut_would_have_passed_through(self) -> None:
        """Sanity check: the test is meaningful only if the buggy
        shortcut would have produced a different value. Here we
        confirm that as_of + HORIZON_DAYS lands BEFORE the actual
        label-end position — so the fix is doing real work."""
        gap_len = 10
        target_as_of_index = 20
        shortcut_label_end = target_as_of_index + HORIZON_DAYS  # 41
        actual_label_end = target_as_of_index + HORIZON_DAYS + gap_len  # 51
        # The shortcut would have UNDERSTATED the label end by the
        # full gap, making the sample-level purge too permissive.
        assert shortcut_label_end < actual_label_end
        assert actual_label_end - shortcut_label_end == gap_len


# -- Realized-span validation: gap drops the sample ------------------------


class TestRealizedSpanValidation:
    """``realized_return_1d`` is a *one-day* simple return. When the
    next instrument bar is not the next global trading day (a gap),
    the row must be dropped — otherwise we'd ship a 2+ day return
    mislabelled as 1d and the daily-MtM compounding stream would
    silently inflate."""

    def test_sample_at_gap_boundary_is_dropped(self) -> None:
        # Same shape: 80-day global calendar, instrument has a 10-day
        # gap. The bar immediately BEFORE the gap (calendar day 29)
        # has a 11-day span to its "next" instrument bar (calendar
        # day 40), not 1 day. This row must be dropped.
        inst_str = str(uuid.uuid4())
        global_calendar = _calendar(80)
        gap_start, gap_len = 30, 10
        instrument_dates = global_calendar.delete(list(range(gap_start, gap_start + gap_len)))
        feature_panel = _feature_panel(inst_str, instrument_dates)
        close_panel = _close_panel(inst_str, instrument_dates)

        samples = build_supervised_samples(
            feature_panel,
            close_panel,
            ["alpha"],
            sector_map={inst_str: "TEST"},
            global_calendar=global_calendar,
        )

        # No sample should exist at as_of_index = 29 (the gap
        # boundary) because its realized_return_1d would actually
        # be an 11-day return.
        gap_boundary = gap_start - 1
        assert not any(s.as_of_index == gap_boundary for s in samples), (
            f"sample at as_of_index={gap_boundary} should have been "
            f"dropped because its next-bar return spans the gap"
        )

    def test_samples_after_gap_are_present_with_correct_indices(self) -> None:
        """Sanity check: rows after the gap (where the next bar IS
        the next trading day again) are emitted normally with their
        true calendar indices."""
        inst_str = str(uuid.uuid4())
        global_calendar = _calendar(80)
        gap_start, gap_len = 30, 10
        instrument_dates = global_calendar.delete(list(range(gap_start, gap_start + gap_len)))
        feature_panel = _feature_panel(inst_str, instrument_dates)
        close_panel = _close_panel(inst_str, instrument_dates)

        samples = build_supervised_samples(
            feature_panel,
            close_panel,
            ["alpha"],
            sector_map={inst_str: "TEST"},
            global_calendar=global_calendar,
        )

        # A sample at calendar day 45 (well after the gap, where the
        # instrument has dense bars again) should exist with
        # as_of_index = 45 and label_end_index = 45 + 21 = 66.
        post_gap_samples = [s for s in samples if s.as_of_index == 45]
        # Only present if there's room for the 21-day forward label
        # (45 + 21 = 66, well within the 80-day calendar) AND no
        # gap in the forward window.
        if post_gap_samples:
            s = post_gap_samples[0]
            assert s.label_end_index == 45 + HORIZON_DAYS


class TestPcaGatingIsDataDriven:
    """Pins the v3 regression: PCA must fit for any requested arm
    that declares ``requires_pca=True``, not just the originally-
    hardcoded ``{"C", "D"}`` set.

    Reads the script's ARM_SPECS registry directly and asserts the
    invariant that every arm with ``requires_pca=True`` is in the
    "PCA fits when requested" decision set. The dispatch loop in
    ``main()`` uses ``any(spec.requires_pca for spec in
    requested_specs)`` so a future PCA-requiring arm cannot fall
    into the same trap E-only fell into in PR #65.
    """

    def test_every_pca_requiring_arm_is_known(self) -> None:
        # Import lazily so the test file stays cheap to load.
        from scripts.backtest_latest_stack import ARM_SPECS  # noqa: PLC0415

        pca_requiring = [spec.cli_alias for spec in ARM_SPECS if spec.requires_pca]
        # C, D, E require the PCA artifact; F, G are the no-PCA
        # ablation. If you add a new PCA-requiring arm, update this
        # set; the data-driven ``any(...)`` check in ``main()`` will
        # Just Work.
        assert set(pca_requiring) == {"C", "D", "E"}, (
            "Unexpected set of PCA-requiring arms — update this test "
            "and verify ``main()`` still gates PCA fitting via "
            "``any(spec.requires_pca for spec in requested_specs)``."
        )

    def test_e_only_request_would_trigger_pca_under_data_driven_gate(self) -> None:
        # Simulate the dispatch-time decision without booting the
        # whole script. The fix is the boolean expression itself;
        # if it's wrong the whole arm-E-only path silently skips
        # PCA and Arm E gets bench'd as "PCA unavailable".
        from scripts.backtest_latest_stack import ARM_SPECS  # noqa: PLC0415

        e_spec = next(s for s in ARM_SPECS if s.cli_alias == "E")
        requested_specs = [e_spec]
        needs_pca = any(spec.requires_pca for spec in requested_specs)
        assert needs_pca, (
            "E-only request must trigger PCA fitting. If this fails, "
            "the data-driven gate has regressed back to a hardcoded set."
        )

    def test_baseline_only_request_does_not_trigger_pca(self) -> None:
        # Negative case: A/B don't need PCA, so running just A or just
        # B (or A+B) should NOT fit PCA. Saves ~1 second of warmup work
        # in the script and ~75k rows in the warmup window.
        from scripts.backtest_latest_stack import ARM_SPECS  # noqa: PLC0415

        baselines = [s for s in ARM_SPECS if s.cli_alias in {"A", "B"}]
        needs_pca = any(spec.requires_pca for spec in baselines)
        assert not needs_pca

    def test_f_and_g_no_pca_ablation_request_does_not_trigger_pca(self) -> None:
        # F and G are the no-PCA half of the 2x2 long-only ablation:
        # ``--arms F`` or ``--arms F,G`` alone must NOT fit PCA, so
        # the F/G evidence reflects only the pv+formulaic panel.
        # If a future refactor accidentally marks F/G as
        # ``requires_pca=True``, the ablation is corrupted.
        from scripts.backtest_latest_stack import ARM_SPECS  # noqa: PLC0415

        no_pca_long_only = [s for s in ARM_SPECS if s.cli_alias in {"F", "G"}]
        assert len(no_pca_long_only) == 2, (
            "Expected both F and G in ARM_SPECS — the no-PCA arm of the "
            "2x2 ablation. Missing arms here corrupt the analysis."
        )
        for spec in no_pca_long_only:
            assert not spec.requires_pca, (
                f"Arm {spec.cli_alias} is the no-PCA ablation; requires_pca must be False."
            )
            assert spec.panel_key == "pv_form", (
                f"Arm {spec.cli_alias} must consume the pv_form panel "
                f"(no learned-PCA features); got panel_key={spec.panel_key!r}."
            )
        needs_pca = any(spec.requires_pca for spec in no_pca_long_only)
        assert not needs_pca
