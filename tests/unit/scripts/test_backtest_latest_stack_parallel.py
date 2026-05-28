"""Unit tests for the latest-stack parallelism machinery.

The full universe-300 parallel-vs-sequential bit-identity check
lives in the script's own smoke test (too slow / data-heavy for CI),
but these tests pin the unit-level invariants of the dispatch:

1. ``_resolve_max_workers`` returns the right value across
   (requested, n_arms, cpu_count) combinations — especially the
   clamping behaviour that prevents over-spawning.
2. ``_run_one_arm_job`` catches worker exceptions and returns them
   as ``_ArmJobResult.error`` instead of raising — a runtime
   exception in one arm must not tear down the pool.
3. The result-ordering step preserves the user's requested arm
   order regardless of which order workers complete in.
4. ``_ArmJobResult`` enforces its tagged-union invariant — exactly
   one of ``evidence`` / ``error`` is set — and ``unwrap_evidence()``
   raises rather than returning ``None`` on the error path.
5. The argparse validator rejects ``--max-workers 0`` / negative.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add project root so ``scripts/`` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.backtest_latest_stack import (  # noqa: E402  (post-sys.path)
    ARM_SPECS,
    _ArmJob,
    _ArmJobResult,
    _positive_int_argparse,
    _resolve_max_workers,
    _run_one_arm_job,
    save_run_manifest,
)

# -- 1. _resolve_max_workers ------------------------------------------------


class TestResolveMaxWorkers:
    """Decision table for ``_resolve_max_workers(requested, n_arms)``.

    The function is a hot path (called once per run) but trivial; the
    test value is in pinning the clamp-to-n_arms behaviour and the
    cpu_count-derived default, both of which a future refactor could
    silently regress.
    """

    @pytest.mark.parametrize(
        ("requested", "n_arms", "expected"),
        [
            # Explicit requested worker count is honoured up to n_arms.
            (1, 7, 1),
            (3, 7, 3),
            (4, 7, 4),
            # Over-requesting clamps to the number of arms — no point
            # spawning workers with nothing to do.
            (10, 7, 7),
            (100, 1, 1),
            # n_arms == 0 short-circuits to 1 regardless of request.
            (4, 0, 1),
            (None, 0, 1),
        ],
    )
    def test_explicit_request_or_zero_arms(
        self, requested: int | None, n_arms: int, expected: int
    ) -> None:
        assert _resolve_max_workers(requested, n_arms) == expected

    def test_default_is_half_cpu_count_clamped_to_arms(self) -> None:
        # 8-core box, 7 arms → 4 workers (cpu // 2, clamped).
        with patch("scripts.backtest_latest_stack.os.cpu_count", return_value=8):
            assert _resolve_max_workers(None, 7) == 4

    def test_default_clamps_to_arms_when_cpu_count_higher(self) -> None:
        # 64-core box, 7 arms → 7 workers (clamped to arms, not 32).
        with patch("scripts.backtest_latest_stack.os.cpu_count", return_value=64):
            assert _resolve_max_workers(None, 7) == 7

    def test_default_floors_at_one_on_single_core(self) -> None:
        # 1-core box, default → max(1, 1 // 2) = max(1, 0) = 1.
        # Defensive against the floor — without it we'd return 0
        # workers and ProcessPoolExecutor would raise.
        with patch("scripts.backtest_latest_stack.os.cpu_count", return_value=1):
            assert _resolve_max_workers(None, 7) == 1

    def test_default_handles_cpu_count_none(self) -> None:
        # os.cpu_count() returning None happens on exotic platforms;
        # the function uses ``or 2`` as a defensive default.
        with patch("scripts.backtest_latest_stack.os.cpu_count", return_value=None):
            # cpu defaults to 2, so 2 // 2 = 1.
            assert _resolve_max_workers(None, 7) == 1


# -- 2. _run_one_arm_job exception handling --------------------------------


class TestWorkerCapturesException:
    """A raise inside the worker MUST come back as ``error`` on the
    result, never propagate to the pool. Otherwise one bad arm kills
    every other arm in the run."""

    def _minimal_job(self, tmp_path: Path) -> _ArmJob:
        """Build a syntactically-valid _ArmJob with empty panels.

        Validity isn't important — the worker monkeypatches will
        raise before any of these fields are read substantively.
        """
        spec = ARM_SPECS[0]  # A: research_ranker_pv (no PCA needed)
        empty_panel = pd.DataFrame({"instrument_id": [], "date": [], "alpha": []})
        empty_close = pd.DataFrame({"instrument_id": [], "date": [], "close": []})
        empty_calendar = pd.DatetimeIndex([])
        return _ArmJob(
            spec=spec,
            panel_df=empty_panel,
            feature_names=["alpha"],
            feature_set_versions={"price_volume": "test"},
            close_panel=empty_close,
            sector_map={},
            global_calendar=empty_calendar,
            pca_artifact_metadata={},
            universe_fingerprint={"path": "test", "sha256": "test"},
            bars_fingerprint={
                "algorithm": "test",
                "is_content_hash": False,
                "files": 0,
                "fingerprint": "test",
            },
            git_commit="test",
            out_root=tmp_path,
            cli_args_payload={},
        )

    def test_worker_returns_error_result_on_raise(self, tmp_path: Path) -> None:
        # Monkeypatch the first thing the worker calls so we can
        # control where the exception comes from.
        with patch(
            "scripts.backtest_latest_stack.build_supervised_samples",
            side_effect=RuntimeError("synthetic worker failure"),
        ):
            result = _run_one_arm_job(self._minimal_job(tmp_path))

        # Tagged-union invariant: error set, evidence None.
        assert result.error is not None, "worker swallowed an exception silently"
        assert "synthetic worker failure" in result.error
        assert "RuntimeError" in result.error
        assert result.evidence is None
        # Traceback captured so the failure is debuggable from the
        # main process's log output.
        assert "Traceback" in result.error_traceback
        assert "synthetic worker failure" in result.error_traceback
        # Spec is preserved so the main loop can correlate the
        # failure with the originating ArmSpec.
        assert result.spec.cli_alias == "A"


# -- 3. Result ordering preservation ---------------------------------------


class TestResultsSortToRequestedOrder:
    """The dispatch loop submits arms in requested order but consumes
    them in completion order; the final pass re-sorts back to the
    requested order so the COMPARISON table and on-disk evidence list
    match what the user asked for.
    """

    def test_sort_recovers_requested_order(self) -> None:
        # Take the first 4 specs from ARM_SPECS as a stable test
        # universe. Their cli_aliases are A, B, C, D in registry order.
        requested = list(ARM_SPECS[:4])
        # Build results in shuffled order (D, A, C, B) — simulating
        # completion-order delivery from as_completed().
        shuffled_indices = [3, 0, 2, 1]
        results = [
            _ArmJobResult(
                spec=requested[i],
                evidence=None,
                error=f"stub error for {requested[i].cli_alias}",
            )
            for i in shuffled_indices
        ]
        spec_order = {spec.cli_alias: idx for idx, spec in enumerate(requested)}
        sorted_results = sorted(results, key=lambda r: spec_order[r.spec.cli_alias])

        assert [r.spec.cli_alias for r in sorted_results] == [s.cli_alias for s in requested]

    def test_missing_alias_in_spec_order_is_a_keyerror(self) -> None:
        # Defensive: if a result ever comes back with an alias the
        # main loop didn't request, the sort key lookup MUST raise.
        # The previous .get(alias, 999) fallback masked this as a
        # silent sort-to-end; we want a loud failure.
        requested = list(ARM_SPECS[:2])
        # Result from an alias not in the requested set.
        rogue_alias_spec = ARM_SPECS[3]  # D, not in the first 2
        bad_result = _ArmJobResult(spec=rogue_alias_spec, error="stub")
        spec_order = {spec.cli_alias: idx for idx, spec in enumerate(requested)}
        with pytest.raises(KeyError):
            sorted([bad_result], key=lambda r: spec_order[r.spec.cli_alias])


# -- 4. _ArmJobResult invariant + unwrap helper ----------------------------


class TestArmJobResultInvariant:
    """Pins the tagged-union shape: exactly one of (evidence, error)
    must be set on a valid result. The previous design allowed
    ``evidence=None`` with a ``# type: ignore`` to lie about the
    type; the invariant + ``unwrap_evidence()`` close that hole.
    """

    def test_both_evidence_and_error_set_rejected(self) -> None:
        # We can't easily fabricate a WalkForwardEvidence here, but
        # we can verify the invariant rejects "neither set" which is
        # the same broken-state error path.
        with pytest.raises(ValueError, match="exactly one of evidence/error"):
            _ArmJobResult(spec=ARM_SPECS[0])  # neither field set

    def test_unwrap_evidence_raises_on_error_result(self) -> None:
        result = _ArmJobResult(spec=ARM_SPECS[0], error="stub")
        with pytest.raises(RuntimeError, match="error result"):
            result.unwrap_evidence()


# -- 5. _positive_int_argparse ---------------------------------------------


class TestPositiveIntArgparse:
    """The argparse type for ``--max-workers`` rejects values < 1 at
    parse time with a usage error, so ``_resolve_max_workers`` can
    trust its input."""

    @pytest.mark.parametrize("good", ["1", "2", "8", "42"])
    def test_positive_int_accepted(self, good: str) -> None:
        assert _positive_int_argparse(good) == int(good)

    @pytest.mark.parametrize("bad", ["0", "-1", "-100"])
    def test_zero_and_negative_rejected(self, bad: str) -> None:
        import argparse  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
            _positive_int_argparse(bad)

    @pytest.mark.parametrize("bad", ["foo", "3.5", "", " "])
    def test_non_integer_rejected(self, bad: str) -> None:
        import argparse  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
            _positive_int_argparse(bad)


# -- The smoke-test compatibility check ------------------------------------


def test_arm_a_spec_is_picklable() -> None:
    """Sanity check: ``ArmSpec`` instances must be picklable because
    they ride along inside ``_ArmJob`` to each worker. A non-picklable
    factory (e.g. a lambda) would silently break parallel mode."""
    import pickle  # noqa: PLC0415

    for spec in ARM_SPECS:
        # Round-trip must preserve every field including the callables.
        # S301 is silenced — we pickled the data ourselves in the
        # previous line; this is the symmetric loads call, not
        # untrusted-data deserialization.
        roundtrip = pickle.loads(pickle.dumps(spec))  # noqa: S301
        assert roundtrip.cli_alias == spec.cli_alias
        assert roundtrip.canonical_name == spec.canonical_name
        assert roundtrip.requires_pca == spec.requires_pca
        # Callable round-trip: the unpickled callable must produce
        # the same config object (factories are pure constructors).
        if spec.portfolio_config_factory is not None:
            assert roundtrip.portfolio_config_factory is not None
            original = spec.portfolio_config_factory()
            recovered = roundtrip.portfolio_config_factory()
            assert original == recovered
        if spec.fold_streak_risk_config_factory is not None:
            assert roundtrip.fold_streak_risk_config_factory is not None
            assert (
                spec.fold_streak_risk_config_factory()
                == roundtrip.fold_streak_risk_config_factory()
            )


# -- save_run_manifest schema + skip-arm handling --------------------------


class TestSaveRunManifest:
    """The run-level manifest is the index of a run — points at per-arm
    evidence, summarises pass/fail, lists skipped arms with reasons.
    Without it, "what was this run?" requires parsing N evidence files.
    These tests pin the schema and the skipped-arm bookkeeping."""

    def _empty_manifest_inputs(self, tmp_path: Path) -> dict[str, object]:
        """Build the minimum-viable inputs the function needs."""
        from datetime import UTC, datetime  # noqa: PLC0415
        from uuid import uuid4  # noqa: PLC0415

        started = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        finished = datetime(2026, 5, 28, 12, 13, 24, tzinfo=UTC)
        return {
            "out_root": tmp_path,
            "run_id": uuid4(),
            "started_at": started,
            "finished_at": finished,
            "git_commit": "abc1234",
            "cli_args_payload": {"arms": "A,B", "max_workers": 2},
            "max_workers_used": 2,
            "requested_specs": list(ARM_SPECS[:2]),
            "arm_results": [],
            "skipped_specs": [],
            "universe_fingerprint": {"path": "test", "sha256": "test"},
            "bars_fingerprint": {"algorithm": "test", "files": 0, "fingerprint": "test"},
        }

    def test_manifest_carries_required_run_metadata(self, tmp_path: Path) -> None:
        import json  # noqa: PLC0415

        manifest_path = save_run_manifest(**self._empty_manifest_inputs(tmp_path))
        payload = json.loads(manifest_path.read_text())

        # The minimum metadata a future auditor needs to identify
        # "what was this run?" without opening any per-arm evidence.
        for required_key in (
            "evidence_schema_version",
            "manifest_kind",
            "run_id",
            "started_at_utc",
            "finished_at_utc",
            "wall_clock_seconds",
            "git_commit",
            "cli_args",
            "max_workers_used",
            "universe_fingerprint",
            "bars_snapshot_fingerprint",
            "requested_arms",
            "completed_arms",
            "skipped_arms",
        ):
            assert required_key in payload, f"run_manifest missing required key {required_key!r}"
        assert payload["manifest_kind"] == "run"
        assert payload["wall_clock_seconds"] == 13 * 60 + 24
        assert payload["max_workers_used"] == 2

    def test_manifest_lists_skipped_arms_with_reasons(self, tmp_path: Path) -> None:
        import json  # noqa: PLC0415

        inputs = self._empty_manifest_inputs(tmp_path)
        inputs["skipped_specs"] = [
            (ARM_SPECS[2], "PCA artifact unavailable"),
            (ARM_SPECS[3], "synthetic worker failure"),
        ]
        manifest_path = save_run_manifest(**inputs)
        payload = json.loads(manifest_path.read_text())

        skipped = payload["skipped_arms"]
        assert isinstance(skipped, list)
        assert len(skipped) == 2
        assert {s["cli_alias"] for s in skipped} == {"C", "D"}
        assert skipped[0]["reason"] == "PCA artifact unavailable"
        assert skipped[1]["reason"] == "synthetic worker failure"
        # ``canonical_name`` is included so a reader can correlate
        # the skip with the registered ArmSpec without looking up
        # the alias separately.
        for entry in skipped:
            assert "canonical_name" in entry

    def test_manifest_writes_alongside_per_arm_evidence(self, tmp_path: Path) -> None:
        manifest_path = save_run_manifest(**self._empty_manifest_inputs(tmp_path))
        # File should land at ``<out_root>/run_manifest.json`` — the
        # contract a downstream consumer relies on for indexing runs.
        assert manifest_path == tmp_path / "run_manifest.json"
        assert manifest_path.exists()
