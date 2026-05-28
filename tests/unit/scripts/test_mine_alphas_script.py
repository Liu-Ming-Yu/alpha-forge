"""Unit tests for ``scripts/mine_alphas.py``.

Focus on the importable helpers (arg parser, search/gate/fold-config
builders, JSONL writer) rather than the end-to-end ``main()`` flow,
which needs real parquet files. The end-to-end path is exercised
indirectly via the existing mining tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.mining import (
    AlphaGrammar,
    EvolutionarySearch,
    MiningFoldConfig,
    RandomSearch,
    make_forward_return_labels,
    mine_alphas,
)
from quant_platform.research.features.formulaic.mining.admission import (
    AdmissionGate,
    AdmissionThresholds,
)
from quant_platform.research.features.formulaic.mining.evidence import (
    CandidateEvidence,
)
from quant_platform.research.features.formulaic.mining.provenance import (
    AutoAlphaProvenance,
)
from quant_platform.research.features.formulaic.operators import rank
from quant_platform.research.features.formulaic.panel import build_market_panel

# Load the script as a module via importlib because ``scripts/`` is not
# on the package path. This mirrors how pytest would discover it if it
# were a proper module, while keeping the script itself importable as
# `python scripts/mine_alphas.py ...`.
SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "mine_alphas.py"
_spec = importlib.util.spec_from_file_location("mine_alphas_script", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
mine_alphas_script = importlib.util.module_from_spec(_spec)
sys.modules["mine_alphas_script"] = mine_alphas_script
_spec.loader.exec_module(mine_alphas_script)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def test_arg_parser_requires_contracts_start_end_output() -> None:
    parser = mine_alphas_script.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    # With required flags, parse succeeds.
    args = parser.parse_args(
        [
            "--contracts-file",
            "/tmp/x.json",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--output",
            "/tmp/out.jsonl",
        ]
    )
    assert args.contracts_file == Path("/tmp/x.json")
    assert args.output == Path("/tmp/out.jsonl")


def test_iso_date_parser_handles_date_and_datetime() -> None:
    parse = mine_alphas_script._parse_iso_date
    d = parse("2024-01-01")
    assert d == datetime(2024, 1, 1, tzinfo=UTC)
    dt = parse("2024-01-01T15:30:00+00:00")
    assert dt == datetime(2024, 1, 1, 15, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> object:
    parser = mine_alphas_script.build_arg_parser()
    args = parser.parse_args(
        [
            "--contracts-file",
            "/tmp/x.json",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--output",
            "/tmp/out.jsonl",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_build_search_random() -> None:
    args = _args(search="random", n_candidates=42)
    search = mine_alphas_script.build_search(args)
    assert isinstance(search, RandomSearch)
    assert search.n_candidates == 42


def test_build_search_evolutionary() -> None:
    args = _args(search="evolutionary", population=20, generations=3)
    search = mine_alphas_script.build_search(args)
    assert isinstance(search, EvolutionarySearch)
    assert search.population_size == 20
    assert search.n_generations == 3


def test_build_search_rejects_unknown_algorithm() -> None:
    args = _args(search="bogus")
    with pytest.raises(ValueError, match="unknown --search"):
        mine_alphas_script.build_search(args)


def test_build_gate_passes_thresholds_through() -> None:
    args = _args(
        min_rank_ic=0.05,
        min_icir=0.25,
        max_turnover=0.3,
        max_correlation_to_admitted=0.5,
    )
    gate = mine_alphas_script.build_gate(args)
    assert isinstance(gate, AdmissionGate)
    assert gate.thresholds.min_rank_ic == pytest.approx(0.05)
    assert gate.thresholds.min_icir == pytest.approx(0.25)
    assert gate.thresholds.max_turnover == pytest.approx(0.3)
    assert gate.thresholds.max_correlation_to_admitted == pytest.approx(0.5)


def test_build_fold_config_disabled_when_n_folds_zero() -> None:
    args = _args(n_folds=0)
    assert mine_alphas_script.build_fold_config(args) is None


def test_build_fold_config_defaults_embargo_to_horizon() -> None:
    args = _args(n_folds=5, embargo_days=None, horizon_days=7, min_test_days=15)
    config = mine_alphas_script.build_fold_config(args)
    assert isinstance(config, MiningFoldConfig)
    assert config.n_folds == 5
    assert config.embargo_days == 7  # defaulted from horizon
    assert config.min_test_days == 15


def test_build_fold_config_honours_explicit_embargo() -> None:
    args = _args(n_folds=4, embargo_days=3, horizon_days=10, min_test_days=20)
    config = mine_alphas_script.build_fold_config(args)
    assert isinstance(config, MiningFoldConfig)
    assert config.embargo_days == 3  # not overridden by horizon


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def _sample_provenance() -> AutoAlphaProvenance:
    evidence = CandidateEvidence(
        mean_ic=0.04,
        rank_ic=0.05,
        icir=0.21,
        turnover=0.18,
        coverage=600,
        correlation_to_baseline_max=float("nan"),
        n_dates=120,
    )
    return AutoAlphaProvenance(
        name="auto_alpha_000001",
        expression=rank(Var("close")),
        generation=0,
        seed=42,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=evidence,
        created_at=datetime(2026, 5, 25, 12, 0, tzinfo=UTC),
        admitted=True,
        admission_reason="admitted",
    )


def test_provenance_to_dict_includes_full_expression() -> None:
    prov = _sample_provenance()
    payload = mine_alphas_script.provenance_to_dict(prov)
    assert payload["name"] == "auto_alpha_000001"
    assert payload["admitted"] is True
    assert payload["expression"]["kind"] == "OpCall"
    assert payload["expression"]["name"] == "rank"
    # Evidence is flattened by field name.
    assert payload["evidence"]["rank_ic"] == pytest.approx(0.05)
    # NaN floats become null so the line is valid JSON.
    assert payload["evidence"]["correlation_to_baseline_max"] is None


def test_provenance_to_dict_carries_wf_fields_when_present() -> None:
    from quant_platform.research.features.formulaic.mining.walk_forward import (
        WalkForwardEvidence,
    )

    wf_evidence = WalkForwardEvidence(
        mean_ic=0.04,
        rank_ic=0.05,
        icir=0.21,
        fold_ics=(0.03, -0.01, 0.04, 0.05),
        fold_rank_ics=(0.03, -0.01, 0.04, 0.05),
        fold_negative_ic_streak=1,
        n_folds_valid=4,
        n_dates=120,
        turnover=0.18,
        coverage=600,
        correlation_to_baseline_max=float("nan"),
    )
    prov = replace(_sample_provenance(), evidence=wf_evidence)
    payload = mine_alphas_script.provenance_to_dict(prov)
    assert payload["evidence"]["fold_ics"] == [0.03, -0.01, 0.04, 0.05]
    assert payload["evidence"]["fold_negative_ic_streak"] == 1
    assert payload["evidence"]["n_folds_valid"] == 4


def test_write_jsonl_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    provenances = [_sample_provenance() for _ in range(3)]
    n_written = mine_alphas_script.write_jsonl(output_path=output, provenances=provenances)
    assert n_written == 3
    assert output.exists()
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        # Each line must be valid JSON.
        payload = json.loads(line)
        assert payload["name"] == "auto_alpha_000001"
        assert payload["admitted"] is True


def test_write_jsonl_creates_missing_parent_dir(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "nested" / "run.jsonl"
    n_written = mine_alphas_script.write_jsonl(
        output_path=output,
        provenances=[_sample_provenance()],
    )
    assert n_written == 1
    assert output.exists()


# ---------------------------------------------------------------------------
# Integration: build a synthetic panel + run the miner end-to-end
# ---------------------------------------------------------------------------


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(seed=1)
    rows = []
    dates = pd.bdate_range(start="2024-01-02", periods=200)
    for inst in range(8):
        closes = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=len(dates)))
        for i, d in enumerate(dates):
            close = float(closes[i])
            rows.append(
                {
                    "instrument_id": f"I{inst}",
                    "date": d,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + 5 * inst + rng.normal(0, 50),
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_mine_alphas_jsonl_output(tmp_path: Path) -> None:
    """Drive the miner with a synthetic panel and verify the JSONL
    output the script would produce is parseable + carries the
    expression payloads."""
    panel = build_market_panel(_synthetic_panel())
    labels = make_forward_return_labels(panel, horizon=5)
    result = mine_alphas(
        grammar=AlphaGrammar(max_depth=3, max_total_lookback=60),
        panel=panel,
        labels=labels,
        search=RandomSearch(n_candidates=5),
        gate=AdmissionGate(
            thresholds=AdmissionThresholds(
                min_rank_ic=-1.0,
                min_icir=-10.0,
                min_n_dates=1,
                min_n_folds_valid=1,
                max_fold_negative_ic_streak=10,
            )
        ),
        seed=42,
        fold_config=MiningFoldConfig(n_folds=4, embargo_days=5, min_test_days=15),
    )
    output = tmp_path / "run.jsonl"
    n_written = mine_alphas_script.write_jsonl(output_path=output, provenances=result.history)
    assert n_written == result.n_evaluated == 5
    # Every line must parse and carry a non-trivial expression.
    for line in output.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert payload["expression"]["kind"] in {
            "OpCall",
            "BinOp",
            "UnaryOp",
            "Var",
        }
        assert "fold_ics" in payload["evidence"]
        # NaN scrubbed to None, integers preserved.
        assert isinstance(payload["evidence"]["n_folds_valid"], int)
