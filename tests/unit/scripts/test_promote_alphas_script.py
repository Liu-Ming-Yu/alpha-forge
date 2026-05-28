"""Unit tests for ``scripts/promote_alphas.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_platform.research.features.formulaic.ast import Var
from quant_platform.research.features.formulaic.auto_library import (
    PromotedAlphaRecord,
    load_promoted_library,
)
from quant_platform.research.features.formulaic.mining.evidence import (
    CandidateEvidence,
)
from quant_platform.research.features.formulaic.mining.provenance import (
    AutoAlphaProvenance,
)
from quant_platform.research.features.formulaic.mining.walk_forward import (
    WalkForwardEvidence,
)
from quant_platform.research.features.formulaic.operators import rank

# Load the script via importlib (mirrors test_mine_alphas_script.py).
SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "promote_alphas.py"
_spec = importlib.util.spec_from_file_location("promote_alphas_script", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
promote_alphas_script = importlib.util.module_from_spec(_spec)
sys.modules["promote_alphas_script"] = promote_alphas_script
_spec.loader.exec_module(promote_alphas_script)


# Also load the mining script — we use its provenance_to_dict to build
# realistic input JSONL for the promotion CLI to parse.
MINE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "mine_alphas.py"
_mine_spec = importlib.util.spec_from_file_location("mine_alphas_for_promote", MINE_PATH)
assert _mine_spec is not None and _mine_spec.loader is not None
mine_alphas_script = importlib.util.module_from_spec(_mine_spec)
sys.modules["mine_alphas_for_promote"] = mine_alphas_script
_mine_spec.loader.exec_module(mine_alphas_script)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _wf_provenance(name: str = "auto_alpha_a", rank_ic: float = 0.06) -> AutoAlphaProvenance:
    evidence = WalkForwardEvidence(
        mean_ic=rank_ic,
        rank_ic=rank_ic,
        icir=0.5,
        fold_ics=(0.05, 0.06, 0.07, 0.06),
        fold_rank_ics=(0.05, 0.06, 0.07, 0.06),
        fold_negative_ic_streak=0,
        n_folds_valid=4,
        n_dates=200,
        turnover=0.2,
        coverage=600,
        correlation_to_baseline_max=float("nan"),
    )
    return AutoAlphaProvenance(
        name=name,
        expression=rank(Var("close")),
        generation=0,
        seed=42,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=evidence,
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
        admitted=True,
        admission_reason="admitted",
    )


def _write_mining_jsonl(path: Path, provenances: list[AutoAlphaProvenance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for prov in provenances:
            fh.write(json.dumps(mine_alphas_script.provenance_to_dict(prov)))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------


def test_arg_parser_requires_input() -> None:
    parser = promote_alphas_script.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_defaults_output_to_promoted_path() -> None:
    parser = promote_alphas_script.build_arg_parser()
    args = parser.parse_args(["--input", "/tmp/x.jsonl"])
    assert args.output.name == "promoted_alphas.jsonl"


def test_arg_parser_supports_dry_run() -> None:
    parser = promote_alphas_script.build_arg_parser()
    args = parser.parse_args(["--input", "/tmp/x.jsonl", "--dry-run"])
    assert args.dry_run is True


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def test_parse_mining_jsonl_round_trips_wf_evidence(tmp_path: Path) -> None:
    target = tmp_path / "run.jsonl"
    _write_mining_jsonl(target, [_wf_provenance()])
    pairs = promote_alphas_script.parse_mining_jsonl(target)
    assert len(pairs) == 1
    prov, _ = pairs[0]
    assert isinstance(prov.evidence, WalkForwardEvidence)
    assert prov.evidence.fold_negative_ic_streak == 0
    assert prov.evidence.n_folds_valid == 4


def test_parse_mining_jsonl_handles_single_pass_evidence(tmp_path: Path) -> None:
    target = tmp_path / "run.jsonl"
    sp_evidence = CandidateEvidence(
        mean_ic=0.04,
        rank_ic=0.04,
        icir=0.3,
        turnover=0.2,
        coverage=500,
        correlation_to_baseline_max=float("nan"),
        n_dates=150,
    )
    prov = AutoAlphaProvenance(
        name="auto_alpha_single",
        expression=rank(Var("close")),
        generation=0,
        seed=7,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=sp_evidence,
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
        admitted=True,
        admission_reason="admitted",
    )
    _write_mining_jsonl(target, [prov])
    pairs = promote_alphas_script.parse_mining_jsonl(target)
    assert len(pairs) == 1
    parsed, _ = pairs[0]
    assert isinstance(parsed.evidence, CandidateEvidence)


def test_parse_mining_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    target = tmp_path / "run.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    good_prov = _wf_provenance(name="auto_alpha_good")
    with target.open("w", encoding="utf-8") as fh:
        fh.write("definitely not JSON\n")
        fh.write(json.dumps(mine_alphas_script.provenance_to_dict(good_prov)))
        fh.write("\n")
        fh.write('{"missing": "fields"}\n')
    pairs = promote_alphas_script.parse_mining_jsonl(target)
    # Only the well-formed line survives.
    assert len(pairs) == 1
    assert pairs[0][0].name == "auto_alpha_good"


def test_parse_mining_jsonl_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        promote_alphas_script.parse_mining_jsonl(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def test_make_description_carries_metrics() -> None:
    prov = _wf_provenance()
    desc = promote_alphas_script.make_description(prov, run_label="test_run")
    assert "test_run" in desc
    assert "seed=42" in desc
    assert "rank-IC=0.0600" in desc
    assert "ICIR=0.5000" in desc
    assert "fold_streak=0/4" in desc


def test_build_promoted_record_uses_content_hash_name() -> None:
    prov = _wf_provenance(name="ignored_input_name")
    record = promote_alphas_script.build_promoted_record(prov, run_label="run")
    assert record.name.startswith("auto_alpha_")
    # Same expression → same hash, regardless of the input name.
    second = promote_alphas_script.build_promoted_record(
        _wf_provenance(name="other"), run_label="run"
    )
    assert second.name == record.name


def test_evidence_to_dict_scrubs_nan() -> None:
    prov = _wf_provenance()
    payload = promote_alphas_script._evidence_to_dict(prov.evidence)
    assert payload["correlation_to_baseline_max"] is None  # NaN scrubbed


def test_evidence_to_dict_renders_fold_ics_as_list() -> None:
    prov = _wf_provenance()
    payload = promote_alphas_script._evidence_to_dict(prov.evidence)
    assert isinstance(payload["fold_ics"], list)
    assert payload["fold_ics"] == [0.05, 0.06, 0.07, 0.06]


# ---------------------------------------------------------------------------
# End-to-end through main()
# ---------------------------------------------------------------------------


def test_main_dry_run_writes_no_file(tmp_path: Path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    _write_mining_jsonl(input_path, [_wf_provenance()])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    # Dry-run should not write the output file.
    assert not output_path.exists()


def test_main_writes_promotable_candidates(tmp_path: Path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    _write_mining_jsonl(input_path, [_wf_provenance()])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.exists()
    alphas = load_promoted_library(path=output_path)
    assert len(alphas) == 1
    # The record's name is content-hashed, not the original.
    assert alphas[0].name.startswith("auto_alpha_")


def test_main_filters_with_thresholds(tmp_path: Path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    # Two provenances: one passes, one has too-low rank-IC.
    good = _wf_provenance(name="auto_alpha_good", rank_ic=0.08)
    bad = _wf_provenance(name="auto_alpha_bad", rank_ic=0.001)
    _write_mining_jsonl(input_path, [good, bad])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-oos-rank-ic",
            "0.05",
        ]
    )
    assert rc == 0
    alphas = load_promoted_library(path=output_path)
    # Only one passes.
    assert len(alphas) == 1


def test_main_no_promotable_candidates_writes_nothing(tmp_path: Path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    # Threshold too high — nothing passes.
    _write_mining_jsonl(input_path, [_wf_provenance(rank_ic=0.001)])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert not output_path.exists()


def test_main_rejects_single_pass_evidence_by_default(tmp_path: Path) -> None:
    """Default thresholds require WF — a single-pass-evidence mining
    file produces zero promotions."""
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    sp_evidence = CandidateEvidence(
        mean_ic=0.08,
        rank_ic=0.08,
        icir=0.6,
        turnover=0.2,
        coverage=500,
        correlation_to_baseline_max=float("nan"),
        n_dates=150,
    )
    prov = AutoAlphaProvenance(
        name="auto_alpha_x",
        expression=rank(Var("close")),
        generation=0,
        seed=42,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=sp_evidence,
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
        admitted=True,
        admission_reason="admitted",
    )
    _write_mining_jsonl(input_path, [prov])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert not output_path.exists()


def test_main_allow_single_pass_overrides_wf_requirement(tmp_path: Path) -> None:
    """``--allow-single-pass`` lets non-WF candidates through if
    their pooled metrics qualify."""
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "promoted.jsonl"
    sp_evidence = CandidateEvidence(
        mean_ic=0.08,
        rank_ic=0.08,
        icir=0.6,
        turnover=0.2,
        coverage=500,
        correlation_to_baseline_max=float("nan"),
        n_dates=150,
    )
    prov = AutoAlphaProvenance(
        name="auto_alpha_x",
        expression=rank(Var("close")),
        generation=0,
        seed=42,
        parent_name=None,
        mutation_kind=None,
        operator_set_version="operator-set-v1",
        feature_set_version="formulaic-alpha-v1",
        evidence=sp_evidence,
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
        admitted=True,
        admission_reason="admitted",
    )
    _write_mining_jsonl(input_path, [prov])
    rc = promote_alphas_script.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--allow-single-pass",
            "--min-n-folds-valid",
            "1",  # SP evidence has no folds; relax the floor too
        ]
    )
    assert rc == 0
    assert output_path.exists()
    alphas = load_promoted_library(path=output_path)
    assert len(alphas) == 1


# ---------------------------------------------------------------------------
# Integration: promote-then-import wires alphas into the family
# ---------------------------------------------------------------------------


def test_promoted_alpha_lands_in_family_via_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting QUANT_PROMOTED_ALPHAS_PATH to the freshly-written
    JSONL means load_promoted_library returns the alpha (and the
    family's MANIFEST would include it on next import)."""
    target = tmp_path / "promoted.jsonl"
    rec = PromotedAlphaRecord(
        name="auto_alpha_via_env",
        expression_payload={"kind": "Var", "name": "close", "version": "v1"},
        description="env-driven test",
        promotion_evidence={"rank_ic": 0.05},
        promoted_from_seed=0,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rec.to_jsonl_line() + "\n", encoding="utf-8")
    monkeypatch.setenv("QUANT_PROMOTED_ALPHAS_PATH", str(target))
    monkeypatch.delenv("QUANT_DISABLE_AUTO_PROMOTED_LIBRARY", raising=False)
    alphas = load_promoted_library()
    assert any(a.name == "auto_alpha_via_env" for a in alphas)
