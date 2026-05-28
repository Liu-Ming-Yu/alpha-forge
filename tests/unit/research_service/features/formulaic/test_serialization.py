"""Round-trip tests for formulaic-AST serialization."""

from __future__ import annotations

import pytest

from quant_platform.research.features.formulaic.ast import (
    BinOp,
    Compare,
    Const,
    OpCall,
    UnaryOp,
    Var,
    Where,
)
from quant_platform.research.features.formulaic.operators import (
    absolute,
    delta,
    rank,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_rank,
    ts_zscore,
)
from quant_platform.research.features.formulaic.serialization import (
    SERIALIZATION_VERSION,
    expression_from_dict,
    expression_to_dict,
)

# ---------------------------------------------------------------------------
# Per-node-type round trips
# ---------------------------------------------------------------------------


def test_var_round_trip() -> None:
    original = Var("close")
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert round_tripped == original
    assert hash(round_tripped) == hash(original)


def test_const_round_trip() -> None:
    original = Const(3.14)
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert round_tripped == original
    assert isinstance(round_tripped, Const)
    assert round_tripped.value == pytest.approx(3.14)


def test_unary_op_round_trip() -> None:
    original = -Var("close")
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert isinstance(round_tripped, UnaryOp)
    assert round_tripped == original


def test_bin_op_round_trip_all_operators() -> None:
    for op_str in ("+", "-", "*", "/"):
        a, b = Var("close"), Var("open")
        original: BinOp
        if op_str == "+":
            original = a + b
        elif op_str == "-":
            original = a - b
        elif op_str == "*":
            original = a * b
        else:
            original = a / b
        round_tripped = expression_from_dict(expression_to_dict(original))
        assert round_tripped == original


def test_compare_round_trip_all_operators() -> None:
    a = Var("close")
    for original in (a < 0, a <= 0, a > 0, a >= 0, a.cmp_eq(0), a.cmp_ne(0)):
        round_tripped = expression_from_dict(expression_to_dict(original))
        assert isinstance(round_tripped, Compare)
        assert round_tripped == original


def test_where_round_trip() -> None:
    original = Where(
        condition=Var("close") > 100,
        then_branch=Var("volume"),
        else_branch=Const(0.0),
    )
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert isinstance(round_tripped, Where)
    assert round_tripped == original


def test_opcall_round_trip_carries_int_args() -> None:
    original = delta(Var("close"), 7)
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert isinstance(round_tripped, OpCall)
    assert round_tripped == original
    assert round_tripped.int_args == (7,)


def test_opcall_round_trip_carries_float_args() -> None:
    original = signed_power(Var("returns"), 2.5)
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert isinstance(round_tripped, OpCall)
    assert round_tripped == original
    assert round_tripped.float_args == (2.5,)


def test_opcall_round_trip_carries_str_args() -> None:
    from quant_platform.research.features.formulaic.operators import group_rank

    original = group_rank(Var("close"), "sector")
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert isinstance(round_tripped, OpCall)
    assert round_tripped == original
    assert round_tripped.str_args == ("sector",)


# ---------------------------------------------------------------------------
# Nested + library-shape round trips
# ---------------------------------------------------------------------------


def test_nested_ast_round_trip() -> None:
    """The full wq_alpha_001 shape round-trips structurally + by hash."""
    original = rank(ts_argmax(signed_power(absolute(Var("returns")), 2.0), 5)) - 0.5
    payload = expression_to_dict(original)
    round_tripped = expression_from_dict(payload)
    assert round_tripped == original
    assert hash(round_tripped) == hash(original)
    assert round_tripped.required_inputs() == original.required_inputs()
    assert round_tripped.lookback_days() == original.lookback_days()


def test_ts_corr_round_trip_preserves_two_args() -> None:
    original = -ts_corr(rank(Var("volume")), rank(Var("high")), 5)
    round_tripped = expression_from_dict(expression_to_dict(original))
    assert round_tripped == original
    assert round_tripped.lookback_days() == original.lookback_days()


def test_library_smoke_round_trip_all_starter_alphas() -> None:
    """Every alpha in the curated library round-trips intact."""
    from quant_platform.research.features.formulaic.library import LIBRARY

    for alpha in LIBRARY:
        payload = expression_to_dict(alpha.expression)
        back = expression_from_dict(payload)
        assert back == alpha.expression, alpha.name
        assert hash(back) == hash(alpha.expression), alpha.name


# ---------------------------------------------------------------------------
# Versioning + error paths
# ---------------------------------------------------------------------------


def test_payload_carries_schema_version() -> None:
    payload = expression_to_dict(Var("close"))
    assert payload["version"] == SERIALIZATION_VERSION


def test_unknown_version_rejected() -> None:
    payload = expression_to_dict(Var("close"))
    payload["version"] = "v9999"
    with pytest.raises(ValueError, match="unsupported serialization version"):
        expression_from_dict(payload)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node kind"):
        expression_from_dict({"kind": "Bogus", "version": SERIALIZATION_VERSION})


def test_missing_kind_rejected() -> None:
    with pytest.raises(ValueError, match="missing 'kind' field"):
        expression_from_dict({"version": SERIALIZATION_VERSION})


def test_non_dict_rejected() -> None:
    with pytest.raises(ValueError, match="expected dict"):
        expression_from_dict("not a dict")  # type: ignore[arg-type]


def test_var_missing_name_rejected() -> None:
    payload = {"kind": "Var", "version": SERIALIZATION_VERSION}
    with pytest.raises(ValueError, match="must be a string"):
        expression_from_dict(payload)


def test_bin_op_invalid_operator_rejected() -> None:
    payload = {
        "kind": "BinOp",
        "version": SERIALIZATION_VERSION,
        "op": "%",
        "left": expression_to_dict(Var("close")),
        "right": expression_to_dict(Var("open")),
    }
    with pytest.raises(ValueError, match="BinOp.op invalid"):
        expression_from_dict(payload)


def test_opcall_carries_window_lookback() -> None:
    original = ts_rank(Var("close"), 60)
    payload = expression_to_dict(original)
    assert payload["window_lookback"] == 60
    back = expression_from_dict(payload)
    assert isinstance(back, OpCall)
    assert back.window_lookback == 60


def test_ts_zscore_round_trip() -> None:
    original = ts_zscore(Var("close"), 20)
    back = expression_from_dict(expression_to_dict(original))
    assert back == original
    assert isinstance(back, OpCall)
    assert back.int_args == (20,)
