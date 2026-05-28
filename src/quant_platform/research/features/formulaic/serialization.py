"""JSON-friendly serialization of formulaic-AST expressions.

The mining pipeline produces :class:`AutoAlphaProvenance` records that
must survive a round-trip through disk (JSONL files, evidence bundles
in object storage, candidate review tooling). The dataclass fields are
mostly trivially serialisable — strings, ints, datetimes — but the
:attr:`AutoAlphaProvenance.expression` field is an
:class:`Expression` *tree*, which JSON cannot encode natively.

This module provides the two-way bridge:

* :func:`expression_to_dict` walks an :class:`Expression` and emits a
  nested dict whose leaves are JSON-friendly (str, int, float,
  list[str], list[dict], None). Every node carries a ``kind`` tag so
  :func:`expression_from_dict` can route to the right constructor
  without guessing.
* :func:`expression_from_dict` is the inverse. It validates the
  ``kind`` tag, recursively rebuilds children, and constructs the
  frozen dataclass instance with the same field values the original
  carried.

The round-trip is **structurally lossless** — for any AST built from
the public operator builders, ``expression_from_dict(
expression_to_dict(e)) == e`` (and the same hash, so cache entries
keyed by AST equality survive serialisation). Tests pin this on every
node type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from quant_platform.research.features.formulaic.ast import (
    BinOp,
    Compare,
    Const,
    Expression,
    OpCall,
    UnaryOp,
    Var,
    Where,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quant_platform.research.features.formulaic.ast import BinaryOp, ComparisonOp


#: Schema version pinned into every serialised node. Bumped when the
#: dict shape changes; old payloads can be migrated by inspecting
#: this field. v1 is the current shape.
SERIALIZATION_VERSION: str = "v1"


def expression_to_dict(expression: Expression) -> dict[str, Any]:
    """Serialise an AST node to a JSON-friendly dict.

    The returned dict has a ``kind`` discriminator plus per-node-type
    fields. ``version`` is pinned at the root so a deserialiser can
    reject incompatible payloads early.

    Parameters
    ----------
    expression:
        Any :class:`Expression` subclass instance.

    Returns
    -------
    dict
        Nested dict with JSON-friendly leaf types.
    """
    payload = _encode(expression)
    payload["version"] = SERIALIZATION_VERSION
    return payload


def expression_from_dict(payload: dict[str, Any]) -> Expression:
    """Rebuild an AST from a :func:`expression_to_dict` payload.

    Parameters
    ----------
    payload:
        The dict produced by :func:`expression_to_dict`, or anything
        structurally compatible.

    Returns
    -------
    Expression
        The reconstructed AST. Equality and hash match the original.

    Raises
    ------
    ValueError
        When the dict's ``version`` or ``kind`` is unknown, or when a
        required field is missing for the node type.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"expression_from_dict: expected dict, got {type(payload).__name__}")
    version = payload.get("version", SERIALIZATION_VERSION)
    if version != SERIALIZATION_VERSION:
        raise ValueError(
            f"expression_from_dict: unsupported serialization version {version!r}; "
            f"this build expects {SERIALIZATION_VERSION!r}"
        )
    return _decode(payload)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def _encode(expression: Expression) -> dict[str, Any]:
    if isinstance(expression, Var):
        return {"kind": "Var", "name": expression.name}
    if isinstance(expression, Const):
        return {"kind": "Const", "value": float(expression.value)}
    if isinstance(expression, UnaryOp):
        return {
            "kind": "UnaryOp",
            "op": expression.op,
            "operand": _encode(expression.operand),
        }
    if isinstance(expression, BinOp):
        return {
            "kind": "BinOp",
            "op": expression.op,
            "left": _encode(expression.left),
            "right": _encode(expression.right),
        }
    if isinstance(expression, Compare):
        return {
            "kind": "Compare",
            "op": expression.op,
            "left": _encode(expression.left),
            "right": _encode(expression.right),
        }
    if isinstance(expression, Where):
        return {
            "kind": "Where",
            "condition": _encode(expression.condition),
            "then_branch": _encode(expression.then_branch),
            "else_branch": _encode(expression.else_branch),
        }
    if isinstance(expression, OpCall):
        return {
            "kind": "OpCall",
            "name": expression.name,
            "args": [_encode(arg) for arg in expression.args],
            "window_lookback": int(expression.window_lookback),
            "int_args": list(expression.int_args),
            "str_args": list(expression.str_args),
            "float_args": [float(v) for v in expression.float_args],
        }
    raise TypeError(f"Cannot serialise expression node of type {type(expression).__name__!r}")


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _decode(payload: dict[str, Any]) -> Expression:
    kind = payload.get("kind")
    if kind is None:
        raise ValueError(f"expression_from_dict: payload missing 'kind' field: {payload!r}")
    decoder = _DECODERS.get(kind)
    if decoder is None:
        known = sorted(_DECODERS)
        raise ValueError(
            f"expression_from_dict: unknown node kind {kind!r}; known kinds: {known!r}"
        )
    return decoder(payload)


def _decode_var(payload: dict[str, Any]) -> Var:
    name = _require_str(payload, "name")
    return Var(name=name)


def _decode_const(payload: dict[str, Any]) -> Const:
    value = payload.get("value")
    if value is None:
        raise ValueError(f"Const payload missing 'value': {payload!r}")
    return Const(value=float(value))


def _decode_unary_op(payload: dict[str, Any]) -> UnaryOp:
    op = _require_str(payload, "op")
    if op != "-":
        raise ValueError(f"UnaryOp.op must be '-'; got {op!r}")
    operand = _decode_child(payload, "operand")
    return UnaryOp(op="-", operand=operand)


def _decode_bin_op(payload: dict[str, Any]) -> BinOp:
    op = _require_str(payload, "op")
    if op not in {"+", "-", "*", "/"}:
        raise ValueError(f"BinOp.op invalid: {op!r}")
    left = _decode_child(payload, "left")
    right = _decode_child(payload, "right")
    return BinOp(op=cast("BinaryOp", op), left=left, right=right)


def _decode_compare(payload: dict[str, Any]) -> Compare:
    op = _require_str(payload, "op")
    if op not in {"<", "<=", ">", ">=", "==", "!="}:
        raise ValueError(f"Compare.op invalid: {op!r}")
    left = _decode_child(payload, "left")
    right = _decode_child(payload, "right")
    return Compare(op=cast("ComparisonOp", op), left=left, right=right)


def _decode_where(payload: dict[str, Any]) -> Where:
    return Where(
        condition=_decode_child(payload, "condition"),
        then_branch=_decode_child(payload, "then_branch"),
        else_branch=_decode_child(payload, "else_branch"),
    )


def _decode_op_call(payload: dict[str, Any]) -> OpCall:
    name = _require_str(payload, "name")
    raw_args = payload.get("args", [])
    if not isinstance(raw_args, list):
        raise ValueError(f"OpCall.args must be a list; got {type(raw_args).__name__}")
    args = tuple(_decode(arg) for arg in raw_args)
    window_lookback = int(payload.get("window_lookback", 0))
    int_args = tuple(int(v) for v in payload.get("int_args", ()))
    str_args = tuple(str(v) for v in payload.get("str_args", ()))
    float_args = tuple(float(v) for v in payload.get("float_args", ()))
    return OpCall(
        name=name,
        args=args,
        window_lookback=window_lookback,
        int_args=int_args,
        str_args=str_args,
        float_args=float_args,
    )


def _decode_child(payload: dict[str, Any], field: str) -> Expression:
    child = payload.get(field)
    if child is None:
        raise ValueError(f"Payload missing required child {field!r}: {payload!r}")
    if not isinstance(child, dict):
        raise ValueError(f"Child {field!r} must be a dict; got {type(child).__name__}")
    return _decode(child)


def _require_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Payload field {field!r} must be a string; got {type(value).__name__}")
    return value


_DECODERS: dict[str, Callable[[dict[str, Any]], Expression]] = {
    "Var": _decode_var,
    "Const": _decode_const,
    "UnaryOp": _decode_unary_op,
    "BinOp": _decode_bin_op,
    "Compare": _decode_compare,
    "Where": _decode_where,
    "OpCall": _decode_op_call,
}


__all__ = [
    "SERIALIZATION_VERSION",
    "expression_from_dict",
    "expression_to_dict",
]
