"""Domain-event JSON serialization helpers."""

from __future__ import annotations

import json
import types
import uuid
from dataclasses import fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import quant_platform.core.events as event_defs
from quant_platform.core.events import DomainEvent

_EVENT_TYPES: dict[str, type[DomainEvent]] = {
    name: cls
    for name, cls in vars(event_defs).items()
    if isinstance(cls, type) and issubclass(cls, DomainEvent) and cls is not DomainEvent
}


def _unwrap_optional(tp: object) -> object:
    origin = get_origin(tp)
    if origin in (types.UnionType, Union):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _serialize_value(value: object) -> object:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def _coerce_value(value: object, expected_type: object) -> object:
    if value is None:
        return None
    target_type = _unwrap_optional(expected_type)
    origin = get_origin(target_type)
    if target_type is uuid.UUID:
        return uuid.UUID(str(value))
    if target_type is datetime:
        return datetime.fromisoformat(str(value))
    if target_type is Decimal:
        return Decimal(str(value))
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict payload for {target_type!r}")
        args = get_args(target_type)
        key_type, val_type = args if len(args) == 2 else (object, object)
        return {_coerce_value(k, key_type): _coerce_value(v, val_type) for k, v in value.items()}
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"Expected list payload for {target_type!r}")
        args = get_args(target_type)
        item_type = args[0] if args else object
        return [_coerce_value(v, item_type) for v in value]
    return value


def serialize_event(event: DomainEvent) -> str:
    payload = {
        "event_type": type(event).__name__,
        "data": {f.name: _serialize_value(getattr(event, f.name)) for f in fields(event)},
    }
    return json.dumps(payload, separators=(",", ":"))


def deserialize_event(payload: str) -> DomainEvent:
    raw = json.loads(payload)
    event_type_name = raw["event_type"]
    data = raw["data"]
    event_type = _EVENT_TYPES.get(event_type_name)
    if event_type is None:
        raise ValueError(f"Unknown event_type: {event_type_name}")
    resolved_types = get_type_hints(event_type)
    kwargs = {
        f.name: _coerce_value(data.get(f.name), resolved_types.get(f.name, f.type))
        for f in fields(event_type)
    }
    return cast("DomainEvent", cast("type[Any]", event_type)(**kwargs))


_deserialize_event = deserialize_event
_serialize_event = serialize_event
