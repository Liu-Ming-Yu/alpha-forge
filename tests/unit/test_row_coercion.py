from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from quant_platform.infrastructure.postgres.row_coercion import (
    optional_datetime,
    optional_mapping,
    optional_sequence,
    require_date,
    require_datetime,
    require_mapping,
    require_sequence,
)


def test_require_datetime_accepts_datetime_and_iso_string() -> None:
    now = datetime(2026, 5, 8, tzinfo=UTC)

    assert require_datetime({"as_of": now}, "as_of") is now
    assert require_datetime({"as_of": now.isoformat()}, "as_of") == now


def test_optional_datetime_accepts_none() -> None:
    assert optional_datetime({"revoked_at": None}, "revoked_at") is None


def test_require_date_accepts_date_and_iso_string() -> None:
    value = date(2026, 5, 8)

    assert require_date({"trade_date": value}, "trade_date") is value
    assert require_date({"trade_date": value.isoformat()}, "trade_date") == value


def test_require_date_rejects_datetime() -> None:
    with pytest.raises(TypeError, match="trade_date must be date-compatible"):
        require_date({"trade_date": datetime(2026, 5, 8, tzinfo=UTC)}, "trade_date")


def test_mapping_helpers_validate_payload_shape() -> None:
    assert require_mapping({"a": 1}, name="payload") == {"a": 1}
    assert optional_mapping(None, name="payload") == {}
    with pytest.raises(TypeError, match="payload must be a mapping"):
        require_mapping([("a", 1)], name="payload")


def test_sequence_helpers_reject_strings() -> None:
    assert tuple(require_sequence([1, 2], name="ids")) == (1, 2)
    assert optional_sequence(None, name="ids") == ()
    with pytest.raises(TypeError, match="ids must be a non-string sequence"):
        require_sequence("abc", name="ids")
