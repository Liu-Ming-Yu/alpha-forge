"""Edge-case validation tests (F1).

Parametrized tests covering domain validation, config validators,
and guard conditions identified during the production audit.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.orders import (
    FillEvent,
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from quant_platform.session import _SessionDrawdownGuard

_NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
_RUN_ID = uuid.uuid4()
_INSTR_ID = uuid.uuid4()
_TARGET_ID = uuid.uuid4()


def _intent(**overrides) -> OrderIntent:
    defaults = dict(
        order_id=uuid.uuid4(),
        strategy_run_id=_RUN_ID,
        portfolio_target_id=_TARGET_ID,
        instrument_id=_INSTR_ID,
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        created_at=_NOW,
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


# ---------------------------------------------------------------------------
# OrderIntent domain validation
# ---------------------------------------------------------------------------


class TestOrderIntentValidation:
    def test_valid_intent_constructs(self) -> None:
        intent = _intent()
        assert intent.quantity == 10

    @pytest.mark.parametrize("quantity", [0, -1, -100])
    def test_zero_or_negative_quantity_raises(self, quantity: int) -> None:
        with pytest.raises(ValueError, match="quantity"):
            _intent(quantity=quantity)

    @pytest.mark.parametrize("order_type", [OrderType.LIMIT, OrderType.LOC])
    def test_limit_types_require_limit_price(self, order_type: OrderType) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            _intent(order_type=order_type, limit_price=None)

    @pytest.mark.parametrize("price", [Decimal("0"), Decimal("-0.01"), Decimal("-100")])
    def test_non_positive_limit_price_raises(self, price: Decimal) -> None:
        with pytest.raises(ValueError, match="limit_price"):
            _intent(order_type=OrderType.LIMIT, limit_price=price)

    def test_naive_created_at_raises(self) -> None:
        naive = datetime(2026, 1, 2, 12, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone"):
            _intent(created_at=naive)

    def test_positive_limit_price_accepted(self) -> None:
        intent = _intent(
            order_type=OrderType.LIMIT,
            limit_price=Decimal("150.25"),
        )
        assert intent.limit_price == Decimal("150.25")


# ---------------------------------------------------------------------------
# FillEvent domain validation
# ---------------------------------------------------------------------------


class TestFillEventValidation:
    def _fill(self, **overrides) -> FillEvent:
        defaults = dict(
            fill_id=uuid.uuid4(),
            order_id=uuid.uuid4(),
            broker_order_id="IB-001",
            instrument_id=_INSTR_ID,
            side=OrderSide.BUY,
            quantity=10,
            fill_price=Decimal("150.00"),
            commission=Decimal("1.00"),
            currency="USD",
            executed_at=_NOW,
            received_at=_NOW,
        )
        defaults.update(overrides)
        return FillEvent(**defaults)

    def test_valid_fill_constructs(self) -> None:
        fill = self._fill()
        assert fill.quantity == 10

    @pytest.mark.parametrize("quantity", [0, -1])
    def test_zero_or_negative_quantity_raises(self, quantity: int) -> None:
        with pytest.raises(ValueError, match="quantity"):
            self._fill(quantity=quantity)

    @pytest.mark.parametrize("price", [Decimal("0"), Decimal("-1.00")])
    def test_non_positive_fill_price_raises(self, price: Decimal) -> None:
        with pytest.raises(ValueError, match="fill_price"):
            self._fill(fill_price=price)

    def test_negative_commission_raises(self) -> None:
        with pytest.raises(ValueError, match="commission"):
            self._fill(commission=Decimal("-0.01"))


# ---------------------------------------------------------------------------
# _SessionDrawdownGuard — negative HWM edge case (E5)
# ---------------------------------------------------------------------------


class TestDrawdownGuardNegativeHWM:
    def test_negative_hwm_raises_runtime_error(self) -> None:
        guard = _SessionDrawdownGuard(Decimal("-0.15"))
        # Force a negative HWM directly (pathological: margin account with
        # more debt than assets — impossible via normal update_and_check since
        # negative NAV can never exceed the initial HWM of 0).
        guard._hwm = Decimal("-10_000")  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="Negative high-water-mark"):
            guard.update_and_check(Decimal("-11_000"))

    def test_zero_nav_after_positive_hwm_is_full_drawdown(self) -> None:
        guard = _SessionDrawdownGuard(Decimal("-0.15"))
        guard.update_and_check(Decimal("100_000"))
        ok, dd = guard.update_and_check(Decimal("0"))
        assert ok is False
        assert dd == Decimal("1")


# ---------------------------------------------------------------------------
# PortfolioConstructor — top_n guard (B4)
# ---------------------------------------------------------------------------


class TestPortfolioConstructorTopNGuard:
    def test_top_n_zero_raises_value_error(self) -> None:
        from quant_platform.services.portfolio_service.portfolio_constructor import (
            LongOnlyPortfolioConstructor,
        )

        with pytest.raises(ValueError, match="top_n"):
            LongOnlyPortfolioConstructor(top_n=0)

    def test_top_n_negative_raises_value_error(self) -> None:
        from quant_platform.services.portfolio_service.portfolio_constructor import (
            LongOnlyPortfolioConstructor,
        )

        with pytest.raises(ValueError, match="top_n"):
            LongOnlyPortfolioConstructor(top_n=-5)


# ---------------------------------------------------------------------------
# Config validators (B6)
# ---------------------------------------------------------------------------


class TestConfigValidators:
    def test_tiingo_token_required_when_fallback_selected(self) -> None:
        from pydantic import ValidationError

        from quant_platform.config import DataIngestSettings

        with pytest.raises(ValidationError, match="tiingo_api_token"):
            DataIngestSettings(bar_fetch_fallback="tiingo", tiingo_api_token="")

    def test_tiingo_token_valid_when_provided(self) -> None:
        from quant_platform.config import DataIngestSettings

        settings = DataIngestSettings(
            bar_fetch_fallback="tiingo",
            tiingo_api_token="tok_abc123",
        )
        assert settings.tiingo_api_token == "tok_abc123"

    def test_llm_timeout_must_be_positive(self) -> None:
        from pydantic import ValidationError

        from quant_platform.config import LLMSettings

        with pytest.raises(ValidationError):
            LLMSettings(timeout_seconds=0)

    def test_llm_timeout_must_not_exceed_300(self) -> None:
        from pydantic import ValidationError

        from quant_platform.config import LLMSettings

        with pytest.raises(ValidationError):
            LLMSettings(timeout_seconds=301)

    def test_regime_settings_requires_proxy_when_seed_enforced(self) -> None:
        from pydantic import ValidationError

        from quant_platform.config import RegimeSettings

        with pytest.raises(ValidationError, match="market_proxy_instrument_id"):
            RegimeSettings(
                enabled=True,
                require_seed_on_cycle=True,
                market_proxy_instrument_id="",
            )

    def test_regime_settings_valid_with_proxy_set(self) -> None:
        from quant_platform.config import RegimeSettings

        settings = RegimeSettings(
            enabled=True,
            require_seed_on_cycle=True,
            market_proxy_instrument_id=str(uuid.uuid4()),
        )
        assert settings.require_seed_on_cycle is True

    @pytest.mark.parametrize("seconds", [0.0, 300.5])
    def test_llm_timeout_boundary_values(self, seconds: float) -> None:
        from pydantic import ValidationError

        from quant_platform.config import LLMSettings

        with pytest.raises(ValidationError):
            LLMSettings(timeout_seconds=seconds)
