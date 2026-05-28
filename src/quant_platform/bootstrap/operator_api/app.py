"""Operator API composition helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.views.operator_api.app import create_app_from_session

if TYPE_CHECKING:
    from decimal import Decimal

    from fastapi import FastAPI

    from quant_platform.config import PlatformSettings


def build_operator_api_app(settings: PlatformSettings, *, initial_cash: Decimal) -> FastAPI:
    session = create_paper_session(settings, initial_cash=initial_cash)
    return create_app_from_session(session)
