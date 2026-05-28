"""Entry point for the quant platform CLI."""

from __future__ import annotations

from quant_platform.cli.app import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
