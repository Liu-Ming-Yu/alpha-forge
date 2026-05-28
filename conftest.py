"""Root conftest — test import path and deterministic timeout guard."""

import asyncio
import faulthandler
import os
import signal
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

# psycopg (async) requires SelectorEventLoop on Windows; ProactorEventLoop is the
# default on Windows and is incompatible with asyncio-based PostgreSQL drivers.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--qp-test-timeout",
        action="store",
        default=os.getenv("QP_TEST_TIMEOUT_SECONDS", "180"),
        help="Per-test call timeout in seconds; set 0 to disable.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    raw_timeout = item.config.getoption("--qp-test-timeout")
    try:
        timeout_seconds = int(raw_timeout)
    except (TypeError, ValueError):
        timeout_seconds = 180

    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout_handler(signum: int, frame: object) -> None:
        message = f"\npytest timeout: test call exceeded {timeout_seconds}s: {item.nodeid}\n"
        os.write(2, message.encode("utf-8", errors="replace"))
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(124)

    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
