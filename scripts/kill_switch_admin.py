"""Operator kill-switch admin — inspect or clear the durable kill switch.

The durable kill switch (``kill_switch_state``, Alembic 004) blocks the engine
loop until an operator clears it. Recovery assessment is read-only by design;
clearing is a deliberate operator action. This is the CLI for that action.

    python scripts/kill_switch_admin.py status
    python scripts/kill_switch_admin.py clear --operator <id> --reason "<why>"

``clear`` requires an explicit ``--confirm`` flag so it is never accidental.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

# psycopg3 async requires SelectorEventLoop; Windows defaults to ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from quant_platform.config import PlatformSettings  # noqa: E402
from quant_platform.services.execution_service.stores.kill_switch_store import (  # noqa: E402
    PostgresKillSwitchStore,
)


async def _run(args: argparse.Namespace) -> int:
    settings = PlatformSettings()
    dsn = settings.storage.postgres_dsn
    if not dsn:
        print("ERROR: QP__STORAGE__POSTGRES_DSN not set", file=sys.stderr)
        return 2
    engine = create_async_engine(dsn)
    try:
        store = PostgresKillSwitchStore(engine)
        state = await store.get()
        print("=== durable kill-switch state ===")
        print(f"  active      : {state.active}")
        print(f"  reason      : {state.reason}")
        print(f"  activated_at: {state.activated_at}")
        print(f"  activated_by: {state.activated_by}")
        print(f"  cleared_at  : {state.cleared_at}")
        if args.command == "status":
            return 0
        # clear
        if not state.active:
            print("\nkill switch already inactive; nothing to clear")
            return 0
        if not args.confirm:
            print(
                "\nrefusing to clear without --confirm "
                "(clearing a durable kill switch is an operator decision)",
                file=sys.stderr,
            )
            return 3
        await store.clear(operator_id=args.operator, as_of=datetime.now(tz=UTC))
        after = await store.get()
        print("\n=== after clear ===")
        print(f"  active      : {after.active}")
        print(f"  cleared_at  : {after.cleared_at}")
        print(f"  cleared_by  : {after.activated_by}")
        print("\nCLEARED")
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the durable kill-switch state")
    clear = sub.add_parser("clear", help="clear the durable kill switch (operator action)")
    clear.add_argument("--operator", required=True, help="operator id recorded on the clear")
    clear.add_argument("--reason", default="operator-cleared via CLI", help="audit reason")
    clear.add_argument("--confirm", action="store_true", help="required to actually clear")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
