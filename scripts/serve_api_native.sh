#!/usr/bin/env bash
# Serve the operator API NATIVELY (with ibapi for live IBKR broker sync) against
# the Dockerized Postgres + Redis.
#
# The Docker image has no ibapi, so the containerized API cannot pull live IBKR
# positions/NAV. This runs the same API in your venv (ibapi installed) on
# 127.0.0.1 — which the TWS API trusts by default, unlike a container's bridge
# address — while reusing the durable Postgres + Redis that Docker provides.
#
# Env vars: PORT (default 8000), API_HOST (default 127.0.0.1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"   # Git Bash on Windows
[ -x "$PY" ] || { echo "ERROR: no .venv found. Run scripts/setup.sh first." >&2; exit 2; }

# 1. ensure ibapi is present
if ! "$PY" -c "import ibapi" 2>/dev/null; then
  echo "ibapi not installed; installing now ..."
  PYTHON="$PY" bash "$ROOT/scripts/install_ibapi.sh"
fi

# 2. ensure datastores are up
echo "Ensuring Postgres + Redis are up ..."
docker compose up -d postgres redis >/dev/null
ready=0
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -U quant -d quant_platform >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" = "1" ] || { echo "ERROR: Postgres did not become ready in 60s." >&2; exit 1; }

# 3. free the port — stop the Dockerized API if running
if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q "quant-platform-api"; then
  echo "Stopping Dockerized API to free port ${PORT} ..."
  docker compose stop quant-platform-api >/dev/null
fi

# 3b. Propagate the IBKR contracts file from .env into the process env. broker
# sync reads QP__LIVE_IBKR__CONTRACTS_FILE from the environment directly (not
# pydantic settings), so without this the console shows 0 positions natively.
CONTRACTS_LINE="$(grep -E '^[[:space:]]*QP__LIVE_IBKR__CONTRACTS_FILE[[:space:]]*=' "$ROOT/.env" 2>/dev/null | head -1 || true)"
if [ -n "$CONTRACTS_LINE" ]; then
  CV="$(printf '%s' "$CONTRACTS_LINE" | cut -d= -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//')"
  if [ -n "$CV" ]; then
    case "$CV" in /* | ?:*) : ;; *) CV="$ROOT/$CV" ;; esac
    export QP__LIVE_IBKR__CONTRACTS_FILE="$CV"
    echo "Contracts file (position mapping) -> $CV"
  fi
fi

# 4. serve natively
echo ""
echo "Starting native operator API at http://${API_HOST}:${PORT}  (ibapi enabled)"
echo "Console: http://${API_HOST}:${PORT}/app/"
echo "Reminder: TWS / IB Gateway must be running per .env QP__BROKER__* (paper TWS = 7497),"
echo "          with 127.0.0.1 in the API 'Trusted IPs'. Ctrl+C to stop."
echo ""
exec "$PY" -m quant_platform serve-api --host "$API_HOST" --port "$PORT"
