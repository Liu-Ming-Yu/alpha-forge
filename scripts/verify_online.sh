#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1

if [ -f ".env" ]; then
  while IFS= read -r line; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" == *"="* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [ -z "${!key:-}" ] && [ -n "$value" ]; then
      export "$key=$value"
    fi
  done < ".env"
fi

die() {
  echo "verify-online: $*" >&2
  exit 2
}

require_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    die "$cmd is required. $hint"
  fi
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    die "$name is required for the online gate."
  fi
}

require_int_env() {
  local name="$1"
  require_env "$name"
  if ! [[ "${!name}" =~ ^[0-9]+$ ]]; then
    die "$name must be a numeric value."
  fi
}

python_check_tcp() {
  local host="$1"
  local port="$2"
  python3.11 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect((host, port))
except OSError as exc:
    raise SystemExit(f"{host}:{port} is not reachable: {exc}") from exc
finally:
    sock.close()
PY
}

require_command python3.11 "Install Python 3.11 before running make verify-online."

export QP__STORAGE__POSTGRES_DSN="${QP__STORAGE__POSTGRES_DSN:-postgresql+psycopg://quant:quant@localhost:5432/quant_platform}"
export QP__STORAGE__REDIS_URL="${QP__STORAGE__REDIS_URL:-redis://localhost:6379/0}"
export QP__STORAGE__EVENT_BUS_BACKEND="${QP__STORAGE__EVENT_BUS_BACKEND:-redis_streams}"
export QP__LIVE_IBKR__CONTRACTS_FILE="${QP__LIVE_IBKR__CONTRACTS_FILE:-infra/config/paper_contracts.json}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-quant}"

require_env QP__STORAGE__POSTGRES_DSN
require_env QP__STORAGE__REDIS_URL
require_env QP__STORAGE__EVENT_BUS_BACKEND
if [ "$QP__STORAGE__EVENT_BUS_BACKEND" != "redis_streams" ]; then
  die "QP__STORAGE__EVENT_BUS_BACKEND must be redis_streams for the online gate."
fi

require_env QP__BROKER__HOST
require_int_env QP__BROKER__PORT
require_int_env QP__BROKER__CLIENT_ID
require_env QP__BROKER__ACCOUNT_ID
require_env QP__LIVE_IBKR__CONTRACTS_FILE
if [ "${QP_LIVE_IBKR_ALLOW_PAPER_ORDERS:-0}" = "1" ]; then
  paper_raw="${QP__BROKER__PAPER_TRADING:-true}"
  case "${paper_raw,,}" in
    1|true|yes|on) ;;
    *) die "paper-order verification requires QP__BROKER__PAPER_TRADING=true." ;;
  esac
  case "$QP__BROKER__PORT" in
    4002|7497) ;;
    *) die "paper-order verification requires paper IBKR port 4002 or 7497." ;;
  esac
  case "${QP__BROKER__ACCOUNT_ID^^}" in
    DU*) ;;
    *) die "paper-order verification requires an IBKR paper account id beginning with DU." ;;
  esac
fi

if [ ! -f "$QP__LIVE_IBKR__CONTRACTS_FILE" ]; then
  die "QP__LIVE_IBKR__CONTRACTS_FILE does not exist: $QP__LIVE_IBKR__CONTRACTS_FILE"
fi

if [ -n "${IBAPI_PACKAGE_PATH:-}" ] && [ ! -e "$IBAPI_PACKAGE_PATH" ]; then
  die "IBAPI_PACKAGE_PATH is set but does not exist: $IBAPI_PACKAGE_PATH"
fi

if [ -z "${IBAPI_PACKAGE_PATH:-}" ]; then
  python3.11 - <<'PY' || echo "verify-online: ibapi is not importable from python3.11; verify_project.sh will install ibapi from PyPI into .venv-verify."
try:
    import ibapi  # noqa: F401
except Exception as exc:
    raise SystemExit(str(exc)) from exc
PY
fi

if [ "${QP_ONLINE_SKIP_DOCKER:-0}" != "1" ]; then
  require_command docker "Install Docker Desktop and enable WSL integration on Windows, or install Docker Desktop for Mac."
  if ! docker compose version >/dev/null 2>&1; then
    die "docker compose is required. On WSL2, enable Docker Desktop integration for this distro and restart WSL."
  fi
  docker compose up -d postgres redis
fi

python_check_tcp 127.0.0.1 5432
python_check_tcp 127.0.0.1 6379
python_check_tcp "$QP__BROKER__HOST" "$QP__BROKER__PORT"

LIVE_BROKER_HOST="$QP__BROKER__HOST"
LIVE_BROKER_PORT="$QP__BROKER__PORT"
LIVE_BROKER_CLIENT_ID="$QP__BROKER__CLIENT_ID"
LIVE_BROKER_ACCOUNT_ID="$QP__BROKER__ACCOUNT_ID"
LIVE_BROKER_REQUEST_TIMEOUT_SECONDS="${QP__BROKER__REQUEST_TIMEOUT_SECONDS:-}"
LIVE_CONTRACTS_FILE="$QP__LIVE_IBKR__CONTRACTS_FILE"
LIVE_TEST_SYMBOL="${QP__LIVE_IBKR__TEST_SYMBOL:-}"
LIVE_TEST_CON_ID="${QP__LIVE_IBKR__TEST_CON_ID:-}"
LIVE_TEST_EXCHANGE="${QP__LIVE_IBKR__TEST_EXCHANGE:-}"
LIVE_TEST_CURRENCY="${QP__LIVE_IBKR__TEST_CURRENCY:-}"

env \
  -u QP__BROKER__HOST \
  -u QP__BROKER__PORT \
  -u QP__BROKER__CLIENT_ID \
  -u QP__BROKER__ACCOUNT_ID \
  -u QP__BROKER__REQUEST_TIMEOUT_SECONDS \
  -u QP__LIVE_IBKR__CONTRACTS_FILE \
  -u QP__LIVE_IBKR__TEST_SYMBOL \
  -u QP__LIVE_IBKR__TEST_CON_ID \
  -u QP__LIVE_IBKR__TEST_EXCHANGE \
  -u QP__LIVE_IBKR__TEST_CURRENCY \
  -u QP_VERIFY_LIVE_IBKR \
  -u QP_LIVE_IBKR_REQUIRED \
  -u QP_LIVE_IBKR_ALLOW_PAPER_ORDERS \
  QP_VERIFY_DURABLE=1 \
  make verify

PY="${VERIFY_VENV:-.venv-verify}/bin/python"
if [ ! -x "$PY" ]; then
  die "verification virtualenv was not created at $PY"
fi

if [ -n "${IBAPI_PACKAGE_PATH:-}" ]; then
  "$PY" -m pip install "$IBAPI_PACKAGE_PATH"
else
  "$PY" -m pip install -c constraints/py311.txt ibapi
fi

export QP_VERIFY_LIVE_IBKR=1
export QP_LIVE_IBKR_REQUIRED=1
export QP__BROKER__HOST="$LIVE_BROKER_HOST"
export QP__BROKER__PORT="$LIVE_BROKER_PORT"
export QP__BROKER__CLIENT_ID="$LIVE_BROKER_CLIENT_ID"
export QP__BROKER__ACCOUNT_ID="$LIVE_BROKER_ACCOUNT_ID"
export QP__LIVE_IBKR__CONTRACTS_FILE="$LIVE_CONTRACTS_FILE"

if [ -n "$LIVE_BROKER_REQUEST_TIMEOUT_SECONDS" ]; then
  export QP__BROKER__REQUEST_TIMEOUT_SECONDS="$LIVE_BROKER_REQUEST_TIMEOUT_SECONDS"
fi
if [ -n "$LIVE_TEST_SYMBOL" ]; then
  export QP__LIVE_IBKR__TEST_SYMBOL="$LIVE_TEST_SYMBOL"
fi
if [ -n "$LIVE_TEST_CON_ID" ]; then
  export QP__LIVE_IBKR__TEST_CON_ID="$LIVE_TEST_CON_ID"
fi
if [ -n "$LIVE_TEST_EXCHANGE" ]; then
  export QP__LIVE_IBKR__TEST_EXCHANGE="$LIVE_TEST_EXCHANGE"
fi
if [ -n "$LIVE_TEST_CURRENCY" ]; then
  export QP__LIVE_IBKR__TEST_CURRENCY="$LIVE_TEST_CURRENCY"
fi

if [ -f ".env" ]; then
  "$PY" -m dotenv -f .env run --no-override -- "$PY" -m pytest -q -m "ibapi and not ibapi_orders"
else
  "$PY" -m pytest -q -m "ibapi and not ibapi_orders"
fi

if [ "${QP_LIVE_IBKR_ALLOW_PAPER_ORDERS:-0}" = "1" ]; then
  if [ -f ".env" ]; then
    "$PY" -m dotenv -f .env run --no-override -- "$PY" -m pytest -q -m "ibapi_orders"
  else
    "$PY" -m pytest -q -m "ibapi_orders"
  fi
else
  echo "Skipping paper-order IBKR tests; set QP_LIVE_IBKR_ALLOW_PAPER_ORDERS=1 to run them."
fi
