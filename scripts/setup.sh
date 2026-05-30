#!/usr/bin/env bash
# One-command setup for a new machine: Python 3.11 venv + dependencies + ibapi
# (IBKR TWS API) + .env. After this, run scripts/serve_api_native.sh for the
# broker-capable API, or scripts/deploy.sh for the full Docker stack.
#
# Flags:
#   --extras    also install heavy research extras: ml (XGBoost), backtest (vectorbt)
#   --no-ibapi  skip installing ibapi (if this machine never connects to IBKR)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXTRAS=0; NO_IBAPI=0
for arg in "$@"; do
  case "$arg" in
    --extras) EXTRAS=1 ;;
    --no-ibapi) NO_IBAPI=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# --- 1. locate Python 3.11 -------------------------------------------------
echo "==> Locating Python 3.11"
PYBOOT=""
for cand in python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" --version 2>&1 | grep -q "3\.11"; then
    PYBOOT="$cand"; break
  fi
done
[ -n "$PYBOOT" ] || { echo "ERROR: Python 3.11 not found. Install 3.11.x from https://www.python.org/downloads/." >&2; exit 1; }
echo "    Using: $PYBOOT ($($PYBOOT --version 2>&1))"

# --- 2. create venv --------------------------------------------------------
echo "==> Creating .venv"
if [ -x ".venv/bin/python" ]; then
  echo "    .venv already exists (kept)"
else
  "$PYBOOT" -m venv .venv
  echo "    Created .venv"
fi
PY="$ROOT/.venv/bin/python"

# --- 3. install dependencies ----------------------------------------------
echo "==> Installing dependencies"
"$PY" -m pip install --upgrade pip
if [ "$EXTRAS" = "1" ]; then SPEC=".[dev,api,ml,backtest]"; else SPEC=".[dev,api]"; fi
echo "    pip install -e \"$SPEC\""
"$PY" -m pip install -e "$SPEC"

# --- 4. ibapi (IBKR TWS API) ----------------------------------------------
if [ "$NO_IBAPI" = "0" ]; then
  echo "==> Installing ibapi (IBKR TWS API)"
  PYTHON="$PY" bash "$ROOT/scripts/install_ibapi.sh"
fi

# --- 5. .env ---------------------------------------------------------------
echo "==> Preparing .env"
if [ -f ".env" ]; then
  echo "    .env already exists (kept)"
elif [ -f "infra/config/settings.example.env" ]; then
  cp "infra/config/settings.example.env" ".env"
  echo "    Created .env from infra/config/settings.example.env"
  echo "    EDIT IT: set POSTGRES_PASSWORD, QP__API__OPERATOR_API_KEY, and your API keys."
else
  echo "    No settings.example.env found; create .env manually."
fi

echo ""
echo "============================================================"
echo "  Setup complete. Next:"
echo "    1. Edit .env (secrets + POSTGRES_PASSWORD)."
echo "    2. Broker-capable API (native):  bash scripts/serve_api_native.sh"
echo "       or full Docker stack:         bash scripts/deploy.sh"
echo "============================================================"
