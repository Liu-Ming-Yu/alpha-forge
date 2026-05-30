#!/usr/bin/env bash
# Install the IBKR TWS API Python client (ibapi) into the project venv.
#
# ibapi is NOT on PyPI, so `pip install -e ".[...]"` cannot provide it. This
# downloads the official pinned TWS API release from IBKR, extracts the bundled
# pythonclient, and installs it. Idempotent: no-op if ibapi already imports
# unless IBAPI_FORCE=1.
#
# Env vars:
#   IBAPI_VERSION  TWS API zip-encoding (default 1046.01 = API 10.46.1). If IBKR
#                  has removed this build, set a current one from
#                  https://interactivebrokers.github.io/ (e.g. 1045.01, 1047.01).
#   PYTHON         python to install into (default: .venv/bin/python)
#   IBAPI_FORCE    set to 1 to reinstall even if ibapi already imports
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${IBAPI_VERSION:-1046.01}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$ROOT/.venv/Scripts/python.exe"   # Git Bash on Windows
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: python not found at '$PYTHON'. Create the venv first (scripts/setup.sh) or set PYTHON." >&2
  exit 2
fi

if [ "${IBAPI_FORCE:-0}" != "1" ] && "$PYTHON" -c "import ibapi" 2>/dev/null; then
  V="$("$PYTHON" -c "from ibapi import get_version_string; print(get_version_string())" 2>/dev/null || echo '?')"
  echo "ibapi already installed ($V). Set IBAPI_FORCE=1 to reinstall."
  exit 0
fi

URL="https://interactivebrokers.github.io/downloads/twsapi_macunix.${VERSION}.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading TWS API ${VERSION} ..."
echo "  $URL"
if ! curl -fsSL --max-time 180 -o "$TMP/twsapi.zip" "$URL"; then
  echo "ERROR: download failed for version '${VERSION}'. IBKR may have removed this build;" >&2
  echo "       pick a current one from https://interactivebrokers.github.io/ and set IBAPI_VERSION." >&2
  exit 1
fi

echo "Extracting ..."
unzip -q "$TMP/twsapi.zip" -d "$TMP"
CLIENT="$TMP/IBJts/source/pythonclient"
if [ ! -f "$CLIENT/setup.py" ]; then
  echo "ERROR: pythonclient/setup.py not found — TWS API layout for ${VERSION} may differ." >&2
  exit 1
fi

echo "Installing ibapi into $PYTHON ..."
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install "$CLIENT"
elif command -v uv >/dev/null 2>&1; then
  echo "  pip not in venv; falling back to uv ..."
  uv pip install --python "$PYTHON" "$CLIENT"
else
  echo "ERROR: neither pip nor uv available to install ibapi." >&2
  exit 1
fi

VER="$("$PYTHON" -c "from ibapi import get_version_string; print(get_version_string())")"
echo "ibapi ${VER} installed OK."
