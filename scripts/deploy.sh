#!/usr/bin/env bash
# One-command full-stack deploy: Postgres + Redis + the operator API serving
# BOTH the JSON API and the built browser console (ADR-013). Windows twin:
# scripts/deploy.ps1. No host Node or Python required — everything builds in
# Docker.
#
# Usage: scripts/deploy.sh [--workers] [--paper] [--rebuild]
set -euo pipefail

WORKERS=0
PAPER=0
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --workers) WORKERS=1 ;;
    --paper)   PAPER=1 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) echo "usage: scripts/deploy.sh [--workers] [--paper] [--rebuild]"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ENV_FILE="$REPO_ROOT/.env"
EXAMPLE="$REPO_ROOT/.env.example"

step() { printf '\n==> %s\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }

gen_secret() { # $1=bytes  $2=hex|b64
  if command -v openssl >/dev/null 2>&1; then
    if [ "${2:-b64}" = "hex" ]; then openssl rand -hex "$1"; else openssl rand -base64 "$1" | tr -d '+/='; fi
  elif [ "${2:-b64}" = "hex" ]; then
    head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'
  else
    head -c "$1" /dev/urandom | base64 | tr -d '+/=\n'
  fi
}

get_env() { # $1=key
  [ -f "$ENV_FILE" ] || return 0
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$ENV_FILE" | head -n1 | sed -E 's/^[^=]*=//'
}

set_env() { # $1=key  $2=value (only called with shell-safe values)
  local key="$1" val="$2"
  if [ -f "$ENV_FILE" ] && grep -qE "^[[:space:]]*$key[[:space:]]*=" "$ENV_FILE"; then
    awk -v k="$key" -v v="$val" '$0 ~ "^[[:space:]]*"k"[[:space:]]*=" {print k"="v; next} {print}' \
      "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
  fi
}

step "Checking Docker"
command -v docker >/dev/null 2>&1 || { echo "Docker not found. Install Docker and retry." >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable. Start Docker and retry." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 ('docker compose') not found." >&2; exit 1; }
ok "Docker is running"

step "Preparing .env"
if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$EXAMPLE" ]; then cp "$EXAMPLE" "$ENV_FILE"; ok "Created .env from .env.example"; else : > "$ENV_FILE"; ok "Created empty .env"; fi
fi
PG="$(get_env POSTGRES_PASSWORD || true)"
if [ -z "$PG" ] || [ "$PG" = "change_me_before_running_compose" ]; then
  PG="$(gen_secret 24 hex)"; set_env POSTGRES_PASSWORD "$PG"; ok "Generated POSTGRES_PASSWORD"
else ok "POSTGRES_PASSWORD already set (kept)"; fi
APIKEY="$(get_env QP__API__OPERATOR_API_KEY || true)"
if [ -z "$APIKEY" ]; then
  APIKEY="$(gen_secret 32 b64)"; set_env QP__API__OPERATOR_API_KEY "$APIKEY"; ok "Generated QP__API__OPERATOR_API_KEY"
else ok "QP__API__OPERATOR_API_KEY already set (kept)"; fi

step "Building images (backend + console SPA)"
export QP_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export QP_BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUILD=(docker compose build)
if [ "$REBUILD" = "1" ]; then BUILD+=(--no-cache); fi
"${BUILD[@]}"
ok "Image built"

step "Starting Postgres and Redis"
docker compose up -d postgres redis

step "Waiting for Postgres to accept connections"
ready=0
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -U quant -d quant_platform >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[ "$ready" = "1" ] || { echo "Postgres did not become ready in time" >&2; exit 1; }
ok "Postgres ready"

step "Applying database migrations"
docker compose run --rm --no-deps quant-platform-api python -m quant_platform migrate
ok "Schema migrated"

step "Starting the operator API"
UP=(docker compose)
if [ "$WORKERS" = "1" ]; then UP+=(--profile workers); fi
if [ "$PAPER" = "1" ]; then UP+=(--profile paper); fi
UP+=(up -d)
"${UP[@]}"

step "Waiting for the API to report ready"
api_ready=0
for _ in $(seq 1 60); do
  if curl -sf -H "X-API-Key: $APIKEY" http://localhost:8000/health/ready >/dev/null 2>&1; then api_ready=1; break; fi
  sleep 2
done

echo
if [ "$api_ready" = "1" ]; then
  echo "============================================================"
  echo "  Quant platform is up."
  echo "  Console : http://localhost:8000/app/"
  echo "  API     : http://localhost:8000/"
  echo "  API key : $APIKEY"
  echo "============================================================"
  echo "  Open the console and paste the API key into the connect screen."
else
  echo "API did not report ready within the timeout. Recent logs:" >&2
  docker compose logs --tail 40 quant-platform-api || true
  exit 1
fi
