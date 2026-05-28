#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required for project verification." >&2
  echo "Install Python 3.11, then rerun: make verify" >&2
  exit 127
fi

VERIFY_VENV="${VERIFY_VENV:-.venv-verify}"
PY="$VERIFY_VENV/bin/python"

rm -rf "$VERIFY_VENV"
python3.11 -m venv "$VERIFY_VENV"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -c constraints/py311.txt -e ".[dev,api,ml]"
"$PY" -m pip install -c constraints/py311.txt detect-secrets

git diff --check
"$PY" -m ruff check scripts src tests
"$PY" -m ruff format --check scripts src tests
"$PY" scripts/check_generated_artifacts.py --clean
"$PY" scripts/check_import_boundaries.py
"$PY" scripts/check_composition_layering.py
"$PY" scripts/check_service_coupling.py
"$PY" scripts/check_service_bootstrap_coupling.py
"$PY" scripts/check_composition_complexity.py
"$PY" scripts/check_module_size.py
"$PY" scripts/check_type_debt.py
"$PY" scripts/check_lint_debt.py

if ! grep -Eq '^script_location = src/quant_platform/alembic$' alembic.ini; then
  echo "alembic.ini must use src/quant_platform/alembic as the canonical migration tree." >&2
  exit 1
fi
if [ -d alembic ] && find alembic -path '*/__pycache__' -prune -o -type f -name '*.py' -print -quit | grep -q .; then
  echo "Root alembic migration files are retired; use src/quant_platform/alembic only." >&2
  exit 1
fi

tracked="$(git ls-files \
  'data/parquet/**' '*.parquet' \
  '**/__pycache__/**' '*.pyc' \
  '.mypy_cache/**' '.pytest_cache/**' '.ruff_cache/**' || true)"
if [ -n "$tracked" ]; then
  echo "Generated artifacts must not be committed:" >&2
  echo "$tracked" >&2
  exit 1
fi

"$PY" scripts/check_secrets.py

"$PY" -m mypy src

# P2 architecture ratchet: application use cases should stay directly covered
# even while broader service/adaptor coverage is paid down incrementally.
"$PY" -m pytest \
  tests/unit/test_cli_inputs.py \
  tests/unit/test_application_contracts.py \
  tests/unit/test_alpha_ensemble.py \
  tests/unit/research_service/feature_quality/test_feature_audit.py \
  tests/unit/research_service/campaigns/test_research_campaign_policy.py \
  tests/unit/research_service/text/test_text_candidate_screening.py \
  tests/unit/test_feature_family_plugins.py \
  tests/unit/test_feature_registry.py \
  tests/unit/test_operator_research_queries.py \
  --cov=src/quant_platform/application \
  --cov-report=term-missing \
  --cov-fail-under=80 \
  -q

# The default gate is intentionally offline.  Force in-memory infrastructure so
# QP_VERIFY_DURABLE=1 or a developer .env with real DSNs cannot make the broad
# non-durable suite touch Postgres/Redis before migrations have run.
QP__STORAGE__POSTGRES_DSN= \
QP__STORAGE__REDIS_URL= \
QP__STORAGE__EVENT_BUS_BACKEND=in_memory \
"$PY" -m pytest \
  --cov=src/quant_platform \
  --cov-report=term-missing \
  --cov-fail-under=75 \
  -m "not ibapi and not integration_durable"

if [ "${QP_VERIFY_DURABLE:-0}" = "1" ]; then
  "$PY" -m quant_platform migrate
  "$PY" -m pytest -q -m "integration_durable"
else
  echo "Skipping durable Postgres/Redis tests; set QP_VERIFY_DURABLE=1 to run them."
fi

if [ "${QP_VERIFY_LIVE_IBKR:-0}" = "1" ]; then
  if [ -n "${IBAPI_PACKAGE_PATH:-}" ]; then
    "$PY" -m pip install "$IBAPI_PACKAGE_PATH"
  else
    "$PY" -m pip install -c constraints/py311.txt ibapi
  fi
  export QP_LIVE_IBKR_REQUIRED=1
  if [ -f ".env" ]; then
    "$PY" -m dotenv -f .env run --no-override -- "$PY" -m pytest -q -m "ibapi"
  else
    "$PY" -m pytest -q -m "ibapi"
  fi
else
  echo "Skipping live IBKR tests; set QP_VERIFY_LIVE_IBKR=1 to run them."
fi

"$PY" scripts/check_generated_artifacts.py
