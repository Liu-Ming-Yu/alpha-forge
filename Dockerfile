# ── Stage 1: build dependencies ──────────────────────────────────────────────
#
# Pin to a specific Python 3.11 patch version (matching the CI runner
# at ``actions/setup-python``'s cached version). The bare
# ``python:3.11-slim`` tag is a moving target — a base-image rebuild
# can silently change wheels resolved against ``constraints/py311.txt``.
# Bumping this explicitly is the only signal we can trust.
FROM python:3.11.15-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY constraints/py311.txt ./constraints/py311.txt
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --prefix=/install -c constraints/py311.txt ".[api]"

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11.15-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ``curl`` powers the HEALTHCHECK. ``libgomp1`` is required at import
# time by both numpy's OpenBLAS backend and (if [ml] is layered in
# later) xgboost — a few hundred KB each on top of slim, but avoids
# silent runtime ImportErrors when wheel dependencies shift between
# upstream releases.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 quant

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=quant:quant src ./src
COPY --chown=quant:quant alembic.ini ./alembic.ini
COPY --chown=quant:quant scripts ./scripts
COPY --chown=quant:quant infra ./infra

USER quant

# OCI labels — populated at build time via ``docker build --build-arg
# QP_GIT_COMMIT=$(git rev-parse HEAD) --build-arg
# QP_BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) ...``. Optional but
# cheap to carry: ``docker inspect`` then surfaces the exact commit
# the running image was built from, which the live-paper soak logs
# already correlate against by SHA.
ARG QP_GIT_COMMIT=unknown
ARG QP_BUILD_DATE=unknown
LABEL org.opencontainers.image.title="quant-platform" \
      org.opencontainers.image.source="https://github.com/Liu-Ming-Yu/Quant" \
      org.opencontainers.image.revision="${QP_GIT_COMMIT}" \
      org.opencontainers.image.created="${QP_BUILD_DATE}" \
      org.opencontainers.image.licenses="Proprietary"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD if [ -n "$QP__API__OPERATOR_API_KEY" ]; then curl -sf -H "X-API-Key: $QP__API__OPERATOR_API_KEY" http://localhost:8000/health/ready; else curl -sf http://localhost:8000/health/ready; fi || exit 1

CMD ["python", "-m", "quant_platform", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
