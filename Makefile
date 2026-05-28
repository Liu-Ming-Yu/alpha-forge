PYTHON ?= python3

.PHONY: setup lint format-check typecheck test test-cov diff-check verify verify-online migrate-check docker-build-test backup-drill

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,api]"

lint:
	$(PYTHON) -m ruff check scripts src tests

format-check:
	$(PYTHON) -m ruff format --check scripts src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	QP__STORAGE__POSTGRES_DSN= \
	QP__STORAGE__REDIS_URL= \
	QP__STORAGE__EVENT_BUS_BACKEND=in_memory \
	$(PYTHON) -m pytest -q -m "not ibapi and not integration_durable"

test-cov:
	QP__STORAGE__POSTGRES_DSN= \
	QP__STORAGE__REDIS_URL= \
	QP__STORAGE__EVENT_BUS_BACKEND=in_memory \
	$(PYTHON) -m pytest \
		--cov=src/quant_platform \
		--cov-report=term-missing \
		--cov-fail-under=75 \
		-m "not ibapi and not integration_durable"

diff-check:
	git diff --check

verify:
	bash scripts/verify_project.sh

verify-online:
	bash scripts/verify_online.sh

migrate-check:
	$(PYTHON) -m quant_platform migrate
	$(PYTHON) -m quant_platform verify-schema

docker-build-test:
	docker build --target runtime -t quant-platform:ci-test .

backup-drill:
	bash scripts/local_backup_drill.sh
