from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quant_platform.bootstrap import text_events as text_event_ops
from quant_platform.config import AlphaSettings, LLMSettings, PlatformSettings, StorageSettings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_v3_exhibit_extraction_requires_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QP__STORAGE__POSTGRES_DSN", "postgresql://quant@localhost/db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    async def _verify_postgres_schema(_settings: PlatformSettings) -> None:
        return None

    monkeypatch.setattr(
        text_event_ops,
        "verify_postgres_schema",
        _verify_postgres_schema,
    )

    payload = await text_event_ops.extract_text_features(
        PlatformSettings(_env_file=None),
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2026, 4, 17, tzinfo=UTC),
        prompt_version="v3",
        document_role="exhibit",
        source_data_manifest=None,
        artifact_root=None,
    )

    assert payload["passed"] is False
    assert payload["reason"] == "text-v3 exhibit extraction requires --source-data-manifest"


@pytest.mark.asyncio
async def test_v4_primary_extraction_requires_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QP__STORAGE__POSTGRES_DSN", "postgresql://quant@localhost/db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    async def _verify_postgres_schema(_settings: PlatformSettings) -> None:
        return None

    monkeypatch.setattr(
        text_event_ops,
        "verify_postgres_schema",
        _verify_postgres_schema,
    )

    payload = await text_event_ops.extract_text_features(
        PlatformSettings(_env_file=None),
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2026, 4, 17, tzinfo=UTC),
        prompt_version="v4",
        document_role="primary",
        source_data_manifest=None,
        artifact_root=None,
    )

    assert payload["passed"] is False
    assert payload["reason"] == "text-v4 primary extraction requires --source-data-manifest"


@pytest.mark.asyncio
async def test_v5_primary_extraction_requires_source_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QP__STORAGE__POSTGRES_DSN", "postgresql://quant@localhost/db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    async def _verify_postgres_schema(_settings: PlatformSettings) -> None:
        return None

    monkeypatch.setattr(
        text_event_ops,
        "verify_postgres_schema",
        _verify_postgres_schema,
    )

    payload = await text_event_ops.extract_text_features(
        PlatformSettings(_env_file=None),
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2026, 4, 17, tzinfo=UTC),
        prompt_version="v5",
        document_role="primary",
        source_data_manifest=None,
        artifact_root=None,
    )

    assert payload["passed"] is False
    assert payload["reason"] == "text-v5 primary extraction requires --source-data-manifest"


@pytest.mark.asyncio
async def test_deepseek_credential_preflight_reads_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QP__STORAGE__POSTGRES_DSN", "postgresql://quant@localhost/db")
    monkeypatch.setenv("QP__LLM__PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-dotenv\n", encoding="utf-8")

    async def _verify_postgres_schema(_settings: PlatformSettings) -> None:
        return None

    monkeypatch.setattr(
        text_event_ops,
        "verify_postgres_schema",
        _verify_postgres_schema,
    )

    payload = await text_event_ops.extract_text_features(
        PlatformSettings(_env_file=None),
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2026, 4, 17, tzinfo=UTC),
        prompt_version="v4",
        document_role="primary",
        source_data_manifest=None,
        artifact_root=None,
    )

    assert payload["passed"] is False
    assert payload["reason"] == "text-v4 primary extraction requires --source-data-manifest"


@pytest.mark.asyncio
async def test_live_llm_extraction_uses_repositories_before_startup_assertion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)

    async def _verify_postgres_schema(_settings: PlatformSettings) -> None:
        return None

    class _Repositories:
        text_event_store = object()
        feature_repo = object()

    repositories = _Repositories()

    class _Result:
        passed = True

        def to_payload(self) -> dict[str, object]:
            return {"events_scanned": 0, "vectors_written": 0}

    async def _extract_text_event_features(**kwargs: object) -> _Result:
        assert kwargs["text_event_store"] is repositories.text_event_store
        assert kwargs["feature_repo"] is repositories.feature_repo
        return _Result()

    monkeypatch.setattr(text_event_ops, "verify_postgres_schema", _verify_postgres_schema)
    monkeypatch.setattr(
        text_event_ops,
        "build_runtime_repositories",
        lambda _settings: repositories,
    )
    monkeypatch.setattr(
        text_event_ops,
        "extract_text_event_features",
        _extract_text_event_features,
    )

    payload = await text_event_ops.extract_text_features(
        PlatformSettings(
            _env_file=None,
            storage=StorageSettings(
                object_store_root=str(tmp_path),
                postgres_dsn="postgresql://quant@localhost/db",
            ),
            alpha=AlphaSettings(
                ensemble_mode="live",
                source_weights={"classical": 0.99, "text": 0.01},
                max_non_classical_weight=0.01,
            ),
            llm=LLMSettings(live_mode_enabled=True, live_rehearsal_enabled=True),
        ),
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2026, 4, 17, tzinfo=UTC),
        prompt_version="v1",
        document_role="primary",
        source_data_manifest=None,
        artifact_root=None,
    )

    assert payload["passed"] is True
    assert payload["vectors_written"] == 0
    assert "text_extraction_manifest" in payload
