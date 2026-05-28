"""Unit tests for PlatformSettings and configure_logging."""

from __future__ import annotations

import os
import tempfile
from decimal import Decimal

from quant_platform.config import (
    LoggingSettings,
    PlatformSettings,
    configure_logging,
)


class TestPlatformSettingsDefaults:
    def test_defaults_match_original_constants(self) -> None:
        """Defaults should preserve the values that were module-level constants."""
        s = PlatformSettings(_env_file=None)

        assert s.broker.host == "127.0.0.1"
        assert s.broker.port == 7497
        assert s.broker.client_id == 1
        assert s.broker.paper_trading is True
        assert s.broker.request_timeout_seconds == 10.0
        assert s.broker.primary_broker_path == "tws"
        assert s.broker.heartbeat_interval_seconds == 5.0
        assert s.production.signal_gate_max_drawdown == -0.10
        assert s.production.signal_gate_max_turnover == 1.0
        assert s.production.heartbeat_stale_after_minutes == 10
        assert s.broker.max_consecutive_health_failures == 3
        assert s.broker.reconcile_on_reconnect is True
        assert s.broker.stale_day_order_cleanup_minutes == 30
        assert s.execution.post_submit_lifecycle_drain_seconds == 2.0
        assert s.execution.lifecycle_drain_poll_seconds == 0.1

        assert s.throttle.capacity == 10
        assert s.throttle.refill_rate == 2.0

        assert s.cash.reservation_buffer_pct == Decimal("0.01")
        assert s.cash.reservation_ttl_minutes == 30
        assert s.cash.drift_tolerance_usd == Decimal("1.00")

        assert s.risk.auto_correct_threshold == 1
        assert s.risk.max_single_name_weight == Decimal("0.05")

        assert s.logging.log_level == "INFO"
        assert s.logging.log_format == "json"
        assert s.api.operator_api_key == ""
        assert s.llm.provider == "anthropic"
        assert s.llm.deepseek_base_url == "https://api.deepseek.com/anthropic"
        assert s.llm.text_model_manifest == ""
        assert s.llm.text_feature_set_version == "paper-alpha-catalyst-v10"
        assert tuple(s.llm.text_feature_weights) == (
            "v10_stability_abs_text_specificity_event_surprise_21d",
            "v10_stability_abs_text_specificity_forward_outlook_21d",
            "v10_stability_abs_text_tone_cov40_minus_vol_tone_21d",
        )
        assert s.llm.extraction_artifact_root == ""
        assert s.llm.live_startup_assertion_stale_after_hours == 24
        assert s.llm.live_rehearsal_enabled is False
        assert s.llm.replay_only_live is True
        # Raised from 30s to 120s alongside timeout_seconds (ADR-001 item 4): the
        # v4 catalyst prompt routinely takes 60–90s on DeepSeek and the platform
        # invariant max_request_latency_seconds <= timeout_seconds requires both
        # knobs to move together.
        assert s.llm.max_request_latency_seconds == 120.0
        assert s.llm.max_daily_calls == 1000
        assert s.llm.max_daily_estimated_cost_usd == 25.0
        assert s.alpha.source_weights == {
            "classical": 0.70,
            "xgboost": 0.15,
            "text": 0.05,
            "event": 0.05,
            "intraday": 0.05,
        }
        assert s.alpha.max_non_classical_weight == 0.01
        assert s.alpha.paper_max_non_classical_weight == 0.30
        assert s.alpha.live_ramp_initial == Decimal("0.01")


class TestPlatformSettingsEnvOverride:
    def test_env_vars_override_defaults(self, monkeypatch: object) -> None:
        """QP__BROKER__PORT env var should override the default port."""
        os.environ["QP__BROKER__PORT"] = "4002"
        os.environ["QP__RISK__MAX_SINGLE_NAME_WEIGHT"] = "0.10"
        os.environ["QP__THROTTLE__CAPACITY"] = "20"
        os.environ["QP__CASH__RESERVATION_TTL_MINUTES"] = "60"
        os.environ["QP__CASH__DRIFT_TOLERANCE_USD"] = "2.50"
        os.environ["QP__LOGGING__LOG_LEVEL"] = "DEBUG"
        os.environ["QP__BROKER__PRIMARY_BROKER_PATH"] = "dual"
        os.environ["QP__API__OPERATOR_API_KEY"] = "test-key"
        os.environ["QP__LLM__PROVIDER"] = "deepseek"
        os.environ["QP__LLM__MODEL"] = "deepseek-v4-flash"
        try:
            s = PlatformSettings(_env_file=None)
            assert s.broker.port == 4002
            assert s.risk.max_single_name_weight == Decimal("0.10")
            assert s.throttle.capacity == 20
            assert s.cash.reservation_ttl_minutes == 60
            assert s.cash.drift_tolerance_usd == Decimal("2.50")
            assert s.logging.log_level == "DEBUG"
            assert s.broker.primary_broker_path == "dual"
            assert s.api.operator_api_key == "test-key"
            assert s.llm.provider == "deepseek"
            assert s.llm.model == "deepseek-v4-flash"
        finally:
            for key in [
                "QP__BROKER__PORT",
                "QP__RISK__MAX_SINGLE_NAME_WEIGHT",
                "QP__THROTTLE__CAPACITY",
                "QP__CASH__RESERVATION_TTL_MINUTES",
                "QP__CASH__DRIFT_TOLERANCE_USD",
                "QP__LOGGING__LOG_LEVEL",
                "QP__BROKER__PRIMARY_BROKER_PATH",
                "QP__API__OPERATOR_API_KEY",
                "QP__LLM__PROVIDER",
                "QP__LLM__MODEL",
            ]:
                os.environ.pop(key, None)

    def test_env_file_override(self, tmp_path: object) -> None:
        """Settings can load from an explicit .env file."""
        env_content = "QP__BROKER__PORT=9999\nQP__BROKER__HOST=10.0.0.1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, dir=str(tmp_path)
        ) as f:
            f.write(env_content)
            env_path = f.name

        try:
            s = PlatformSettings(_env_file=env_path)
            assert s.broker.port == 9999
            assert s.broker.host == "10.0.0.1"
        finally:
            os.unlink(env_path)


class TestConfigureLogging:
    def test_json_logging(self) -> None:
        """configure_logging with json format should not raise."""
        configure_logging(LoggingSettings(log_level="WARNING", log_format="json"))

    def test_console_logging(self) -> None:
        """configure_logging with console format should not raise."""
        configure_logging(LoggingSettings(log_level="DEBUG", log_format="console"))

    def test_default_logging(self) -> None:
        """configure_logging with no args should use defaults."""
        configure_logging()
