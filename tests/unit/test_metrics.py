"""Unit tests for ``quant_platform.infrastructure.metrics``.

These tests exercise the Prometheus exposition surface when
``prometheus_client`` is available.  They also assert graceful degradation
when the optional dependency is missing (the no-op shim path).
"""

from __future__ import annotations

import importlib

import pytest

from quant_platform.infrastructure import metrics as metrics_module


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Re-import the module so each test sees a fresh registry."""
    importlib.reload(metrics_module)


def _counter_value(name: str, labels: dict[str, str]) -> float | None:
    """Read a labelled counter from the module-level registry."""
    registry = metrics_module.get_registry()
    if registry is None:
        return None
    # Counter exposes both ``*`` and ``*_total`` — match the stored family.
    for family in registry.collect():
        if family.name != name and not name.startswith(family.name):
            continue
        for sample in family.samples:
            if not sample.name.endswith("_total"):
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


def _gauge_value(name: str, labels: dict[str, str] | None = None) -> float | None:
    registry = metrics_module.get_registry()
    if registry is None:
        return None
    labels = labels or {}
    for family in registry.collect():
        if family.name != name and not name.startswith(family.name):
            continue
        for sample in family.samples:
            if sample.name != family.name and not sample.name.endswith(family.name):
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


def test_record_order_submitted_increments_counter() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.record_order_submitted("engine_a")
    value = _counter_value("quant_orders_submitted", {"engine": "engine_a"})
    assert value == 1.0
    metrics_module.record_order_submitted("engine_a")
    assert _counter_value("quant_orders_submitted", {"engine": "engine_a"}) == 2.0


def test_record_order_rejected_labels_reason() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.record_order_rejected("engine_a", "execution_policy")
    metrics_module.record_order_rejected("engine_a", "broker_unavailable")
    assert (
        _counter_value(
            "quant_orders_rejected",
            {"engine": "engine_a", "reason": "execution_policy"},
        )
        == 1.0
    )
    assert (
        _counter_value(
            "quant_orders_rejected",
            {"engine": "engine_a", "reason": "broker_unavailable"},
        )
        == 1.0
    )


def test_record_lease_loss_increments_lock_lease_counter() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.record_lease_loss("strategy_cycle:run-1")
    assert (
        _counter_value(
            "quant_lock_lease_lost",
            {"resource": "strategy_cycle:run-1"},
        )
        == 1.0
    )


def test_record_lock_operation_labels_outcome() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.record_lock_operation("lock-x", "acquire", "ok")
    metrics_module.record_lock_operation("lock-x", "acquire", "timeout")
    assert (
        _counter_value(
            "quant_lock_operations",
            {"resource": "lock-x", "op": "acquire", "outcome": "ok"},
        )
        == 1.0
    )
    assert (
        _counter_value(
            "quant_lock_operations",
            {"resource": "lock-x", "op": "acquire", "outcome": "timeout"},
        )
        == 1.0
    )


def test_set_kill_switch_flips_gauge() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.set_kill_switch(True)
    assert _gauge_value("quant_kill_switch_active") == 1.0
    metrics_module.set_kill_switch(False)
    assert _gauge_value("quant_kill_switch_active") == 0.0


def test_time_phase_records_cycle_duration() -> None:
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    with metrics_module.time_phase("engine_x", "signals"):
        pass
    registry = metrics_module.get_registry()
    count = 0
    for family in registry.collect():
        if family.name != "quant_cycle_duration_seconds":
            continue
        for sample in family.samples:
            if (
                sample.name.endswith("_count")
                and sample.labels.get("engine") == "engine_x"
                and sample.labels.get("phase") == "signals"
            ):
                count = int(sample.value)
    assert count == 1


def test_render_latest_returns_bytes_and_content_type() -> None:
    body, content_type = metrics_module.render_latest()
    assert isinstance(body, bytes)
    assert "text/plain" in content_type


def test_noop_metric_accepts_all_operations() -> None:
    """When prometheus_client is not installed, helpers must not raise."""
    noop = metrics_module._NOOP
    noop.labels(a="b").inc()
    noop.labels(a="b").dec()
    noop.labels(a="b").set(1.0)
    noop.labels(a="b").observe(1.0)


def _histogram_count(name: str, labels: dict[str, str]) -> float | None:
    """Read the _count bucket on a histogram family matching the labels."""
    registry = metrics_module.get_registry()
    if registry is None:
        return None
    for family in registry.collect():
        if family.name != name:
            continue
        for sample in family.samples:
            if not sample.name.endswith("_count"):
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


def test_observe_fill_latency_records_histogram() -> None:
    """After observe_fill_latency, _count must be at least 1 for that engine."""
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.observe_fill_latency("engine_fill", 0.42)
    count = _histogram_count("quant_fill_latency_seconds", {"engine": "engine_fill"})
    assert count == 1.0


def test_observe_order_submit_latency_records_histogram() -> None:
    """Submit-latency is tagged with engine + outcome and observed."""
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    metrics_module.observe_order_submit_latency("engine_submit", "acked", 0.05)
    metrics_module.observe_order_submit_latency("engine_submit", "error", 0.12)
    acked = _histogram_count(
        "quant_order_submit_latency_seconds",
        {"engine": "engine_submit", "outcome": "acked"},
    )
    err = _histogram_count(
        "quant_order_submit_latency_seconds",
        {"engine": "engine_submit", "outcome": "error"},
    )
    assert acked == 1.0
    assert err == 1.0


def test_observe_fill_latency_clamps_negative_to_zero() -> None:
    """A fill delivered before the intent (clock skew) must not underflow."""
    if metrics_module.get_registry() is None:
        pytest.skip("prometheus_client not installed")
    # Should not raise; the helper clamps negative values to 0.
    metrics_module.observe_fill_latency("engine_skew", -5.0)
    metrics_module.observe_order_submit_latency("engine_skew", "acked", -1.0)
    assert _histogram_count("quant_fill_latency_seconds", {"engine": "engine_skew"}) == 1.0
