"""Unit tests for the learned-representations-v1 feature family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.features.learned import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_N_COMPONENTS,
    FEATURE_NAMES,
    FEATURE_SPECS,
    MANIFEST,
    LearnedConfig,
    PCAArtifact,
    compute_learned_features,
    load_pca_artifact,
    save_pca_artifact,
)
from quant_platform.research.features.learned.config import FEATURE_SET_VERSION

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _identity_artifact(
    *,
    n_features: int = 5,
    n_components: int | None = None,
    family_version: str | None = None,
    scale: tuple[float, ...] | None = None,
) -> PCAArtifact:
    """Build a synthetic artifact whose components are the canonical
    basis vectors. Projecting a row equals the row's mean-centred and
    scale-divided values; reconstruction error is zero when ``n_components
    == n_features`` — useful for hand-checking feature semantics
    without depending on sklearn.

    The default ``scale`` is a tuple of ``1.0`` so the standardisation
    step reduces to identity and the artifact behaves like a pre-v2
    centred-only PCA, keeping the catalogue's existing tests valid.
    """
    nc = n_components if n_components is not None else DEFAULT_N_COMPONENTS
    if nc > n_features:
        raise ValueError("n_components cannot exceed n_features for identity test")
    fv = family_version if family_version is not None else FEATURE_SET_VERSION
    feature_names = tuple(f"feature_{i}" for i in range(n_features))
    mean = tuple(0.0 for _ in range(n_features))
    scale_tuple = scale if scale is not None else tuple(1.0 for _ in range(n_features))
    if len(scale_tuple) != n_features:
        raise ValueError("scale must have length n_features")
    components = tuple(tuple(1.0 if j == i else 0.0 for j in range(n_features)) for i in range(nc))
    explained_variance_ratio = tuple(1.0 / nc for _ in range(nc))
    return PCAArtifact(
        artifact_version=ARTIFACT_SCHEMA_VERSION,
        family_version=fv,
        n_components=nc,
        feature_names=feature_names,
        mean=mean,
        scale=scale_tuple,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        fit_metadata={"source": "test"},
    )


def _make_panel(
    *,
    n_features: int = 5,
    n_rows: int = 4,
    instruments: tuple[str, ...] = ("AAPL", "MSFT"),
) -> pd.DataFrame:
    """Make a deterministic panel that aligns with ``_identity_artifact``."""
    rows = []
    for inst_idx, inst in enumerate(instruments):
        for r in range(n_rows):
            row = {
                "instrument_id": inst,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r),
            }
            for j in range(n_features):
                # Distinct value per (instrument, row, feature) so the
                # projection result is informative.
                row[f"feature_{j}"] = float(inst_idx * 100 + r * 10 + j)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


def test_artifact_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValueError, match="artifact_version must equal"):
        PCAArtifact(
            artifact_version="something-else",
            family_version=FEATURE_SET_VERSION,
            n_components=2,
            feature_names=("a", "b"),
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
            components=((1.0, 0.0), (0.0, 1.0)),
            explained_variance_ratio=(0.6, 0.4),
        )


def test_artifact_rejects_mismatched_component_row_length() -> None:
    with pytest.raises(ValueError, match="components row 0 must have length"):
        PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version=FEATURE_SET_VERSION,
            n_components=1,
            feature_names=("a", "b", "c"),
            mean=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            components=((1.0, 0.0),),  # only 2 entries, should be 3
            explained_variance_ratio=(1.0,),
        )


def test_artifact_rejects_duplicate_feature_names() -> None:
    with pytest.raises(ValueError, match="feature_names must be unique"):
        PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version=FEATURE_SET_VERSION,
            n_components=1,
            feature_names=("a", "a"),
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
            components=((1.0, 0.0),),
            explained_variance_ratio=(1.0,),
        )


def test_artifact_rejects_zero_n_components() -> None:
    with pytest.raises(ValueError, match="n_components must be > 0"):
        PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version=FEATURE_SET_VERSION,
            n_components=0,
            feature_names=("a",),
            mean=(0.0,),
            scale=(1.0,),
            components=(),
            explained_variance_ratio=(),
        )


def test_artifact_rejects_mismatched_scale_length() -> None:
    with pytest.raises(ValueError, match="scale must have length 3"):
        PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version=FEATURE_SET_VERSION,
            n_components=1,
            feature_names=("a", "b", "c"),
            mean=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0),  # only 2 entries, should be 3
            components=((1.0, 0.0, 0.0),),
            explained_variance_ratio=(1.0,),
        )


def test_artifact_rejects_non_positive_scale_entry() -> None:
    with pytest.raises(ValueError, match=r"scale\[1\] \(feature 'b'\) must be > 0"):
        PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version=FEATURE_SET_VERSION,
            n_components=1,
            feature_names=("a", "b"),
            mean=(0.0, 0.0),
            scale=(1.0, 0.0),  # zero scale on column 'b' would divide-by-zero
            components=((1.0, 0.0),),
            explained_variance_ratio=(1.0,),
        )


def test_artifact_round_trips_to_dict_and_back() -> None:
    original = _identity_artifact(n_features=4, n_components=2)
    payload = original.to_dict()
    rebuilt = PCAArtifact.from_dict(payload)
    assert rebuilt == original


def test_artifact_components_as_numpy_returns_2d_array() -> None:
    artifact = _identity_artifact(n_features=4, n_components=2)
    arr = artifact.components_as_numpy()
    assert arr.shape == (2, 4)
    assert arr.dtype == float


# ---------------------------------------------------------------------------
# Loader (disk round-trip)
# ---------------------------------------------------------------------------


def test_loader_round_trip_preserves_artifact(tmp_path: Path) -> None:
    original = _identity_artifact(n_features=4, n_components=2)
    path = tmp_path / "subdir" / "pca.json"
    save_pca_artifact(original, path)
    assert path.exists()
    rebuilt = load_pca_artifact(path)
    assert rebuilt == original


def test_loader_rejects_payload_with_wrong_schema_version(tmp_path: Path) -> None:
    # Hand-write a JSON file with a stale artifact_version.
    import json

    bad_payload = {
        "artifact_version": "pca-artifact-v0",
        "family_version": FEATURE_SET_VERSION,
        "n_components": 1,
        "feature_names": ["a"],
        "mean": [0.0],
        "components": [[1.0]],
        "explained_variance_ratio": [1.0],
        "fit_metadata": {},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported artifact_version"):
        load_pca_artifact(path)


# ---------------------------------------------------------------------------
# Catalogue + manifest
# ---------------------------------------------------------------------------


def test_feature_specs_total_nine() -> None:
    assert len(FEATURE_SPECS) == 9  # 8 PCs + 1 reconstruction error
    assert len(FEATURE_NAMES) == 9
    assert len(set(FEATURE_NAMES)) == 9


def test_feature_set_version_is_v1() -> None:
    assert FEATURE_SET_VERSION == "learned-representations-v1"


def test_feature_specs_carry_version_and_family() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == FEATURE_SET_VERSION
        assert spec.family == "learned"


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_names_include_expected_set() -> None:
    expected = {f"learned_pc_{i}" for i in range(1, DEFAULT_N_COMPONENTS + 1)}
    expected.add("learned_reconstruction_error")
    assert set(FEATURE_NAMES) == expected


def test_manifest_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    assert registry.has_family("learned", FEATURE_SET_VERSION)
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)


def test_manifest_contract_holds() -> None:
    assert MANIFEST.name == "learned"
    assert MANIFEST.version == FEATURE_SET_VERSION
    assert set(MANIFEST.feature_names) == set(FEATURE_NAMES)
    assert MANIFEST.key_columns == ("instrument_id", "date")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_rejects_zero_n_components() -> None:
    with pytest.raises(ValueError, match="expected_n_components must be > 0"):
        LearnedConfig(expected_n_components=0)


def test_default_config_uses_v1_defaults() -> None:
    assert DEFAULT_CONFIG.version == "learned-representations-v1"
    assert DEFAULT_CONFIG.expected_n_components == 8


# ---------------------------------------------------------------------------
# Compute — deterministic transform, NO fitting
# ---------------------------------------------------------------------------


def test_compute_rejects_artifact_family_version_mismatch() -> None:
    """An artifact trained for v2 must NOT be applied under v1."""
    artifact = _identity_artifact(
        n_features=10, n_components=8, family_version="learned-representations-v2"
    )
    panel = _make_panel(n_features=10)
    with pytest.raises(ValueError, match="family_version=.* does not match"):
        compute_learned_features(panel=panel, artifact=artifact)


def test_compute_rejects_artifact_with_wrong_n_components() -> None:
    """The v1 family emits exactly DEFAULT_N_COMPONENTS PC columns; an
    artifact with a different n_components is rejected."""
    artifact = _identity_artifact(n_features=10, n_components=4)
    panel = _make_panel(n_features=10)
    with pytest.raises(ValueError, match="n_components=.* does not match"):
        compute_learned_features(panel=panel, artifact=artifact)


def test_compute_rejects_panel_missing_source_features() -> None:
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=5)  # missing feature_5..feature_9
    with pytest.raises(ValueError, match="missing source feature columns"):
        compute_learned_features(panel=panel, artifact=artifact)


def test_compute_does_not_fit_during_compute_path() -> None:
    """Smoke test: the compute path must not import sklearn. If it
    accidentally did, ``import sklearn`` would succeed in the test
    process and the family would silently grow a runtime dependency.
    We assert sklearn is NOT in sys.modules before compute, then
    again after — proving the compute path doesn't trigger a fit."""
    import sys

    sklearn_was_loaded_before = "sklearn" in sys.modules

    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    _ = compute_learned_features(panel=panel, artifact=artifact)

    sklearn_loaded_after = "sklearn" in sys.modules
    # The compute path must not introduce sklearn as a dependency.
    # If sklearn was already loaded (e.g. another test imported it),
    # we just verify compute didn't add it (no-op). If it wasn't
    # loaded, compute must not load it.
    assert sklearn_loaded_after == sklearn_was_loaded_before


def test_compute_identity_artifact_pc_scores_equal_centered_features() -> None:
    """With an identity-basis artifact (components = I, mean = 0),
    learned_pc_i == feature_i for every row."""
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    ff = compute_learned_features(panel=panel, artifact=artifact)
    # learned_pc_1 should equal feature_0 (1-indexed in feature names,
    # 0-indexed in the identity components).
    for i in range(1, 9):
        np.testing.assert_array_equal(
            ff.frame[f"learned_pc_{i}"].to_numpy(),
            panel[f"feature_{i - 1}"].to_numpy(),
        )


def test_compute_identity_artifact_reconstruction_perfect_for_full_rank() -> None:
    """If n_components == n_features, identity PCA reconstructs the
    input perfectly → reconstruction error is zero on every row."""
    artifact = _identity_artifact(n_features=8, n_components=8)
    panel = _make_panel(n_features=8)
    ff = compute_learned_features(panel=panel, artifact=artifact)
    np.testing.assert_allclose(ff.frame["learned_reconstruction_error"].to_numpy(), 0.0, atol=1e-12)


def test_compute_reconstruction_error_nonzero_when_rank_deficient() -> None:
    """If n_components < n_features, the artifact's principal subspace
    doesn't span the full feature space → reconstruction error > 0
    on rows where the orthogonal complement is non-trivial."""
    # 10 features, 8 PCs → reconstruction error captures feature_8 and
    # feature_9 (which the identity-8 projection drops).
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    ff = compute_learned_features(panel=panel, artifact=artifact)
    # The reconstruction residual on each row is sqrt(feature_8^2 +
    # feature_9^2). Let's check a few rows:
    for i in range(len(panel)):
        f8 = panel.iloc[i]["feature_8"]
        f9 = panel.iloc[i]["feature_9"]
        expected = float(np.sqrt(f8**2 + f9**2))
        assert ff.frame.iloc[i]["learned_reconstruction_error"] == pytest.approx(
            expected, abs=1e-10
        )


def test_compute_propagates_nan_per_row() -> None:
    """If any source feature is NaN on a row, all 9 output features
    on that row are NaN — conservative PIT-safe propagation."""
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    # Inject a NaN on row 1's feature_3.
    panel.loc[1, "feature_3"] = np.nan
    ff = compute_learned_features(panel=panel, artifact=artifact)
    # Row 1 should have NaN on every output feature.
    nan_row = ff.frame.iloc[1]
    for name in FEATURE_NAMES:
        assert pd.isna(nan_row[name]), name
    # Adjacent rows should be unaffected.
    intact_row = ff.frame.iloc[0]
    assert pd.notna(intact_row["learned_pc_1"])
    assert pd.notna(intact_row["learned_reconstruction_error"])


def test_compute_empty_panel_returns_empty_frame() -> None:
    artifact = _identity_artifact(n_features=10, n_components=8)
    empty = pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "date": pd.Series(dtype="datetime64[ns]"),
            **{f"feature_{i}": pd.Series(dtype=float) for i in range(10)},
        }
    )
    ff = compute_learned_features(panel=empty, artifact=artifact)
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_compute_is_deterministic() -> None:
    """Same artifact × same panel = same output, byte-for-byte. No
    randomness anywhere."""
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    ff1 = compute_learned_features(panel=panel, artifact=artifact)
    ff2 = compute_learned_features(panel=panel, artifact=artifact)
    pd.testing.assert_frame_equal(ff1.frame, ff2.frame)


def test_compute_handles_extra_panel_columns_gracefully() -> None:
    """Extra panel columns beyond the artifact's feature_names are
    ignored. The operator can pass a richer panel than the
    artifact's training set."""
    artifact = _identity_artifact(n_features=8, n_components=8)
    panel = _make_panel(n_features=12)  # 12 features, artifact only needs 8
    ff = compute_learned_features(panel=panel, artifact=artifact)
    assert len(ff.frame) == len(panel)


def test_compute_multi_instrument_panels_keep_independent_rows() -> None:
    """Two instruments in the panel produce independent rows; no
    cross-contamination."""
    artifact = _identity_artifact(n_features=8, n_components=8)
    panel = _make_panel(n_features=8, instruments=("AAPL", "MSFT", "GOOG"))
    ff = compute_learned_features(panel=panel, artifact=artifact)
    # 3 instruments × 4 rows each = 12 rows.
    assert len(ff.frame) == 12
    # The first row of each instrument's slice should reflect the
    # instrument's distinct fixture values (recall: row_value =
    # inst_idx * 100 + r * 10 + feature_idx; feature_0 of row 0 is
    # inst_idx*100).
    aapl_first = ff.frame[ff.frame["instrument_id"] == "AAPL"].iloc[0]
    msft_first = ff.frame[ff.frame["instrument_id"] == "MSFT"].iloc[0]
    googl_first = ff.frame[ff.frame["instrument_id"] == "GOOG"].iloc[0]
    assert aapl_first["learned_pc_1"] == 0.0
    assert msft_first["learned_pc_1"] == 100.0
    assert googl_first["learned_pc_1"] == 200.0


def test_compute_replaces_infs_with_nan() -> None:
    artifact = _identity_artifact(n_features=10, n_components=8)
    panel = _make_panel(n_features=10)
    ff = compute_learned_features(panel=panel, artifact=artifact)
    for name in FEATURE_NAMES:
        column = ff.frame[name]
        assert not np.isinf(column.dropna()).any(), name


# ---------------------------------------------------------------------------
# Trainer (lazy sklearn) — only smoke-test that the family doesn't
# import it eagerly. We don't fit-test here; that's research-side.
# ---------------------------------------------------------------------------


def test_trainer_lazy_imports_sklearn() -> None:
    """The family's main API surface (compute, config, artifact,
    loader) must not import sklearn. Only the trainer does, and only
    when called.

    Note: this in-process variant only fires reliably when sklearn
    wasn't already loaded by another test in the same session.
    :func:`test_family_main_api_does_not_load_sklearn_cold_start`
    below is the unconditional version — it spawns a fresh Python
    process so the assertion holds regardless of test ordering.
    """
    import sys

    sklearn_initially_present = "sklearn" in sys.modules

    # Import the family's public surface in isolation.
    import importlib

    importlib.import_module("quant_platform.research.features.learned.artifact")
    importlib.import_module("quant_platform.research.features.learned.config")
    importlib.import_module("quant_platform.research.features.learned.features")
    importlib.import_module("quant_platform.research.features.learned.loader")

    sklearn_after_family_import = "sklearn" in sys.modules

    # The family's main API surface must NOT bring sklearn in.
    if not sklearn_initially_present:
        assert not sklearn_after_family_import, (
            "Importing the learned family's main API surface must not load "
            "sklearn. Only the trainer should."
        )


def test_family_main_api_does_not_load_sklearn_cold_start() -> None:
    """Cold-start invariant: in a fresh Python process, importing the
    family's main API surface AND running a full compute path does
    NOT load sklearn. This is the unconditional version of the
    in-process tests above — it bypasses sys.modules pollution from
    other tests by spawning a subprocess.

    The script in the subprocess:
      1. Asserts sklearn is not loaded after main-API imports.
      2. Runs compute_learned_features end-to-end with a synthetic
         identity artifact.
      3. Asserts sklearn is STILL not loaded after compute.
      4. Exits with status 0 on success.

    Subprocess isolation is heavyweight but is the only test that
    survives test-ordering reshuffles + parallel pytest workers.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys

        # 1. Importing the family's main API must NOT load sklearn.
        from quant_platform.research.features.learned import (
            ARTIFACT_SCHEMA_VERSION,
            DEFAULT_CONFIG,
            PCAArtifact,
            compute_learned_features,
            load_pca_artifact,
            save_pca_artifact,
        )

        assert "sklearn" not in sys.modules, (
            "Importing the learned family's main API loaded sklearn. "
            "Only the trainer should. Modules: "
            + str([m for m in sys.modules if m.startswith("sklearn")])
        )

        # 2. Build a synthetic identity artifact + panel and compute features.
        import pandas as pd

        feature_names = tuple(f"f_{i}" for i in range(10))
        artifact = PCAArtifact(
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            family_version="learned-representations-v1",
            n_components=8,
            feature_names=feature_names,
            mean=tuple(0.0 for _ in range(10)),
            scale=tuple(1.0 for _ in range(10)),
            components=tuple(
                tuple(1.0 if j == i else 0.0 for j in range(10))
                for i in range(8)
            ),
            explained_variance_ratio=tuple(1.0 / 8 for _ in range(8)),
            fit_metadata={"source": "subprocess-test"},
        )
        panel = pd.DataFrame(
            [
                {
                    "instrument_id": "AAPL",
                    "date": pd.Timestamp("2024-01-01"),
                    **{f"f_{i}": float(i) for i in range(10)},
                }
            ]
        )
        ff = compute_learned_features(panel=panel, artifact=artifact)
        assert len(ff.frame) == 1

        # 3. After running compute, sklearn STILL must not be loaded.
        assert "sklearn" not in sys.modules, (
            "compute_learned_features loaded sklearn. The compute path "
            "must stay sklearn-free. Modules: "
            + str([m for m in sys.modules if m.startswith("sklearn")])
        )

        # 4. Reaching here = success.
        print("OK")
        """
    )

    result = subprocess.run(  # noqa: S603 - controlled subprocess invocation, test-only
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"Cold-start sklearn invariant failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Catalogue-vs-config consistency invariants (review findings A6, C9)
# ---------------------------------------------------------------------------


def test_feature_specs_count_matches_default_n_components_plus_one() -> None:
    """**Drift invariant**: the family ships exactly
    ``DEFAULT_CONFIG.expected_n_components + 1`` features
    (N PCs + 1 reconstruction error). A future contributor who bumps
    ``DEFAULT_N_COMPONENTS`` without updating the spec builder (or
    vice versa) breaks this — and the test catches it.
    """
    assert len(FEATURE_SPECS) == DEFAULT_CONFIG.expected_n_components + 1, (
        f"FEATURE_SPECS count ({len(FEATURE_SPECS)}) must equal "
        f"DEFAULT_CONFIG.expected_n_components ({DEFAULT_CONFIG.expected_n_components}) "
        "+ 1 (for learned_reconstruction_error). One of the two has drifted."
    )


def test_specs_for_config_returns_default_specs_when_config_matches() -> None:
    """``_specs_for_config`` short-circuits to the cached
    ``FEATURE_SPECS`` tuple when the config matches the default —
    callers should get the exact same object identity, not a freshly-
    built copy."""
    from quant_platform.research.features.learned.features import _specs_for_config

    specs = _specs_for_config(DEFAULT_CONFIG)
    assert specs is FEATURE_SPECS


def test_specs_for_config_rebuilds_when_n_components_differs() -> None:
    """A non-default ``expected_n_components`` triggers a fresh spec
    rebuild with the new count. The rebuilt specs carry the
    config's version string and the new count of PC features."""
    from quant_platform.research.features.learned.features import _specs_for_config

    custom = LearnedConfig(
        version="learned-representations-experiment-1",
        expected_n_components=4,
    )
    specs = _specs_for_config(custom)
    assert specs is not FEATURE_SPECS  # genuinely rebuilt
    # 4 PCs + 1 reconstruction error.
    assert len(specs) == 5
    # All carry the custom version.
    for spec in specs:
        assert spec.version == "learned-representations-experiment-1"


# ---------------------------------------------------------------------------
# Trainer registry-validation opt-in (review finding C3/A2)
# ---------------------------------------------------------------------------


def test_trainer_validates_feature_names_against_registry_when_opted_in() -> None:
    """With ``validate_against_registry=True``, a feature name not in
    the global registry is rejected at training time — catches typos
    and stale references before they hit the artifact."""
    # Skip cleanly if sklearn isn't installed; the trainer's lazy
    # import raises ImportError before validation runs.
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "definitely_not_a_registered_feature_xyz": np.random.default_rng(42).normal(size=10),
        }
    )
    with pytest.raises(ValueError, match="not present in the global"):
        fit_pca_artifact(
            panel=panel,
            feature_names=("definitely_not_a_registered_feature_xyz",),
            n_components=1,
            family_version="learned-representations-v1",
            validate_against_registry=True,
        )


def test_trainer_skips_registry_validation_when_disabled() -> None:
    """The default (``validate_against_registry=False``) accepts any
    feature_names — operators with hand-curated panels not tied to
    the registry shouldn't need to register synthetic columns."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    rng = np.random.default_rng(42)
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * 50,
            "date": pd.date_range("2024-01-01", periods=50),
            "synthetic_a": rng.normal(size=50),
            "synthetic_b": rng.normal(size=50),
        }
    )
    # Should not raise even though "synthetic_a" / "synthetic_b" aren't
    # in the global registry.
    artifact = fit_pca_artifact(
        panel=panel,
        feature_names=("synthetic_a", "synthetic_b"),
        n_components=2,
        family_version="learned-representations-v1",
        validate_against_registry=False,
    )
    assert artifact.n_components == 2


def test_trainer_drops_inf_rows_under_drop_nan_rows() -> None:
    """``drop_nan_rows=True`` (the default) coerces ±inf to NaN and
    drops those rows. sklearn rejects ±inf with a hard error before
    fit, and upstream feature panels occasionally carry ±inf from
    safe_div edge cases on zero / near-zero denominators — the
    trainer now closes that gap so callers don't have to remember
    to sanitize."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    rng = np.random.default_rng(42)
    n_rows = 50
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    # Inject one +inf and one -inf into otherwise-clean columns.
    a[3] = np.inf
    b[17] = -np.inf
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_rows,
            "date": pd.date_range("2024-01-01", periods=n_rows),
            "synthetic_a": a,
            "synthetic_b": b,
        }
    )
    artifact = fit_pca_artifact(
        panel=panel,
        feature_names=("synthetic_a", "synthetic_b"),
        n_components=2,
        family_version="learned-representations-v1",
        validate_against_registry=False,
    )
    # 50 rows minus 2 inf-bearing rows = 48 fit samples.
    assert artifact.fit_metadata["n_samples_fit"] == "48"


def test_trainer_drops_mixed_nan_and_inf_rows() -> None:
    """A panel with NaN and ±inf in different rows produces the
    correct survivor count — both are stripped by the same dropna
    step under ``drop_nan_rows=True``."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    rng = np.random.default_rng(42)
    n_rows = 60
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    a[1] = np.nan
    a[2] = np.inf
    b[3] = -np.inf
    b[4] = np.nan
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n_rows,
            "date": pd.date_range("2024-01-01", periods=n_rows),
            "synthetic_a": a,
            "synthetic_b": b,
        }
    )
    artifact = fit_pca_artifact(
        panel=panel,
        feature_names=("synthetic_a", "synthetic_b"),
        n_components=2,
        family_version="learned-representations-v1",
        validate_against_registry=False,
    )
    # 60 - 4 distinct bad rows = 56 survivors.
    assert artifact.fit_metadata["n_samples_fit"] == "56"


# ---------------------------------------------------------------------------
# v2 schema: standardisation + scale field
# (added 2026-05-26 after the universe-300 backtest revealed the v1
# trainer collapsed EVR onto PC1 on the mixed-scale source panel)
# ---------------------------------------------------------------------------


def test_trainer_default_standardises_mixed_scale_panel() -> None:
    """Trainer default (``standardise=True``) divides per-feature
    std-dev before the PCA fit, so the EVR distribution reflects
    *relative* variance directions rather than the absolute scale of
    the highest-magnitude column.

    Without standardisation, a column scaled by 1e6 next to a column
    scaled by 1e-2 makes PCA project the entire variance budget onto
    PC1; PC2 carries effectively zero EVR. With standardisation, the
    two synthetic columns contribute comparable variance and PC2
    carries a meaningful share."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    rng = np.random.default_rng(0)
    n = 500
    # Two columns of comparable RELATIVE variance, but wildly
    # different absolute scale. After standardisation they look
    # near-equal to PCA; without it PC1 just is the big-scale column.
    raw_a = rng.normal(size=n)
    raw_b = rng.normal(size=n)
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n,
            "date": pd.date_range("2024-01-01", periods=n),
            "scaled_big": raw_a * 1e6,
            "scaled_small": raw_b * 1e-2,
        }
    )
    artifact = fit_pca_artifact(
        panel=panel,
        feature_names=("scaled_big", "scaled_small"),
        n_components=2,
        family_version="learned-representations-v1",
    )
    assert artifact.explained_variance_ratio[1] > 0.001, (
        "PC2 carries effectively zero EVR — standardisation didn't run "
        f"(EVR={artifact.explained_variance_ratio})"
    )
    # And the scale matches the raw std-devs (within sampling tolerance).
    assert artifact.scale[0] == pytest.approx(np.std(raw_a * 1e6), rel=0.05)
    assert artifact.scale[1] == pytest.approx(np.std(raw_b * 1e-2), rel=0.05)


def test_trainer_standardise_false_skips_scaling() -> None:
    """With ``standardise=False`` the artifact's ``scale`` is written
    as a tuple of ``1.0`` and the compute path's
    ``(matrix - mean) / scale`` reduces to ``matrix - mean`` —
    equivalent to the pre-v2 centre-only transform."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    rng = np.random.default_rng(0)
    n = 200
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n,
            "date": pd.date_range("2024-01-01", periods=n),
            "scaled_big": rng.normal(size=n) * 1e6,
            "scaled_small": rng.normal(size=n) * 1e-2,
        }
    )
    artifact = fit_pca_artifact(
        panel=panel,
        feature_names=("scaled_big", "scaled_small"),
        n_components=2,
        family_version="learned-representations-v1",
        standardise=False,
    )
    assert artifact.scale == (1.0, 1.0)
    assert artifact.fit_metadata["standardised"] == "False"


def test_trainer_rejects_constant_column() -> None:
    """A constant column has zero std-dev; the trainer must reject the
    fit rather than divide-by-zero inside the StandardScaler. The
    error message names the offending column so the operator can drop
    it without debugging the math."""
    pytest.importorskip("sklearn")
    from quant_platform.research.features.learned.trainer import fit_pca_artifact

    n = 50
    panel = pd.DataFrame(
        {
            "instrument_id": ["AAPL"] * n,
            "date": pd.date_range("2024-01-01", periods=n),
            "varying": np.linspace(-1.0, 1.0, n),
            "constant_col": np.full(n, 3.14),
        }
    )
    with pytest.raises(ValueError, match="constant_col"):
        fit_pca_artifact(
            panel=panel,
            feature_names=("varying", "constant_col"),
            n_components=2,
            family_version="learned-representations-v1",
        )


def test_artifact_schema_version_v2_round_trips(tmp_path: Path) -> None:
    """Schema bumped to v2 to admit the ``scale`` field. An artifact
    with a non-trivial scale round-trips through to_dict/from_dict
    and through the on-disk loader without losing the scale tuple."""
    assert ARTIFACT_SCHEMA_VERSION == "pca-artifact-v2"
    artifact = PCAArtifact(
        artifact_version=ARTIFACT_SCHEMA_VERSION,
        family_version=FEATURE_SET_VERSION,
        n_components=2,
        feature_names=("a", "b", "c"),
        mean=(0.1, 0.2, 0.3),
        scale=(2.0, 4.0, 8.0),
        components=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        explained_variance_ratio=(0.6, 0.4),
        fit_metadata={"source": "v2-roundtrip"},
    )
    # In-memory round-trip.
    rebuilt = PCAArtifact.from_dict(artifact.to_dict())
    assert rebuilt == artifact
    assert rebuilt.scale == (2.0, 4.0, 8.0)
    # Disk round-trip.
    path = tmp_path / "v2.json"
    save_pca_artifact(artifact, path)
    loaded = load_pca_artifact(path)
    assert loaded == artifact


def test_loader_rejects_v1_artifact(tmp_path: Path) -> None:
    """A persisted ``pca-artifact-v1`` payload (no ``scale`` field)
    must be cleanly rejected — the compute path would otherwise have
    no scale to apply and produce silently wrong results."""
    import json

    v1_payload = {
        "artifact_version": "pca-artifact-v1",
        "family_version": FEATURE_SET_VERSION,
        "n_components": 1,
        "feature_names": ["a"],
        "mean": [0.0],
        "components": [[1.0]],
        "explained_variance_ratio": [1.0],
        "fit_metadata": {},
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported artifact_version"):
        load_pca_artifact(path)


def test_compute_path_applies_scaling() -> None:
    """The compute path standardises before projecting: with an
    identity-PCA artifact and a known ``(mean, scale)`` pair, the PC
    scores equal ``(input - mean) / scale`` projected through I.

    Hand-checked example:
        input row     = [10.0, 8.0, 6.0, 4.0]
        mean          = [ 0.0, 0.0, 0.0, 0.0]
        scale         = [ 2.0, 4.0, 6.0, 8.0]
        standardised  = [ 5.0, 2.0, 1.0, 0.5]
        identity-PCA  → pc_i == standardised_i
    """
    artifact = _identity_artifact(n_features=4, n_components=4, scale=(2.0, 4.0, 6.0, 8.0))
    panel = pd.DataFrame(
        [
            {
                "instrument_id": "AAPL",
                "date": pd.Timestamp("2024-01-01"),
                "feature_0": 10.0,
                "feature_1": 8.0,
                "feature_2": 6.0,
                "feature_3": 4.0,
            }
        ]
    )
    config = LearnedConfig(expected_n_components=4)
    ff = compute_learned_features(panel=panel, artifact=artifact, config=config)
    row = ff.frame.iloc[0]
    assert row["learned_pc_1"] == pytest.approx(5.0)
    assert row["learned_pc_2"] == pytest.approx(2.0)
    assert row["learned_pc_3"] == pytest.approx(1.0)
    assert row["learned_pc_4"] == pytest.approx(0.5)
    # Full-rank identity reconstruction is perfect → error is 0.
    assert row["learned_reconstruction_error"] == pytest.approx(0.0, abs=1e-12)
