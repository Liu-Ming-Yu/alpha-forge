"""Fit + persist the production learned-PCA artifact for the live D arm (ADR-012).

Arm D projects the 36 pv+formulaic features through an 8-component PCA artifact
(+ a reconstruction-error feature) fit on a warmup window. The backtest fits this
per-run; for live inference the artifact must be a **persisted, versioned**
file the engine loads (never re-fit live) so replay is deterministic.

This operator job fits the artifact on all available history for the configured
universe (the live warmup) and writes:

* ``pca_d_artifact.json`` — the ``PCAArtifact`` (schema ``pca-artifact-v2``),
  JSON via ``loader.save_pca_artifact``.
* ``pca_d_manifest.json`` — provenance the live loader checks: artifact + family
  versions, the 36-name source-feature contract, the 9 learned output names, the
  fit window, universe, bar fingerprint, git commit, and timestamp.

Re-run quarterly (or on a feature-set-version bump) to mint a new artifact
version; the live engine pins one version for deterministic replay.

Usage::

    python scripts/build_live_pca_artifact.py
    python scripts/build_live_pca_artifact.py --universe infra/config/universe_300.json \\
        --out-dir data/parquet/research/live_artifacts --as-of 2026-05-29
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_platform.research.features.learned.artifact import ARTIFACT_SCHEMA_VERSION
from quant_platform.research.features.learned.config import (
    DEFAULT_CONFIG as LEARNED_CONFIG,
)
from quant_platform.research.features.learned.config import (
    FEATURE_SET_VERSION as LEARNED_FEATURE_SET_VERSION,
)
from quant_platform.research.features.learned.features import compute_learned_features
from quant_platform.research.features.learned.loader import load_pca_artifact, save_pca_artifact

# ``scripts/`` is not an installed package; add the project root so the backtest
# feature helpers import. The artifact must be fit on the SAME pv+formulaic
# compute the backtest uses — reuse its functions, never a divergent copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_latest_stack import (  # noqa: E402
    FORMULAIC_FEATURE_SET_VERSION,
    PV_FEATURE_SET_VERSION,
    compute_formulaic_alphas,
    compute_pv_features,
    fit_warmup_pca_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAR_ROOT = PROJECT_ROOT / "data" / "parquet" / "bars"
DEFAULT_UNIVERSE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
# Tracked location (not data/, which is gitignored): the live engine loads a
# versioned artifact, so it must travel with the repo for deterministic deploy.
DEFAULT_OUT_DIR = PROJECT_ROOT / "infra" / "artifacts" / "learned_pca"


def _load_universe_bars(universe_path: Path) -> pd.DataFrame:
    """Load every configured instrument's daily bars into one OHLCV frame."""
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    instrument_ids = list(universe)
    frames: list[pd.DataFrame] = []
    for instrument_id in instrument_ids:
        files = sorted(glob.glob(str(BAR_ROOT / instrument_id / "daily" / "*.parquet")))
        if not files:
            continue
        for file in files:
            df = pd.read_parquet(file)
            df = df[df["bar_seconds"] == 86400]
            if df.empty:
                continue
            df = df.copy()
            df["instrument_id"] = instrument_id
            df["date"] = (
                pd.to_datetime(df["timestamp"], utc=True)
                .dt.tz_convert("UTC")
                .dt.normalize()
                .dt.tz_localize(None)
            )
            frames.append(df[["instrument_id", "date", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise SystemExit(f"no daily bars found under {BAR_ROOT} for {universe_path}")
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.sort_values(["instrument_id", "date"]).drop_duplicates(
        subset=["instrument_id", "date"]
    )
    return bars.reset_index(drop=True)


def _git_commit() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build(*, universe_path: Path, out_dir: Path, as_of: str | None) -> tuple[Path, Path]:
    """Fit the production PCA artifact + write artifact and manifest. Returns their paths."""
    print(f"[1] Loading universe bars from {universe_path} ...")
    bars = _load_universe_bars(universe_path)
    print(f"    {len(bars):,} bar rows, {bars['instrument_id'].nunique()} instruments")

    print("[2] Computing pv + formulaic features ...")
    pv_df, pv_names = compute_pv_features(bars)
    form_df, form_names = compute_formulaic_alphas(bars)
    pv_form = pv_df.merge(form_df, on=["instrument_id", "date"], how="inner")
    source_names = list(pv_names) + list(form_names)
    print(f"    pv+formulaic: {len(pv_form):,} rows × {len(source_names)} features")

    warmup_end = pd.Timestamp(as_of) if as_of else pv_form["date"].max()
    print(f"[3] Fitting PCA artifact (warmup_end={warmup_end.date()}, all history) ...")
    artifact = fit_warmup_pca_artifact(pv_form, source_names, warmup_end)

    # Reproducibility self-check: the persisted artifact must reproduce the transform.
    learned = compute_learned_features(panel=pv_form, artifact=artifact, config=LEARNED_CONFIG)
    learned_names = tuple(c for c in learned.frame.columns if c not in ("instrument_id", "date"))

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "pca_d_artifact.json"
    save_pca_artifact(artifact, artifact_path)
    reloaded = load_pca_artifact(artifact_path)
    if reloaded.to_dict() != artifact.to_dict():
        raise SystemExit(
            "artifact round-trip mismatch — refusing to write a non-reproducible artifact"
        )

    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(bars[["instrument_id", "date", "close"]], index=False).values
    ).hexdigest()
    manifest = {
        "artifact_file": artifact_path.name,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "learned_feature_set_version": LEARNED_FEATURE_SET_VERSION,
        "source_feature_set_versions": {
            "price_volume": PV_FEATURE_SET_VERSION,
            "formulaic": FORMULAIC_FEATURE_SET_VERSION,
        },
        "source_feature_names": list(source_names),
        "learned_feature_names": list(learned_names),
        "n_components": artifact.n_components,
        "fit_window": {
            "start": pv_form["date"].min().date().isoformat(),
            "end": warmup_end.date().isoformat(),
        },
        "universe_file": universe_path.relative_to(PROJECT_ROOT).as_posix(),
        "n_instruments": int(bars["instrument_id"].nunique()),
        "bars_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = out_dir / "pca_d_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[4] Wrote artifact → {artifact_path}")
    print(f"    Wrote manifest → {manifest_path}")
    print(
        f"    n_components={artifact.n_components} | source_features={len(source_names)} | "
        f"learned_features={len(learned_names)} | schema={ARTIFACT_SCHEMA_VERSION}"
    )
    return artifact_path, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Warmup end date (ISO). Default: the latest available bar date.",
    )
    args = parser.parse_args(argv)
    build(universe_path=args.universe, out_dir=args.out_dir, as_of=args.as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
