"""XGBoost rank-signal model adapter."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quant_platform.core.domain.signals import SignalScore
from quant_platform.services.research_service.boosting.artifacts import (
    BoostingDevice,
    load_manifest,
    sha256_file,
)
from quant_platform.services.research_service.boosting.device import (
    import_xgboost,
    resolve_xgboost_device,
)
from quant_platform.services.research_service.features.cross_section.cross_section import (
    rank_normalize,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from quant_platform.core.domain.research import FeatureVector, StrategyRun


class XGBoostRankSignalModel:
    """XGBoost ranker adapter for the platform ``SignalModel`` protocol."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        device: BoostingDevice = "auto",
        require_gpu: bool = False,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        self._manifest = load_manifest(self._manifest_path)
        if self._manifest.model_type != "xgboost_ranker":
            raise ValueError(f"unsupported boosting model_type: {self._manifest.model_type}")

        self._xgb = import_xgboost()
        self._device = resolve_xgboost_device(device, require_gpu=require_gpu, xgb=self._xgb)
        xgb_any = cast("Any", self._xgb)
        booster_path = Path(self._manifest.booster_path)
        if not booster_path.is_absolute():
            booster_path = self._manifest_path.parent / booster_path
        if not booster_path.is_file():
            raise ValueError(f"boosting booster artifact not found: {booster_path}")
        if self._manifest.booster_sha256:
            actual_hash = sha256_file(booster_path)
            if actual_hash != self._manifest.booster_sha256:
                raise ValueError("boosting booster artifact hash does not match manifest")
        self._booster = xgb_any.Booster()
        self._booster.load_model(str(booster_path))
        self._booster.set_param({"device": self._device})

    @property
    def model_version(self) -> str:
        return self._manifest.model_version

    @property
    def feature_set_version(self) -> str:
        return self._manifest.feature_set_version

    @property
    def feature_names(self) -> list[str]:
        return list(self._manifest.feature_names)

    @property
    def feature_schema_hash(self) -> str:
        return self._manifest.feature_schema_hash

    @property
    def feature_versions(self) -> dict[str, str]:
        return dict(self._manifest.feature_versions)

    @property
    def device(self) -> str:
        return self._device

    def feature_coverage(self, features: Mapping[str, float]) -> float:
        present = sum(1 for name in self._manifest.feature_names if name in features)
        return present / len(self._manifest.feature_names)

    def _row(self, vector: FeatureVector) -> list[float]:
        if vector.feature_set_version != self._manifest.feature_set_version:
            raise ValueError(
                "FeatureVector feature_set_version mismatch: "
                f"{vector.feature_set_version!r} != {self._manifest.feature_set_version!r}"
            )
        row: list[float] = []
        for name in self._manifest.feature_names:
            if name not in vector.features:
                raise ValueError(f"FeatureVector is missing required boosting feature {name!r}")
            value = float(vector.features[name])
            if not math.isfinite(value):
                raise ValueError(f"FeatureVector feature {name!r} is not finite")
            row.append(value)
        return row

    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        if not vectors:
            return []
        rows = [self._row(vec) for vec in vectors]
        xgb_any = cast("Any", self._xgb)
        dmat = xgb_any.DMatrix(
            np.asarray(rows, dtype=float),
            feature_names=self._manifest.feature_names,
        )
        raw = self._booster.predict(dmat)
        ranked = rank_normalize({idx: float(value) for idx, value in enumerate(raw)})
        scores: list[SignalScore] = []
        for idx, vec in enumerate(vectors):
            scores.append(
                SignalScore(
                    score_id=uuid.uuid4(),
                    instrument_id=vec.instrument_id,
                    strategy_run_id=strategy_run.run_id,
                    as_of=vec.as_of,
                    score=float(ranked[idx]),
                    confidence=1.0,
                    model_version=self._manifest.model_version,
                    feature_vector_id=vec.vector_id,
                )
            )
        return scores
