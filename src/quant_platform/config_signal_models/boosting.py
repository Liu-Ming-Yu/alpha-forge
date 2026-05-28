"""Boosted-tree signal model settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoostingSettings(BaseModel):
    """XGBoost boosted-tree signal model settings."""

    enabled: bool = False
    artifact_manifest: str = Field(
        default="",
        description=(
            "Path to an XGBoost manifest.json produced by ``quant-platform boosting train``."
        ),
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description=(
            "XGBoost execution device. ``auto`` tries CUDA and falls back to "
            "CPU unless require_gpu is true."
        ),
    )
    require_gpu: bool = Field(
        default=False,
        description=(
            "Fail boosted-tree training/scoring startup when CUDA is not usable. "
            "Configured via QP__BOOSTING__REQUIRE_GPU."
        ),
    )
    shadow_artifact_root: str = Field(
        default="data/shadow/boosting",
        description="Directory where shadow boosted-score JSONL artifacts are written.",
    )
    random_seed: int = Field(
        default=17,
        description=(
            "Random seed for XGBoost training. Written into the training manifest "
            "for reproducibility audits. Configured via QP__BOOSTING__RANDOM_SEED."
        ),
    )
