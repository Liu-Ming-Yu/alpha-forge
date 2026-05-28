"""``text-event-v2`` feature family.

LLM-extracted structured features from three document types — news
articles, SEC filings (10-K / 10-Q / 8-K), and earnings-call
transcripts — totalling 27 features.

The family follows the platform-wide contract:

* :data:`FEATURE_SPECS` declares the catalog at module-import time.
* :data:`MANIFEST` is registered into the process-global
  :class:`FamilyRegistry`.
* :func:`compute_text_features` consumes :class:`ExtractedRecord`
  iterables + their :class:`SourceDocument` originals + an optional
  trading-day calendar, and returns a :class:`FeatureFrame`.

The extraction pipeline itself
(:mod:`.client`, :mod:`.extraction`) is intentionally decoupled
from the feature-compute path: the operator runs extraction
offline (mocked or real LLM), persists records to JSONL via
:mod:`.storage`, then the family's compute reads from that file.
This makes feature-compute deterministic and re-runnable without
hitting the LLM provider each time.

How to add a feature
--------------------

1. Add a column to the relevant extraction dataclass
   (:class:`NewsExtraction`, :class:`FilingExtraction`, or
   :class:`EarningsCallExtraction`) — bump :data:`SCHEMA_VERSION`
   if existing extraction payloads need to be re-run.
2. Update the dataclass's ``field_descriptions`` classmethod so the
   prompt template picks it up.
3. Add the new metric to the matching builder in :mod:`.aggregator`
   and a :class:`FeatureSpec` to this package's ``_build_specs``.
   Add a test in ``tests/unit/research_service/features/text/``.
4. Direction stays ``"unknown"`` until walk-forward evidence
   accumulates.
"""

from __future__ import annotations

from quant_platform.research.features.contracts import FamilyManifest
from quant_platform.research.features.registry import register_family
from quant_platform.research.features.text.config import (
    DEFAULT_CONFIG,
    DEFAULT_SENTIMENT_WINDOW,
    DEFAULT_TONE_CHANGE_WINDOW,
    DEFAULT_VOLUME_ZSCORE_WINDOW,
    FEATURE_SET_VERSION,
    TextEventConfig,
)
from quant_platform.research.features.text.features import (
    DEFAULT_TRAINING_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_SPECS,
    REQUIRED_INPUT_COLUMNS,
    compute_text_features,
)
from quant_platform.research.features.text.schemas import (
    SCHEMA_VERSION,
    EarningsCallExtraction,
    ExtractedRecord,
    ExtractionProvenance,
    FailedExtraction,
    FilingExtraction,
    NewsExtraction,
    SourceDocument,
)
from quant_platform.research.features.transforms import DEFAULT_KEY_COLUMNS

MANIFEST: FamilyManifest = FamilyManifest(
    name="text",
    version=FEATURE_SET_VERSION,
    feature_specs=FEATURE_SPECS,
    required_input_columns=REQUIRED_INPUT_COLUMNS,
    key_columns=DEFAULT_KEY_COLUMNS,
    default_training_feature_names=DEFAULT_TRAINING_FEATURE_NAMES,
)

# Side effect: register the family at import time so the global
# registry picks it up before any consumer queries by name.
register_family(MANIFEST)


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SENTIMENT_WINDOW",
    "DEFAULT_TONE_CHANGE_WINDOW",
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "DEFAULT_VOLUME_ZSCORE_WINDOW",
    "FEATURE_NAMES",
    "FEATURE_SET_VERSION",
    "FEATURE_SPECS",
    "MANIFEST",
    "REQUIRED_INPUT_COLUMNS",
    "SCHEMA_VERSION",
    "EarningsCallExtraction",
    "ExtractedRecord",
    "ExtractionProvenance",
    "FailedExtraction",
    "FilingExtraction",
    "NewsExtraction",
    "SourceDocument",
    "TextEventConfig",
    "compute_text_features",
]
