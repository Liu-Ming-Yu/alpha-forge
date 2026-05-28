"""Structured extraction schemas for text-event features.

The brief's Phase 5 rule is non-negotiable: **LLMs produce structured
JSON, never prose recommendations**. Every field this module defines
is a named float (or count) the LLM is allowed to populate; the
downstream pipeline never reads raw LLM prose into model training.

Two kinds of objects live here:

1. **Extraction schemas** — frozen dataclasses describing what a
   model is allowed to emit per source document. Three schemas
   ship in v2 — one per supported document kind:

   * :class:`NewsExtraction` — news articles. 7 fields.
   * :class:`FilingExtraction` — SEC 10-K / 10-Q / 8-K filings.
     10 fields.
   * :class:`EarningsCallExtraction` — earnings-call transcripts.
     7 fields.

   All signed scores live in ``[-1, 1]`` and all unsigned scores in
   ``[0, 1]`` so they compose cleanly through downstream
   aggregation; "no signal" is ``0.0`` (or ``0.5`` for
   :attr:`NewsExtraction.novelty`), not ``None``. Dispatch from
   :attr:`ExtractionProvenance.source_kind` to the right loader
   goes through :func:`extraction_from_dict_for_kind`.

2. **Provenance** — :class:`ExtractionProvenance` carries the
   prompt version, model version, source-document id + kind,
   extraction timestamp, and the LLM's self-reported confidence.
   The brief is explicit that auditors must be able to re-run the
   extraction; this record is what makes that possible.

Stability contract
------------------

Every schema is **versioned**. Renaming, removing, or changing the
meaning of a field requires a *new schema class* under a *new
version string* — never an in-place edit. The mining + feature
audit infrastructure keys evidence bundles by version, so silently
mutating a schema would break cross-run comparability.

Failed-extraction marker
------------------------

When the LLM produces malformed JSON, an out-of-range score, or any
other unrecoverable error, the pipeline persists a
:class:`FailedExtraction` row instead of dropping the document on
the floor. This way the storage layer remains the source of truth
for "every document we tried to process" and downstream feature
aggregators can show coverage gaps explicitly rather than
hallucinating success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, ClassVar

#: Bumped when the dataclass shapes here change. Persisted records
#: carry this value so the loader can reject incompatible payloads.
#: ``v1`` covered news only; ``v2`` adds filings + earnings-calls and
#: routes deserialisation by ``provenance.source_kind``.
SCHEMA_VERSION: str = "v2"

#: Schema versions :meth:`ExtractedRecord.from_payload` will accept.
#: ``v1`` is read-only-compatible: it predates the tagged-union
#: routing, so every ``v1`` extraction is by definition a
#: :class:`NewsExtraction` (the only kind v1 knew about). New writes
#: always carry :data:`SCHEMA_VERSION`.
LOADABLE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionProvenance:
    """Per-record provenance the brief requires.

    Attributes
    ----------
    prompt_version:
        Version pin of the prompt that produced this extraction
        (e.g. ``"news-prompt-v1"``). Lets a future auditor re-run
        the same prompt against the same document.
    model_version:
        Identifier of the LLM that produced this extraction
        (e.g. ``"claude-sonnet-4-5"``). Including this field is the
        only way to compare extractions across model upgrades.
    source_kind:
        ``"news" | "filing-10k" | "filing-10q" | "earnings-call"``
        — categorical label for downstream slicing.
    source_doc_id:
        Stable identifier of the source document (URL hash, EDGAR
        accession number, etc.). Used for dedupe + traceback.
    extracted_at:
        UTC ISO timestamp the extraction ran.
    confidence:
        LLM's self-reported confidence in the extraction, in
        ``[0, 1]``. Reasonable default ``1.0`` when the model
        doesn't ship a confidence and we trust the parse; ``0.0``
        when the response was unparseable and we've fallen back to
        a default.
    raw_response_hash:
        SHA-256 (truncated to 16 hex chars) of the raw LLM response
        text. Doesn't store the prose itself — that's deliberate
        per the brief — but a hash is enough to detect "two
        extractions claim identical provenance but the model said
        different things".
    """

    prompt_version: str
    model_version: str
    source_kind: str
    source_doc_id: str
    extracted_at: str
    confidence: float
    raw_response_hash: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise ValueError("ExtractionProvenance.prompt_version must be non-empty")
        if not self.model_version.strip():
            raise ValueError("ExtractionProvenance.model_version must be non-empty")
        if not self.source_kind.strip():
            raise ValueError("ExtractionProvenance.source_kind must be non-empty")
        if not self.source_doc_id.strip():
            raise ValueError("ExtractionProvenance.source_doc_id must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ExtractionProvenance.confidence must lie in [0, 1]; got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtractionProvenance:
        return cls(**{f.name: payload[f.name] for f in fields(cls)})


# ---------------------------------------------------------------------------
# News extraction schema (v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsExtraction:
    """Structured extraction from one news article about one instrument.

    Every numeric field is in ``[-1, 1]``:

    * ``+1`` = strongly positive for the named dimension
    * ``-1`` = strongly negative
    * ``0`` = neutral / no signal / not mentioned

    Categorical / count fields use their natural ranges (counts are
    non-negative integers; categoricals are strings from a closed
    vocabulary).

    Attributes
    ----------
    sentiment:
        Overall sentiment of the article toward the named
        instrument. The primary aggregation input for
        ``news_sentiment_1d`` / ``news_sentiment_5d``.
    materiality:
        How much of the article's content is *about* the named
        instrument. ``1.0`` = a press release; ``0.1`` = a passing
        mention in a market roundup. The aggregator weights
        sentiment by materiality so a passing mention doesn't drown
        a focused story.
    demand_signal:
        Signal that demand for the company's product/service is
        rising (+) or falling (-). Brief's earnings-call field;
        reused for news because pre-earnings news often signals it.
    margin_pressure:
        Signal that margins are under pressure (-) or expanding
        (+). Reverse-coded so positive is still "good for forward
        return" (less pressure).
    guidance_signal:
        Signal that forward guidance is positive (+) or negative
        (-) — applies when the news quotes management guidance.
    litigation_risk:
        Signal that legal / regulatory risk is rising. Negative is
        "bad for stock" by the usual convention, so we store
        ``-litigation_risk`` semantics: ``-1`` means high risk,
        ``+1`` means low.
    novelty:
        How novel the article's content is vs. the existing news
        stream. ``1.0`` = breaking news; ``0.0`` = restatement of
        already-public information. Feeds ``news_novelty``.
    """

    sentiment: float
    materiality: float
    demand_signal: float = 0.0
    margin_pressure: float = 0.0
    guidance_signal: float = 0.0
    litigation_risk: float = 0.0
    novelty: float = 0.5

    #: Field names whose value must lie in ``[-1, 1]``. Exposed as a
    #: class-level constant so prompts.py can build the LLM
    #: instructions from the same source of truth.
    SIGNED_FIELDS: ClassVar[tuple[str, ...]] = (
        "sentiment",
        "demand_signal",
        "margin_pressure",
        "guidance_signal",
        "litigation_risk",
    )
    #: Fields whose value must lie in ``[0, 1]``.
    UNSIGNED_FIELDS: ClassVar[tuple[str, ...]] = ("materiality", "novelty")

    def __post_init__(self) -> None:
        for name in self.SIGNED_FIELDS:
            value = getattr(self, name)
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"NewsExtraction.{name} must lie in [-1, 1]; got {value}")
        for name in self.UNSIGNED_FIELDS:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"NewsExtraction.{name} must lie in [0, 1]; got {value}")

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NewsExtraction:
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in payload:
                kwargs[f.name] = float(payload[f.name])
        return cls(**kwargs)

    @classmethod
    def field_descriptions(cls) -> dict[str, str]:
        """Mapping ``field_name -> description``. Used by prompts.py
        to compose the LLM's instructions from the schema docstrings
        rather than a parallel string list."""
        return {
            "sentiment": "Overall sentiment toward the instrument in [-1, 1].",
            "materiality": (
                "How much of the article focuses on the instrument, in [0, 1]. "
                "1.0 = press release dedicated to it; 0.1 = passing mention."
            ),
            "demand_signal": (
                "Demand for the company's product/service, in [-1, 1]. Positive = rising demand."
            ),
            "margin_pressure": (
                "Reverse-coded margin pressure in [-1, 1]. Positive = margins "
                "expanding; negative = pressure rising."
            ),
            "guidance_signal": (
                "Forward-guidance signal in [-1, 1]. Only set when the article "
                "quotes management guidance; otherwise 0."
            ),
            "litigation_risk": (
                "Legal / regulatory risk, sign-flipped, in [-1, 1]. Positive = "
                "low risk; negative = elevated risk."
            ),
            "novelty": (
                "Novelty vs. the existing news stream in [0, 1]. 1.0 = breaking "
                "news; 0.0 = restatement of already-public information."
            ),
        }


# ---------------------------------------------------------------------------
# Filing extraction schema (v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingExtraction:
    """Structured extraction from one SEC filing (10-K / 10-Q / 8-K).

    Same scoring convention as :class:`NewsExtraction`: signed fields in
    ``[-1, 1]``, unsigned in ``[0, 1]``, ``0`` (or ``0.5`` for novelty)
    is the explicit "no signal" sentinel — the LLM is forbidden from
    omitting fields.

    Attributes
    ----------
    risk_sentiment:
        Net sentiment of the Risk Factors section, reverse-coded so
        ``+1`` = de-risking language ("we resolved the prior
        litigation") and ``-1`` = elevated risk language ("a
        previously immaterial matter is now material"). Feeds
        ``10k_risk_sentiment``.
    uncertainty_score:
        Density of hedge words in the filing ("may", "could",
        "uncertain", "we are unable to estimate"). ``0`` = none, ``1``
        = pervasive. Feeds ``10q_uncertainty_score``.
    management_tone:
        Tone of the MD&A / Outlook narrative, ``-1`` = defensive /
        cautious, ``+1`` = confident / optimistic. Feeds the
        management-tone-change feature when compared across periods.
    litigation_risk:
        Same convention as :attr:`NewsExtraction.litigation_risk` —
        sign-flipped: ``+1`` = low litigation risk, ``-1`` = high.
    guidance_sentiment:
        Sentiment of explicit guidance statements in the filing.
        Distinct from ``management_tone`` (which captures the whole
        narrative) — ``guidance_sentiment`` is *only* the guidance
        clauses. ``0`` if guidance is not provided.
    supply_chain_risk:
        Signal that supply chains are stressed (negative) or
        normalising (positive). Sign-flipped — ``+1`` is healthy.
    inventory_risk:
        Signal that inventory levels are concerning (negative) or
        normal (positive). Reverse-coded — ``+1`` is healthy.
    margin_pressure:
        Same convention as :attr:`NewsExtraction.margin_pressure` —
        reverse-coded so ``+1`` = expanding margins.
    demand_weakness:
        Signal that end-market demand is weakening (negative) or
        strengthening (positive). Reverse-coded — ``+1`` is healthy.
    financing_stress:
        Signal that the financing profile is stressed (negative —
        debt covenant pressure, going-concern language) or healthy
        (positive). Reverse-coded — ``+1`` is healthy.
    """

    risk_sentiment: float
    uncertainty_score: float
    management_tone: float = 0.0
    litigation_risk: float = 0.0
    guidance_sentiment: float = 0.0
    supply_chain_risk: float = 0.0
    inventory_risk: float = 0.0
    margin_pressure: float = 0.0
    demand_weakness: float = 0.0
    financing_stress: float = 0.0

    SIGNED_FIELDS: ClassVar[tuple[str, ...]] = (
        "risk_sentiment",
        "management_tone",
        "litigation_risk",
        "guidance_sentiment",
        "supply_chain_risk",
        "inventory_risk",
        "margin_pressure",
        "demand_weakness",
        "financing_stress",
    )
    UNSIGNED_FIELDS: ClassVar[tuple[str, ...]] = ("uncertainty_score",)

    def __post_init__(self) -> None:
        for name in self.SIGNED_FIELDS:
            value = getattr(self, name)
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"FilingExtraction.{name} must lie in [-1, 1]; got {value}")
        for name in self.UNSIGNED_FIELDS:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"FilingExtraction.{name} must lie in [0, 1]; got {value}")

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FilingExtraction:
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in payload:
                kwargs[f.name] = float(payload[f.name])
        return cls(**kwargs)

    @classmethod
    def field_descriptions(cls) -> dict[str, str]:
        return {
            "risk_sentiment": (
                "Net sentiment of the Risk Factors section in [-1, 1], reverse-coded. "
                "+1 = de-risking language; -1 = elevated risk."
            ),
            "uncertainty_score": (
                "Hedge-word density in the filing in [0, 1]. 0 = none, 1 = pervasive."
            ),
            "management_tone": (
                "Tone of the MD&A / Outlook narrative in [-1, 1]. -1 = defensive; +1 = confident."
            ),
            "litigation_risk": (
                "Legal / regulatory risk, sign-flipped, in [-1, 1]. +1 = low risk; -1 = elevated."
            ),
            "guidance_sentiment": (
                "Sentiment of explicit forward-guidance statements in [-1, 1]. "
                "0 if no guidance is provided."
            ),
            "supply_chain_risk": (
                "Supply-chain health, sign-flipped, in [-1, 1]. +1 = healthy; -1 = stressed."
            ),
            "inventory_risk": (
                "Inventory health, sign-flipped, in [-1, 1]. +1 = clean; -1 = bloated."
            ),
            "margin_pressure": (
                "Margin trajectory, sign-flipped, in [-1, 1]. +1 = expanding; -1 = pressure."
            ),
            "demand_weakness": (
                "End-market demand, sign-flipped, in [-1, 1]. +1 = strengthening; -1 = weakening."
            ),
            "financing_stress": (
                "Financing-profile health, sign-flipped, in [-1, 1]. +1 = healthy; "
                "-1 = stressed (covenant pressure, going-concern)."
            ),
        }


# ---------------------------------------------------------------------------
# Earnings-call extraction schema (v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarningsCallExtraction:
    """Structured extraction from one earnings-call transcript.

    Earnings calls have a distinctive Q&A structure — analyst pushback
    and management's response carry signal that's absent from filings.
    Same scoring convention: signed in ``[-1, 1]``, unsigned in
    ``[0, 1]``, ``0`` is the explicit no-signal sentinel.

    Attributes
    ----------
    management_confidence:
        Confidence of prepared-remarks tone. ``+1`` = highly confident,
        ``-1`` = defensive / hedging. Feeds the management-tone-change
        feature alongside :attr:`FilingExtraction.management_tone`.
    analyst_pushback:
        Intensity of analyst pushback in Q&A. ``0`` = softball
        questions, ``1`` = sustained skepticism / repeated probing.
        Feeds ``analyst_pushback`` feature.
    guidance_quality:
        Specificity + breadth of forward guidance. ``+1`` = specific
        guidance across multiple periods, ``-1`` = withdrawn /
        explicitly refused, ``0`` = vague reiterations.
    margin_pressure:
        Same convention as :attr:`NewsExtraction.margin_pressure`.
    demand_signal:
        Same convention as :attr:`NewsExtraction.demand_signal`.
    capex_intent:
        Forward capex intent. ``+1`` = significant capex acceleration,
        ``-1`` = explicit capex cuts / deferrals.
    inventory_problem:
        Whether the call surfaced an inventory issue. ``+1`` = explicit
        problem acknowledged, ``-1`` = inventory explicitly described
        as clean. ``0`` = not discussed.
    """

    management_confidence: float
    analyst_pushback: float
    guidance_quality: float = 0.0
    margin_pressure: float = 0.0
    demand_signal: float = 0.0
    capex_intent: float = 0.0
    inventory_problem: float = 0.0

    SIGNED_FIELDS: ClassVar[tuple[str, ...]] = (
        "management_confidence",
        "guidance_quality",
        "margin_pressure",
        "demand_signal",
        "capex_intent",
        "inventory_problem",
    )
    UNSIGNED_FIELDS: ClassVar[tuple[str, ...]] = ("analyst_pushback",)

    def __post_init__(self) -> None:
        for name in self.SIGNED_FIELDS:
            value = getattr(self, name)
            if not (-1.0 <= value <= 1.0):
                raise ValueError(f"EarningsCallExtraction.{name} must lie in [-1, 1]; got {value}")
        for name in self.UNSIGNED_FIELDS:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"EarningsCallExtraction.{name} must lie in [0, 1]; got {value}")

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EarningsCallExtraction:
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in payload:
                kwargs[f.name] = float(payload[f.name])
        return cls(**kwargs)

    @classmethod
    def field_descriptions(cls) -> dict[str, str]:
        return {
            "management_confidence": (
                "Confidence of prepared-remarks tone in [-1, 1]. +1 = confident; -1 = defensive."
            ),
            "analyst_pushback": (
                "Intensity of analyst pushback in Q&A in [0, 1]. 0 = softball; "
                "1 = sustained skepticism."
            ),
            "guidance_quality": (
                "Specificity + breadth of forward guidance in [-1, 1]. +1 = specific "
                "across periods; -1 = withdrawn; 0 = vague reiteration."
            ),
            "margin_pressure": (
                "Margin trajectory, sign-flipped, in [-1, 1]. +1 = expanding; -1 = pressure."
            ),
            "demand_signal": ("Demand for product/service in [-1, 1]. +1 = rising; -1 = falling."),
            "capex_intent": (
                "Forward capex intent in [-1, 1]. +1 = acceleration; -1 = cuts/deferrals."
            ),
            "inventory_problem": (
                "Inventory issue acknowledgement in [-1, 1]. +1 = explicit problem; "
                "-1 = explicitly clean; 0 = not discussed."
            ),
        }


#: Type alias for any concrete extraction kind. Used by
#: :class:`ExtractedRecord` so callers don't have to thread three
#: separate record types around.
ExtractionT = NewsExtraction | FilingExtraction | EarningsCallExtraction


#: Canonical filing ``source_kind`` values the dispatcher accepts.
#: Adding a new SEC form type (6-K, 20-F, etc.) is a one-line change
#: here — the rest of the pipeline reads from this tuple.
KNOWN_FILING_KINDS: tuple[str, ...] = ("filing-10k", "filing-10q", "filing-8k")

#: Canonical ``source_kind`` values the dispatcher accepts at all.
#: Used by validators that want to assert a record's ``source_kind``
#: is something the platform actually knows how to feature-ise.
KNOWN_SOURCE_KINDS: tuple[str, ...] = ("news", *KNOWN_FILING_KINDS, "earnings-call")


def extraction_from_dict_for_kind(kind: str, payload: dict[str, Any]) -> ExtractionT:
    """Dispatch on the provenance ``source_kind`` to the right
    extraction loader.

    Single source of truth for ``source_kind → ExtractionT``
    routing — the prompts module's :func:`get_prompt_for_kind` uses
    the same dispatch rules so persisted extractions can always be
    paired back with the prompt that produced them.

    Only whitelisted kinds are accepted (see
    :data:`KNOWN_SOURCE_KINDS`). An unknown kind raises rather than
    silently mis-routing — a misconfigured feed should fail loudly
    at extraction time, not produce a ``FilingExtraction`` of a news
    article.
    """
    if kind == "news":
        return NewsExtraction.from_dict(payload)
    if kind in KNOWN_FILING_KINDS:
        return FilingExtraction.from_dict(payload)
    if kind == "earnings-call":
        return EarningsCallExtraction.from_dict(payload)
    raise ValueError(
        f"Unsupported source_kind for extraction dispatch: {kind!r}. "
        f"Known kinds: {KNOWN_SOURCE_KINDS!r}"
    )


# ---------------------------------------------------------------------------
# Failed-extraction marker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedExtraction:
    """Sentinel persisted when an LLM call cannot produce a valid extraction.

    The storage layer writes one of these instead of silently
    dropping the row so coverage diagnostics surface the gap.
    Downstream aggregators skip failed extractions but report them
    in their summary stats.

    Attributes
    ----------
    source_doc_id:
        Same as the would-be :class:`ExtractionProvenance`. Lets a
        retry job find documents that failed last time.
    source_kind:
        Same as the would-be provenance.
    failed_at:
        UTC ISO timestamp.
    reason:
        Human-readable error category. Examples: ``"malformed_json"``,
        ``"score_out_of_range"``, ``"empty_response"``,
        ``"client_error"``.
    detail:
        Free-form additional context (LLM error string, validation
        failure message, etc.). Not used by features — only for
        operator diagnostics.
    prompt_version:
        Prompt that was used when the failure occurred. Lets the
        retry strategy know whether to re-attempt with a newer
        prompt version.
    model_version:
        Model that produced the failed response.
    """

    source_doc_id: str
    source_kind: str
    failed_at: str
    reason: str
    detail: str
    prompt_version: str
    model_version: str

    def __post_init__(self) -> None:
        if not self.source_doc_id.strip():
            raise ValueError("FailedExtraction.source_doc_id must be non-empty")
        if not self.reason.strip():
            raise ValueError("FailedExtraction.reason must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FailedExtraction:
        return cls(**{f.name: payload[f.name] for f in fields(cls)})


# ---------------------------------------------------------------------------
# Combined record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedRecord:
    """One row of the extraction storage layer.

    Wraps either a successful :class:`NewsExtraction` + its
    :class:`ExtractionProvenance`, or a :class:`FailedExtraction`.
    Tagged-union shape so callers don't have to thread two list
    types around.

    Attributes
    ----------
    instrument_id:
        Instrument the extraction is about.
    extraction:
        Successful :class:`NewsExtraction` when extraction
        succeeded; ``None`` when failed.
    provenance:
        Provenance of the successful extraction; ``None`` when
        failed.
    failure:
        :class:`FailedExtraction` sentinel when the extraction
        failed; ``None`` on success.
    schema_version:
        :data:`SCHEMA_VERSION` at write time. Pinned per row so
        old + new records coexist if the schema evolves.
    """

    instrument_id: str
    extraction: ExtractionT | None = None
    provenance: ExtractionProvenance | None = None
    failure: FailedExtraction | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("ExtractedRecord.instrument_id must be non-empty")
        extraction_set = self.extraction is not None
        provenance_set = self.provenance is not None
        failure_set = self.failure is not None
        # Order matters: surface "lopsided" success (exactly one of
        # extraction/provenance non-None) before the catch-all "neither
        # success nor failure" check, because callers debugging a
        # construction bug get a more specific error pointing at the
        # actual mismatch.
        if extraction_set != provenance_set:
            raise ValueError("ExtractedRecord: extraction and provenance must be set together")
        has_success = extraction_set and provenance_set
        if has_success and failure_set:
            raise ValueError("ExtractedRecord: cannot carry both extraction and failure")
        if not has_success and not failure_set:
            raise ValueError(
                "ExtractedRecord: must carry either (extraction + provenance) or failure"
            )

    @property
    def succeeded(self) -> bool:
        return self.extraction is not None

    def to_jsonl_line(self) -> str:
        """JSON-serialised single-line representation."""
        import json
        from typing import cast

        # ``__post_init__`` guarantees the tagged-union invariant: a
        # succeeded record carries both extraction + provenance; a
        # failed one carries failure. The casts express this for
        # mypy without re-validating at runtime.
        if self.succeeded:
            extraction = cast("ExtractionT", self.extraction)
            provenance = cast("ExtractionProvenance", self.provenance)
            payload = {
                "instrument_id": self.instrument_id,
                "extraction": extraction.to_dict(),
                "provenance": provenance.to_dict(),
                "schema_version": self.schema_version,
            }
        else:
            failure = cast("FailedExtraction", self.failure)
            payload = {
                "instrument_id": self.instrument_id,
                "failure": failure.to_dict(),
                "schema_version": self.schema_version,
            }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExtractedRecord:
        """Construct from a parsed JSONL line. Raises on malformed input.

        Loadable schema versions are :data:`LOADABLE_SCHEMA_VERSIONS`
        (currently ``v1`` and ``v2``). ``v1`` records predate the
        tagged-union routing — they always carry a
        :class:`NewsExtraction`, so the loader forces news dispatch
        for them regardless of any ``source_kind`` field.
        """
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version not in LOADABLE_SCHEMA_VERSIONS:
            raise ValueError(
                f"ExtractedRecord: unsupported schema_version {version!r}; "
                f"loadable versions: {sorted(LOADABLE_SCHEMA_VERSIONS)!r}"
            )
        instrument_id = payload.get("instrument_id")
        if not isinstance(instrument_id, str):
            raise ValueError(
                f"ExtractedRecord.instrument_id must be a string; "
                f"got {type(instrument_id).__name__}"
            )
        if "failure" in payload:
            return cls(
                instrument_id=instrument_id,
                failure=FailedExtraction.from_dict(payload["failure"]),
                schema_version=version,
            )
        if "extraction" not in payload or "provenance" not in payload:
            raise ValueError(
                f"ExtractedRecord: payload must contain either 'failure' or "
                f"both 'extraction' + 'provenance'; got keys={list(payload)}"
            )
        provenance = ExtractionProvenance.from_dict(payload["provenance"])
        # v1 records predate tagged-union routing — they're always
        # news. Force news dispatch so legacy JSONL loads cleanly even
        # if the operator typo'd the source_kind field.
        dispatch_kind = "news" if version == "v1" else provenance.source_kind
        return cls(
            instrument_id=instrument_id,
            extraction=extraction_from_dict_for_kind(dispatch_kind, payload["extraction"]),
            provenance=provenance,
            schema_version=version,
        )


# ---------------------------------------------------------------------------
# Source document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDocument:
    """One document the extraction pipeline will run against.

    The pipeline doesn't fetch documents — it consumes a list of
    :class:`SourceDocument` instances the operator has already
    assembled (from a news vendor, EDGAR feed, etc.). This module
    is intentionally agnostic to *how* the documents were obtained.

    Attributes
    ----------
    doc_id:
        Stable identifier of the document. Used as
        :attr:`ExtractionProvenance.source_doc_id`.
    instrument_id:
        Instrument this document is about. One document → one
        extraction → one ``(instrument_id, date)`` panel row.
    published_at:
        UTC datetime when the document became publicly available.
        The aggregator uses this (not :attr:`fetched_at`) to assign
        the extraction to a trading date — PIT-safe.
    kind:
        ``"news" | "filing-10k" | ...`` — categorical label.
    text:
        Document body. Length is unbounded; the LLM client is
        responsible for truncation / chunking.
    metadata:
        Free-form additional context (URL, vendor, headline, etc.)
        passed through to provenance recording without
        interpretation.
    """

    doc_id: str
    instrument_id: str
    published_at: datetime
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("SourceDocument.doc_id must be non-empty")
        if not self.instrument_id.strip():
            raise ValueError("SourceDocument.instrument_id must be non-empty")
        if not self.kind.strip():
            raise ValueError("SourceDocument.kind must be non-empty")
        if self.published_at.tzinfo is None:
            raise ValueError("SourceDocument.published_at must be timezone-aware (UTC recommended)")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Helper so call sites don't litter ``datetime.now(UTC).isoformat()``
    everywhere; also makes mocking the clock in tests trivial.
    """
    return datetime.now(UTC).isoformat()


__all__ = [
    "KNOWN_FILING_KINDS",
    "KNOWN_SOURCE_KINDS",
    "LOADABLE_SCHEMA_VERSIONS",
    "SCHEMA_VERSION",
    "EarningsCallExtraction",
    "ExtractedRecord",
    "ExtractionProvenance",
    "ExtractionT",
    "FailedExtraction",
    "FilingExtraction",
    "NewsExtraction",
    "SourceDocument",
    "extraction_from_dict_for_kind",
    "utc_now_iso",
]
