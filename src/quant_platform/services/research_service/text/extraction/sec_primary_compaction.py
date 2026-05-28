"""Deterministic SEC primary filing compaction for LLM text features."""

from __future__ import annotations

import re
from dataclasses import dataclass

SEC_PRIMARY_COMPACTION_POLICY = "sec-primary-compact-v1"
MAX_COMPACTED_TEXT_CHARS = 24_000

_ITEM_HEADING_RE = re.compile(r"(?im)^\s*item\s+\d+[a-z]?(?:\.\d+)?\b[^\n]{0,160}$")

_XBRL_PREFIXES = (
    "dei:",
    "iso4217:",
    "ix:",
    "link:",
    "srt:",
    "us-gaap:",
    "xbrli:",
    "xbrldi:",
    "xlink:",
)


@dataclass(frozen=True)
class CompactedPrimaryText:
    """Compacted source text plus evidence lineage for artifact manifests."""

    text: str
    policy_name: str
    original_chars: int
    compacted_chars: int
    selected_section_labels: tuple[str, ...]

    def to_payload(
        self,
        *,
        raw_source_uri: str,
        raw_artifact_uri: str,
        raw_content_digest: str,
        compacted_content_digest: str,
    ) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "raw_source_uri": raw_source_uri,
            "raw_artifact_uri": raw_artifact_uri,
            "raw_content_digest": raw_content_digest,
            "compacted_content_digest": compacted_content_digest,
            "original_chars": self.original_chars,
            "compacted_chars": self.compacted_chars,
            "selected_section_labels": list(self.selected_section_labels),
        }


@dataclass(frozen=True)
class _SectionPlan:
    label: str
    heading_patterns: tuple[str, ...]
    keyword_patterns: tuple[str, ...] = ()


def compact_sec_primary_text(
    text: str,
    *,
    form_type: str = "",
    max_chars: int = MAX_COMPACTED_TEXT_CHARS,
) -> CompactedPrimaryText:
    """Return a bounded, deterministic primary-filing excerpt for LLM extraction."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    original_chars = len(text)
    normalized = _normalize_sec_text(text)
    plans = _plans_for_form(form_type)
    sections = _select_sections(normalized, plans)
    if not sections:
        sections = _fallback_sections(normalized)
    compacted, labels = _render_sections(sections, max_chars=max_chars)
    return CompactedPrimaryText(
        text=compacted,
        policy_name=SEC_PRIMARY_COMPACTION_POLICY,
        original_chars=original_chars,
        compacted_chars=len(compacted),
        selected_section_labels=tuple(labels),
    )


def _normalize_sec_text(text: str) -> str:
    lines: list[str] = []
    previous = ""
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or _looks_like_xbrl_noise(line):
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)


def _looks_like_xbrl_noise(line: str) -> bool:
    lowered = line.lower()
    prefix_hits = sum(1 for prefix in _XBRL_PREFIXES if prefix in lowered)
    if prefix_hits >= 2:
        return True
    if re.fullmatch(r"(?:[a-z][a-z0-9_-]*:[a-z0-9_.-]+\s*){3,}", lowered):
        return True
    return line.startswith("{") and line.endswith("}") and len(line) > 120


def _plans_for_form(form_type: str) -> tuple[_SectionPlan, ...]:
    form = form_type.upper().strip()
    if form == "10-K":
        return (
            _SectionPlan("item_7_mda", (r"item\s+7\.?\s+management",)),
            _SectionPlan("item_7a_market_risk", (r"item\s+7a\.?\s+quantitative",)),
            _SectionPlan("item_1a_risk_factors", (r"item\s+1a\.?\s+risk",)),
            _SectionPlan("item_1_business_demand", (r"item\s+1\.?\s+business",)),
            _SectionPlan(
                "item_8_revenue_margin_notes",
                (r"item\s+8\.?\s+financial",),
                ("revenue", "net sales", "gross margin", "operating margin"),
            ),
        )
    if form == "8-K":
        return (
            _SectionPlan("item_2_02_results", (r"item\s+2\.02",)),
            _SectionPlan("item_7_01_disclosure", (r"item\s+7\.01",)),
            _SectionPlan("item_8_01_other", (r"item\s+8\.01",)),
            _SectionPlan("item_9_01_exhibits", (r"item\s+9\.01",)),
            _SectionPlan(
                "results_guidance_outlook",
                (),
                ("results", "guidance", "outlook", "revenue", "margin"),
            ),
        )
    return (
        _SectionPlan("item_2_mda", (r"item\s+2\.?\s+management",)),
        _SectionPlan("item_3_market_risk", (r"item\s+3\.?\s+quantitative",)),
        _SectionPlan("item_1a_risk_factors", (r"item\s+1a\.?\s+risk",)),
        _SectionPlan(
            "liquidity_capital_resources",
            (),
            ("liquidity and capital resources", "capital resources"),
        ),
        _SectionPlan("results_of_operations", (), ("results of operations",)),
        _SectionPlan(
            "revenue_margin_notes",
            (),
            ("revenue", "net sales", "gross margin", "operating margin"),
        ),
    )


def _select_sections(
    text: str,
    plans: tuple[_SectionPlan, ...],
) -> list[tuple[str, str, int]]:
    selected: list[tuple[str, str, int]] = []
    used_ranges: list[tuple[int, int]] = []
    for plan in plans:
        section = _find_heading_section(text, plan)
        if section is None and plan.keyword_patterns:
            section = _find_keyword_window(text, plan)
        if section is None:
            continue
        label, body, start, end = section
        if _overlaps_used(start, end, used_ranges):
            continue
        selected.append((label, body, start))
        used_ranges.append((start, end))
    selected.sort(key=lambda item: item[2])
    return selected


def _find_heading_section(
    text: str,
    plan: _SectionPlan,
) -> tuple[str, str, int, int] | None:
    for pattern in plan.heading_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        start = _line_start(text, match.start())
        end = _next_item_heading(text, start + 1) or len(text)
        body = text[start:end].strip()
        if body:
            return plan.label, body, start, end
    return None


def _find_keyword_window(
    text: str,
    plan: _SectionPlan,
    *,
    radius: int = 4_000,
) -> tuple[str, str, int, int] | None:
    lowered = text.lower()
    positions = [
        lowered.find(pattern.lower())
        for pattern in plan.keyword_patterns
        if lowered.find(pattern.lower()) >= 0
    ]
    if not positions:
        return None
    center = min(positions)
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    return plan.label, text[start:end].strip(), start, end


def _line_start(text: str, position: int) -> int:
    previous_newline = text.rfind("\n", 0, position)
    return 0 if previous_newline < 0 else previous_newline + 1


def _next_item_heading(text: str, after: int) -> int | None:
    for match in _ITEM_HEADING_RE.finditer(text, after):
        return match.start()
    return None


def _overlaps_used(
    start: int,
    end: int,
    used_ranges: list[tuple[int, int]],
) -> bool:
    return any(start < used_end and end > used_start for used_start, used_end in used_ranges)


def _fallback_sections(text: str) -> list[tuple[str, str, int]]:
    if not text:
        return [("fallback_empty", "", 0)]
    length = len(text)
    window = min(8_000, max(1_000, length // 3))
    middle_start = max(0, (length // 2) - (window // 2))
    return [
        ("fallback_head", text[:window], 0),
        ("fallback_middle", text[middle_start : middle_start + window], middle_start),
        ("fallback_tail", text[max(0, length - window) :], max(0, length - window)),
    ]


def _render_sections(
    sections: list[tuple[str, str, int]],
    *,
    max_chars: int,
) -> tuple[str, tuple[str, ...]]:
    if not sections:
        return "", ()
    per_section_limit = max(1_200, max_chars // len(sections) - 120)
    while per_section_limit > 200:
        rendered = _render_with_limit(sections, per_section_limit)
        if len(rendered) <= max_chars:
            return rendered, tuple(label for label, _, _ in sections)
        per_section_limit = int(per_section_limit * 0.85)
    rendered = _render_with_limit(sections, per_section_limit)
    return rendered[:max_chars], tuple(label for label, _, _ in sections)


def _render_with_limit(sections: list[tuple[str, str, int]], per_section_limit: int) -> str:
    parts: list[str] = []
    for label, body, _start in sections:
        parts.append(f"[section: {label}]")
        parts.append(_bound_section(body, per_section_limit, label=label))
    return "\n\n".join(parts).strip()


def _bound_section(body: str, limit: int, *, label: str) -> str:
    if len(body) <= limit:
        return body
    head_len = max(1, int(limit * 0.62))
    tail_len = max(1, limit - head_len)
    omitted = len(body) - head_len - tail_len
    marker = f"\n[omitted {omitted} chars from {label}]\n"
    head_len = max(1, head_len - len(marker) // 2)
    tail_len = max(1, tail_len - len(marker) // 2)
    return f"{body[:head_len].rstrip()}{marker}{body[-tail_len:].lstrip()}"


__all__ = [
    "CompactedPrimaryText",
    "MAX_COMPACTED_TEXT_CHARS",
    "SEC_PRIMARY_COMPACTION_POLICY",
    "compact_sec_primary_text",
]
