"""SEC filing document parsing and filtering."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from quant_platform.services.data_service.text.sec_filing_http import (
    absolute_sec_url,
    sec_archive_url,
)
from quant_platform.services.data_service.text.sec_filing_models import SECFilingDocument

_TEXT_DOCUMENT_EXTENSIONS = {".htm", ".html", ".txt", ".xml", ".xhtml"}
_BINARY_DOCUMENT_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".xls",
    ".xlsx",
    ".zip",
}
_BINARY_DOCUMENT_TYPES = {"GRAPHIC", "PDF", "ZIP", "EXCEL", "XML GRAPHIC"}


class _SECHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


def clean_sec_document_text(raw: str) -> str:
    """Convert a SEC HTML/text document into bounded plain text."""
    parser = _SECHTMLTextExtractor()
    parser.feed(raw)
    text = "\n".join(parser.parts) if parser.parts else raw
    return "\n".join(line.strip() for line in unescape(text).splitlines() if line.strip())


class _SECFilingDetailParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[list[str], list[str]]] = []
        self._in_row = False
        self._in_cell = False
        self._cells: list[str] = []
        self._links: list[str] = []
        self._current_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._links = []
            self._current_parts = []
            return
        if self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._current_parts = []
            return
        if self._in_row and tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if self._in_row and tag in {"td", "th"} and self._in_cell:
            self._cells.append(" ".join(self._current_parts).strip())
            self._current_parts = []
            self._in_cell = False
            return
        if tag == "tr" and self._in_row:
            if self._cells:
                self.rows.append((list(self._cells), list(self._links)))
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            stripped = data.strip()
            if stripped:
                self._current_parts.append(stripped)


def parse_sec_filing_detail_documents(
    raw: str,
    *,
    cik: str,
    accession: str,
    primary_document: str,
) -> list[SECFilingDocument]:
    parser = _SECFilingDetailParser()
    parser.feed(raw)
    documents: list[SECFilingDocument] = []
    for cells, links in parser.rows:
        if len(cells) < 4:
            continue
        document_name = cells[2].strip()
        if not document_name or document_name.lower() == "document":
            continue
        document_type = cells[3].strip().upper()
        source_uri = sec_archive_url(cik, accession, document_name)
        for href in links:
            if href.endswith(document_name):
                source_uri = absolute_sec_url(href)
                break
        documents.append(
            SECFilingDocument(
                document_name=document_name,
                document_type=document_type,
                description=cells[1].strip(),
                source_uri=source_uri,
                is_primary_document=document_name == primary_document,
            )
        )
    return documents


def infer_document_type_from_name(name: str) -> str:
    lowered = name.lower()
    if "ex99" in lowered or "ex-99" in lowered or "exhibit99" in lowered:
        return "EX-99"
    return "UNKNOWN"


def is_preferred_exhibit(document: SECFilingDocument) -> bool:
    if document.is_primary_document or not is_textual_sec_document(document):
        return False
    doc_type = document.document_type.upper().strip()
    if doc_type in {"EX-99.1", "EX-99", "EX-99.2"} or doc_type.startswith("EX-99"):
        return True
    haystack = f"{document.document_name} {document.description}".lower()
    return any(
        keyword in haystack
        for keyword in (
            "earnings",
            "results",
            "release",
            "guidance",
            "outlook",
            "financial",
        )
    )


def is_textual_sec_document(document: SECFilingDocument) -> bool:
    document_type = document.document_type.upper().strip()
    suffix = Path(document.document_name.lower()).suffix
    if document_type in _BINARY_DOCUMENT_TYPES or suffix in _BINARY_DOCUMENT_EXTENSIONS:
        return False
    return suffix in _TEXT_DOCUMENT_EXTENSIONS or document_type.startswith("EX-99")


__all__ = [
    "clean_sec_document_text",
    "infer_document_type_from_name",
    "is_preferred_exhibit",
    "is_textual_sec_document",
    "parse_sec_filing_detail_documents",
]
