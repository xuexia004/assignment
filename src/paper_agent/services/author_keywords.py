"""Read explicitly labelled author keyword sections; never generate topic tags."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pymupdf

from paper_agent.models import Paper
from paper_agent.services.pdf_runtime import PDF_PARSE_LOCK

_HEADING = re.compile(r"^\s*(?:key\s*words?|index\s*terms|关键词|关键字)\s*[:：—–-]\s*(.*)$", re.I)
_NEXT_SECTION = re.compile(r"^\s*(?:\d+[.\s]|[IVX]+[.\s]|introduction\b|abstract\b|ACM\b|CCS\b|引言|绪论)", re.I)


def author_keywords(paper: Paper) -> list[str]:
    """Generic keywords, OpenAlex topic tags and arXiv categories are excluded."""
    values = [paper.metadata.get("author_keywords")]
    values.extend(record.get("author_keywords") for record in paper.source_records)
    labels: dict[str, str] = {}
    for group in values:
        for value in group if isinstance(group, list) else []:
            if not isinstance(value, str):
                continue
            value = " ".join(unicodedata.normalize("NFKC", value).split()).strip(" .;；，,")
            if value and len(value) <= 120:
                labels.setdefault(value.casefold(), value)
    return list(labels.values())[:12]


def extract_author_keyword_section(text: str) -> tuple[list[str], str]:
    """Accept only an explicit keyword field; preserve its verbatim provenance."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if not match:
            continue
        parts = [match.group(1)]
        quoted = [line]
        for following in lines[index + 1:index + 5]:
            if not following.strip() or _NEXT_SECTION.match(following):
                break
            parts.append(following.strip())
            quoted.append(following)
        field = " ".join(parts).strip()
        # A keyword list is short. Do not turn a following body paragraph into
        # keywords or infer terms from an abstract without a labelled field.
        if len(field) > 600 or re.search(r"[.!?。！？]\s+\S", field):
            continue
        terms = [item.strip(" .;；，,") for item in re.split(r"[,;，；•]", field)]
        terms = [item for item in terms if item]
        if not terms or len(terms) > 12 or any(len(item) > 120 for item in terms):
            continue
        return list(dict.fromkeys(terms)), "\n".join(quoted)
    return [], ""


def enrich_cached_author_keywords(papers: list[Paper], pdf_dir: Path) -> list[Paper]:
    """Use already acquired PDFs only; retrieval never downloads PDFs for tags."""
    results = []
    for original in papers:
        paper = original.model_copy(deep=True)
        results.append(paper)
        if author_keywords(paper):
            continue
        path = Path(pdf_dir) / (hashlib.sha256(paper.paper_id.encode("utf-8")).hexdigest() + ".pdf")
        if not path.is_file():
            continue
        try:
            with PDF_PARSE_LOCK, pymupdf.open(path) as document:
                if not document.is_pdf:
                    continue
                for page_index in range(min(2, len(document))):
                    # Text blocks retain paragraph boundaries, avoiding a jump
                    # from a keyword line into the following article section.
                    blocks = document[page_index].get_text("blocks")
                    for block in blocks:
                        if not isinstance(block[4], str):
                            continue
                        terms, quote = extract_author_keyword_section(block[4])
                        if terms:
                            paper.metadata["author_keywords"] = terms
                            paper.metadata["author_keywords_provenance"] = {
                                "source": "pdf_keyword_section", "page_number": page_index + 1,
                                "quote": quote, "paper_id": paper.paper_id,
                            }
                            break
                    if author_keywords(paper):
                        break
        except (OSError, RuntimeError, ValueError):
            # Missing/unreadable caches leave title/abstract ranking available.
            continue
    return results
