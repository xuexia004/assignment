"""Annotate the complete supplied source for one-call, quote-grounded analysis.

Only paragraph locator labels are inserted. No source character is removed,
rewritten or reordered, and citation offsets still address the original text.
"""

from __future__ import annotations

import hashlib
import time

from paper_agent.agents.evidence_context import segment_document


def build_grounded_context(text: str) -> tuple[str, dict]:
    """Keep every source character while adding stable paragraph locators."""
    started = time.perf_counter()
    spans = segment_document(text)
    pieces = []
    cursor = 0
    for span in spans:
        pieces.extend((text[cursor:span["start"]], f"[{span['id']}]\n"))
        cursor = span["start"]
    pieces.append(text[cursor:])
    context = "".join(pieces)
    return context, {
        "method_version": "grounded_fulltext_v1",
        "requested_method": "grounded", "actual_method": "grounded",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_bytes": len(text.encode("utf-8")),
        "context_bytes": len(context.encode("utf-8")),
        "parent_count": len(spans), "selected_spans": spans,
        "selection_is_fulltext": True, "fallback_reason": None,
        "duration_seconds": time.perf_counter() - started,
    }
