"""Full-source annotations are reversible and preserve original citation offsets."""

import hashlib

import pytest

from paper_agent.agents.grounded_context import build_grounded_context


@pytest.mark.parametrize("text", [
    "# Title\r\n\r\n  中文条目。\r\nSecond line.  \n\n## Methods\nSame.\n\nSame.\n",
    "# Heading only\n\n", " \n\t", "", "Plain paragraph without a final newline.",
])
def test_complete_source_survives_annotation_with_exact_original_offsets(text):
    context, audit = build_grounded_context(text)
    restored, cursor, inserted = [], 0, 0
    for span in audit["selected_spans"]:
        assert text[span["start"]:span["end"]] == span["text"]
        location = span["start"] + inserted
        label = f"[{span['id']}]\n"
        assert context[location:location + len(label)] == label
        restored.append(context[cursor:location])
        cursor = location + len(label)
        inserted += len(label)
    restored.append(context[cursor:])
    assert "".join(restored) == text
    assert audit["selection_is_fulltext"] is True
    assert audit["source_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert audit["context_bytes"] == len(context.encode("utf-8"))
    assert audit["input_bytes"] == len(text.encode("utf-8"))
