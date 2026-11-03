"""Author-keyword provenance and semantic input checks; no model/API calls."""

import hashlib

import pymupdf

from paper_agent.models import Paper
from paper_agent.services.author_keywords import (
    author_keywords,
    enrich_cached_author_keywords,
    extract_author_keyword_section,
)
from paper_agent.services.retrieval import merge_dedupe_rank
from paper_agent.services.semantic_reranker import _content_key, _paper_text


def test_generated_keywords_and_categories_are_not_author_keywords():
    paper = Paper(paper_id="p1", title="A title", abstract="An abstract", metadata={
        "keywords": ["source generated tag"], "categories": ["cs.AI"],
    }, source_records=[{"source": "openalex", "keywords": [{"display_name": "Generated"}]}])
    assert author_keywords(paper) == []
    assert _paper_text(paper) == "A title\nAn abstract"


def test_author_keywords_survive_source_merge_and_invalidate_old_embedding_key():
    first = Paper(paper_id="a", title="Same title", doi="10.1234/a", abstract="Same abstract")
    second = first.model_copy(deep=True, update={"paper_id": "b", "source_records": [
        {"source": "author_metadata", "author_keywords": ["Memory", "memory", "Abstention"]},
    ]})
    merged = merge_dedupe_rank([first, second], "memory")[0]
    assert author_keywords(merged) == ["Memory", "Abstention"]
    text = _paper_text(merged)
    assert text == "Same title\nAuthor keywords: Memory; Abstention\nSame abstract"
    assert _content_key("paper", text) != _content_key("paper", _paper_text(first))


def test_keyword_extraction_requires_an_explicit_field_not_abstract_inference():
    assert extract_author_keyword_section("Abstract\nWe study uncertainty and memory.") == ([], "")
    terms, quote = extract_author_keyword_section("Keywords: memory, abstention; calibration\n1. Introduction\nBody")
    assert terms == ["memory", "abstention", "calibration"]
    assert quote == "Keywords: memory, abstention; calibration"
    assert extract_author_keyword_section("关键词：长期记忆；工具调用")[0] == ["长期记忆", "工具调用"]


def test_only_existing_pdf_is_read_and_author_keyword_provenance_is_preserved(tmp_path):
    original = Paper(paper_id="cached-paper", title="Cached paper")
    path = tmp_path / (hashlib.sha256(original.paper_id.encode()).hexdigest() + ".pdf")
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 60), "Keywords: memory, abstention\n\n1. Introduction\nActual article text")
        document.save(path)
    enriched, missing = enrich_cached_author_keywords([
        original, Paper(paper_id="not-cached", title="No PDF"),
    ], tmp_path)
    assert author_keywords(enriched) == ["memory", "abstention"]
    assert enriched.metadata["author_keywords_provenance"]["page_number"] == 1
    assert "Keywords: memory" in enriched.metadata["author_keywords_provenance"]["quote"]
    assert original.metadata == {}
    assert author_keywords(missing) == []
