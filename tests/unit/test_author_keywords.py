"""Author-keyword provenance and semantic input checks; no model/API calls."""

import hashlib
import json

import pymupdf
import pytest

from paper_agent.models import Paper
from paper_agent.services.author_keywords import (
    author_keywords,
    enrich_cached_author_keywords,
    extract_author_keyword_section,
)
from paper_agent.services.downloader import pdf_cache_identity, pdf_cache_key
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


def _write_keyword_pdf(path, terms):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 60), f"Keywords: {terms}\n\n1. Introduction\nMock article text")
        document.save(path)


@pytest.mark.parametrize("damage", [None, "version", "hash", "missing_manifest"])
def test_explicit_revision_keywords_require_matching_verified_cache(tmp_path, damage):
    original = Paper(paper_id="arxiv:2601.01234", title="Revision keyword fixture",
                     arxiv_id="2601.01234", arxiv_version="v2")
    legacy = tmp_path / (hashlib.sha256(original.paper_id.encode()).hexdigest() + ".pdf")
    _write_keyword_pdf(legacy, "obsolete")
    assert author_keywords(enrich_cached_author_keywords([original], tmp_path)[0]) == []
    identity, ambiguous = pdf_cache_identity(original)
    assert not ambiguous
    path = tmp_path / (pdf_cache_key(identity) + ".pdf")
    _write_keyword_pdf(path, "current, verified")
    content = path.read_bytes()
    manifest = {"schema": 1, **identity, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if damage == "version":
        manifest["arxiv_version"] = "v1"
    elif damage == "hash":
        manifest["sha256"] = "0" * 64
    if damage != "missing_manifest":
        path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    enriched = enrich_cached_author_keywords([original], tmp_path)[0]
    assert author_keywords(enriched) == ([] if damage else ["current", "verified"])
    if not damage:
        assert enriched.metadata["author_keywords_provenance"]["cache_identity"] == identity
    stale = original.model_copy(deep=True)
    stale.metadata = {
        "author_keywords": ["obsolete"],
        "author_keywords_provenance": {
            "source": "pdf_keyword_section", "cache_identity": {**identity, "arxiv_version": "v1"},
        },
    }
    assert author_keywords(enrich_cached_author_keywords([stale], tmp_path)[0]) == author_keywords(enriched)
    assert stale.metadata["author_keywords"] == ["obsolete"]
    assert original.metadata == {} and legacy.is_file()


@pytest.mark.parametrize("record_identity", [False, True])
def test_revised_paper_discards_only_proven_old_pdf_keywords(tmp_path, record_identity):
    original = Paper(paper_id="arxiv:2601.01234", title="Keyword revision fixture",
                     arxiv_id="2601.01234", arxiv_version="v2", metadata={
                         "author_keywords": ["old PDF term"],
                         "author_keywords_provenance": {"source": "pdf_keyword_section"},
                     }, source_records=[{"author_keywords": ["explicit source term"]}])
    if record_identity:
        old_identity, _ = pdf_cache_identity(original.model_copy(update={"arxiv_version": "v1"}))
        original.metadata["author_keywords_provenance"]["cache_identity"] = old_identity
    enriched = enrich_cached_author_keywords([original], tmp_path)[0]
    assert author_keywords(enriched) == (["explicit source term"] if record_identity
                                        else ["old PDF term", "explicit source term"])
    assert original.metadata["author_keywords"] == ["old PDF term"]
