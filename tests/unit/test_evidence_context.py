"""Synthetic regression fixtures; quality is measured on held-out public evidence."""

import sys
from types import SimpleNamespace

import pytest

from paper_agent.agents import evidence_context as module


def test_markdown_offsets_and_duplicate_paragraphs():
    text = "# Title\r\n\r\n  First line.\nsecond line.  \n\n## Methods\nSame.\n\nSame.\n"
    spans = module.segment_document(text)
    assert [p["id"] for p in spans] == ["P1", "P2", "P3"]
    assert [p["section"] for p in spans] == ["Title", "Methods", "Methods"]
    assert all(text[p["start"]:p["end"]] == p["text"] for p in spans)
    assert spans[1]["start"] != spans[2]["start"]


def test_lexical_selection_keeps_relevant_parent_and_original_order():
    text = "# Results\n\nControl accuracy was 20.\n\n" + "Other method. " * 15 + "\n\nTreatment accuracy was 80 under the same setting."
    context, audit = module.build_evidence_context("control treatment accuracy", text, 130, "bm25")
    assert "Control accuracy was 20." in context
    assert "Treatment accuracy was 80 under the same setting." in context
    assert len(context.encode("utf-8")) <= 130
    assert [s["id"] for s in audit["selected_spans"]] == ["P1", "P3"]
    assert audit["selection_is_fulltext"] is False


def test_neural_rerank_restores_parent_and_records_local_failure(monkeypatch):
    text = "# Experiment\n\n" + "background text. " * 100 + "\n\nrare target and its full condition."
    monkeypatch.setattr(module, "_dense_scores", lambda q, cs, audit: [float(i) for i in range(len(cs))])
    monkeypatch.setattr(module.cross_encoder_reranker, "_predict", lambda q, ps: ([10 if "rare" in p.title else -10 for p in ps], []))
    _, audit = module.build_evidence_context("target", text, 120, "hybrid")
    assert audit["actual_method"] == "hybrid"
    assert any(s["text"] == "rare target and its full condition." for s in audit["selected_spans"])

    def unavailable(*args):
        raise OSError("Local model absent")

    monkeypatch.setattr(module, "_dense_scores", unavailable)
    _, fallback = module.build_evidence_context("target", text, 120, "hybrid")
    assert fallback["actual_method"] == "bm25"
    assert fallback["fallback_reason"] == "local_embedding_unavailable"


def test_oversized_parent_chinese_budget_and_empty_inputs():
    text = "# 条件\n\n" + "控制组条件相同，不能忽略实验限制。" * 200
    context, audit = module.build_evidence_context("实验限制", text, 130, "bm25")
    assert 0 < len(context.encode("utf-8")) <= 130
    span = audit["selected_spans"][0]
    assert span["partial_parent"]
    assert span["text"] == text[span["start"]:span["end"]]
    assert module.build_evidence_context("q", "", 20)[0] == ""
    assert module.build_evidence_context("", "text", 20)[0] == ""
    assert module.build_evidence_context("q", "text", 1, "bm25")[0] == ""
    with pytest.raises(ValueError):
        module.build_evidence_context("q", "text", 0)
    with pytest.raises(ValueError):
        module.build_evidence_context("q", "text", 20, "unknown")


def test_dense_truncation_and_cross_encoder_fallback_are_audited(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        get_num_threads=lambda: 4, set_num_threads=lambda n: None))
    fake = SimpleNamespace(max_seq_length=5, tokenizer=SimpleNamespace(
        encode=lambda value, **kwargs: list(value)))
    monkeypatch.setattr(module.semantic_reranker, "_load_model", lambda: fake)
    monkeypatch.setattr(module.semantic_reranker, "_cache_get", lambda key: [1.0, 0.0])
    audit = {}
    module._dense_scores("a long question", [{"text": "short"}, {"text": "long text"}], audit)
    assert audit["dense_query_truncated"]
    assert audit["dense_truncated_child_count"] == 1

    def unavailable(*args):
        raise OSError("Local reranker absent")

    monkeypatch.setattr(module.cross_encoder_reranker, "_predict", unavailable)
    _, audit = module.build_evidence_context("question", "answer paragraph", 100, "hybrid")
    assert audit["actual_method"] == "hybrid_rrf"
    assert audit["fallback_reason"] == "local_cross_encoder_unavailable"


def test_cascade_reranks_only_lexical_pool_without_dense_encoding(monkeypatch):
    children = [{"text": f"paragraph {index}"} for index in range(48)]
    monkeypatch.setattr(module, "_bm25_scores", lambda q, cs: list(range(len(cs))))
    dense_calls = []

    def no_dense(*args):
        dense_calls.append(args)
        raise AssertionError("Cascade must not encode the document")

    monkeypatch.setattr(module, "_dense_scores", no_dense)
    pools = []

    def predict(query, papers):
        pools.append([int(paper.paper_id) for paper in papers])
        return list(range(len(papers))), []

    monkeypatch.setattr(module.cross_encoder_reranker, "_predict", predict)
    audit = {"actual_method": "cascade"}
    cascade = module._rank("question", children, "cascade", audit)
    assert pools == [list(range(47, 7, -1))]
    assert cascade == list(range(8, 48)) + list(range(7, -1, -1))
    assert not dense_calls
    assert audit["actual_method"] == "cascade"
    assert audit["rerank_pool_count"] == 40

    # Hybrid retains its pre-existing 20-candidate pool and untouched tail.
    monkeypatch.setattr(module, "_dense_scores", lambda q, cs, audit: list(range(len(cs))))
    hybrid = module._rank("question", children, "hybrid", {"actual_method": "hybrid"})
    assert pools[1] == list(range(47, 27, -1))
    assert hybrid == list(range(28, 48)) + list(range(27, -1, -1))


def test_cascade_keeps_offsets_budget_and_explicit_lexical_fallback(monkeypatch):
    text = "# 条件\n\nControl accuracy was 20.\n\nTreatment accuracy was 80."
    monkeypatch.setattr(module.cross_encoder_reranker, "_predict",
                        lambda q, ps: ([1.0] * len(ps), []))
    context, audit = module.build_evidence_context("accuracy", text, 100, "cascade")
    assert 0 < len(context.encode("utf-8")) <= 100
    assert audit["actual_method"] == "cascade"
    assert audit["embedding_model"] is None and audit["rrf_k"] is None
    assert audit["rerank_limit"] == 40 and audit["rerank_pool_count"] == 2
    assert all(text[span["start"]:span["end"]] == span["text"]
               for span in audit["selected_spans"])

    def unavailable(*args):
        raise OSError("Local reranker absent")

    monkeypatch.setattr(module.cross_encoder_reranker, "_predict", unavailable)
    fallback_context, fallback = module.build_evidence_context("accuracy", text, 100, "cascade")
    lexical_context, _ = module.build_evidence_context("accuracy", text, 100, "bm25")
    assert fallback_context == lexical_context
    assert fallback["actual_method"] == "bm25"
    assert fallback["fallback_reason"] == "local_cross_encoder_unavailable"
