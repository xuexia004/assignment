"""Question-conditioned verbatim evidence with small retrieval units and parent context.

Inspired by PaperQA evidence gathering and parent-document retrieval. All neural
models are the application's existing local-only models. No gold evidence or LLM
call enters selection. Offsets always address the original Python string.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter

from paper_agent.models import Paper
from paper_agent.services import cross_encoder_reranker, semantic_reranker

CHILD_CHARS = 900
CHILD_OVERLAP = 120
RERANK_LIMIT = 20
CASCADE_LIMIT = 40
_STOP = set("a an the of to in for and or is are was were be by with as on at from that this these those how what which when where who does do did can using used use".split())


def segment_document(text: str) -> list[dict]:
    """Split Markdown at headings/empty lines, retaining exact paragraph spans."""
    paragraphs = []
    section = ""
    start = None
    offset = 0

    def append(end):
        if start is None:
            return
        value = text[start:end]
        left = start + len(value) - len(value.lstrip())
        right = end - len(value) + len(value.rstrip())
        if left < right:
            paragraphs.append({"id": f"P{len(paragraphs) + 1}", "start": left, "end": right,
                               "section": section, "text": text[left:right]})

    for line in text.splitlines(keepends=True):
        heading = re.match(r"^#{1,6}[ \t]+(\S.*)", line)
        if heading or not line.strip():
            append(offset)
            start = None
            if heading:
                section = heading.group(1).strip()
        elif start is None:
            start = offset
        offset += len(line)
    append(len(text))
    return paragraphs


def _children(parents: list[dict]) -> list[dict]:
    children = []
    for index, parent in enumerate(parents):
        start = parent["start"]
        while start < parent["end"]:
            end = min(parent["end"], start + CHILD_CHARS)
            if end < parent["end"]:
                boundaries = list(re.finditer(r"\s+", parent["text"][start - parent["start"]:end - parent["start"]]))
                if boundaries and boundaries[-1].start() > CHILD_CHARS // 2:
                    end = start + boundaries[-1].start()
            child_text = parent["text"][start - parent["start"]:end - parent["start"]]
            children.append({"parent": index, "start": start, "end": end,
                             "text": parent["section"][:160] + "\n" + child_text})
            if end == parent["end"]:
                break
            start = max(start + 1, end - CHILD_OVERLAP)
    return children


def _tokens(text: str) -> list[str]:
    words = [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in _STOP]
    # English model suitability is audited separately; lexical CJK still works offline.
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words


def _bm25_scores(question: str, children: list[dict]) -> list[float]:
    terms = set(_tokens(question))
    counts = [Counter(_tokens(child["text"])) for child in children]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(len(lengths), 1) or 1
    frequencies = {term: sum(term in count for count in counts) for term in terms}
    scores = []
    for count, length in zip(counts, lengths, strict=True):
        score = 0.0
        for term, frequency in frequencies.items():
            tf = count[term]
            idf = math.log(1 + (len(counts) - frequency + .5) / (frequency + .5))
            score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / average))
        scores.append(score)
    return scores


def _dense_scores(question: str, children: list[dict], audit: dict) -> list[float]:
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    texts = [question] + [child["text"] for child in children]
    model = semantic_reranker._load_model()
    lengths = [len(model.tokenizer.encode(value, add_special_tokens=True, truncation=False,
                                         verbose=False)) for value in texts]
    audit.update(dense_max_tokens=model.max_seq_length,
                 dense_query_truncated=lengths[0] > model.max_seq_length,
                 dense_truncated_child_count=sum(n > model.max_seq_length for n in lengths[1:]))
    keys = [semantic_reranker._content_key("evidence", text) for text in texts]
    vectors = [semantic_reranker._cache_get(key) for key in keys]
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        encoded = semantic_reranker._encode([texts[i] for i in missing])
        for index, vector in zip(missing, encoded, strict=True):
            vectors[index] = vector
            semantic_reranker._cache_put(keys[index], vector)
    query = vectors[0]
    return [sum(a * b for a, b in zip(query, vector, strict=True)) for vector in vectors[1:]]


def _order(scores: list[float]) -> list[int]:
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite evidence scores")
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _rank(question: str, children: list[dict], method: str, audit: dict) -> list[int]:
    lexical = _order(_bm25_scores(question, children))
    if method == "bm25":
        return lexical
    order = lexical
    if method != "cascade":
        try:
            dense = _order(_dense_scores(question, children, audit))
            if len(dense) != len(children):
                raise ValueError("Wrong embedding count")
        except Exception as exc:
            audit.update(actual_method="bm25", fallback_reason="local_embedding_unavailable",
                         error_type=type(exc).__name__)
            return lexical
        if method == "dense":
            return dense
        fused = [0.0] * len(children)
        for ranked in (lexical, dense):
            for rank, index in enumerate(ranked, 1):
                fused[index] += 1 / (60 + rank)
        order = _order(fused)
    rerank_limit = CASCADE_LIMIT if method == "cascade" else RERANK_LIMIT
    pool = order[:rerank_limit]
    audit["rerank_pool_count"] = len(pool)
    try:
        candidates = [Paper(paper_id=str(i), title=children[i]["text"]) for i in pool]
        scores, truncated = cross_encoder_reranker._predict(question, candidates)
        if len(scores) != len(pool):
            raise ValueError("Wrong cross-encoder score count")
        ranked = [pool[index] for index in _order(scores)]
        audit["truncated_child_ids"] = truncated
        return ranked + order[rerank_limit:]
    except Exception as exc:
        audit.update(actual_method="bm25" if method == "cascade" else "hybrid_rrf",
                     fallback_reason="local_cross_encoder_unavailable",
                     error_type=type(exc).__name__)
        return order


def _render(span: dict) -> str:
    return f"[{span['id']}] {span['section'][:160]}\n{span['text']}"


def _window(parent: dict, child: dict, allowance: int) -> dict | None:
    """Only oversized parents need an explicitly partial, verbatim child window."""
    start = child["start"]
    # Decode only complete UTF-8 characters, and retain original character offsets.
    remaining = parent["text"][start - parent["start"]:]
    overhead = len(_render({**parent, "text": ""}).encode("utf-8"))
    value = remaining.encode("utf-8")[:max(0, allowance - overhead)].decode("utf-8", "ignore")
    if not value.strip():
        return None
    return {**parent, "start": start, "end": start + len(value), "text": value,
            "partial_parent": True}


def build_evidence_context(question: str, text: str, max_bytes: int = 12000,
                           method: str = "hybrid") -> tuple[str, dict]:
    """Return selected original passages and their provenance within a UTF-8 cap.

    ``max_bytes`` includes citation IDs, section labels and separators. The caller
    reserves additional prompt/JSON space. Selected text is never a fulltext claim.
    The fixed method is frozen before the held-out evaluation, with no gold inputs.
    """
    if method not in {"bm25", "dense", "hybrid", "cascade"}:
        raise ValueError("Unknown evidence retrieval method")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("Evidence budget must be a positive integer")
    started = time.perf_counter()
    parents = segment_document(text)
    audit = {"method_version": "sparse_cascade_v1" if method == "cascade" else "parent_evidence_v1",
             "requested_method": method,
             "actual_method": method, "retrieval_query": question,
             "input_bytes": len(text.encode("utf-8")), "context_bytes": 0,
             "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "max_bytes": max_bytes, "parent_count": len(parents), "selected_spans": [],
             "fallback_reason": None, "selection_is_fulltext": False,
             "embedding_model": None if method == "cascade" else semantic_reranker.MODEL_NAME,
             "reranker_model": cross_encoder_reranker.MODEL_NAME,
             "reranker_revision": cross_encoder_reranker.MODEL_REVISION,
             "child_chars": CHILD_CHARS, "child_overlap": CHILD_OVERLAP,
             "rerank_limit": CASCADE_LIMIT if method == "cascade" else RERANK_LIMIT,
             "rrf_k": None if method == "cascade" else 60,
             "english_models": True}
    if not parents or not question.strip():
        audit.update(fallback_reason="empty_input", duration_seconds=time.perf_counter() - started)
        return "", audit
    children = _children(parents)
    audit["child_count"] = len(children)
    ranked = _rank(question, children, method, audit)
    selected = []
    seen = set()
    remaining = max_bytes
    for child_index in ranked:
        child = children[child_index]
        parent_index = child["parent"]
        if parent_index in seen:
            continue
        seen.add(parent_index)
        parent = parents[parent_index]
        span = {**parent, "partial_parent": False}
        cost = len(_render(span).encode("utf-8")) + (2 if selected else 0)
        if cost > remaining:
            # Avoid partial paragraphs merely to fill a few unused bytes. A parent
            # exceeding the entire cap can still contribute a tracked source window.
            if selected or len(_render(span).encode("utf-8")) <= max_bytes:
                continue
            span = _window(parent, child, remaining - (2 if selected else 0))
            if span is None:
                continue
            cost = len(_render(span).encode("utf-8")) + (2 if selected else 0)
        selected.append(span)
        remaining -= cost
    selected.sort(key=lambda span: span["start"])
    context = "\n\n".join(_render(span) for span in selected)
    audit.update(selected_spans=selected, context_bytes=len(context.encode("utf-8")),
                 selected_parent_count=len(selected),
                 duration_seconds=time.perf_counter() - started)
    return context, audit
