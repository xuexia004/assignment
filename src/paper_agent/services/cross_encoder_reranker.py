"""Optional retrieve-then-rerank using a local query/document cross-encoder."""

from __future__ import annotations

import asyncio
import math
import threading
import time

from paper_agent.config import SELECTOR_CANDIDATE_LIMIT
from paper_agent.models import Paper
from paper_agent.services.semantic_reranker import _paper_text

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
_MODEL = None
_LOCK = threading.Lock()


def _predict(query: str, papers: list[Paper]):
    global _MODEL
    with _LOCK:
        import torch
        from sentence_transformers import CrossEncoder

        if _MODEL is None:
            # Bound CPU oversubscription in this backend process, including shared torch encoders.
            torch.set_num_threads(min(4, torch.get_num_threads()))
            _MODEL = CrossEncoder(MODEL_NAME, revision=MODEL_REVISION, max_length=512,
                                  device="cpu", local_files_only=True, token=False)
        tokenizer = _MODEL.tokenizer
        query_length = len(tokenizer.encode(query, add_special_tokens=False, verbose=False))
        if query_length >= 509:
            raise ValueError("Query exceeds the cross-encoder pair budget")
        lengths = [query_length + len(tokenizer.encode(_paper_text(p), add_special_tokens=False, verbose=False)) + 3
                   for p in papers]
        scores = _MODEL.predict([(query, _paper_text(p)) for p in papers], batch_size=16,
                                show_progress_bar=False, activation_fn=torch.nn.Identity())
    return [float(s) for s in scores], [p.paper_id for p, length in zip(papers, lengths, strict=True) if length > 512]


async def cross_encoder_rerank(query: str, papers: list[Paper]) -> tuple[list[Paper], dict]:
    started = time.perf_counter()
    # Keep exactly the existing Selector input set; later fallback candidates stay ordered.
    pool = papers[:SELECTOR_CANDIDATE_LIMIT]
    audit = {"method": "local_cross_encoder_v1", "model": MODEL_NAME, "applied": False,
             "model_revision": MODEL_REVISION,
             "pool_limit": SELECTOR_CANDIDATE_LIMIT, "pair_max_tokens": 512, "score_is_probability": False,
             "baseline_ids": [p.paper_id for p in pool]}
    if not query.strip() or not pool:
        return papers, {**audit, "reason": "empty_input"}
    try:
        scores, truncated = await asyncio.to_thread(_predict, query, pool)
        if len(scores) != len(pool) or any(not math.isfinite(s) for s in scores):
            raise ValueError("Cross-encoder returned invalid scores")
    except Exception as exc:
        return papers, {**audit, "reason": "local_reranker_unavailable", "error_type": type(exc).__name__,
                        "duration_seconds": time.perf_counter() - started}
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    output = []
    for rank, index in enumerate(order, 1):
        paper = pool[index].model_copy(deep=True)
        paper.rank = rank
        paper.ranking_details.update(cross_encoder_score=scores[index], cross_encoder_rank=rank,
                                     cross_encoder_model=MODEL_NAME, cross_encoder_original_rank=index + 1)
        output.append(paper)
    audit.update(applied=True, ordered_ids=[p.paper_id for p in output],
                 truncated_pair_ids=truncated, duration_seconds=time.perf_counter() - started)
    return output + papers[SELECTOR_CANDIDATE_LIMIT:], audit
