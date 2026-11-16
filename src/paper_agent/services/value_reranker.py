"""Relevance-constrained citation/recency utility with MMR redundancy control."""

from __future__ import annotations

import math
from datetime import date

from paper_agent.models import Paper


def value_rerank(papers: list[Paper], vectors: dict[str, list[float]], *, as_of: date,
                 prefer_recent: bool = False, anchor_count: int = 5) -> tuple[list[Paper], dict]:
    """Only reorder the first 40 relevant candidates; never use raw citation totals."""
    pool = papers[:40]
    audit = {"method": "relevance_anchored_utility_mmr_v2", "as_of": as_of.isoformat(),
             "pool_limit": 40, "applied": False, "prefer_recent": prefer_recent,
             "protected_count": min(max(0, anchor_count), len(pool)),
             "baseline_ids": [p.paper_id for p in pool], "score_is_probability": False}
    if not pool or any(p.paper_id not in vectors for p in pool):
        return papers, {**audit, "reason": "semantic_vectors_unavailable"}
    if any(not all(math.isfinite(v) for v in vectors[p.paper_id]) for p in pool):
        return papers, {**audit, "reason": "invalid_vectors"}
    dimensions = {len(vectors[p.paper_id]) for p in pool}
    if len(dimensions) != 1 or 0 in dimensions:
        return papers, {**audit, "reason": "invalid_vectors"}
    # Normalize defensively; production vectors are already unit-normalized.
    unit = {}
    for paper in pool:
        vector = vectors[paper.paper_id]
        norm = math.sqrt(sum(v * v for v in vector))
        if not norm:
            return papers, {**audit, "reason": "invalid_vectors"}
        unit[paper.paper_id] = [v / norm for v in vector]
    features = {}
    for index, paper in enumerate(pool):
        percentile = paper.metadata.get("citation_normalized_percentile")
        known_impact = type(percentile) in {int, float} and math.isfinite(percentile) and 0 <= percentile <= 1
        impact = float(percentile) if known_impact else .5
        age = as_of.year - paper.year if paper.year is not None and paper.year <= as_of.year else None
        freshness = math.exp(-age / 3) if age is not None else .5
        relevance = 1 - index / max(1, len(pool) - 1)
        # Recency is a weak default prior; explicit recent-work queries strengthen it.
        weights = (.75, .10, .15) if prefer_recent else (.85, .10, .05)
        utility = weights[0] * relevance + weights[1] * impact + weights[2] * freshness
        features[paper.paper_id] = {"baseline_rank": index + 1, "relevance_prior": relevance,
                                    "impact_percentile": percentile if known_impact else None,
                                    "impact_missing": not known_impact, "freshness": freshness,
                                    "age_years": age, "utility": utility, "weights": list(weights)}
    remaining = list(pool)
    chosen, output = [], []
    while remaining:
        def score(paper):
            similarity = max((max(0., min(1., sum(a * b for a, b in zip(
                unit[paper.paper_id], unit[selected], strict=True)))) for selected in chosen), default=0.)
            return features[paper.paper_id]["utility"] - .10 * similarity, similarity
        winner = (remaining[0] if len(chosen) < audit["protected_count"] else
                  max(remaining, key=lambda p: (score(p)[0], -features[p.paper_id]["baseline_rank"])))
        value, redundancy = score(winner)
        updated = winner.model_copy(deep=True)
        updated.rank = len(chosen) + 1
        updated.ranking_details.update(value_rank=updated.rank, value_score=value,
                                       value_features={**features[winner.paper_id], "redundancy": redundancy,
                                                       "protected_anchor": len(chosen) < audit["protected_count"]},
                                       value_method=audit["method"])
        updated.ranking_reason = ((updated.ranking_reason or "") +
                                  f" 参考价值调整：原相关性名次 {features[winner.paper_id]['baseline_rank']}，"
                                  f"调整后 {updated.rank}；引用指标{'未知' if features[winner.paper_id]['impact_missing'] else '已按领域/年份归一化'}。")
        output.append(updated)
        chosen.append(winner.paper_id)
        remaining.remove(winner)
    audit.update(applied=True, ordered_ids=chosen, impact_known_count=sum(
        not f["impact_missing"] for f in features.values()), redundancy_weight=.10,
                 warning="Utility and diversity signals do not establish scientific correctness.")
    return output + papers[40:], audit
