"""
Evaluation metrics for ranking, coverage, and knowledge tracing.
Implements Equations (14) through (17) from Section IV-E.
"""

from typing import Dict, List, Set
import numpy as np


def precision_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Eq. (14): P@K = |Rel_u ∩ Rec_u^K| / K"""
    if k <= 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for item in rec_k if item in relevant_ids)
    return float(hits / k)


def ndcg_at_k(recommended_ids: List[str], test_ratings: Dict[str, float], k: int) -> float:
    """
    Eq. (15): NDCG@K
    rel_j is engagement score / 5.0 if in test_ratings, else 0.0.
    """
    if k <= 0 or not test_ratings:
        return 0.0

    rec_k = recommended_ids[:k]
    dcg = 0.0
    for j, item_id in enumerate(rec_k):
        rating = test_ratings.get(item_id, 0.0)
        rel_j = rating / 5.0
        gain = (2.0 ** rel_j) - 1.0
        discount = np.log2(j + 2)  # j is 0-indexed, so log2(j + 1 + 1)
        dcg += gain / discount

    # Ideal DCG
    ideal_ratings = sorted(test_ratings.values(), reverse=True)[:k]
    idcg = 0.0
    for j, rating in enumerate(ideal_ratings):
        rel_j = rating / 5.0
        gain = (2.0 ** rel_j) - 1.0
        discount = np.log2(j + 2)
        idcg += gain / discount

    if idcg < 1e-9:
        return 0.0

    return float(dcg / idcg)


def coverage_at_k(
    recommended_items: List[any], truly_weak_topics: Set[str], k: int
) -> float:
    """
    Eq. (17): Cov@K = (1 / K) * sum_{i in Rec_u^K} I[ T(i) ∩ W_u* != empty ]
    Fraction of top-K recommended slots that touch at least one of student's true weak topics.
    """
    if k <= 0 or not recommended_items:
        return 0.0

    rec_k = recommended_items[:k]
    weak_touch_count = 0

    for item in rec_k:
        # Check topic tags of the resource
        topics = getattr(item, "topics", None)
        if topics is None and hasattr(item, "resource"):
            topics = item.resource.topics

        if topics and any(t in truly_weak_topics for t in topics):
            weak_touch_count += 1

    return float(weak_touch_count / len(rec_k))
