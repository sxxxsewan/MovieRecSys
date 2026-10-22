"""Ranking and catalog metrics for offline recommendation evaluation."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from math import log2
from numbers import Integral
from typing import TypeVar

UserId = TypeVar("UserId", bound=Hashable)
ItemId = TypeVar("ItemId", bound=Hashable)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")


def _unique_prefix(items: Sequence[ItemId], k: int) -> list[ItemId]:
    unique: list[ItemId] = []
    seen: set[ItemId] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
            if len(unique) == k:
                break
    return unique


def precision_at_k(
    recommendations: Sequence[ItemId], relevant_items: Collection[ItemId], k: int
) -> float:
    """Return precision at k, dividing hits by k even for short lists."""
    _validate_k(k)
    relevant = set(relevant_items)
    recommended = _unique_prefix(recommendations, k)
    return sum(item in relevant for item in recommended) / k


def recall_at_k(
    recommendations: Sequence[ItemId], relevant_items: Collection[ItemId], k: int
) -> float:
    """Return the fraction of relevant items found in the first k results."""
    _validate_k(k)
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    recommended = _unique_prefix(recommendations, k)
    return sum(item in relevant for item in recommended) / len(relevant)


def average_precision_at_k(
    recommendations: Sequence[ItemId], relevant_items: Collection[ItemId], k: int
) -> float:
    """Return AP@k, normalized by min(number of relevant items, k)."""
    _validate_k(k)
    relevant = set(relevant_items)
    denominator = min(len(relevant), k)
    if denominator == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(_unique_prefix(recommendations, k), start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / denominator


def ndcg_at_k(
    recommendations: Sequence[ItemId], relevant_items: Collection[ItemId], k: int
) -> float:
    """Return binary relevance NDCG at k."""
    _validate_k(k)
    relevant = set(relevant_items)
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, item in enumerate(_unique_prefix(recommendations, k), start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


def mean_average_precision_at_k(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    k: int,
) -> float:
    """Return macro-averaged AP@k over users with at least one relevant item."""
    _validate_k(k)
    evaluable_users = [
        user_id for user_id, relevant in relevant_items.items() if relevant
    ]
    if not evaluable_users:
        raise ValueError("At least one user must have relevant items")
    return sum(
        average_precision_at_k(
            recommendations.get(user_id, ()), relevant_items[user_id], k
        )
        for user_id in evaluable_users
    ) / len(evaluable_users)


def catalog_coverage(
    recommendations: Mapping[UserId, Sequence[ItemId]], catalog_size: int
) -> float:
    """Return the fraction of the catalog represented in recommendations."""
    if isinstance(catalog_size, bool) or not isinstance(catalog_size, Integral):
        raise ValueError("catalog_size must be a positive integer")
    if catalog_size < 1:
        raise ValueError("catalog_size must be a positive integer")
    recommended_items = {
        item for user_items in recommendations.values() for item in user_items
    }
    return len(recommended_items) / catalog_size


def novelty(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    item_popularity: Mapping[ItemId, int],
) -> float:
    """Return mean self-information in bits using training interaction counts.

    Laplace smoothing gives unseen recommended items a finite, well-defined
    novelty score. The denominator includes one pseudo-count per catalog item.
    """
    all_items = [item for user_items in recommendations.values() for item in user_items]
    if not all_items:
        raise ValueError("Novelty requires at least one recommendation")
    if not item_popularity:
        raise ValueError("item_popularity must contain the training catalog")

    counts: dict[ItemId, int] = {}
    for item, count in item_popularity.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Item popularity counts must be nonnegative integers")
        counts[item] = int(count)

    total_interactions = sum(counts.values())
    if total_interactions == 0:
        raise ValueError("item_popularity must contain at least one interaction")
    denominator = total_interactions + len(counts)
    return sum(
        -log2((counts.get(item, 0) + 1) / denominator) for item in all_items
    ) / len(all_items)


def average_popularity(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    item_popularity: Mapping[ItemId, int],
) -> float:
    """Return mean training interaction count for recommended items."""
    all_items = [item for user_items in recommendations.values() for item in user_items]
    if not all_items:
        raise ValueError("Average popularity requires at least one recommendation")
    if any(
        isinstance(count, bool) or not isinstance(count, Integral) or count < 0
        for count in item_popularity.values()
    ):
        raise ValueError("Item popularity counts must be nonnegative integers")
    return sum(item_popularity.get(item, 0) for item in all_items) / len(all_items)


def evaluate_recommendations(
    recommendations: Mapping[UserId, Sequence[ItemId]],
    relevant_items: Mapping[UserId, Collection[ItemId]],
    item_popularity: Mapping[ItemId, int],
    catalog_size: int,
    k: int = 10,
) -> dict[str, float]:
    """Calculate ranking quality and catalog metrics for recommendation lists."""
    _validate_k(k)
    evaluable_users = [
        user_id for user_id, relevant in relevant_items.items() if relevant
    ]
    if not evaluable_users:
        raise ValueError("At least one user must have relevant items")

    return {
        f"ndcg@{k}": sum(
            ndcg_at_k(recommendations.get(user_id, ()), relevant_items[user_id], k)
            for user_id in evaluable_users
        )
        / len(evaluable_users),
        f"recall@{k}": sum(
            recall_at_k(recommendations.get(user_id, ()), relevant_items[user_id], k)
            for user_id in evaluable_users
        )
        / len(evaluable_users),
        f"map@{k}": mean_average_precision_at_k(recommendations, relevant_items, k),
        f"precision@{k}": sum(
            precision_at_k(recommendations.get(user_id, ()), relevant_items[user_id], k)
            for user_id in evaluable_users
        )
        / len(evaluable_users),
        "coverage": catalog_coverage(recommendations, catalog_size),
        "novelty": novelty(recommendations, item_popularity),
        "average_popularity": average_popularity(recommendations, item_popularity),
    }
