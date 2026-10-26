"""Shared interfaces and helpers for recommender models."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd

UserId = TypeVar("UserId", bound=Hashable)
ItemId = TypeVar("ItemId", bound=Hashable)


class Recommender(Protocol):
    """Minimal interface used by offline evaluation and serving adapters."""

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return ranked (item ID, score) pairs for a user."""


def validate_interactions(interactions: pd.DataFrame) -> None:
    """Validate the columns and values expected by collaborative models."""
    required = {"user_id", "movie_id", "rating"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(
            f"Interactions are missing required columns: {', '.join(sorted(missing))}"
        )
    if interactions.empty:
        raise ValueError("Interactions must not be empty")
    if interactions[["user_id", "movie_id", "rating"]].isna().any().any():
        raise ValueError("Interaction user_id, movie_id, and rating must not be null")
    if interactions.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Interactions must have at most one rating per user/movie")


def validate_k(k: int) -> None:
    """Reject invalid recommendation cutoffs."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def rank_scores(
    scores: Sequence[float],
    item_ids: Sequence[ItemId],
    k: int,
    excluded_items: set[ItemId] | None = None,
) -> list[tuple[ItemId, float]]:
    """Sort finite candidate scores, breaking ties deterministically by item ID."""
    validate_k(k)
    excluded = excluded_items or set()
    candidates = [
        (item_id, float(score))
        for item_id, score in zip(item_ids, scores, strict=True)
        if item_id not in excluded
    ]
    candidates.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return candidates[:k]


def rank_score_matrix(
    scores: np.ndarray,
    item_ids: Sequence[ItemId],
    k: int,
    excluded_items: Sequence[set[ItemId]],
) -> list[list[tuple[ItemId, float]]]:
    """Rank score rows with deterministic ties and bounded candidate sorting."""
    validate_k(k)
    if scores.ndim != 2 or scores.shape != (len(excluded_items), len(item_ids)):
        raise ValueError("Score matrix dimensions must match users and item IDs")
    results: list[list[tuple[ItemId, float]]] = []
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    for row, excluded in zip(scores, excluded_items, strict=True):
        ranked_scores = row.copy()
        excluded_indices = [
            item_index[item_id] for item_id in excluded if item_id in item_index
        ]
        ranked_scores[excluded_indices] = -np.inf
        valid_indices = np.flatnonzero(np.isfinite(ranked_scores))
        if valid_indices.size == 0:
            results.append([])
            continue
        valid_scores = ranked_scores[valid_indices]
        selected_count = min(k, valid_indices.size)
        if valid_indices.size > selected_count:
            cutoff_index = np.argpartition(valid_scores, -selected_count)[
                -selected_count
            ]
            cutoff_score = valid_scores[cutoff_index]
            higher = valid_indices[valid_scores > cutoff_score]
            tied = valid_indices[valid_scores == cutoff_score]
            tied = sorted(tied.tolist(), key=lambda index: str(item_ids[index]))
            selected = np.concatenate(
                [
                    higher,
                    np.asarray(tied[: selected_count - len(higher)], dtype=np.intp),
                ]
            )
        else:
            selected = valid_indices
        ranked_indices = sorted(
            selected.tolist(),
            key=lambda index: (-float(ranked_scores[index]), str(item_ids[index])),
        )
        results.append(
            [(item_ids[index], float(ranked_scores[index])) for index in ranked_indices]
        )
    return results
