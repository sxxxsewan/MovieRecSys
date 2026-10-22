"""Shared interfaces and helpers for recommender models."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Protocol, TypeVar

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
