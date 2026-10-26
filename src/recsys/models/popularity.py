"""Popularity baseline recommender."""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from recsys.models.base import (
    rank_score_matrix,
    rank_scores,
    validate_interactions,
    validate_k,
)


class PopularityRecommender:
    """Rank items by training rating count or Bayesian-smoothed rating."""

    def __init__(self, metric: str = "rating_count", smoothing: float = 10.0) -> None:
        if metric not in {"rating_count", "bayesian_rating"}:
            raise ValueError("metric must be 'rating_count' or 'bayesian_rating'")
        if smoothing < 0:
            raise ValueError("smoothing must be nonnegative")
        self.metric = metric
        self.smoothing = float(smoothing)
        self.item_ids: list[Hashable] = []
        self.scores: list[float] = []
        self.user_items: dict[Hashable, set[Hashable]] = {}
        self.item_popularity: dict[Hashable, int] = {}
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> PopularityRecommender:
        """Fit item counts and user history using the supplied training rows."""
        validate_interactions(interactions)
        grouped = interactions.groupby("movie_id", sort=True)
        counts = grouped.size()
        self.item_ids = counts.index.tolist()
        self.item_popularity = counts.astype(int).to_dict()
        self.user_items = interactions.groupby("user_id")["movie_id"].agg(set).to_dict()

        if self.metric == "rating_count":
            self.scores = counts.astype(float).tolist()
        else:
            global_mean = float(interactions["rating"].mean())
            item_means = grouped["rating"].mean()
            smoothed = (counts * item_means + self.smoothing * global_mean) / (
                counts + self.smoothing
            )
            self.scores = smoothed.tolist()
        self.is_fitted = True
        return self

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return the top-k unseen items; unknown users receive the baseline."""
        if not self.is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before use")
        validate_k(k)
        return rank_scores(
            self.scores,
            self.item_ids,
            k,
            excluded_items=self.user_items.get(user_id, set()),
        )

    def scores_for_user(self, user_id: Hashable) -> dict[Hashable, float]:
        """Return popularity scores for all items, including seen items."""
        if not self.is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before use")
        return dict(zip(self.item_ids, self.scores, strict=True))

    def recommend_many(
        self, user_ids: list[Hashable], k: int = 10, batch_size: int = 256
    ) -> dict[Hashable, list[tuple[Hashable, float]]]:
        """Rank users in batches while sharing the baseline score vector."""
        if not self.is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before use")
        validate_k(k)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        recommendations: dict[Hashable, list[tuple[Hashable, float]]] = {}
        item_scores = pd.Series(self.scores, dtype="float64").to_numpy()
        for offset in range(0, len(user_ids), batch_size):
            users = user_ids[offset : offset + batch_size]
            scores = item_scores[None, :].repeat(len(users), axis=0)
            exclusions = [self.user_items.get(user_id, set()) for user_id in users]
            ranked = rank_score_matrix(scores, self.item_ids, k, exclusions)
            recommendations.update(zip(users, ranked, strict=True))
        return recommendations
