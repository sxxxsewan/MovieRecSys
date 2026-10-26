"""Validation-tuned weighted hybrid recommender."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence

import numpy as np

from recsys.evaluation.metrics import mean_average_precision_at_k
from recsys.models.base import rank_score_matrix, rank_scores, validate_k


class HybridRecommender:
    """Combine per-user ALS and content scores after per-user normalization."""

    def __init__(self, als_model: object, content_model: object) -> None:
        if not hasattr(als_model, "scores_for_user") or not hasattr(
            content_model, "scores_for_user"
        ):
            raise TypeError("Both hybrid components must provide scores_for_user")
        self.als_model = als_model
        self.content_model = content_model
        self.als_weight = 0.5
        self.item_ids: list[Hashable] = []
        self.is_tuned = False

    @staticmethod
    def _normalize_scores(scores: Mapping[Hashable, float]) -> dict[Hashable, float]:
        if not scores:
            return {}
        values = np.asarray(list(scores.values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Hybrid component scores must be finite")
        low = float(values.min())
        high = float(values.max())
        if high == low:
            return dict.fromkeys(scores, 0.0)
        return {
            item_id: float((score - low) / (high - low))
            for item_id, score in scores.items()
        }

    def scores_for_user(self, user_id: Hashable) -> dict[Hashable, float]:
        """Return blended normalized scores for all candidates."""
        als_scores = self._normalize_scores(self.als_model.scores_for_user(user_id))
        content_scores = self._normalize_scores(
            self.content_model.scores_for_user(user_id)
        )
        self.item_ids = sorted(
            als_scores.keys() | content_scores.keys(),
            key=str,
        )
        return {
            item_id: self.als_weight * als_scores.get(item_id, 0.0)
            + (1.0 - self.als_weight) * content_scores.get(item_id, 0.0)
            for item_id in self.item_ids
        }

    @staticmethod
    def _normalize_score_matrix(scores: np.ndarray) -> np.ndarray:
        if not np.isfinite(scores).all():
            raise ValueError("Hybrid component scores must be finite")
        minima = scores.min(axis=1, keepdims=True)
        maxima = scores.max(axis=1, keepdims=True)
        ranges = maxima - minima
        normalized = np.zeros_like(scores, dtype=np.float32)
        np.divide(
            scores - minima,
            ranges,
            out=normalized,
            where=ranges > 0,
        )
        return normalized

    def score_matrix_for_users(self, user_ids: list[Hashable]) -> np.ndarray:
        """Blend normalized ALS/content scores over the union item catalog."""
        if not hasattr(self.als_model, "score_matrix_for_users") or not hasattr(
            self.content_model, "score_matrix_for_users"
        ):
            raise TypeError("Hybrid components must support batch user scoring")
        als_ids = list(self.als_model.item_ids)
        content_ids = list(self.content_model.item_ids)
        self.item_ids = sorted(set(als_ids) | set(content_ids), key=str)
        item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}

        als_scores = self._normalize_score_matrix(
            self.als_model.score_matrix_for_users(user_ids)
        )
        content_scores = self._normalize_score_matrix(
            self.content_model.score_matrix_for_users(user_ids)
        )
        combined = np.zeros((len(user_ids), len(self.item_ids)), dtype=np.float32)
        combined[:, [item_index[item_id] for item_id in als_ids]] += (
            self.als_weight * als_scores
        )
        combined[:, [item_index[item_id] for item_id in content_ids]] += (
            1.0 - self.als_weight
        ) * content_scores
        return combined

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return the top unseen items from the blended score."""
        validate_k(k)
        scores = self.scores_for_user(user_id)
        excluded: set[Hashable] = set()
        for model in (self.als_model, self.content_model):
            model_seen = getattr(model, "user_items", {}).get(user_id, set())
            excluded.update(model_seen)
        return rank_scores(
            [scores[item_id] for item_id in self.item_ids],
            self.item_ids,
            k,
            excluded_items=excluded,
        )

    def recommend_many(
        self, user_ids: list[Hashable], k: int = 10, batch_size: int = 256
    ) -> dict[Hashable, list[tuple[Hashable, float]]]:
        """Rank users with batched ALS/content score blending."""
        validate_k(k)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not hasattr(self.als_model, "score_matrix_for_users") or not hasattr(
            self.content_model, "score_matrix_for_users"
        ):
            return {user_id: self.recommend(user_id, k) for user_id in user_ids}
        recommendations: dict[Hashable, list[tuple[Hashable, float]]] = {}
        for offset in range(0, len(user_ids), batch_size):
            users = user_ids[offset : offset + batch_size]
            scores = self.score_matrix_for_users(users)
            exclusions = [
                set(getattr(model, "user_items", {}).get(user_id, set()))
                for user_id in users
                for model in (self.als_model, self.content_model)
            ]
            merged_exclusions = [
                exclusions[index] | exclusions[index + 1]
                for index in range(0, len(exclusions), 2)
            ]
            ranked = rank_score_matrix(scores, self.item_ids, k, merged_exclusions)
            recommendations.update(zip(users, ranked, strict=True))
        return recommendations

    def tune(
        self,
        validation_relevant: Mapping[Hashable, Collection[Hashable]],
        candidate_weights: Sequence[float],
        k: int = 10,
    ) -> float:
        """Choose the ALS blend weight maximizing validation MAP@k."""
        validate_k(k)
        users = [
            user_id for user_id, relevant in validation_relevant.items() if relevant
        ]
        if not users:
            raise ValueError("Validation data must include relevant items")
        if not candidate_weights:
            raise ValueError("At least one candidate ALS weight is required")
        weights = sorted(set(float(weight) for weight in candidate_weights))
        if any(
            not np.isfinite(weight) or weight < 0 or weight > 1 for weight in weights
        ):
            raise ValueError("Candidate ALS weights must be finite values in [0, 1]")

        best_weight = weights[0]
        best_map = -1.0
        for weight in weights:
            self.als_weight = weight
            recommendations = {
                user_id: [item_id for item_id, _ in ranked]
                for user_id, ranked in self.recommend_many(users, k).items()
            }
            score = mean_average_precision_at_k(recommendations, validation_relevant, k)
            if score > best_map:
                best_map = score
                best_weight = weight
        self.als_weight = best_weight
        self.is_tuned = True
        return best_weight
