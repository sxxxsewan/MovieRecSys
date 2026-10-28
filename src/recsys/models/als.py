"""Explicit-feedback Alternating Least Squares and an svds reference model."""

from __future__ import annotations

from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from threadpoolctl import threadpool_limits

from recsys.models.base import (
    rank_score_matrix,
    rank_scores,
    validate_interactions,
    validate_k,
)


def _interaction_matrix(
    interactions: pd.DataFrame,
    user_items: dict[Hashable, set[Hashable]] | None = None,
) -> tuple[
    list[Hashable],
    list[Hashable],
    dict[Hashable, int],
    dict[Hashable, int],
    sparse.csr_matrix,
    dict[Hashable, set[Hashable]],
]:
    validate_interactions(interactions)
    user_ids = sorted(interactions["user_id"].unique().tolist(), key=str)
    item_ids = sorted(interactions["movie_id"].unique().tolist(), key=str)
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    rows = interactions["user_id"].map(user_index).to_numpy(dtype=np.int32)
    columns = interactions["movie_id"].map(item_index).to_numpy(dtype=np.int32)
    ratings = interactions["rating"].astype(np.float32).to_numpy()
    matrix = sparse.csr_matrix(
        (ratings, (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    fitted_user_items = (
        user_items
        if user_items is not None
        else interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
    )
    return user_ids, item_ids, user_index, item_index, matrix, fitted_user_items


class ALSRecommender:
    """Fit explicit-feedback matrix factors by alternating regularized solves."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        if (
            isinstance(factors, bool)
            or not isinstance(factors, Integral)
            or factors < 1
        ):
            raise ValueError("factors must be a positive integer")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, Integral)
            or iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(workers, bool)
            or not isinstance(workers, Integral)
            or workers < 1
        ):
            raise ValueError("workers must be a positive integer")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.workers = int(workers)
        self.user_ids: list[Hashable] = []
        self.item_ids: list[Hashable] = []
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_items: dict[Hashable, set[Hashable]] = {}
        self.is_fitted = False

    def fit(
        self,
        interactions: pd.DataFrame,
        user_items: dict[Hashable, set[Hashable]] | None = None,
    ) -> ALSRecommender:
        """Optimize user and item factors with alternating least squares."""
        (
            self.user_ids,
            self.item_ids,
            self.user_index,
            self.item_index,
            matrix,
            self.user_items,
        ) = _interaction_matrix(interactions, user_items=user_items)
        matrix_csc = matrix.tocsc()
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.factors)
        user_factors = rng.normal(0.0, scale, size=(len(self.user_ids), self.factors))
        item_factors = rng.normal(0.0, scale, size=(len(self.item_ids), self.factors))
        identity = np.eye(self.factors, dtype=np.float64)

        user_ranges = self._work_ranges(matrix.shape[0])
        item_ranges = self._work_ranges(matrix_csc.shape[1])
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for _ in range(self.iterations):
                    list(
                        executor.map(
                            lambda bounds: _update_factor_rows(
                                matrix,
                                item_factors,
                                user_factors,
                                identity,
                                self.regularization,
                                bounds[0],
                                bounds[1],
                            ),
                            user_ranges,
                        )
                    )
                    list(
                        executor.map(
                            lambda bounds: _update_factor_rows(
                                matrix_csc,
                                user_factors,
                                item_factors,
                                identity,
                                self.regularization,
                                bounds[0],
                                bounds[1],
                            ),
                            item_ranges,
                        )
                    )

        self.user_factors = user_factors
        self.item_factors = item_factors
        self.is_fitted = True
        return self

    def _work_ranges(self, entity_count: int) -> list[tuple[int, int]]:
        chunk_size = max(1, (entity_count + self.workers - 1) // self.workers)
        return [
            (start, min(start + chunk_size, entity_count))
            for start in range(0, entity_count, chunk_size)
        ]

    def scores_for_user(self, user_id: Hashable) -> dict[Hashable, float]:
        """Return estimated ratings for every item."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("ALSRecommender must be fitted before use")
        user_index = self.user_index.get(user_id)
        if user_index is None:
            return dict.fromkeys(self.item_ids, 0.0)
        scores = self.item_factors @ self.user_factors[user_index]
        return {
            item_id: float(scores[index]) for index, item_id in enumerate(self.item_ids)
        }

    def score_matrix_for_users(self, user_ids: list[Hashable]) -> np.ndarray:
        """Return dense user-by-item scores for a bounded user batch."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("ALSRecommender must be fitted before use")
        rows = np.zeros((len(user_ids), len(self.item_ids)), dtype=np.float32)
        known_positions = [
            position
            for position, user_id in enumerate(user_ids)
            if user_id in self.user_index
        ]
        if known_positions:
            factor_rows = [
                self.user_index[user_ids[position]] for position in known_positions
            ]
            rows[known_positions] = self.user_factors[factor_rows] @ self.item_factors.T
        return rows

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return top-scoring unseen items."""
        if not self.is_fitted:
            raise RuntimeError("ALSRecommender must be fitted before use")
        validate_k(k)
        scores = self.scores_for_user(user_id)
        return rank_scores(
            [scores[item_id] for item_id in self.item_ids],
            self.item_ids,
            k,
            excluded_items=self.user_items.get(user_id, set()),
        )

    def recommend_many(
        self, user_ids: list[Hashable], k: int = 10, batch_size: int = 256
    ) -> dict[Hashable, list[tuple[Hashable, float]]]:
        """Score and rank users in bounded dense matrix batches."""
        if not self.is_fitted:
            raise RuntimeError("ALSRecommender must be fitted before use")
        validate_k(k)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        recommendations: dict[Hashable, list[tuple[Hashable, float]]] = {}
        for offset in range(0, len(user_ids), batch_size):
            users = user_ids[offset : offset + batch_size]
            scores = self.score_matrix_for_users(users)
            exclusions = [self.user_items.get(user_id, set()) for user_id in users]
            ranked = rank_score_matrix(scores, self.item_ids, k, exclusions)
            recommendations.update(zip(users, ranked, strict=True))
        return recommendations


def _update_factor_rows(
    matrix: sparse.csr_matrix | sparse.csc_matrix,
    known_factors: np.ndarray,
    updated_factors: np.ndarray,
    identity: np.ndarray,
    regularization: float,
    start_row: int,
    end_row: int,
) -> None:
    """Solve a disjoint range of regularized least-squares factor rows."""
    for row_index in range(start_row, end_row):
        start, end = matrix.indptr[row_index : row_index + 2]
        indices = matrix.indices[start:end]
        ratings = matrix.data[start:end]
        factors = known_factors[indices]
        lhs = factors.T @ factors + regularization * identity
        rhs = factors.T @ ratings
        updated_factors[row_index] = np.linalg.solve(lhs, rhs)


class SVDSReferenceRecommender(ALSRecommender):
    """Low-rank truncated-SVD reference for ALS comparison."""

    def fit(
        self,
        interactions: pd.DataFrame,
        user_items: dict[Hashable, set[Hashable]] | None = None,
    ) -> SVDSReferenceRecommender:
        """Fit a rank-limited scipy.sparse.linalg.svds factorization."""
        (
            self.user_ids,
            self.item_ids,
            self.user_index,
            self.item_index,
            matrix,
            self.user_items,
        ) = _interaction_matrix(interactions, user_items=user_items)
        rank = min(self.factors, min(matrix.shape) - 1)
        if rank < 1:
            raise ValueError("svds reference needs at least two users and two items")
        user_vectors, singular_values, item_vectors_t = svds(
            matrix,
            k=rank,
            which="LM",
            random_state=self.seed,
        )
        order = np.argsort(singular_values)[::-1]
        roots = np.sqrt(np.maximum(singular_values[order], 0.0))
        self.user_factors = user_vectors[:, order] * roots
        self.item_factors = item_vectors_t[order, :].T * roots
        self.is_fitted = True
        return self
