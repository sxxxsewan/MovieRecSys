"""Explicit-feedback Alternating Least Squares and an svds reference model."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from recsys.models.base import rank_scores, validate_interactions, validate_k


def _interaction_matrix(
    interactions: pd.DataFrame,
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
    rows = interactions["user_id"].map(user_index).to_numpy()
    columns = interactions["movie_id"].map(item_index).to_numpy()
    ratings = interactions["rating"].astype(np.float64).to_numpy()
    matrix = sparse.csr_matrix(
        (ratings, (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    user_items = interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
    return user_ids, item_ids, user_index, item_index, matrix, user_items


class ALSRecommender:
    """Fit explicit-feedback matrix factors by alternating regularized solves."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        seed: int = 42,
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
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.user_ids: list[Hashable] = []
        self.item_ids: list[Hashable] = []
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_items: dict[Hashable, set[Hashable]] = {}
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> ALSRecommender:
        """Optimize user and item factors with alternating least squares."""
        (
            self.user_ids,
            self.item_ids,
            self.user_index,
            self.item_index,
            matrix,
            self.user_items,
        ) = _interaction_matrix(interactions)
        matrix_csc = matrix.tocsc()
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.factors)
        user_factors = rng.normal(0.0, scale, size=(len(self.user_ids), self.factors))
        item_factors = rng.normal(0.0, scale, size=(len(self.item_ids), self.factors))
        identity = np.eye(self.factors, dtype=np.float64)

        for _ in range(self.iterations):
            for user_row in range(matrix.shape[0]):
                start, end = matrix.indptr[user_row : user_row + 2]
                indices = matrix.indices[start:end]
                ratings = matrix.data[start:end]
                factors = item_factors[indices]
                lhs = factors.T @ factors + self.regularization * identity
                rhs = factors.T @ ratings
                user_factors[user_row] = np.linalg.solve(lhs, rhs)

            for item_column in range(matrix_csc.shape[1]):
                start, end = matrix_csc.indptr[item_column : item_column + 2]
                indices = matrix_csc.indices[start:end]
                ratings = matrix_csc.data[start:end]
                factors = user_factors[indices]
                lhs = factors.T @ factors + self.regularization * identity
                rhs = factors.T @ ratings
                item_factors[item_column] = np.linalg.solve(lhs, rhs)

        self.user_factors = user_factors
        self.item_factors = item_factors
        self.is_fitted = True
        return self

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


class SVDSReferenceRecommender(ALSRecommender):
    """Low-rank truncated-SVD reference for ALS comparison."""

    def fit(self, interactions: pd.DataFrame) -> SVDSReferenceRecommender:
        """Fit a rank-limited scipy.sparse.linalg.svds factorization."""
        (
            self.user_ids,
            self.item_ids,
            self.user_index,
            self.item_index,
            matrix,
            self.user_items,
        ) = _interaction_matrix(interactions)
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
