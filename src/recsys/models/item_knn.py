"""Sparse cosine item-item k-nearest-neighbor recommender."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from recsys.models.base import (
    rank_score_matrix,
    rank_scores,
    validate_interactions,
    validate_k,
)


class ItemKNNRecommender:
    """Recommend items through cosine neighbors in a sparse user-item matrix."""

    def __init__(self, k_neighbors: int = 50, min_similarity: float = 0.0) -> None:
        if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, Integral):
            raise ValueError("k_neighbors must be a positive integer")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")
        if not np.isfinite(min_similarity) or min_similarity < 0:
            raise ValueError("min_similarity must be finite and nonnegative")
        self.k_neighbors = int(k_neighbors)
        self.min_similarity = float(min_similarity)
        self.user_ids: list[Hashable] = []
        self.item_ids: list[Hashable] = []
        self.user_index: dict[Hashable, int] = {}
        self.item_index: dict[Hashable, int] = {}
        self.user_items: dict[Hashable, set[Hashable]] = {}
        self.user_item_matrix: sparse.csr_matrix | None = None
        self.similarity_matrix: sparse.csr_matrix | None = None
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> ItemKNNRecommender:
        """Build a sparse rating matrix and its truncated cosine graph."""
        validate_interactions(interactions)
        self.user_ids = sorted(interactions["user_id"].unique().tolist(), key=str)
        self.item_ids = sorted(interactions["movie_id"].unique().tolist(), key=str)
        self.user_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.item_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        rows = interactions["user_id"].map(self.user_index).to_numpy()
        columns = interactions["movie_id"].map(self.item_index).to_numpy()
        values = interactions["rating"].astype(np.float32).to_numpy()
        matrix = sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        self.user_item_matrix = matrix
        self.user_items = interactions.groupby("user_id")["movie_id"].agg(set).to_dict()

        item_vectors = normalize(matrix.T.tocsr(), norm="l2", axis=1)
        neighbor_count = min(self.k_neighbors + 1, len(self.item_ids))
        neighbors = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(item_vectors)
        rows_out: list[int] = []
        columns_out: list[int] = []
        values_out: list[float] = []
        with config_context(working_memory=256):
            for start in range(0, len(self.item_ids), 1024):
                stop = min(start + 1024, len(self.item_ids))
                distances, indices = neighbors.kneighbors(
                    item_vectors[start:stop], return_distance=True
                )
                for local_index, (item_distances, item_neighbors) in enumerate(
                    zip(distances, indices, strict=True)
                ):
                    item_index = start + local_index
                    added = 0
                    for distance, neighbor_index in zip(
                        item_distances, item_neighbors, strict=True
                    ):
                        similarity = max(0.0, 1.0 - float(distance))
                        if (
                            neighbor_index == item_index
                            or similarity < self.min_similarity
                        ):
                            continue
                        rows_out.append(item_index)
                        columns_out.append(int(neighbor_index))
                        values_out.append(similarity)
                        added += 1
                        if added == self.k_neighbors:
                            break
        self.similarity_matrix = sparse.csr_matrix(
            (values_out, (rows_out, columns_out)),
            shape=(len(self.item_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        self.is_fitted = True
        return self

    def scores_for_user(self, user_id: Hashable) -> dict[Hashable, float]:
        """Score candidates from a user's observed ratings and item neighbors."""
        if not self.is_fitted or self.user_item_matrix is None:
            raise RuntimeError("ItemKNNRecommender must be fitted before use")
        user_index = self.user_index.get(user_id)
        if user_index is None:
            return dict.fromkeys(self.item_ids, 0.0)
        scores = self.user_item_matrix.getrow(user_index) @ self.similarity_matrix
        dense_scores = np.asarray(scores.toarray()).ravel()
        return {
            item_id: float(dense_scores[index])
            for index, item_id in enumerate(self.item_ids)
        }

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return top-scoring unseen items."""
        if not self.is_fitted:
            raise RuntimeError("ItemKNNRecommender must be fitted before use")
        validate_k(k)
        scores = self.scores_for_user(user_id)
        ordered = [scores[item_id] for item_id in self.item_ids]
        return rank_scores(
            ordered,
            self.item_ids,
            k,
            excluded_items=self.user_items.get(user_id, set()),
        )

    def similar_items(
        self, movie_id: Hashable, k: int = 10
    ) -> list[tuple[Hashable, float]]:
        """Return the nearest items by precomputed cosine similarity."""
        if not self.is_fitted or self.similarity_matrix is None:
            raise RuntimeError("ItemKNNRecommender must be fitted before use")
        validate_k(k)
        item_index = self.item_index.get(movie_id)
        if item_index is None:
            return []
        scores = self.similarity_matrix.getrow(item_index).toarray().ravel()
        return rank_scores(
            scores,
            self.item_ids,
            k,
            excluded_items={movie_id},
        )

    def recommend_many(
        self, user_ids: list[Hashable], k: int = 10, batch_size: int = 256
    ) -> dict[Hashable, list[tuple[Hashable, float]]]:
        """Score and rank user batches using sparse matrix multiplication."""
        if (
            not self.is_fitted
            or self.user_item_matrix is None
            or self.similarity_matrix is None
        ):
            raise RuntimeError("ItemKNNRecommender must be fitted before use")
        validate_k(k)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        recommendations: dict[Hashable, list[tuple[Hashable, float]]] = {}
        for offset in range(0, len(user_ids), batch_size):
            users = user_ids[offset : offset + batch_size]
            row_indices = [self.user_index.get(user_id) for user_id in users]
            known_positions = [
                position
                for position, row_index in enumerate(row_indices)
                if row_index is not None
            ]
            scores = np.zeros((len(users), len(self.item_ids)), dtype=np.float32)
            if known_positions:
                known_rows = [row_indices[position] for position in known_positions]
                batch_scores = (
                    self.user_item_matrix[known_rows] @ self.similarity_matrix
                ).toarray()
                scores[known_positions] = batch_scores
            exclusions = [self.user_items.get(user_id, set()) for user_id in users]
            ranked = rank_score_matrix(scores, self.item_ids, k, exclusions)
            recommendations.update(zip(users, ranked, strict=True))
        return recommendations
