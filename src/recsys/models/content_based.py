"""Content-based recommendation from genres, user tags, and genome scores."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from recsys.models.base import rank_score_matrix, rank_scores, validate_k


class ContentBasedRecommender:
    """Recommend by cosine similarity to a rating-weighted user content profile."""

    def __init__(
        self,
        max_features: int = 20_000,
        svd_components: int = 128,
        seed: int = 42,
    ) -> None:
        if (
            isinstance(max_features, bool)
            or not isinstance(max_features, Integral)
            or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer")
        if (
            isinstance(svd_components, bool)
            or not isinstance(svd_components, Integral)
            or svd_components < 0
        ):
            raise ValueError("svd_components must be a nonnegative integer")
        self.max_features = int(max_features)
        self.svd_components = int(svd_components)
        self.seed = int(seed)
        self.item_ids: list[Hashable] = []
        self.item_index: dict[Hashable, int] = {}
        self.item_features: np.ndarray | sparse.csr_matrix | None = None
        self.user_profiles: dict[Hashable, np.ndarray] = {}
        self.user_items: dict[Hashable, set[Hashable]] = {}
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.scaler: StandardScaler | None = None
        self.is_fitted = False

    def fit(
        self,
        movies: pd.DataFrame,
        interactions: pd.DataFrame,
        tags: pd.DataFrame | None = None,
        genome_scores: pd.DataFrame | None = None,
        genome_tags: pd.DataFrame | None = None,
    ) -> ContentBasedRecommender:
        """Build item vectors, optional SVD features, and rating-weighted profiles."""
        movie_columns = {"movie_id", "genres"}
        if not movie_columns.issubset(movies.columns):
            raise ValueError("movies must contain movie_id and genres columns")
        interaction_columns = {"user_id", "movie_id", "rating"}
        if not interaction_columns.issubset(interactions.columns):
            raise ValueError(
                "interactions must contain user_id, movie_id, and rating columns"
            )
        if movies.empty or interactions.empty:
            raise ValueError("movies and interactions must not be empty")
        if movies["movie_id"].duplicated().any() or movies["movie_id"].isna().any():
            raise ValueError("movies must have unique, non-null movie_id values")
        if interactions[["user_id", "movie_id", "rating"]].isna().any().any():
            raise ValueError(
                "Interaction user_id, movie_id, and rating must not be null"
            )
        if interactions.duplicated(["user_id", "movie_id"]).any():
            raise ValueError("Interactions must have at most one rating per user/movie")

        ordered_movies = movies.sort_values("movie_id", key=lambda ids: ids.astype(str))
        self.item_ids = ordered_movies["movie_id"].tolist()
        self.item_index = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        text_by_movie = pd.Series("", index=self.item_ids, dtype="object")
        text = ordered_movies["genres"].fillna("").astype(str)
        if "title" in ordered_movies.columns:
            text = ordered_movies["title"].fillna("").astype(str) + " " + text
        text_by_movie.loc[ordered_movies["movie_id"].tolist()] = text.to_numpy()
        if tags is not None and not tags.empty:
            if not {"movie_id", "tag"}.issubset(tags.columns):
                raise ValueError("tags must contain movie_id and tag columns")
            tag_text = (
                tags.dropna(subset=["movie_id", "tag"])
                .groupby("movie_id")["tag"]
                .agg(lambda values: " ".join(values.astype(str)))
            )
            for movie_id, tag_value in tag_text.items():
                if movie_id in text_by_movie.index:
                    text_by_movie.loc[movie_id] += f" {tag_value}"

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=r"(?u)\b\w+\b",
        )
        text_features = self.vectorizer.fit_transform(text_by_movie.tolist()).tocsr()
        blocks: list[sparse.csr_matrix] = [text_features]

        if genome_scores is not None and not genome_scores.empty:
            if not {"movie_id", "tag_id", "relevance"}.issubset(genome_scores.columns):
                raise ValueError(
                    "genome_scores must contain movie_id, tag_id, and relevance columns"
                )
            genome = genome_scores.loc[genome_scores["movie_id"].isin(self.item_index)]
            if genome[["movie_id", "tag_id", "relevance"]].isna().any().any():
                raise ValueError(
                    "genome_scores movie_id, tag_id, and relevance must not be null"
                )
            tag_ids = sorted(genome["tag_id"].dropna().unique().tolist(), key=str)
            tag_index = {tag_id: index for index, tag_id in enumerate(tag_ids)}
            genome_rows = genome["movie_id"].map(self.item_index).to_numpy()
            genome_columns = genome["tag_id"].map(tag_index).to_numpy()
            genome_values = genome["relevance"].astype(np.float32).to_numpy()
            genome_matrix = sparse.csr_matrix(
                (genome_values, (genome_rows, genome_columns)),
                shape=(len(self.item_ids), len(tag_ids)),
                dtype=np.float32,
            )
            if genome_tags is not None and not genome_tags.empty:
                if not {"tag_id", "tag"}.issubset(genome_tags.columns):
                    raise ValueError("genome_tags must contain tag_id and tag columns")
                genome_tag_ids = set(genome_tags["tag_id"].dropna())
                unknown_tag_ids = set(tag_ids) - genome_tag_ids
                if unknown_tag_ids:
                    raise ValueError(
                        "genome_scores contains tag IDs missing from genome_tags"
                    )
            blocks.append(normalize(genome_matrix.tocsr(), norm="l2", axis=1))

        features = sparse.hstack(blocks, format="csr", dtype=np.float32)
        component_count = min(
            self.svd_components, min(features.shape[0], features.shape[1]) - 1
        )
        if component_count > 0:
            self.svd = TruncatedSVD(
                n_components=component_count,
                random_state=self.seed,
            )
            reduced = self.svd.fit_transform(features)
            self.scaler = StandardScaler()
            reduced = self.scaler.fit_transform(reduced)
            self.item_features = normalize(reduced, norm="l2", axis=1)
        else:
            self.item_features = normalize(features, norm="l2", axis=1).tocsr()

        self.user_items = interactions.groupby("user_id")["movie_id"].agg(set).to_dict()
        self.user_profiles.clear()
        for user_id, history in interactions.groupby("user_id"):
            indices = history["movie_id"].map(self.item_index)
            known = indices.notna()
            if not known.any():
                continue
            item_rows = indices.loc[known].astype(int).to_numpy()
            rating_values = history.loc[known, "rating"].astype(float).to_numpy()
            weights = np.maximum(rating_values - 2.5, 0.1)
            profile = self.item_features[item_rows].T @ weights / weights.sum()
            profile = np.asarray(profile).ravel()
            norm = np.linalg.norm(profile)
            if norm:
                profile = profile / norm
            self.user_profiles[user_id] = profile
        self.is_fitted = True
        return self

    def scores_for_user(self, user_id: Hashable) -> dict[Hashable, float]:
        """Return cosine-profile scores for all catalog items."""
        if not self.is_fitted or self.item_features is None:
            raise RuntimeError("ContentBasedRecommender must be fitted before use")
        profile = self.user_profiles.get(user_id)
        if profile is None:
            return dict.fromkeys(self.item_ids, 0.0)
        scores = self.item_features @ profile
        values = np.asarray(scores).ravel()
        return {
            item_id: float(values[index]) for index, item_id in enumerate(self.item_ids)
        }

    def score_matrix_for_users(self, user_ids: list[Hashable]) -> np.ndarray:
        """Return dense content scores for a bounded list of user profiles."""
        if not self.is_fitted or self.item_features is None:
            raise RuntimeError("ContentBasedRecommender must be fitted before use")
        profile_size = (
            self.item_features.shape[1] if self.item_features.ndim == 2 else 0
        )
        profiles = np.zeros((len(user_ids), profile_size), dtype=np.float32)
        for row, user_id in enumerate(user_ids):
            profile = self.user_profiles.get(user_id)
            if profile is not None:
                profiles[row] = profile
        scores = self.item_features @ profiles.T
        return np.asarray(scores, dtype=np.float32).T

    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[Hashable, float]]:
        """Return the top unseen content matches for a user."""
        if not self.is_fitted:
            raise RuntimeError("ContentBasedRecommender must be fitted before use")
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
        """Score and rank users in bounded feature-matrix batches."""
        if not self.is_fitted:
            raise RuntimeError("ContentBasedRecommender must be fitted before use")
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

    def recommend_from_favorites(
        self, favorite_movie_ids: list[Hashable], k: int = 10
    ) -> list[tuple[Hashable, float]]:
        """Recommend from provided favorite movie IDs for a cold-start user."""
        if not self.is_fitted or self.item_features is None:
            raise RuntimeError("ContentBasedRecommender must be fitted before use")
        validate_k(k)
        indices = [
            self.item_index[movie_id]
            for movie_id in favorite_movie_ids
            if movie_id in self.item_index
        ]
        if not indices:
            raise ValueError("At least one favorite movie must exist in the catalog")
        profile = np.asarray(self.item_features[indices].mean(axis=0)).ravel()
        norm = np.linalg.norm(profile)
        if norm:
            profile /= norm
        scores = self.item_features @ profile
        return rank_scores(
            np.asarray(scores).ravel(),
            self.item_ids,
            k,
            excluded_items={self.item_ids[index] for index in indices},
        )
