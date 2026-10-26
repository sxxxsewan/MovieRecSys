from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from recsys.models import (
    ALSRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    SVDSReferenceRecommender,
)
from recsys.models.base import rank_score_matrix


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, 10, 5.0),
            (1, 11, 4.0),
            (1, 12, 2.0),
            (2, 10, 4.0),
            (2, 12, 5.0),
            (2, 13, 3.0),
            (3, 11, 5.0),
            (3, 12, 3.0),
            (3, 13, 4.0),
            (4, 10, 3.0),
            (4, 11, 4.0),
            (4, 13, 5.0),
        ],
        columns=["user_id", "movie_id", "rating"],
    )


def _assert_batch_recommendations_match_single(model: object) -> None:
    user_ids = [1, 999]
    batched = model.recommend_many(user_ids, k=2)
    for user_id in user_ids:
        single = model.recommend(user_id, k=2)
        assert [item_id for item_id, _ in batched[user_id]] == [
            item_id for item_id, _ in single
        ]
        np.testing.assert_allclose(
            [score for _, score in batched[user_id]],
            [score for _, score in single],
            rtol=1e-6,
            atol=1e-7,
        )


def test_rank_score_matrix_breaks_ties_and_applies_exclusions() -> None:
    ranked = rank_score_matrix(
        np.array([[1.0, 2.0, 2.0, 3.0], [2.0, 2.0, np.nan, 2.0]]),
        [40, 20, 30, 10],
        k=2,
        excluded_items=[{10}, {20}],
    )

    assert ranked == [[(20, 2.0), (30, 2.0)], [(10, 2.0), (40, 2.0)]]


def test_popularity_ranks_training_items_and_excludes_seen(
    interactions: pd.DataFrame,
) -> None:
    model = PopularityRecommender().fit(interactions)

    recommendations = model.recommend(user_id=1, k=3)

    assert [item for item, _ in recommendations] == [13]
    assert model.recommend(user_id=999, k=2)[0][0] == 10
    assert model.item_popularity == {10: 3, 11: 3, 12: 3, 13: 3}
    _assert_batch_recommendations_match_single(model)


def test_item_knn_returns_unseen_items_and_similars(
    interactions: pd.DataFrame,
) -> None:
    model = ItemKNNRecommender(k_neighbors=2).fit(interactions)

    recommendations = model.recommend(user_id=1, k=2)
    similar = model.similar_items(movie_id=10, k=2)

    assert all(item not in {10, 11, 12} for item, _ in recommendations)
    assert len(similar) == 2
    assert all(item != 10 for item, _ in similar)
    _assert_batch_recommendations_match_single(model)


def test_als_and_svds_reference_fit_small_matrix(
    interactions: pd.DataFrame,
) -> None:
    als = ALSRecommender(factors=2, iterations=2, seed=3).fit(interactions)
    reference = SVDSReferenceRecommender(factors=2, seed=3).fit(interactions)

    assert np.isfinite(list(als.scores_for_user(1).values())).all()
    assert np.isfinite(list(reference.scores_for_user(1).values())).all()
    assert len(als.recommend(1, k=2)) == 1
    assert len(reference.recommend(1, k=2)) == 1
    _assert_batch_recommendations_match_single(als)


def test_content_recommender_uses_genre_tags_and_genome(
    interactions: pd.DataFrame,
) -> None:
    movies = pd.DataFrame(
        [
            (10, "Space Drama", "Drama|Sci-Fi"),
            (11, "Space Journey", "Sci-Fi"),
            (12, "Funny Story", "Comedy"),
            (13, "Funny Drama", "Comedy|Drama"),
        ],
        columns=["movie_id", "title", "genres"],
    )
    tags = pd.DataFrame(
        [(1, 10, "space travel"), (2, 12, "funny")],
        columns=["user_id", "movie_id", "tag"],
    )
    genome_scores = pd.DataFrame(
        [
            (movie_id, tag_id, relevance)
            for movie_id in [10, 11, 12, 13]
            for tag_id, relevance in [(1, 0.9), (2, 0.1)]
        ],
        columns=["movie_id", "tag_id", "relevance"],
    )
    genome_tags = pd.DataFrame([(1, "space"), (2, "humor")], columns=["tag_id", "tag"])

    model = ContentBasedRecommender(svd_components=0).fit(
        movies,
        interactions,
        tags=tags,
        genome_scores=genome_scores,
        genome_tags=genome_tags,
    )

    assert len(model.recommend(1, k=2)) == 1
    assert len(model.recommend_from_favorites([10], k=2)) == 2
    _assert_batch_recommendations_match_single(model)


def test_hybrid_tunes_weight_using_validation_relevance(
    interactions: pd.DataFrame,
) -> None:
    als = ALSRecommender(factors=2, iterations=1, seed=5).fit(interactions)
    content = ContentBasedRecommender(svd_components=0).fit(
        pd.DataFrame(
            [
                (10, "Drama", "Drama"),
                (11, "Sci-Fi", "Sci-Fi"),
                (12, "Comedy", "Comedy"),
                (13, "Mystery", "Mystery"),
            ],
            columns=["movie_id", "title", "genres"],
        ),
        interactions,
    )
    hybrid = HybridRecommender(als, content)

    selected_weight = hybrid.tune(
        {1: {13}, 2: {11}, 3: {10}},
        candidate_weights=[0.0, 0.5, 1.0],
        k=2,
    )

    assert selected_weight in {0.0, 0.5, 1.0}
    assert hybrid.is_tuned
    _assert_batch_recommendations_match_single(hybrid)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PopularityRecommender(metric="unknown"), "metric"),
        (lambda: ItemKNNRecommender(k_neighbors=0), "k_neighbors"),
        (lambda: ALSRecommender(factors=0), "factors"),
        (lambda: ContentBasedRecommender(svd_components=-1), "svd_components"),
    ],
)
def test_models_validate_constructor_parameters(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_hybrid_combines_component_catalogs() -> None:
    class ScoreModel:
        user_items: dict[int, set[int]] = {}

        def __init__(self, scores: dict[int, float]) -> None:
            self.scores = scores

        def scores_for_user(self, user_id: int) -> dict[int, float]:
            return self.scores

    hybrid = HybridRecommender(
        ScoreModel({10: 1.0, 11: 0.0}),
        ScoreModel({11: 1.0, 12: 0.0}),
    )

    assert hybrid.scores_for_user(7) == {10: 0.5, 11: 0.5, 12: 0.0}
