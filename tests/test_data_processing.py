from __future__ import annotations

import pandas as pd
import pytest

from recsys.data import (
    filter_interactions,
    split_and_filter_interactions,
    time_based_split,
)


def test_time_based_split_orders_partitions_and_keeps_timestamp_ties() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "movie_id": [10, 11, 12, 13, 14, 15, 16, 17],
            "timestamp": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )

    split = time_based_split(interactions, 0.5, 0.25, 0.25)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert split.train["timestamp"].tolist() == [1, 1, 2, 2]
    assert split.validation["timestamp"].tolist() == [3, 3]
    assert split.test["timestamp"].tolist() == [4, 4]
    assert sum(map(len, split)) == len(interactions)


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ((0.8, 0.1, 0.2), "sum to 1"),
        ((1.0, 0.0, 0.0), "positive"),
        ((float("nan"), 0.1, 0.9), "finite positive"),
    ],
)
def test_time_based_split_rejects_invalid_fractions(
    fractions: tuple[float, float, float], message: str
) -> None:
    interactions = pd.DataFrame({"timestamp": [1, 2, 3]})

    with pytest.raises(ValueError, match=message):
        time_based_split(interactions, *fractions)


def test_time_based_split_requires_multiple_distinct_timestamps() -> None:
    interactions = pd.DataFrame({"timestamp": [1, 1, 2, 2]})

    with pytest.raises(ValueError, match="three distinct timestamps"):
        time_based_split(interactions)


def test_split_and_filter_uses_training_support_only() -> None:
    interactions = pd.DataFrame(
        [
            (1, "a", 1),
            (2, "a", 1),
            (1, "b", 2),
            (2, "b", 2),
            (1, "a", 3),
            (1, "c", 3),
            (3, "a", 3),
            (3, "c", 3),
            (2, "b", 4),
            (2, "c", 4),
            (3, "b", 4),
            (3, "c", 4),
        ],
        columns=["user_id", "movie_id", "timestamp"],
    )

    split = split_and_filter_interactions(
        interactions,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        min_user_ratings=2,
        min_movie_ratings=2,
    )

    assert set(split.train["user_id"]) == {1, 2}
    assert set(split.train["movie_id"]) == {"a", "b"}
    assert set(split.validation["user_id"]) <= set(split.train["user_id"])
    assert set(split.validation["movie_id"]) <= set(split.train["movie_id"])
    assert set(split.test["user_id"]) <= set(split.train["user_id"])
    assert set(split.test["movie_id"]) <= set(split.train["movie_id"])
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_filter_interactions_repeats_until_minimums_stabilize() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": ["a", "b", "a", "b", "b", "c"],
            "rating": [5.0, 4.0, 4.0, 3.0, 5.0, 3.0],
        }
    )

    filtered = filter_interactions(
        interactions,
        min_user_ratings=2,
        min_movie_ratings=2,
    )

    assert filtered[["user_id", "movie_id"]].to_records(index=False).tolist() == [
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
    ]
    assert len(interactions) == 6


def test_filter_interactions_validates_thresholds_and_required_columns() -> None:
    interactions = pd.DataFrame({"user_id": [1], "movie_id": [10]})

    with pytest.raises(ValueError, match="positive integers"):
        filter_interactions(interactions, min_user_ratings=0)
    with pytest.raises(ValueError, match="positive integers"):
        filter_interactions(interactions, min_user_ratings=1.5)
    with pytest.raises(ValueError, match="required columns"):
        filter_interactions(pd.DataFrame({"timestamp": [1]}))


def test_filter_interactions_reports_when_thresholds_remove_everything() -> None:
    interactions = pd.DataFrame({"user_id": [1], "movie_id": [10]})

    with pytest.raises(ValueError, match="removed all interactions"):
        filter_interactions(interactions, min_user_ratings=2)
