"""Deterministic timestamp-based interaction splitting."""

from __future__ import annotations

from math import isclose, isfinite
from typing import NamedTuple

import pandas as pd

from recsys.data.filtering import filter_interactions


class TimeSplit(NamedTuple):
    """Chronological training, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _closest_group_boundary(
    cumulative_counts: list[int],
    target_count: float,
    first_boundary: int,
    last_boundary: int,
) -> int:
    """Choose a nonempty timestamp-group boundary nearest a target row count."""
    return min(
        range(first_boundary, last_boundary + 1),
        key=lambda boundary: (
            abs(cumulative_counts[boundary - 1] - target_count),
            boundary,
        ),
    )


def time_based_split(
    interactions: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> TimeSplit:
    """Split interactions chronologically without dividing equal timestamps.

    Fractions define desired row-count proportions. Split boundaries are chosen
    between timestamp groups, so ties may make actual proportions differ.
    """
    if "timestamp" not in interactions.columns:
        raise ValueError("Interactions must include a timestamp column")
    if interactions.empty:
        raise ValueError("Cannot split an empty interactions dataframe")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not isfinite(fraction) or fraction <= 0 for fraction in fractions):
        raise ValueError("Split fractions must be finite positive values")
    if not isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1")

    timestamps = pd.to_numeric(interactions["timestamp"], errors="raise")
    if timestamps.isna().any():
        raise ValueError("Interaction timestamps must not be null")
    if not timestamps.map(isfinite).all():
        raise ValueError("Interaction timestamps must be finite")

    timestamp_counts = timestamps.value_counts(sort=False).sort_index()
    timestamp_values = timestamp_counts.index.tolist()
    group_count = len(timestamp_values)
    if group_count < 3:
        raise ValueError(
            "At least three distinct timestamps are required for nonempty "
            "train, validation, and test splits"
        )

    cumulative_counts = timestamp_counts.cumsum().tolist()
    row_count = len(interactions)
    train_boundary = _closest_group_boundary(
        cumulative_counts,
        row_count * train_fraction,
        first_boundary=1,
        last_boundary=group_count - 2,
    )
    validation_boundary = _closest_group_boundary(
        cumulative_counts,
        row_count * (train_fraction + validation_fraction),
        first_boundary=train_boundary + 1,
        last_boundary=group_count - 1,
    )

    train_cutoff = timestamp_values[train_boundary]
    test_cutoff = timestamp_values[validation_boundary]
    train_mask = timestamps < train_cutoff
    validation_mask = (timestamps >= train_cutoff) & (timestamps < test_cutoff)
    test_mask = timestamps >= test_cutoff

    def chronological_frame(mask: pd.Series) -> pd.DataFrame:
        return (
            interactions.loc[mask]
            .assign(timestamp=timestamps.loc[mask])
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )

    return TimeSplit(
        train=chronological_frame(train_mask),
        validation=chronological_frame(validation_mask),
        test=chronological_frame(test_mask),
    )


def split_and_filter_interactions(
    interactions: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
) -> TimeSplit:
    """Split first, filter training support, and exclude unseen holdout entities."""
    split = time_based_split(
        interactions,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train = filter_interactions(
        split.train,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
    )
    train_users = train["user_id"].unique()
    train_movies = train["movie_id"].unique()

    def supported_holdout(frame: pd.DataFrame) -> pd.DataFrame:
        supported = frame["user_id"].isin(train_users) & frame["movie_id"].isin(
            train_movies
        )
        return frame.loc[supported].copy().reset_index(drop=True)

    return TimeSplit(
        train=train,
        validation=supported_holdout(split.validation),
        test=supported_holdout(split.test),
    )
