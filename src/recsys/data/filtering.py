"""Filtering utilities for user-movie interactions."""

from __future__ import annotations

import logging
from numbers import Integral

import pandas as pd

LOGGER = logging.getLogger(__name__)


def filter_interactions(
    interactions: pd.DataFrame,
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
) -> pd.DataFrame:
    """Filter interactions to the stable user/movie minimum-rating core.

    Filtering repeats until every retained user and movie meets its threshold,
    since removing one side can make entities on the other side ineligible.
    """
    required_columns = {"user_id", "movie_id"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Interactions are missing required columns: {missing}")
    thresholds = (min_user_ratings, min_movie_ratings)
    if any(
        isinstance(threshold, bool)
        or not isinstance(threshold, Integral)
        or threshold < 1
        for threshold in thresholds
    ):
        raise ValueError("Minimum rating counts must be positive integers")
    if interactions.empty:
        raise ValueError("Cannot filter an empty interactions dataframe")
    if interactions[["user_id", "movie_id"]].isna().any().any():
        raise ValueError("Interaction user_id and movie_id values must not be null")

    filtered = interactions
    iteration = 0
    while True:
        previous_size = len(filtered)
        eligible_users = filtered["user_id"].value_counts()
        eligible_users = eligible_users[eligible_users >= min_user_ratings].index
        filtered = filtered.loc[filtered["user_id"].isin(eligible_users)]

        eligible_movies = filtered["movie_id"].value_counts()
        eligible_movies = eligible_movies[eligible_movies >= min_movie_ratings].index
        filtered = filtered.loc[filtered["movie_id"].isin(eligible_movies)]

        iteration += 1
        LOGGER.info(
            "Interaction filtering iteration %d retained %d of %d rows",
            iteration,
            len(filtered),
            previous_size,
        )
        if filtered.empty:
            raise ValueError(
                "Minimum user/movie rating thresholds removed all interactions"
            )
        if len(filtered) == previous_size:
            return filtered.copy()
