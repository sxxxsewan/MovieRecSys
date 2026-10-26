"""Offline A/B experiment simulation from historical rating outcomes."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from recsys.ab.assignment import assign_group
from recsys.ab.statistics import (
    bootstrap_mean_difference_ci,
    cuped_adjust,
    independent_t_test,
    mann_whitney_test,
    required_sample_size_per_group,
)
from recsys.db.models import ExperimentResult

RATING_COLUMNS = {"user_id", "movie_id", "rating"}


def simulate_experiment(
    test_ratings: pd.DataFrame,
    experiment_name: str,
    rating_click_threshold: float,
    minimum_detectable_effect: float,
    baseline_click_rate: float,
    control_name: str = "control",
    treatment_name: str = "treatment",
    alpha: float = 0.05,
    power: float = 0.8,
    bootstrap_samples: int = 5000,
    seed: int = 42,
    use_mann_whitney: bool = False,
    pre_period_ratings: pd.DataFrame | None = None,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Simulate an experiment; a test-period rating above threshold is a proxy click."""
    _validate_ratings(test_ratings, "test_ratings")
    if not np.isfinite(rating_click_threshold):
        raise ValueError("rating_click_threshold must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    user_metrics = _user_click_metrics(test_ratings, rating_click_threshold)
    if len(user_metrics) < 4:
        raise ValueError("At least four users are required to compare two groups")
    groups = {
        user_id: assign_group(
            user_id,
            experiment_name,
            control=control_name,
            treatment=treatment_name,
        )
        for user_id in user_metrics
    }
    control_users = [
        user_id for user_id, group in groups.items() if group == control_name
    ]
    treatment_users = [
        user_id for user_id, group in groups.items() if group == treatment_name
    ]
    if len(control_users) < 2 or len(treatment_users) < 2:
        raise ValueError("Assignment must place at least two users in each group")

    control_outcomes = np.asarray(
        [user_metrics[user_id]["ctr"] for user_id in control_users], dtype=np.float64
    )
    treatment_outcomes = np.asarray(
        [user_metrics[user_id]["ctr"] for user_id in treatment_users],
        dtype=np.float64,
    )
    cuped_applied = pre_period_ratings is not None
    if pre_period_ratings is not None:
        _validate_ratings(pre_period_ratings, "pre_period_ratings")
        pre_metrics = _user_click_metrics(pre_period_ratings, rating_click_threshold)
        pooled_covariate = (
            float(np.mean([metric["ctr"] for metric in pre_metrics.values()]))
            if pre_metrics
            else 0.0
        )
        covariates = {user_id: metric["ctr"] for user_id, metric in pre_metrics.items()}
        control_covariates = np.asarray(
            [covariates.get(user_id, pooled_covariate) for user_id in control_users],
            dtype=np.float64,
        )
        treatment_covariates = np.asarray(
            [covariates.get(user_id, pooled_covariate) for user_id in treatment_users],
            dtype=np.float64,
        )
        combined_outcomes = cuped_adjust(
            np.concatenate([control_outcomes, treatment_outcomes]),
            np.concatenate([control_covariates, treatment_covariates]),
        )
        control_outcomes = combined_outcomes[: len(control_users)]
        treatment_outcomes = combined_outcomes[len(control_users) :]

    if use_mann_whitney:
        effect, p_value = mann_whitney_test(control_outcomes, treatment_outcomes)
        statistical_test = "mann_whitney_u"
    else:
        effect, p_value = independent_t_test(control_outcomes, treatment_outcomes)
        statistical_test = "welch_t_test"
    ci_low, ci_high = bootstrap_mean_difference_ci(
        control_outcomes,
        treatment_outcomes,
        resamples=bootstrap_samples,
        seed=seed,
    )
    sample_size = required_sample_size_per_group(
        minimum_detectable_effect=minimum_detectable_effect,
        baseline_rate=baseline_click_rate,
        alpha=alpha,
        power=power,
    )
    control_impressions = sum(
        int(user_metrics[user_id]["impressions"]) for user_id in control_users
    )
    treatment_impressions = sum(
        int(user_metrics[user_id]["impressions"]) for user_id in treatment_users
    )
    control_clicks = sum(
        int(user_metrics[user_id]["clicks"]) for user_id in control_users
    )
    treatment_clicks = sum(
        int(user_metrics[user_id]["clicks"]) for user_id in treatment_users
    )
    result: dict[str, Any] = {
        "experiment_name": experiment_name,
        "control_name": control_name,
        "treatment_name": treatment_name,
        "control_users": len(control_users),
        "treatment_users": len(treatment_users),
        "control_impressions": control_impressions,
        "treatment_impressions": treatment_impressions,
        "control_clicks": control_clicks,
        "treatment_clicks": treatment_clicks,
        "control_ctr": control_clicks / control_impressions,
        "treatment_ctr": treatment_clicks / treatment_impressions,
        "effect": effect,
        "p_value": p_value,
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high,
        "sample_size_per_group": sample_size,
        "statistical_test": statistical_test,
        "cuped_applied": cuped_applied,
        "is_simulated": True,
    }
    if engine is not None:
        persist_experiment_result(engine, result)
    return result


def persist_experiment_result(engine: Engine, result: Mapping[str, Any]) -> None:
    """Persist one offline experiment result using the configured database engine."""
    with Session(engine) as session, session.begin():
        session.add(ExperimentResult(**result))


def _user_click_metrics(
    ratings: pd.DataFrame, threshold: float
) -> dict[Hashable, dict[str, float | int]]:
    clicks = ratings.assign(click=(ratings["rating"] >= threshold).astype("int64"))
    grouped = clicks.groupby("user_id")["click"].agg(["sum", "count"])
    return {
        user_id: {
            "clicks": int(row["sum"]),
            "impressions": int(row["count"]),
            "ctr": float(row["sum"] / row["count"]),
        }
        for user_id, row in grouped.iterrows()
    }


def _validate_ratings(ratings: pd.DataFrame, name: str) -> None:
    if not RATING_COLUMNS.issubset(ratings.columns):
        raise ValueError(f"{name} must contain user_id, movie_id, and rating columns")
    if ratings.empty:
        raise ValueError(f"{name} must not be empty")
    if ratings[["user_id", "movie_id", "rating"]].isna().any().any():
        raise ValueError(f"{name} user_id, movie_id, and rating must not be null")
    if not np.isfinite(ratings["rating"].astype(float)).all():
        raise ValueError(f"{name} ratings must be finite")
