from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

from recsys.ab import (
    aa_false_positive_rate,
    assign_group,
    bootstrap_mean_difference_ci,
    cuped_adjust,
    independent_t_test,
    mann_whitney_test,
    required_sample_size_per_group,
    simulate_experiment,
)
from recsys.db.base import Base
from recsys.db.models import ExperimentResult


def test_assignment_is_deterministic_and_experiment_scoped() -> None:
    group = assign_group(42, "ranking-v1")

    assert assign_group(42, "ranking-v1") == group
    assert assign_group(42, "ranking-v2") in {"control", "treatment"}
    assert any(
        assign_group(user_id, "ranking-v1") != assign_group(user_id, "ranking-v2")
        for user_id in range(100)
    )


def test_assignment_validates_group_parameters() -> None:
    with pytest.raises(ValueError, match="distinct"):
        assign_group(1, "experiment", control="same", treatment="same")
    with pytest.raises(ValueError, match="strictly between"):
        assign_group(1, "experiment", treatment_fraction=1.0)


def test_sample_size_for_proportions() -> None:
    assert required_sample_size_per_group(0.02, 0.1, alpha=0.05, power=0.8) > 1000
    with pytest.raises(ValueError, match="baseline_rate"):
        required_sample_size_per_group(0.02, 1.1)


def test_tests_and_bootstrap_return_expected_direction() -> None:
    control = [0.0, 0.1, 0.0, 0.2, 0.1]
    treatment = [0.4, 0.5, 0.4, 0.6, 0.5]

    difference, t_p_value = independent_t_test(control, treatment)
    rank_difference, rank_p_value = mann_whitney_test(control, treatment)
    interval = bootstrap_mean_difference_ci(control, treatment, resamples=500, seed=2)

    assert difference > 0
    assert rank_difference > 0
    assert t_p_value < 0.05
    assert rank_p_value < 0.05
    assert interval[0] > 0


def test_cuped_adjustment_preserves_mean_and_reduces_covariate_effect() -> None:
    covariates = np.arange(10, dtype=float)
    outcomes = 2 * covariates + np.asarray([0, 1, -1, 0, 1, -1, 0, 1, -1, 0])

    adjusted = cuped_adjust(outcomes, covariates)

    assert adjusted.mean() == pytest.approx(outcomes.mean())
    assert adjusted.var() < outcomes.var()


def test_simulation_counts_rating_proxy_clicks_and_persists(tmp_path: Path) -> None:
    records = [
        (user_id, movie_id, 5.0 if movie_id % 2 else 3.0)
        for user_id in range(1, 81)
        for movie_id in range(1, 6)
    ]
    ratings = pd.DataFrame(records, columns=["user_id", "movie_id", "rating"])
    engine = create_engine(f"sqlite:///{tmp_path / 'ab.sqlite'}")
    Base.metadata.create_all(engine)

    result = simulate_experiment(
        ratings,
        experiment_name="synthetic-ranking-test",
        rating_click_threshold=4.0,
        minimum_detectable_effect=0.02,
        baseline_click_rate=0.1,
        bootstrap_samples=200,
        seed=7,
        engine=engine,
    )

    assert result["is_simulated"] is True
    assert result["control_users"] + result["treatment_users"] == 80
    assert result["control_clicks"] + result["treatment_clicks"] == 240
    assert result["control_ctr"] == pytest.approx(0.6)
    assert result["treatment_ctr"] == pytest.approx(0.6)
    assert inspect(engine).has_table("experiments")
    with engine.connect() as connection:
        stored = (
            connection.execute(ExperimentResult.__table__.select()).mappings().one()
        )
    assert stored["experiment_name"] == "synthetic-ranking-test"
    assert stored["is_simulated"] is True


def test_simulation_supports_cuped_and_rejects_invalid_rating_frames() -> None:
    frame = pd.DataFrame(
        [
            (user_id, movie_id, 5.0 if (user_id + movie_id) % 3 == 0 else 3.0)
            for user_id in range(1, 41)
            for movie_id in range(1, 5)
        ],
        columns=["user_id", "movie_id", "rating"],
    )
    result = simulate_experiment(
        frame,
        experiment_name="cuped-test",
        rating_click_threshold=4.0,
        minimum_detectable_effect=0.02,
        baseline_click_rate=0.1,
        bootstrap_samples=100,
        pre_period_ratings=frame,
    )

    assert result["cuped_applied"] is True
    with pytest.raises(ValueError, match="must contain"):
        simulate_experiment(
            frame[["user_id", "rating"]],
            experiment_name="invalid",
            rating_click_threshold=4.0,
            minimum_detectable_effect=0.02,
            baseline_click_rate=0.1,
        )


def test_aa_check_is_approximately_five_percent() -> None:
    false_positive_rate = aa_false_positive_rate(
        simulations=1000,
        users_per_group=200,
        baseline_rate=0.1,
        alpha=0.05,
        seed=42,
    )

    assert false_positive_rate == pytest.approx(0.05, abs=0.02)
