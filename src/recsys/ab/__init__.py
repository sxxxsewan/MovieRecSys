"""Offline A/B testing utilities."""

from recsys.ab.assignment import assign_group
from recsys.ab.simulation import persist_experiment_result, simulate_experiment
from recsys.ab.statistics import (
    aa_false_positive_rate,
    bootstrap_mean_difference_ci,
    cuped_adjust,
    independent_t_test,
    mann_whitney_test,
    required_sample_size_per_group,
)

__all__ = [
    "aa_false_positive_rate",
    "assign_group",
    "bootstrap_mean_difference_ci",
    "cuped_adjust",
    "independent_t_test",
    "mann_whitney_test",
    "persist_experiment_result",
    "required_sample_size_per_group",
    "simulate_experiment",
]
