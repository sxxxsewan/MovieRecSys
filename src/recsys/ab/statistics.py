"""Statistical utilities for experiment planning and analysis."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
from scipy import stats


def required_sample_size_per_group(
    minimum_detectable_effect: float,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Estimate per-arm sample size for a two-sided difference in proportions."""
    if not 0 < minimum_detectable_effect:
        raise ValueError("minimum_detectable_effect must be positive")
    if not 0 <= baseline_rate <= 1:
        raise ValueError("baseline_rate must be in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    treatment_rate = baseline_rate + minimum_detectable_effect
    if treatment_rate > 1:
        treatment_rate = baseline_rate - minimum_detectable_effect
    if treatment_rate < 0 or treatment_rate == baseline_rate:
        raise ValueError(
            "minimum_detectable_effect must keep the alternative rate in [0, 1]"
        )
    critical_value = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    variance_sum = baseline_rate * (1 - baseline_rate) + treatment_rate * (
        1 - treatment_rate
    )
    return ceil(critical_value**2 * variance_sum / minimum_detectable_effect**2)


def independent_t_test(
    control: Sequence[float], treatment: Sequence[float]
) -> tuple[float, float]:
    """Return treatment-minus-control mean difference and Welch t-test p-value."""
    control_values = _validated_values(control, "control")
    treatment_values = _validated_values(treatment, "treatment")
    if len(control_values) < 2 or len(treatment_values) < 2:
        raise ValueError("Both groups must contain at least two observations")
    difference = float(np.mean(treatment_values) - np.mean(control_values))
    if np.var(control_values) <= 1e-15 and np.var(treatment_values) <= 1e-15:
        return difference, 1.0 if difference == 0 else 0.0
    test = stats.ttest_ind(treatment_values, control_values, equal_var=False)
    return difference, float(test.pvalue)


def mann_whitney_test(
    control: Sequence[float], treatment: Sequence[float]
) -> tuple[float, float]:
    """Return treatment-minus-control mean difference and Mann-Whitney p-value."""
    control_values = _validated_values(control, "control")
    treatment_values = _validated_values(treatment, "treatment")
    test = stats.mannwhitneyu(treatment_values, control_values, alternative="two-sided")
    difference = float(np.mean(treatment_values) - np.mean(control_values))
    return difference, float(test.pvalue)


def bootstrap_mean_difference_ci(
    control: Sequence[float],
    treatment: Sequence[float],
    confidence_level: float = 0.95,
    resamples: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean difference."""
    control_values = _validated_values(control, "control")
    treatment_values = _validated_values(treatment, "treatment")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_control = rng.choice(control_values, len(control_values), replace=True)
        sampled_treatment = rng.choice(
            treatment_values, len(treatment_values), replace=True
        )
        differences[index] = sampled_treatment.mean() - sampled_control.mean()
    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(differences, [tail, 1 - tail])
    return float(lower), float(upper)


def cuped_adjust(outcomes: Sequence[float], covariates: Sequence[float]) -> np.ndarray:
    """Apply CUPED adjustment using a pre-experiment covariate."""
    outcome_values = _validated_values(outcomes, "outcomes")
    covariate_values = _validated_values(covariates, "covariates")
    if len(outcome_values) != len(covariate_values):
        raise ValueError("outcomes and covariates must have equal lengths")
    variance = (
        float(np.var(covariate_values, ddof=1)) if len(covariate_values) > 1 else 0
    )
    if variance == 0:
        return outcome_values.copy()
    theta = float(np.cov(outcome_values, covariate_values, ddof=1)[0, 1] / variance)
    return outcome_values - theta * (covariate_values - np.mean(covariate_values))


def aa_false_positive_rate(
    simulations: int = 1000,
    users_per_group: int = 200,
    baseline_rate: float = 0.1,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """Estimate type-I error from equal-rate synthetic control/treatment groups."""
    if simulations < 1 or users_per_group < 2:
        raise ValueError("simulations must be positive and users_per_group at least 2")
    if not 0 <= baseline_rate <= 1:
        raise ValueError("baseline_rate must be in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, baseline_rate, (simulations, users_per_group))
    treatment = rng.binomial(1, baseline_rate, (simulations, users_per_group))
    control_variance = np.var(control, axis=1, ddof=1)
    treatment_variance = np.var(treatment, axis=1, ddof=1)
    standard_error = np.sqrt(
        control_variance / users_per_group + treatment_variance / users_per_group
    )
    difference = np.mean(treatment, axis=1) - np.mean(control, axis=1)
    degrees_of_freedom = np.divide(
        (control_variance / users_per_group + treatment_variance / users_per_group)
        ** 2,
        (control_variance / users_per_group) ** 2 / (users_per_group - 1)
        + (treatment_variance / users_per_group) ** 2 / (users_per_group - 1),
        out=np.full(simulations, np.inf),
        where=standard_error > 0,
    )
    p_values = np.ones(simulations, dtype=np.float64)
    has_variance = standard_error > 0
    p_values[has_variance] = 2 * stats.t.sf(
        np.abs(difference[has_variance] / standard_error[has_variance]),
        degrees_of_freedom[has_variance],
    )
    p_values[(~has_variance) & (difference != 0)] = 0.0
    return float(np.mean(p_values < alpha))


def _validated_values(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result
