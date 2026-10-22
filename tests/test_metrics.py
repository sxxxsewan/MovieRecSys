from __future__ import annotations

from math import isclose, log2

import pytest

from recsys.evaluation import (
    average_popularity,
    average_precision_at_k,
    catalog_coverage,
    evaluate_recommendations,
    mean_average_precision_at_k,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall_average_precision_and_ndcg_at_k() -> None:
    recommendations = ["a", "x", "b", "c"]
    relevant = {"a", "b", "c"}

    assert precision_at_k(recommendations, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(recommendations, relevant, 3) == pytest.approx(2 / 3)
    assert average_precision_at_k(recommendations, relevant, 3) == pytest.approx(
        (1 + 2 / 3) / 3
    )
    ideal_dcg = 1 + 1 / log2(3) + 1 / log2(4)
    actual_dcg = 1 + 1 / log2(4)
    assert ndcg_at_k(recommendations, relevant, 3) == pytest.approx(
        actual_dcg / ideal_dcg
    )


def test_rank_metrics_handle_short_lists_and_duplicate_items() -> None:
    recommendations = ["a", "a"]

    assert precision_at_k(recommendations, {"a"}, 3) == pytest.approx(1 / 3)
    assert recall_at_k(recommendations, {"a", "b"}, 3) == pytest.approx(1 / 2)
    assert average_precision_at_k(recommendations, {"a", "b"}, 3) == pytest.approx(
        1 / 2
    )
    assert ndcg_at_k(recommendations, {"a", "b"}, 3) == pytest.approx(
        1 / (1 + 1 / log2(3))
    )


def test_mean_average_precision_ignores_users_without_test_relevance() -> None:
    recommendations = {1: ["a", "b"], 2: ["x"]}
    relevant_items = {1: {"a"}, 2: set()}

    assert mean_average_precision_at_k(recommendations, relevant_items, 2) == 1.0


def test_catalog_metrics_calculate_coverage_novelty_and_popularity() -> None:
    recommendations = {1: ["popular", "rare"], 2: ["rare", "unseen"]}
    popularity = {"popular": 8, "rare": 2}

    assert catalog_coverage(recommendations, catalog_size=4) == pytest.approx(0.75)
    assert average_popularity(recommendations, popularity) == pytest.approx(3.0)
    expected_novelty = (-log2(9 / 12) - log2(3 / 12) - log2(3 / 12) - log2(1 / 12)) / 4
    assert novelty(recommendations, popularity) == pytest.approx(expected_novelty)


def test_evaluate_recommendations_returns_all_required_metrics() -> None:
    recommendations = {1: ["a", "b"], 2: ["b", "c"]}
    relevant = {1: {"a"}, 2: {"c"}}
    popularity = {"a": 3, "b": 2, "c": 1}

    result = evaluate_recommendations(
        recommendations,
        relevant,
        item_popularity=popularity,
        catalog_size=4,
        k=2,
    )

    assert set(result) == {
        "ndcg@2",
        "recall@2",
        "map@2",
        "precision@2",
        "coverage",
        "novelty",
        "average_popularity",
    }
    assert result["ndcg@2"] == pytest.approx((1 + 1 / log2(3)) / 2)
    assert result["recall@2"] == pytest.approx(1.0)
    assert result["map@2"] == pytest.approx(0.75)
    assert result["precision@2"] == pytest.approx(0.5)
    assert result["coverage"] == pytest.approx(0.75)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_ranking_metrics_reject_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(["a"], {"a"}, k)


def test_coverage_rejects_invalid_catalog_size() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        catalog_coverage({1: ["a"]}, 0)


def test_catalog_metrics_reject_missing_or_invalid_popularity_data() -> None:
    with pytest.raises(ValueError, match="at least one interaction"):
        novelty({1: ["a"]}, {"a": 0})
    with pytest.raises(ValueError, match="nonnegative integers"):
        average_popularity({1: ["a"]}, {"a": -1})


def test_catalog_metrics_require_recommendations() -> None:
    with pytest.raises(ValueError, match="at least one recommendation"):
        novelty({}, {"a": 1})
    with pytest.raises(ValueError, match="at least one recommendation"):
        average_popularity({}, {"a": 1})


def test_aggregate_metrics_reject_empty_evaluation_users() -> None:
    with pytest.raises(ValueError, match="relevant items"):
        evaluate_recommendations({}, {1: set()}, {"a": 1}, catalog_size=1)


def test_ndcg_is_one_for_perfect_ordering() -> None:
    assert isclose(ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
