"""Re-evaluate the exported MLflow model against the configured test split.

Run from the repository root with:
    python -m recsys.evaluation.best_model_check
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import yaml

from recsys.data import split_and_filter_interactions
from recsys.evaluation import evaluate_recommendations

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRIC_KEYS = (
    "ndcg@10",
    "recall@10",
    "map@10",
    "precision@10",
    "coverage",
    "novelty",
    "average_popularity",
)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section must be a mapping: {name}")
    return value


def _relevant_items(
    interactions: pd.DataFrame, threshold: float
) -> dict[int, set[int]]:
    relevant = interactions.loc[interactions["rating"] >= threshold]
    return relevant.groupby("user_id")["movie_id"].agg(set).to_dict()


def _evaluate_exported_model(
    model_path: Path,
    ratings_path: Path,
    movies_path: Path,
    data_config_path: Path,
    experiment_config_path: Path,
) -> tuple[dict[str, float], str, bool]:
    data_config = _mapping(_read_yaml(data_config_path).get("data"), "data")
    split_config = _mapping(data_config.get("split"), "data.split")
    filtering_config = _mapping(data_config.get("filtering"), "data.filtering")
    experiments = _mapping(
        _read_yaml(experiment_config_path).get("experiments"), "experiments"
    )
    evaluation = _mapping(experiments.get("evaluation"), "experiments.evaluation")
    k = int(evaluation.get("k", 10))
    threshold = float(evaluation.get("relevance_rating_threshold", 4.0))
    batch_size = int(evaluation.get("batch_size", 32))
    if k != 10:
        raise ValueError(
            "This comparison expects the saved model_metrics.csv @10 metrics; "
            f"configured k is {k}"
        )
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive")

    ratings = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int32",
        },
    ).rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
        }
    )
    split = split_and_filter_interactions(
        ratings,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        min_user_ratings=int(filtering_config["min_user_ratings"]),
        min_movie_ratings=int(filtering_config["min_movie_ratings"]),
    )
    del ratings

    relevant = _relevant_items(split.test, threshold)
    if not relevant:
        raise ValueError("The configured test split has no relevant ratings")
    popularity = split.train["movie_id"].value_counts().astype(int).to_dict()
    catalog_size = int(
        pd.read_csv(movies_path, usecols=["movieId"], dtype={"movieId": "int32"})[
            "movieId"
        ].nunique()
    )

    model = mlflow.pyfunc.load_model(str(model_path))
    python_model = model.unwrap_python_model()
    recommender = getattr(python_model, "recommender", None)
    if recommender is None or not hasattr(recommender, "recommend_many"):
        raise TypeError(
            "The exported model does not expose a compatible batched recommender"
        )

    sample_user = int(next(iter(relevant)))
    prediction = model.predict(pd.DataFrame([{"user_id": sample_user, "k": k}]))
    if (
        len(prediction) != 1
        or "recommendations" not in prediction.columns
        or not isinstance(prediction.iloc[0]["recommendations"], list)
        or not prediction.iloc[0]["recommendations"]
    ):
        raise ValueError("The exported model failed its MLflow prediction smoke check")

    recommendations: dict[int, list[int]] = {}
    user_ids = list(relevant)
    for offset in range(0, len(user_ids), batch_size):
        users = user_ids[offset : offset + batch_size]
        ranked = recommender.recommend_many(users, k=k, batch_size=batch_size)
        recommendations.update(
            {
                user_id: [movie_id for movie_id, _ in ranked.get(user_id, [])]
                for user_id in users
            }
        )

    metrics = evaluate_recommendations(
        recommendations,
        relevant,
        popularity,
        catalog_size=catalog_size,
        k=k,
    )
    metadata = _read_yaml(model_path / "MLmodel")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"MLmodel metadata has no valid run_id: {model_path}")
    return metrics, run_id, True


def _build_report(
    metrics: Mapping[str, float],
    run_id: str,
    metrics_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    comparison = pd.read_csv(metrics_path)
    required = {
        "model",
        "run_id",
        "validation_ndcg_at_10",
        "test_ndcg_at_10",
        *(f"test_{key.replace('@', '_at_')}" for key in METRIC_KEYS),
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            f"Metrics table is missing required columns: {', '.join(sorted(missing))}"
        )
    selected_rows = comparison.loc[comparison["run_id"] == run_id]
    if len(selected_rows) != 1:
        raise ValueError(
            f"Expected exactly one metrics row for exported model run {run_id}; "
            f"found {len(selected_rows)}"
        )

    selected = selected_rows.iloc[0]
    validation_winner = comparison.loc[comparison["validation_ndcg_at_10"].idxmax()]
    test_winner = comparison.loc[comparison["test_ndcg_at_10"].idxmax()]
    validation_order = comparison["validation_ndcg_at_10"].rank(
        method="min", ascending=False
    )
    test_order = comparison["test_ndcg_at_10"].rank(method="min", ascending=False)
    selected_index = selected_rows.index[0]
    comparison_rows = []
    for index, row in comparison.iterrows():
        validation_ndcg = float(row["validation_ndcg_at_10"])
        test_ndcg = float(row["test_ndcg_at_10"])
        comparison_rows.append(
            {
                "model": str(row["model"]),
                "validation_ndcg_at_10": validation_ndcg,
                "test_ndcg_at_10": test_ndcg,
                "validation_ndcg_rank": int(validation_order.loc[index]),
                "test_ndcg_rank": int(test_order.loc[index]),
                "ndcg_gap": validation_ndcg - test_ndcg,
                "ndcg_drop_percent": (
                    100 * (validation_ndcg - test_ndcg) / validation_ndcg
                    if validation_ndcg
                    else None
                ),
            }
        )
    logged_test_metrics = {
        key: float(selected[f"test_{key.replace('@', '_at_')}"]) for key in METRIC_KEYS
    }
    metric_differences = {
        key: abs(float(metrics[key]) - logged_test_metrics[key]) for key in METRIC_KEYS
    }
    return {
        "exported_model_run_id": run_id,
        "exported_model_name": str(selected["model"]),
        "selected_by_validation_ndcg_at_10": run_id == str(validation_winner["run_id"]),
        "validation_ndcg_winner": str(validation_winner["model"]),
        "test_ndcg_winner_in_saved_comparison": str(test_winner["model"]),
        "validation_ndcg_rank": int(validation_order.loc[selected_index]),
        "test_ndcg_rank_in_saved_comparison": int(test_order.loc[selected_index]),
        "all_model_ndcg_comparison": comparison_rows,
        "recomputed_test_metrics": dict(metrics),
        "logged_test_metrics": logged_test_metrics,
        "absolute_metric_differences": metric_differences,
        "matches_logged_metrics_within_tolerance": all(
            difference <= tolerance for difference in metric_differences.values()
        ),
        "validation_to_test_ndcg_gap": float(
            selected["validation_ndcg_at_10"] - selected["test_ndcg_at_10"]
        ),
        "validation_to_test_ndcg_drop_percent": (
            100
            * (selected["validation_ndcg_at_10"] - selected["test_ndcg_at_10"])
            / selected["validation_ndcg_at_10"]
            if selected["validation_ndcg_at_10"]
            else None
        ),
        "als_ndcg_near_zero_warning": bool(
            (
                comparison["model"].eq("als")
                & comparison["validation_ndcg_at_10"].le(0.001)
                & comparison["test_ndcg_at_10"].le(0.001)
            ).any()
        ),
        "note": (
            "Test metrics are a final diagnostic only; do not use them to tune "
            "or select another model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the exported best model on the fixed test split."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "best_model" / "model",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=PROJECT_ROOT / "data" / "ratings.csv",
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=PROJECT_ROOT / "data" / "movies.csv",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments.yaml",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model_metrics.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "best_model_evaluation.json",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for path in (
        args.model / "MLmodel",
        args.ratings,
        args.movies,
        args.data_config,
        args.experiment_config,
        args.metrics,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input not found: {path}")
    if args.tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")

    metrics, run_id, pyfunc_smoke_check_passed = _evaluate_exported_model(
        args.model,
        args.ratings,
        args.movies,
        args.data_config,
        args.experiment_config,
    )
    report = _build_report(metrics, run_id, args.metrics, args.tolerance)
    report["mlflow_prediction_smoke_check_passed"] = pyfunc_smoke_check_passed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Evaluation report saved to %s", args.report)
    LOGGER.info("Evaluation summary:\n%s", json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
