"""Fit, evaluate, and register classical MovieLens recommenders."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from recsys.data import split_and_filter_interactions
from recsys.db import create_engine_from_url, get_database_url
from recsys.evaluation import evaluate_recommendations
from recsys.models.als import ALSRecommender, SVDSReferenceRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.tracking import (
    configure_tracking,
    log_recommender_run,
    register_best_model,
)

LOGGER = logging.getLogger(__name__)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def _relevant_items(
    interactions: pd.DataFrame, threshold: float
) -> dict[Hashable, set[Hashable]]:
    positives = interactions.loc[interactions["rating"] >= threshold]
    return positives.groupby("user_id")["movie_id"].agg(set).to_dict()


def _recommendations(
    recommender: Any, users: Collection[Hashable], k: int
) -> dict[Hashable, list[Hashable]]:
    return {
        user_id: [item_id for item_id, _ in recommender.recommend(user_id, k)]
        for user_id in users
    }


def _evaluate_partition(
    recommender: Any,
    interactions: pd.DataFrame,
    item_popularity: Mapping[Hashable, int],
    catalog_size: int,
    relevance_rating_threshold: float,
    k: int,
) -> dict[str, float]:
    relevant = _relevant_items(interactions, relevance_rating_threshold)
    if not relevant:
        raise ValueError(
            "Evaluation partition has no ratings meeting the relevance threshold"
        )
    ranked = _recommendations(recommender, relevant.keys(), k)
    return evaluate_recommendations(
        ranked,
        relevant,
        item_popularity,
        catalog_size=catalog_size,
        k=k,
    )


def train_and_log(
    data_config_path: Path,
    model_config_path: Path,
    experiment_config_path: Path,
) -> str:
    """Run configured training, compare on validation, and register the best model."""
    data_config = _read_yaml(data_config_path)["data"]
    model_config = _read_yaml(model_config_path)["models"]
    experiment_config = _read_yaml(experiment_config_path)["experiments"]
    database_config = data_config["database"]
    engine = create_engine_from_url(
        get_database_url(str(database_config.get("url_env", "DATABASE_URL")))
    )

    LOGGER.info("Loading persisted MovieLens tables for training")
    ratings = pd.read_sql_table("ratings", engine)
    movies = pd.read_sql_table("movies", engine)
    tags = pd.read_sql_table("tags", engine)
    genome_scores = pd.read_sql_table("genome_scores", engine)
    genome_tags = pd.read_sql_table("genome_tags", engine)
    split = split_and_filter_interactions(
        ratings,
        train_fraction=float(data_config["split"]["train_fraction"]),
        validation_fraction=float(data_config["split"]["validation_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
        min_user_ratings=int(data_config["filtering"]["min_user_ratings"]),
        min_movie_ratings=int(data_config["filtering"]["min_movie_ratings"]),
    )
    LOGGER.info(
        "Chronological split rows: train=%d validation=%d test=%d",
        len(split.train),
        len(split.validation),
        len(split.test),
    )
    evaluation_config = experiment_config.get("evaluation", {})
    relevance_rating_threshold = float(
        evaluation_config.get("relevance_rating_threshold", 4.0)
    )
    evaluation_k = int(evaluation_config.get("k", 10))
    validation_relevant = _relevant_items(split.validation, relevance_rating_threshold)
    if not validation_relevant:
        raise ValueError(
            "Validation split has no ratings meeting the configured relevance threshold"
        )

    # Fit candidates in order: baseline first, then neighborhood, factor, and
    # content methods. Hybrid tuning uses only the validation partition.
    popularity_config = model_config["popularity"]
    candidates: dict[str, tuple[Any, dict[str, Any]]] = {}
    popularity = PopularityRecommender(
        metric=popularity_config["metric"],
        smoothing=float(popularity_config["smoothing"]),
    ).fit(split.train)
    candidates["popularity"] = (
        popularity,
        {"metric": popularity.metric, "smoothing": popularity.smoothing},
    )

    remaining = _fit_candidates_without_hybrid(
        split.train,
        movies,
        tags,
        genome_scores,
        genome_tags,
        model_config,
    )
    candidates.update(remaining)
    als = candidates["als"][0]
    content = candidates["content_based"][0]
    hybrid = HybridRecommender(als, content)
    hybrid_config = model_config["hybrid"]
    hybrid.tune(
        validation_relevant,
        candidate_weights=hybrid_config.get(
            "candidate_als_weights", [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
        k=evaluation_k,
    )
    candidates["hybrid"] = (
        hybrid,
        {
            "als_weight": hybrid.als_weight,
            "content_weight": 1.0 - hybrid.als_weight,
            "tuned_on": f"validation_map_at_{evaluation_k}",
        },
    )

    item_popularity = split.train["movie_id"].value_counts().astype(int).to_dict()
    catalog_size = int(movies["movie_id"].nunique())
    tracking_env = str(experiment_config.get("tracking_uri_env", "MLFLOW_TRACKING_URI"))
    configure_tracking(tracking_env)
    experiment_name = str(experiment_config["default_name"])
    mlflow.set_experiment(experiment_name)

    run_ids: dict[str, str] = {}
    validation_scores: dict[str, float] = {}
    for model_name, (model, parameters) in candidates.items():
        validation_metrics = _evaluate_partition(
            model,
            split.validation,
            item_popularity,
            catalog_size,
            relevance_rating_threshold,
            evaluation_k,
        )
        test_metrics = _evaluate_partition(
            model,
            split.test,
            item_popularity,
            catalog_size,
            relevance_rating_threshold,
            evaluation_k,
        )
        logged_metrics = {
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        logged_parameters = {
            **parameters,
            "evaluation_k": evaluation_k,
            "relevance_rating_threshold": relevance_rating_threshold,
        }
        run_ids[model_name] = log_recommender_run(
            run_name=model_name,
            recommender=model,
            parameters=logged_parameters,
            metrics=logged_metrics,
        )
        validation_scores[model_name] = validation_metrics[f"ndcg@{evaluation_k}"]
        LOGGER.info(
            "%s validation NDCG@%d=%.6f test NDCG@%d=%.6f",
            model_name,
            evaluation_k,
            validation_scores[model_name],
            evaluation_k,
            test_metrics[f"ndcg@{evaluation_k}"],
        )

    best_name = max(validation_scores, key=validation_scores.__getitem__)
    registry_name = f"{experiment_name}-best"
    version = register_best_model(run_ids[best_name], registry_name)
    LOGGER.info(
        "Registered validation-selected model %s as %s version %s",
        best_name,
        registry_name,
        version,
    )
    return best_name


def _fit_candidates_without_hybrid(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
    genome_scores: pd.DataFrame,
    genome_tags: pd.DataFrame,
    model_config: Mapping[str, Any],
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Fit all non-baseline candidates after popularity has been logged first."""
    knn_config = model_config["item_knn"]
    als_config = model_config["als"]
    content_config = model_config["content_based"]
    item_knn = ItemKNNRecommender(
        k_neighbors=int(knn_config["k_neighbors"]),
        min_similarity=float(knn_config["min_similarity"]),
    ).fit(train)
    als = ALSRecommender(
        factors=int(als_config["factors"]),
        regularization=float(als_config["regularization"]),
        iterations=int(als_config["iterations"]),
        seed=int(als_config["seed"]),
    ).fit(train)
    svds_reference = SVDSReferenceRecommender(
        factors=int(als_config["factors"]),
        seed=int(als_config["seed"]),
    ).fit(train)
    content = ContentBasedRecommender(
        max_features=int(content_config["max_features"]),
        svd_components=int(content_config["svd_components"]),
        seed=int(als_config["seed"]),
    ).fit(
        movies=movies,
        interactions=train,
        tags=tags,
        genome_scores=genome_scores,
        genome_tags=genome_tags,
    )
    return {
        "item_knn": (
            item_knn,
            {
                "k_neighbors": item_knn.k_neighbors,
                "min_similarity": item_knn.min_similarity,
            },
        ),
        "als": (
            als,
            {
                "factors": als.factors,
                "regularization": als.regularization,
                "iterations": als.iterations,
                "seed": als.seed,
            },
        ),
        "svds_reference": (
            svds_reference,
            {"factors": svds_reference.factors, "seed": svds_reference.seed},
        ),
        "content_based": (
            content,
            {
                "max_features": content.max_features,
                "svd_components": content.svd_components,
                "seed": content.seed,
            },
        ),
    }


def main() -> None:
    """Parse config paths and start the explicit training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_and_log(
        args.data_config,
        args.model_config,
        args.experiment_config,
    )


if __name__ == "__main__":
    main()
