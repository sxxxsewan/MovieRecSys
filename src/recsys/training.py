"""Fit, evaluate, and register classical MovieLens recommenders."""

from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from sqlalchemy import Engine, Table, select

from recsys.data import split_and_filter_interactions
from recsys.db import create_engine_from_url, get_database_url
from recsys.db.models import GenomeScore, GenomeTag, Movie, Rating, Tag
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
    recommender: Any, users: Collection[Hashable], k: int, batch_size: int
) -> dict[Hashable, list[Hashable]]:
    user_ids = list(users)
    if hasattr(recommender, "recommend_many"):
        ranked = recommender.recommend_many(user_ids, k, batch_size=batch_size)
        return {
            user_id: [item_id for item_id, _ in ranked.get(user_id, [])]
            for user_id in user_ids
        }
    return {
        user_id: [item_id for item_id, _ in recommender.recommend(user_id, k)]
        for user_id in user_ids
    }


def _evaluate_partition(
    recommender: Any,
    relevant: Mapping[Hashable, set[Hashable]],
    item_popularity: Mapping[Hashable, int],
    catalog_size: int,
    k: int,
    batch_size: int,
) -> dict[str, float]:
    if not relevant:
        raise ValueError(
            "Evaluation partition has no ratings meeting the relevance threshold"
        )
    ranked = _recommendations(recommender, relevant.keys(), k, batch_size)
    return evaluate_recommendations(
        ranked,
        relevant,
        item_popularity,
        catalog_size=catalog_size,
        k=k,
    )


def _evaluate_and_log_candidate(
    model_name: str,
    model: Any,
    parameters: Mapping[str, Any],
    validation_relevant: Mapping[Hashable, set[Hashable]],
    test_relevant: Mapping[Hashable, set[Hashable]],
    item_popularity: Mapping[Hashable, int],
    catalog_size: int,
    relevance_rating_threshold: float,
    evaluation_k: int,
    evaluation_batch_size: int,
) -> tuple[str, float]:
    validation_metrics = _evaluate_partition(
        model,
        validation_relevant,
        item_popularity,
        catalog_size,
        evaluation_k,
        evaluation_batch_size,
    )
    test_metrics = _evaluate_partition(
        model,
        test_relevant,
        item_popularity,
        catalog_size,
        evaluation_k,
        evaluation_batch_size,
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
    run_id = log_recommender_run(
        run_name=model_name,
        recommender=model,
        parameters=logged_parameters,
        metrics=logged_metrics,
    )
    validation_score = validation_metrics[f"ndcg@{evaluation_k}"]
    LOGGER.info(
        "%s validation_metrics=%s test_metrics=%s",
        model_name,
        validation_metrics,
        test_metrics,
    )
    return run_id, validation_score


def train_and_log(
    data_config_path: Path,
    model_config_path: Path,
    experiment_config_path: Path,
    smoke: bool = False,
) -> str:
    """Run configured training, compare on validation, and register the best model."""
    data_config = _read_yaml(data_config_path)["data"]
    model_config = _read_yaml(model_config_path)["models"]
    experiment_config = _read_yaml(experiment_config_path)["experiments"]
    if smoke:
        LOGGER.info("Running bounded synthetic smoke workflow")
        ratings, movies, tags, genome_scores, genome_tags = _make_smoke_data()
        data_config = _smoke_data_config(data_config)
        model_config = _smoke_model_config(model_config)
    else:
        database_config = data_config["database"]
        engine = create_engine_from_url(
            get_database_url(str(database_config.get("url_env", "DATABASE_URL")))
        )
        LOGGER.info("Loading persisted MovieLens tables for training")
        try:
            ratings = _read_sql_frame(
                engine,
                Rating.__table__,
                {
                    "user_id": "int32",
                    "movie_id": "int32",
                    "rating": "float32",
                    "timestamp": "int32",
                },
            )
            movies = _read_sql_frame(engine, Movie.__table__, {"movie_id": "int32"})
            tags = _read_sql_frame(
                engine,
                Tag.__table__,
                {"user_id": "int32", "movie_id": "int32", "timestamp": "int32"},
            )
            genome_scores = _read_sql_frame(
                engine,
                GenomeScore.__table__,
                {"movie_id": "int32", "tag_id": "int32", "relevance": "float32"},
            )
            genome_tags = _read_sql_frame(
                engine, GenomeTag.__table__, {"tag_id": "int32"}
            )
        finally:
            engine.dispose()
    split = split_and_filter_interactions(
        ratings,
        train_fraction=float(data_config["split"]["train_fraction"]),
        validation_fraction=float(data_config["split"]["validation_fraction"]),
        test_fraction=float(data_config["split"]["test_fraction"]),
        min_user_ratings=int(data_config["filtering"]["min_user_ratings"]),
        min_movie_ratings=int(data_config["filtering"]["min_movie_ratings"]),
    )
    del ratings
    gc.collect()
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
    evaluation_batch_size = int(evaluation_config.get("batch_size", 256))
    if evaluation_batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    validation_relevant = _relevant_items(split.validation, relevance_rating_threshold)
    if not validation_relevant:
        raise ValueError(
            "Validation split has no ratings meeting the configured relevance threshold"
        )
    test_relevant = _relevant_items(split.test, relevance_rating_threshold)
    if not test_relevant:
        raise ValueError(
            "Test split has no ratings meeting the configured relevance threshold"
        )

    item_popularity = split.train["movie_id"].value_counts().astype(int).to_dict()
    catalog_size = int(movies["movie_id"].nunique())
    shared_user_items = split.train.groupby("user_id")["movie_id"].agg(set).to_dict()
    tracking_env = str(experiment_config.get("tracking_uri_env", "MLFLOW_TRACKING_URI"))
    configure_tracking(tracking_env)
    experiment_name = str(experiment_config["default_name"])
    mlflow.set_experiment(experiment_name)

    run_ids: dict[str, str] = {}
    validation_scores: dict[str, float] = {}

    # Fit, evaluate, and release models as soon as possible to bound peak RAM.
    popularity_config = model_config["popularity"]
    popularity = PopularityRecommender(
        metric=popularity_config["metric"],
        smoothing=float(popularity_config["smoothing"]),
    ).fit(split.train, user_items=shared_user_items)
    run_ids["popularity"], validation_scores["popularity"] = (
        _evaluate_and_log_candidate(
            "popularity",
            popularity,
            {"metric": popularity.metric, "smoothing": popularity.smoothing},
            validation_relevant,
            test_relevant,
            item_popularity,
            catalog_size,
            relevance_rating_threshold,
            evaluation_k,
            evaluation_batch_size,
        )
    )
    del popularity
    gc.collect()

    knn_config = model_config["item_knn"]
    item_knn = ItemKNNRecommender(
        k_neighbors=int(knn_config["k_neighbors"]),
        min_similarity=float(knn_config["min_similarity"]),
    ).fit(split.train, user_items=shared_user_items)
    run_ids["item_knn"], validation_scores["item_knn"] = _evaluate_and_log_candidate(
        "item_knn",
        item_knn,
        {
            "k_neighbors": item_knn.k_neighbors,
            "min_similarity": item_knn.min_similarity,
        },
        validation_relevant,
        test_relevant,
        item_popularity,
        catalog_size,
        relevance_rating_threshold,
        evaluation_k,
        evaluation_batch_size,
    )
    del item_knn
    gc.collect()

    als_config = model_config["als"]
    als = ALSRecommender(
        factors=int(als_config["factors"]),
        regularization=float(als_config["regularization"]),
        iterations=int(als_config["iterations"]),
        seed=int(als_config["seed"]),
        workers=int(als_config.get("workers", 1)),
    ).fit(split.train, user_items=shared_user_items)
    run_ids["als"], validation_scores["als"] = _evaluate_and_log_candidate(
        "als",
        als,
        {
            "factors": als.factors,
            "regularization": als.regularization,
            "iterations": als.iterations,
            "seed": als.seed,
            "workers": als.workers,
        },
        validation_relevant,
        test_relevant,
        item_popularity,
        catalog_size,
        relevance_rating_threshold,
        evaluation_k,
        evaluation_batch_size,
    )

    svds_reference = SVDSReferenceRecommender(
        factors=int(als_config["factors"]),
        seed=int(als_config["seed"]),
    ).fit(split.train, user_items=shared_user_items)
    run_ids["svds_reference"], validation_scores["svds_reference"] = (
        _evaluate_and_log_candidate(
            "svds_reference",
            svds_reference,
            {"factors": svds_reference.factors, "seed": svds_reference.seed},
            validation_relevant,
            test_relevant,
            item_popularity,
            catalog_size,
            relevance_rating_threshold,
            evaluation_k,
            evaluation_batch_size,
        )
    )
    del svds_reference
    gc.collect()

    content_config = model_config["content_based"]
    content = ContentBasedRecommender(
        max_features=int(content_config["max_features"]),
        svd_components=int(content_config["svd_components"]),
        seed=int(als_config["seed"]),
    ).fit(
        movies=movies,
        interactions=split.train,
        tags=tags,
        genome_scores=genome_scores,
        genome_tags=genome_tags,
        user_items=shared_user_items,
    )
    del tags, genome_scores, genome_tags
    del movies
    gc.collect()
    run_ids["content_based"], validation_scores["content_based"] = (
        _evaluate_and_log_candidate(
            "content_based",
            content,
            {
                "max_features": content.max_features,
                "svd_components": content.svd_components,
                "seed": content.seed,
            },
            validation_relevant,
            test_relevant,
            item_popularity,
            catalog_size,
            relevance_rating_threshold,
            evaluation_k,
            evaluation_batch_size,
        )
    )

    hybrid = HybridRecommender(als, content)
    hybrid_config = model_config["hybrid"]
    hybrid.tune(
        validation_relevant,
        candidate_weights=hybrid_config.get(
            "candidate_als_weights", [0.0, 0.25, 0.5, 0.75, 1.0]
        ),
        k=evaluation_k,
    )
    run_ids["hybrid"], validation_scores["hybrid"] = _evaluate_and_log_candidate(
        "hybrid",
        hybrid,
        {
            "als_weight": hybrid.als_weight,
            "content_weight": 1.0 - hybrid.als_weight,
            "tuned_on": f"validation_map_at_{evaluation_k}",
        },
        validation_relevant,
        test_relevant,
        item_popularity,
        catalog_size,
        relevance_rating_threshold,
        evaluation_k,
        evaluation_batch_size,
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


def _read_sql_frame(
    engine: Engine, table: Table, dtypes: Mapping[str, str], chunksize: int = 100_000
) -> pd.DataFrame:
    """Read a SQL table in bounded chunks and downcast numeric columns."""
    statement = select(*table.columns)
    chunks = pd.read_sql_query(statement, engine, chunksize=chunksize)
    frames = [
        frame.astype({key: value for key, value in dtypes.items() if key in frame})
        for frame in chunks
    ]
    if not frames:
        return pd.DataFrame(columns=[column.name for column in table.columns]).astype(
            {key: value for key, value in dtypes.items() if key in table.columns}
        )
    return pd.concat(frames, ignore_index=True, copy=False)


SmokeDataTuple = tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]


def _make_smoke_data() -> SmokeDataTuple:
    """Create a small deterministic MovieLens-shaped dataset for the smoke path."""
    user_count = 30
    movie_count = 50
    ratings_records: list[tuple[int, int, float, int]] = []
    for round_index in range(20):
        for user_id in range(1, user_count + 1):
            movie_id = ((user_id * 7 + round_index * 11) % movie_count) + 1
            rating = (
                5.0
                if (movie_id + user_id) % 4 == 0
                else 3.0 + ((user_id * movie_id) % 3) * 0.5
            )
            ratings_records.append(
                (user_id, movie_id, rating, round_index * user_count + user_id)
            )
    ratings = pd.DataFrame(
        ratings_records,
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    movies = pd.DataFrame(
        [
            (
                movie_id,
                f"Synthetic Movie {movie_id}",
                f"Genre{movie_id % 6}|Genre{(movie_id + 1) % 6}",
            )
            for movie_id in range(1, movie_count + 1)
        ],
        columns=["movie_id", "title", "genres"],
    )
    tags = pd.DataFrame(
        [
            (user_id, movie_id, f"tag{movie_id % 8}")
            for user_id, movie_id, _, _ in ratings_records[::3]
        ],
        columns=["user_id", "movie_id", "tag"],
    )
    genome_scores = pd.DataFrame(
        [
            (movie_id, tag_id, float(((movie_id * tag_id) % 10) / 10))
            for movie_id in range(1, movie_count + 1)
            for tag_id in range(1, 7)
        ],
        columns=["movie_id", "tag_id", "relevance"],
    )
    genome_tags = pd.DataFrame(
        [(tag_id, f"genome{tag_id}") for tag_id in range(1, 7)],
        columns=["tag_id", "tag"],
    )
    return ratings, movies, tags, genome_scores, genome_tags


def _smoke_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    smoke_config = dict(config)
    smoke_config["split"] = {
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
    }
    smoke_config["filtering"] = {
        "min_user_ratings": 1,
        "min_movie_ratings": 1,
    }
    return smoke_config


def _smoke_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    smoke_config = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in config.items()
    }
    smoke_config["item_knn"].update({"k_neighbors": 5})
    smoke_config["als"].update({"factors": 4, "iterations": 2, "workers": 2})
    smoke_config["content_based"].update({"max_features": 1000, "svd_components": 8})
    smoke_config["hybrid"]["candidate_als_weights"] = [0.0, 0.5, 1.0]
    return smoke_config


def main() -> None:
    """Parse config paths and start the explicit training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the complete workflow on a small deterministic synthetic dataset",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_and_log(
        args.data_config,
        args.model_config,
        args.experiment_config,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
