"""Database and model adapters used by the FastAPI endpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from recsys.ab.assignment import assign_group
from recsys.db import create_engine_from_url, get_database_url
from recsys.db.models import ExperimentResult, Feedback, Movie


class RecommendationService:
    """Serve configured models and persist/read API data."""

    def __init__(
        self,
        engine: Engine,
        models: Mapping[str, Any],
        experiment_name: str,
        control_name: str,
        treatment_name: str,
    ) -> None:
        self.engine = engine
        self.models = dict(models)
        self.experiment_name = experiment_name
        self.control_name = control_name
        self.treatment_name = treatment_name

    def recommend(self, user_id: int, k: int) -> tuple[str, list[tuple[int, float]]]:
        group = assign_group(
            user_id,
            self.experiment_name,
            control=self.control_name,
            treatment=self.treatment_name,
        )
        model = self.models.get(group)
        if model is None:
            raise LookupError(f"No model is configured for experiment group {group!r}")
        return group, _recommend_with_model(model, user_id, k)

    def similar_items(self, movie_id: int, k: int) -> list[tuple[int, float]]:
        model = self.models.get("similar")
        if model is None:
            raise LookupError("No item-similarity model is configured")
        if hasattr(model, "similar_items"):
            similar_items = model.similar_items(movie_id, k)
        elif hasattr(model, "predict"):
            prediction = model.predict(pd.DataFrame({"movie_id": [movie_id], "k": [k]}))
            if prediction.empty or "similar_movies" not in prediction:
                raise ValueError("Similarity model returned an invalid prediction")
            similar_items = prediction.iloc[0]["similar_movies"]
        else:
            raise TypeError("Configured similarity model must expose similar_items()")
        return [(int(item_id), float(score)) for item_id, score in similar_items]

    def get_movie(self, movie_id: int) -> Movie | None:
        with Session(self.engine) as session:
            return session.get(Movie, movie_id)

    def get_movies(self, movie_ids: Sequence[int]) -> dict[int, Movie]:
        if not movie_ids:
            return {}
        with Session(self.engine) as session:
            movies = session.scalars(
                select(Movie).where(Movie.movie_id.in_(movie_ids))
            ).all()
            return {movie.movie_id: movie for movie in movies}

    def record_feedback(
        self, user_id: int, movie_id: int, event_type: str, rating: float | None
    ) -> Feedback:
        with Session(self.engine) as session, session.begin():
            feedback = Feedback(
                user_id=user_id,
                movie_id=movie_id,
                event_type=event_type,
                rating=rating,
            )
            session.add(feedback)
            session.flush()
            session.refresh(feedback)
            session.expunge(feedback)
            return feedback

    def experiment_results(self, experiment_name: str) -> list[ExperimentResult]:
        with Session(self.engine) as session:
            results = session.scalars(
                select(ExperimentResult)
                .where(ExperimentResult.experiment_name == experiment_name)
                .order_by(
                    ExperimentResult.created_at.desc(),
                    ExperimentResult.experiment_id.desc(),
                )
            ).all()
            for result in results:
                session.expunge(result)
            return results


def build_service() -> RecommendationService:
    """Build the API service using environment-configured DB and MLflow models."""
    experiment_config = _read_ab_config()
    ab_config = experiment_config.get("ab", {})
    control_name = str(ab_config.get("control", "control"))
    treatment_name = str(ab_config.get("treatment", "treatment"))
    experiment_name = str(ab_config.get("name", "recommendation-ranking-v1"))

    model_uris = {
        control_name: os.getenv("RECSYS_CONTROL_MODEL_URI"),
        treatment_name: os.getenv("RECSYS_TREATMENT_MODEL_URI"),
        "similar": os.getenv("RECSYS_SIMILAR_MODEL_URI"),
    }
    models = {
        name: mlflow.pyfunc.load_model(uri) for name, uri in model_uris.items() if uri
    }
    return RecommendationService(
        engine=create_engine_from_url(get_database_url()),
        models=models,
        experiment_name=experiment_name,
        control_name=control_name,
        treatment_name=treatment_name,
    )


def _recommend_with_model(model: Any, user_id: int, k: int) -> list[tuple[int, float]]:
    if hasattr(model, "recommend"):
        recommendations = model.recommend(user_id, k)
    elif hasattr(model, "predict"):
        prediction = model.predict(pd.DataFrame({"user_id": [user_id], "k": [k]}))
        if prediction.empty or "recommendations" not in prediction:
            raise ValueError("Recommendation model returned an invalid prediction")
        recommendations = prediction.iloc[0]["recommendations"]
    else:
        raise TypeError("Configured model must expose recommend() or predict()")
    return [(int(item_id), float(score)) for item_id, score in recommendations]


def _read_ab_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "experiments.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), dict):
        raise ValueError(f"Invalid experiment configuration: {config_path}")
    return config["experiments"]
