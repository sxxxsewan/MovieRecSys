"""MLflow logging and registration for recommendation models."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types import ColSpec, Schema


class RecommendationPyfuncModel(PythonModel):
    """Serve a recommender through a small MLflow pyfunc input contract."""

    def __init__(self, recommender: Any) -> None:
        self.recommender = recommender

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        has_user_id = "user_id" in model_input.columns
        has_movie_id = "movie_id" in model_input.columns
        if has_user_id == has_movie_id:
            raise ValueError(
                "Model input must include exactly one of user_id or movie_id"
            )
        predictions: list[dict[str, Any]] = []
        for row in model_input.itertuples(index=False):
            if "k" in model_input.columns:
                k = int(row.k)
            else:
                k = int((params or {}).get("k", 10))
            if has_movie_id:
                if not hasattr(self.recommender, "similar_items"):
                    raise TypeError("This recommender does not support similar_items")
                predictions.append(
                    {
                        "movie_id": row.movie_id,
                        "similar_movies": self.recommender.similar_items(
                            row.movie_id, k
                        ),
                    }
                )
            else:
                predictions.append(
                    {
                        "user_id": row.user_id,
                        "recommendations": self.recommender.recommend(row.user_id, k),
                    }
                )
        return pd.DataFrame(predictions)


def configure_tracking(tracking_uri_env: str = "MLFLOW_TRACKING_URI") -> None:
    """Set MLflow to the configured tracking store or a local file store."""
    tracking_uri = os.getenv(tracking_uri_env) or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
    if registry_uri is None and tracking_uri.startswith("file:"):
        tracking_path = Path(tracking_uri.removeprefix("file:")).expanduser()
        tracking_path.mkdir(parents=True, exist_ok=True)
        registry_path = tracking_path.resolve().parent / "mlruns_registry.db"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_uri = f"sqlite:///{registry_path.as_posix()}"
    mlflow.set_registry_uri(registry_uri or tracking_uri)


def log_recommender_run(
    run_name: str,
    recommender: Any,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> str:
    """Log one fitted recommender and return its MLflow run ID."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {key: _mlflow_parameter(value) for key, value in parameters.items()}
        )
        mlflow.log_metrics(
            {_mlflow_metric_name(key): float(value) for key, value in metrics.items()}
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RecommendationPyfuncModel(recommender),
            signature=ModelSignature(
                inputs=Schema(
                    [
                        ColSpec("long", "user_id"),
                        ColSpec("long", "k"),
                    ]
                )
            ),
        )
        return run.info.run_id


def register_best_model(run_id: str, model_name: str) -> str:
    """Register the selected run's pyfunc artifact in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(registered.version)


def _mlflow_parameter(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _mlflow_metric_name(name: str) -> str:
    """Encode ranking cutoffs with MLflow-safe characters."""
    return name.replace("@", "_at_")
