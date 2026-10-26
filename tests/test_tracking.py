from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from recsys.models.tracking import _mlflow_metric_name, configure_tracking


def test_configure_tracking_uses_local_registry_for_file_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_path = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tracking_path}")
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)

    configure_tracking()

    assert mlflow.get_tracking_uri().startswith("file:")
    assert mlflow.get_registry_uri().startswith("sqlite:///")
    assert tracking_path.is_dir()


def test_configure_tracking_respects_registry_uri_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_uri = f"sqlite:///{tmp_path / 'registry.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", registry_uri)

    configure_tracking()

    assert mlflow.get_tracking_uri() == "http://mlflow:5000"
    assert mlflow.get_registry_uri() == registry_uri


def test_recommendation_pyfunc_supports_item_similarity() -> None:
    from recsys.models.tracking import RecommendationPyfuncModel

    class SimilarityModel:
        def similar_items(self, movie_id: int, k: int) -> list[tuple[int, float]]:
            return [(movie_id + 1, 0.9)][:k]

    model = RecommendationPyfuncModel(SimilarityModel())
    result = model.predict(
        None,
        pd.DataFrame({"movie_id": [5], "k": [1]}),
    )

    assert result.iloc[0]["similar_movies"] == [(6, 0.9)]


def test_mlflow_metric_names_encode_ranking_cutoffs() -> None:
    assert _mlflow_metric_name("validation_ndcg@10") == "validation_ndcg_at_10"
