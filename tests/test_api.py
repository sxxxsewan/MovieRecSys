from __future__ import annotations

from collections.abc import Hashable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from recsys.ab.assignment import assign_group
from recsys.api.app import create_app
from recsys.api.service import RecommendationService
from recsys.db import Base, ExperimentResult, Movie


class FakeRecommender:
    def recommend(self, user_id: Hashable, k: int = 10) -> list[tuple[int, float]]:
        return [(2, 0.9), (3, 0.7)][:k]

    def similar_items(self, movie_id: Hashable, k: int = 10) -> list[tuple[int, float]]:
        return [(3, 0.8), (2, 0.6)][:k]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    engine = create_engine(f"sqlite:///{tmp_path / 'api-test.sqlite'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            Movie.__table__.insert(),
            [
                {"movie_id": 1, "title": "Source Movie", "genres": "Drama"},
                {"movie_id": 2, "title": "Recommended Movie", "genres": "Comedy"},
                {"movie_id": 3, "title": "Similar Movie", "genres": "Sci-Fi"},
            ],
        )
        connection.execute(
            ExperimentResult.__table__.insert(),
            {
                "experiment_name": "recommendation-ranking-v1",
                "control_name": "popularity",
                "treatment_name": "hybrid",
                "control_users": 20,
                "treatment_users": 21,
                "control_impressions": 100,
                "treatment_impressions": 105,
                "control_clicks": 10,
                "treatment_clicks": 15,
                "control_ctr": 0.1,
                "treatment_ctr": 15 / 105,
                "effect": 15 / 105 - 0.1,
                "p_value": 0.04,
                "confidence_interval_low": 0.001,
                "confidence_interval_high": 0.08,
                "sample_size_per_group": 1000,
                "statistical_test": "welch_t_test",
                "cuped_applied": False,
                "is_simulated": True,
                "created_at": datetime.now(UTC),
            },
        )
    service = RecommendationService(
        engine=engine,
        models={
            "popularity": FakeRecommender(),
            "hybrid": FakeRecommender(),
            "similar": FakeRecommender(),
        },
        experiment_name="recommendation-ranking-v1",
        control_name="popularity",
        treatment_name="hybrid",
    )
    with TestClient(create_app(service)) as test_client:
        yield test_client
    engine.dispose()


def test_recommendation_route_assigns_group_and_returns_movie_explanations(
    client: TestClient,
) -> None:
    user_id = next(
        user_id
        for user_id in range(1, 100)
        if assign_group(
            user_id,
            "recommendation-ranking-v1",
            control="popularity",
            treatment="hybrid",
        )
        == "hybrid"
    )

    response = client.get(f"/recommend/{user_id}?k=1")

    assert response.status_code == 200
    body = response.json()
    assert body["group"] == "hybrid"
    assert body["recommendations"][0]["movie_id"] == 2
    assert body["recommendations"][0]["title"] == "Recommended Movie"
    assert body["recommendations"][0]["explanation"]


def test_similar_movie_and_movie_detail_routes(client: TestClient) -> None:
    similar_response = client.get("/similar/1?k=1")
    movie_response = client.get("/movies/1")

    assert similar_response.status_code == 200
    assert similar_response.json()["similar_movies"][0]["movie_id"] == 3
    assert movie_response.json() == {
        "movie_id": 1,
        "title": "Source Movie",
        "genres": "Drama",
    }
    assert client.get("/movies/999").status_code == 404


def test_feedback_route_validates_and_persists_rating_and_click(
    client: TestClient,
) -> None:
    rating_response = client.post(
        "/feedback",
        json={"user_id": 5, "movie_id": 2, "event_type": "rating", "rating": 4.5},
    )
    click_response = client.post(
        "/feedback",
        json={"user_id": 5, "movie_id": 3, "event_type": "click"},
    )

    assert rating_response.status_code == 201
    assert rating_response.json()["rating"] == 4.5
    assert click_response.status_code == 201
    assert click_response.json()["rating"] is None
    assert (
        client.post(
            "/feedback",
            json={"user_id": 5, "movie_id": 2, "event_type": "rating"},
        ).status_code
        == 422
    )


def test_experiment_results_and_openapi(client: TestClient) -> None:
    response = client.get("/experiments/recommendation-ranking-v1/results")

    assert response.status_code == 200
    assert response.json()["results"][0]["is_simulated"] is True
    assert client.get("/experiments/missing/results").status_code == 404
    assert client.get("/openapi.json").status_code == 200


def test_unconfigured_recommendation_model_returns_service_unavailable(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'api-empty.sqlite'}")
    Base.metadata.create_all(engine)
    service = RecommendationService(
        engine=engine,
        models={},
        experiment_name="experiment",
        control_name="control",
        treatment_name="treatment",
    )
    with TestClient(create_app(service)) as test_client:
        response = test_client.get("/recommend/1")
    engine.dispose()

    assert response.status_code == 503
