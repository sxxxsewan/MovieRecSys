"""FastAPI application for MovieLens recommendations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Query, Request, status

from recsys.api.schemas import (
    ExperimentResultResponse,
    ExperimentResultsResponse,
    FeedbackRequest,
    FeedbackResponse,
    MovieDetails,
    RecommendationItem,
    RecommendationResponse,
    SimilarMovieItem,
    SimilarMoviesResponse,
)
from recsys.api.service import RecommendationService, build_service


def create_app(service: RecommendationService | None = None) -> FastAPI:
    """Create an app with an optional injected service for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            runtime_service = build_service()
            app.state.recommendation_service = runtime_service
        else:
            runtime_service = service
            app.state.recommendation_service = runtime_service
        try:
            yield
        finally:
            if service is None:
                runtime_service.engine.dispose()

    app = FastAPI(
        title="MovieLens Recommender API",
        description=(
            "Classical movie recommendations. Experiment assignment and historical "
            "rating-based A/B results are offline simulations."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/recommend/{user_id}", response_model=RecommendationResponse)
    def recommend(
        user_id: int,
        request: Request,
        k: int = Query(default=10, ge=1, le=100),
    ) -> RecommendationResponse:
        api_service = _service(request)
        try:
            group, ranked = api_service.recommend(user_id, k)
        except LookupError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        movies = api_service.get_movies([movie_id for movie_id, _ in ranked])
        return RecommendationResponse(
            user_id=user_id,
            experiment_name=api_service.experiment_name,
            group=group,
            recommendations=[
                RecommendationItem(
                    movie_id=movie_id,
                    title=movies[movie_id].title if movie_id in movies else None,
                    genres=movies[movie_id].genres if movie_id in movies else None,
                    score=score,
                    explanation=f"Ranked by the {group} recommender.",
                )
                for movie_id, score in ranked
            ],
        )

    @app.get("/similar/{movie_id}", response_model=SimilarMoviesResponse)
    def similar(
        movie_id: int,
        request: Request,
        k: int = Query(default=10, ge=1, le=100),
    ) -> SimilarMoviesResponse:
        api_service = _service(request)
        try:
            ranked = api_service.similar_items(movie_id, k)
        except LookupError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        movies = api_service.get_movies([item_id for item_id, _ in ranked])
        return SimilarMoviesResponse(
            movie_id=movie_id,
            similar_movies=[
                SimilarMovieItem(
                    movie_id=item_id,
                    title=movies[item_id].title if item_id in movies else None,
                    genres=movies[item_id].genres if item_id in movies else None,
                    score=score,
                )
                for item_id, score in ranked
            ],
        )

    @app.get("/movies/{movie_id}", response_model=MovieDetails)
    def movie_details(movie_id: int, request: Request) -> MovieDetails:
        movie = _service(request).get_movie(movie_id)
        if movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return MovieDetails(
            movie_id=movie.movie_id,
            title=movie.title,
            genres=movie.genres,
        )

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def record_feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
        feedback = _service(request).record_feedback(
            payload.user_id,
            payload.movie_id,
            payload.event_type,
            payload.rating,
        )
        return FeedbackResponse.model_validate(feedback)

    @app.get("/experiments/{name}/results", response_model=ExperimentResultsResponse)
    def experiment_results(name: str, request: Request) -> ExperimentResultsResponse:
        results = _service(request).experiment_results(name)
        if not results:
            raise HTTPException(
                status_code=404, detail="No results found for this experiment"
            )
        return ExperimentResultsResponse(
            experiment_name=name,
            results=[
                ExperimentResultResponse.model_validate(result) for result in results
            ],
        )

    return app


def _service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Recommendation service is not initialized"
        )
    return cast(RecommendationService, service)


app = create_app()
