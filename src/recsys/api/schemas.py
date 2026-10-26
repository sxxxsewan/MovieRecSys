"""Validated HTTP request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MovieDetails(BaseModel):
    """Movie metadata returned by the API."""

    movie_id: int
    title: str
    genres: str


class RecommendationItem(BaseModel):
    """A ranked movie and a concise explanation of its placement."""

    movie_id: int
    title: str | None
    genres: str | None
    score: float
    explanation: str


class RecommendationResponse(BaseModel):
    """Recommendations returned with stable experiment assignment."""

    user_id: int
    experiment_name: str
    group: str
    recommendations: list[RecommendationItem]


class SimilarMovieItem(BaseModel):
    """A similar movie with its model similarity score."""

    movie_id: int
    title: str | None
    genres: str | None
    score: float


class SimilarMoviesResponse(BaseModel):
    """Similar movies for one catalog item."""

    movie_id: int
    similar_movies: list[SimilarMovieItem]


class FeedbackRequest(BaseModel):
    """Rating or click event submitted by a client."""

    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)
    event_type: Literal["rating", "click"]
    rating: float | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def validate_event_rating(self) -> FeedbackRequest:
        if self.event_type == "rating" and self.rating is None:
            raise ValueError("rating is required for rating feedback")
        if self.event_type == "click" and self.rating is not None:
            raise ValueError("rating must be omitted for click feedback")
        return self


class FeedbackResponse(BaseModel):
    """Acknowledgement for a persisted feedback event."""

    model_config = ConfigDict(from_attributes=True)

    feedback_id: int
    user_id: int
    movie_id: int
    event_type: Literal["rating", "click"]
    rating: float | None
    created_at: datetime


class ExperimentResultResponse(BaseModel):
    """One persisted offline experiment result."""

    model_config = ConfigDict(from_attributes=True)

    experiment_id: int
    experiment_name: str
    control_name: str
    treatment_name: str
    control_users: int
    treatment_users: int
    control_impressions: int
    treatment_impressions: int
    control_clicks: int
    treatment_clicks: int
    control_ctr: float
    treatment_ctr: float
    effect: float
    p_value: float
    confidence_interval_low: float
    confidence_interval_high: float
    sample_size_per_group: int
    statistical_test: str
    cuped_applied: bool
    is_simulated: bool
    created_at: datetime


class ExperimentResultsResponse(BaseModel):
    """Results for an experiment, newest first."""

    experiment_name: str
    results: list[ExperimentResultResponse]
