"""SQLAlchemy models for MovieLens source data."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from recsys.db.base import Base


class User(Base):
    """A user identified by their MovieLens user ID."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)


class Movie(Base):
    """Movie metadata supplied by MovieLens."""

    __tablename__ = "movies"

    movie_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    genres: Mapped[str] = mapped_column(Text, nullable=False)


class Rating(Base):
    """A historical user rating for a movie."""

    __tablename__ = "ratings"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "movie_id", name="pk_ratings"),
        Index("ix_ratings_user_id", "user_id"),
        Index("ix_ratings_movie_id", "movie_id"),
        Index("ix_ratings_timestamp", "timestamp"),
    )

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)


class Tag(Base):
    """A free-form user tag attached to a movie."""

    __tablename__ = "tags"
    __table_args__ = (
        PrimaryKeyConstraint("user_id", "movie_id", "tag", "timestamp", name="pk_tags"),
        Index("ix_tags_user_id", "user_id"),
        Index("ix_tags_movie_id", "movie_id"),
        Index("ix_tags_timestamp", "timestamp"),
    )

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False
    )
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)


class Link(Base):
    """External identifier mappings for a MovieLens movie."""

    __tablename__ = "links"

    movie_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("movies.movie_id", ondelete="CASCADE"),
        primary_key=True,
    )
    imdb_id: Mapped[int | None] = mapped_column(BigInteger)
    tmdb_id: Mapped[int | None] = mapped_column(BigInteger)


class GenomeTag(Base):
    """A tag in the MovieLens genome."""

    __tablename__ = "genome_tags"

    tag_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tag: Mapped[str] = mapped_column(Text, nullable=False)


class GenomeScore(Base):
    """Relevance score for a movie and genome tag."""

    __tablename__ = "genome_scores"
    __table_args__ = (
        PrimaryKeyConstraint("movie_id", "tag_id", name="pk_genome_scores"),
        Index("ix_genome_scores_tag_id", "tag_id"),
    )

    movie_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False
    )
    tag_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("genome_tags.tag_id", ondelete="CASCADE"),
        nullable=False,
    )
    relevance: Mapped[float] = mapped_column(Float, nullable=False)


class ExperimentResult(Base):
    """Persisted outcome of one offline-simulated experiment."""

    __tablename__ = "experiments"

    experiment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    control_name: Mapped[str] = mapped_column(String(255), nullable=False)
    treatment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    control_users: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_users: Mapped[int] = mapped_column(Integer, nullable=False)
    control_impressions: Mapped[int] = mapped_column(BigInteger, nullable=False)
    treatment_impressions: Mapped[int] = mapped_column(BigInteger, nullable=False)
    control_clicks: Mapped[int] = mapped_column(BigInteger, nullable=False)
    treatment_clicks: Mapped[int] = mapped_column(BigInteger, nullable=False)
    control_ctr: Mapped[float] = mapped_column(Float, nullable=False)
    treatment_ctr: Mapped[float] = mapped_column(Float, nullable=False)
    effect: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_interval_low: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_interval_high: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size_per_group: Mapped[int] = mapped_column(Integer, nullable=False)
    statistical_test: Mapped[str] = mapped_column(String(64), nullable=False)
    cuped_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Feedback(Base):
    """User-submitted rating or click feedback event."""

    __tablename__ = "feedback"

    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    movie_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    rating: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
