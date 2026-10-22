from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from recsys.data.loader import load_data
from recsys.db import GenomeScore, GenomeTag, Link, Movie, Tag, User


@pytest.fixture
def movielens_csvs(tmp_path: Path) -> Path:
    data = {
        "ratings.csv": pd.DataFrame(
            [(1, 10, 4.5, 100), (2, 10, 3.0, 200)],
            columns=["userId", "movieId", "rating", "timestamp"],
        ),
        "movies.csv": pd.DataFrame(
            [(10, "Example, Movie (2000)", "Drama|Comedy")],
            columns=["movieId", "title", "genres"],
        ),
        "tags.csv": pd.DataFrame(
            [(1, 10, "favorite", 300)],
            columns=["userId", "movieId", "tag", "timestamp"],
        ),
        "links.csv": pd.DataFrame(
            [(10, 123, 456)], columns=["movieId", "imdbId", "tmdbId"]
        ),
        "genome-scores.csv": pd.DataFrame(
            [(10, 1, 0.75)], columns=["movieId", "tagId", "relevance"]
        ),
        "genome-tags.csv": pd.DataFrame(
            [(1, "thought-provoking")], columns=["tagId", "tag"]
        ),
    }
    for filename, frame in data.items():
        frame.to_csv(tmp_path / filename, index=False)
    return tmp_path


def test_load_data_inserts_all_csv_entities(
    movielens_csvs: Path, tmp_path: Path
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    from recsys.db import Base

    Base.metadata.create_all(engine)

    counts = load_data(movielens_csvs, engine, chunk_size=1)

    assert counts == {
        "users": 2,
        "movies": 1,
        "links": 1,
        "genome_tags": 1,
        "ratings": 2,
        "tags": 1,
        "genome_scores": 1,
    }
    with engine.connect() as connection:
        movie_title = connection.scalar(select(Movie.title))
        user_ids = connection.scalars(select(User.user_id).order_by(User.user_id)).all()
        tag = connection.scalar(select(Tag.tag))
        link = connection.execute(select(Link.imdb_id, Link.tmdb_id)).one()
        genome_relevance = connection.scalar(select(GenomeScore.relevance))
        genome_tag = connection.scalar(select(GenomeTag.tag))

    assert movie_title == "Example, Movie (2000)"
    assert user_ids == [1, 2]
    assert tag == "favorite"
    assert link == (123, 456)
    assert genome_relevance == pytest.approx(0.75)
    assert genome_tag == "thought-provoking"


def test_load_data_requires_replace_for_existing_rows(
    movielens_csvs: Path, tmp_path: Path
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    from recsys.db import Base

    Base.metadata.create_all(engine)
    load_data(movielens_csvs, engine)

    with pytest.raises(RuntimeError, match="Use --replace"):
        load_data(movielens_csvs, engine)
