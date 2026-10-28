from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, select

from recsys.data.loader import _load_config, load_data
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


def test_load_data_drops_null_and_blank_tags(
    movielens_csvs: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    pd.DataFrame(
        [
            (1, 10, "  favorite  ", 300),
            (1, 10, "   ", 301),
            (2, 10, None, 302),
        ],
        columns=["userId", "movieId", "tag", "timestamp"],
    ).to_csv(movielens_csvs / "tags.csv", index=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    from recsys.db import Base

    Base.metadata.create_all(engine)

    counts = load_data(movielens_csvs, engine, chunk_size=1)

    assert counts["tags"] == 1
    with engine.connect() as connection:
        tags = connection.scalars(select(Tag.tag)).all()
    assert tags == ["favorite"]
    assert (
        "Dropped 2 rows with missing or blank required values from tags" in caplog.text
    )


def test_load_config_resolves_project_data_path_and_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": "../data",
                    "raw_dir_env": "MOVIELENS_DATA_DIR",
                    "files": {"ratings": "ratings.csv"},
                    "database": {"chunk_size": 50000},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("MOVIELENS_DATA_DIR", raising=False)
    default_dir, *_ = _load_config(config_path)
    override_dir = tmp_path / "drive" / "raw"
    monkeypatch.setenv("MOVIELENS_DATA_DIR", str(override_dir))
    configured_dir, *_ = _load_config(config_path)

    assert default_dir == tmp_path / "data"
    assert configured_dir == override_dir
