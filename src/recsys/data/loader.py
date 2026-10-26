"""Chunked MovieLens CSV ingestion."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import Engine, Table, delete, insert, select
from sqlalchemy.engine import Connection

from recsys.db import (
    GenomeScore,
    GenomeTag,
    Link,
    Movie,
    Rating,
    Tag,
    User,
    create_engine_from_url,
    get_database_url,
)

LOGGER = logging.getLogger(__name__)
TABLES_BY_NAME: dict[str, type[Any]] = {
    "movies": Movie,
    "ratings": Rating,
    "tags": Tag,
    "links": Link,
    "genome_scores": GenomeScore,
    "genome_tags": GenomeTag,
}
FILE_COLUMNS: dict[str, tuple[str, ...]] = {
    "ratings": ("userId", "movieId", "rating", "timestamp"),
    "movies": ("movieId", "title", "genres"),
    "tags": ("userId", "movieId", "tag", "timestamp"),
    "links": ("movieId", "imdbId", "tmdbId"),
    "genome_scores": ("movieId", "tagId", "relevance"),
    "genome_tags": ("tagId", "tag"),
}
COLUMN_NAMES = {
    "userId": "user_id",
    "movieId": "movie_id",
    "tagId": "tag_id",
    "imdbId": "imdb_id",
    "tmdbId": "tmdb_id",
}
INTEGER_COLUMNS = {"user_id", "movie_id", "tag_id", "imdb_id", "tmdb_id", "timestamp"}
FLOAT_COLUMNS = {"rating", "relevance"}
DELETE_ORDER = (GenomeScore, Tag, Rating, Link, GenomeTag, Movie, User)


def _csv_path(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required MovieLens input file was not found: {path}")
    return path


def _validate_headers(data_dir: Path, files: dict[str, str]) -> None:
    for key, columns in FILE_COLUMNS.items():
        path = _csv_path(data_dir, files[key])
        actual_columns = set(pd.read_csv(path, nrows=0).columns)
        missing = set(columns) - actual_columns
        if missing:
            raise ValueError(
                f"{path} is missing required CSV columns: {', '.join(sorted(missing))}"
            )


def _iter_csv_records(
    path: Path, columns: Sequence[str], chunk_size: int
) -> Iterator[list[dict[str, Any]]]:
    for frame in pd.read_csv(path, usecols=list(columns), chunksize=chunk_size):
        frame = frame.rename(columns=COLUMN_NAMES)
        records: list[dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            normalized: dict[str, Any] = {}
            for column, value in record.items():
                if pd.isna(value):
                    normalized[column] = None
                elif column in INTEGER_COLUMNS:
                    normalized[column] = int(value)
                elif column in FLOAT_COLUMNS:
                    normalized[column] = float(value)
                else:
                    normalized[column] = str(value)
            records.append(normalized)
        if records:
            yield records


def _insert_csv(
    connection: Connection,
    table: Table,
    path: Path,
    columns: Sequence[str],
    chunk_size: int,
) -> int:
    inserted = 0
    for records in _iter_csv_records(path, columns, chunk_size):
        connection.execute(insert(table), records)
        inserted += len(records)
    LOGGER.info("Loaded %s rows into %s from %s", inserted, table.name, path)
    return inserted


def _read_user_ids(path: Path, chunk_size: int) -> list[int]:
    user_ids: set[int] = set()
    for frame in pd.read_csv(path, usecols=["userId"], chunksize=chunk_size):
        user_ids.update(frame["userId"].dropna().astype("int64").tolist())
    if not user_ids:
        raise ValueError(f"No user IDs were found in ratings file: {path}")
    return sorted(user_ids)


def _database_has_data(connection: Connection) -> bool:
    for model in TABLES_BY_NAME.values():
        table = model.__table__
        if connection.scalar(select(1).select_from(table).limit(1)):
            return True
    if connection.scalar(select(1).select_from(User.__table__).limit(1)):
        return True
    return False


def load_data(
    data_dir: Path,
    engine: Engine,
    chunk_size: int = 10_000,
    replace: bool = False,
    files: dict[str, str] | None = None,
) -> dict[str, int]:
    """Load all configured MovieLens CSVs using bounded-size executemany batches."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    input_files = files or {
        "ratings": "ratings.csv",
        "movies": "movies.csv",
        "tags": "tags.csv",
        "links": "links.csv",
        "genome_scores": "genome-scores.csv",
        "genome_tags": "genome-tags.csv",
    }
    _validate_headers(data_dir, input_files)
    ratings_path = _csv_path(data_dir, input_files["ratings"])
    user_ids = _read_user_ids(ratings_path, chunk_size)
    counts: dict[str, int] = {}

    with engine.begin() as connection:
        if _database_has_data(connection):
            if not replace:
                raise RuntimeError(
                    "Database already contains MovieLens rows. Use --replace to "
                    "replace the current dataset."
                )
            for model in DELETE_ORDER:
                connection.execute(delete(model.__table__))

        connection.execute(
            insert(User.__table__), [{"user_id": user_id} for user_id in user_ids]
        )
        counts["users"] = len(user_ids)

        for name in (
            "movies",
            "links",
            "genome_tags",
            "ratings",
            "tags",
            "genome_scores",
        ):
            model = TABLES_BY_NAME[name]
            counts[name] = _insert_csv(
                connection,
                model.__table__,
                _csv_path(data_dir, input_files[name]),
                FILE_COLUMNS[name],
                chunk_size,
            )

    LOGGER.info("MovieLens ingestion completed: %s", counts)
    return counts


def _load_config(config_path: Path) -> tuple[Path, int, str, dict[str, str]]:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    database_config = data_config["database"]
    raw_dir_env = str(data_config.get("raw_dir_env", "MOVIELENS_DATA_DIR"))
    configured_path = os.getenv(raw_dir_env) or str(data_config["raw_dir"])
    data_dir = Path(configured_path).expanduser()
    if not data_dir.is_absolute():
        data_dir = (config_path.resolve().parent / data_dir).resolve()
    return (
        data_dir,
        int(database_config["chunk_size"]),
        str(database_config.get("url_env", "DATABASE_URL")),
        {key: str(value) for key, value in data_config["files"].items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace existing MovieLens rows before loading",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_dir, chunk_size, database_url_env, files = _load_config(args.config)
    database_url = get_database_url(database_url_env)
    engine = create_engine_from_url(database_url)
    load_data(
        data_dir,
        engine,
        chunk_size=chunk_size,
        replace=args.replace,
        files=files,
    )


if __name__ == "__main__":
    main()
