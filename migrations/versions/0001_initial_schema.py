"""Create the MovieLens source-data tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", name="pk_users"),
    )
    op.create_table(
        "movies",
        sa.Column("movie_id", sa.BigInteger(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("genres", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("movie_id", name="pk_movies"),
    )
    op.create_table(
        "links",
        sa.Column(
            "movie_id",
            sa.BigInteger(),
            sa.ForeignKey("movies.movie_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("imdb_id", sa.BigInteger(), nullable=True),
        sa.Column("tmdb_id", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("movie_id", name="pk_links"),
    )
    op.create_table(
        "genome_tags",
        sa.Column("tag_id", sa.BigInteger(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("tag_id", name="pk_genome_tags"),
    )
    op.create_table(
        "ratings",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "movie_id",
            sa.BigInteger(),
            sa.ForeignKey("movies.movie_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "movie_id", name="pk_ratings"),
    )
    op.create_index("ix_ratings_user_id", "ratings", ["user_id"])
    op.create_index("ix_ratings_movie_id", "ratings", ["movie_id"])
    op.create_index("ix_ratings_timestamp", "ratings", ["timestamp"])
    op.create_table(
        "tags",
        sa.Column(
            "user_id",
            sa.BigInteger(),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "movie_id",
            sa.BigInteger(),
            sa.ForeignKey("movies.movie_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint(
            "user_id", "movie_id", "tag", "timestamp", name="pk_tags"
        ),
    )
    op.create_index("ix_tags_user_id", "tags", ["user_id"])
    op.create_index("ix_tags_movie_id", "tags", ["movie_id"])
    op.create_index("ix_tags_timestamp", "tags", ["timestamp"])
    op.create_table(
        "genome_scores",
        sa.Column(
            "movie_id",
            sa.BigInteger(),
            sa.ForeignKey("movies.movie_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tag_id",
            sa.BigInteger(),
            sa.ForeignKey("genome_tags.tag_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("movie_id", "tag_id", name="pk_genome_scores"),
    )
    op.create_index("ix_genome_scores_tag_id", "genome_scores", ["tag_id"])


def downgrade() -> None:
    op.drop_index("ix_genome_scores_tag_id", table_name="genome_scores")
    op.drop_table("genome_scores")
    op.drop_index("ix_tags_timestamp", table_name="tags")
    op.drop_index("ix_tags_movie_id", table_name="tags")
    op.drop_index("ix_tags_user_id", table_name="tags")
    op.drop_table("tags")
    op.drop_index("ix_ratings_timestamp", table_name="ratings")
    op.drop_index("ix_ratings_movie_id", table_name="ratings")
    op.drop_index("ix_ratings_user_id", table_name="ratings")
    op.drop_table("ratings")
    op.drop_table("genome_tags")
    op.drop_table("links")
    op.drop_table("movies")
    op.drop_table("users")
