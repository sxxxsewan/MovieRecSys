"""Add offline experiment results."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_experiment_results"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("experiment_name", sa.String(length=255), nullable=False),
        sa.Column("control_name", sa.String(length=255), nullable=False),
        sa.Column("treatment_name", sa.String(length=255), nullable=False),
        sa.Column("control_users", sa.Integer(), nullable=False),
        sa.Column("treatment_users", sa.Integer(), nullable=False),
        sa.Column("control_impressions", sa.BigInteger(), nullable=False),
        sa.Column("treatment_impressions", sa.BigInteger(), nullable=False),
        sa.Column("control_clicks", sa.BigInteger(), nullable=False),
        sa.Column("treatment_clicks", sa.BigInteger(), nullable=False),
        sa.Column("control_ctr", sa.Float(), nullable=False),
        sa.Column("treatment_ctr", sa.Float(), nullable=False),
        sa.Column("effect", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("confidence_interval_low", sa.Float(), nullable=False),
        sa.Column("confidence_interval_high", sa.Float(), nullable=False),
        sa.Column("sample_size_per_group", sa.Integer(), nullable=False),
        sa.Column("statistical_test", sa.String(length=64), nullable=False),
        sa.Column("cuped_applied", sa.Boolean(), nullable=False),
        sa.Column("is_simulated", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("experiment_id", name="pk_experiments"),
    )
    op.create_index(
        "ix_experiments_experiment_name", "experiments", ["experiment_name"]
    )


def downgrade() -> None:
    op.drop_index("ix_experiments_experiment_name", table_name="experiments")
    op.drop_table("experiments")
