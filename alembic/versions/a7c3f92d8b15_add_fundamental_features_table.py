"""add fundamental_features table and snapshot columns

Revision ID: a7c3f92d8b15
Revises: fe53eadd321b
Create Date: 2026-03-17 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a7c3f92d8b15"
down_revision: Union[str, None] = "e9f4a7b2c3d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fundamental_features",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("as_of_date", sa.Date, nullable=False),
        sa.Column("accruals", sa.Float),
        sa.Column("roe_trend", sa.Float),
        sa.Column("earnings_momentum", sa.Float),
        sa.Column("revenue_surprise", sa.Float),
        sa.Column("gross_margin_change", sa.Float),
        sa.Column("operating_leverage", sa.Float),
        sa.Column("source", sa.String(20)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("symbol", "as_of_date"),
    )
    op.create_index(
        "ix_fundamental_features_as_of_date",
        "fundamental_features",
        ["as_of_date"],
    )

    op.add_column("features_snapshot", sa.Column("accruals", sa.Float))
    op.add_column("features_snapshot", sa.Column("roe_trend", sa.Float))
    op.add_column("features_snapshot", sa.Column("earnings_momentum", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "earnings_momentum")
    op.drop_column("features_snapshot", "roe_trend")
    op.drop_column("features_snapshot", "accruals")
    op.drop_index("ix_fundamental_features_as_of_date", table_name="fundamental_features")
    op.drop_table("fundamental_features")
