"""add momentum horizon and reversal feature columns

Revision ID: d4e7c2a1f9b3
Revises: c1f2a9d8e7b6
Create Date: 2026-03-12 21:35:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e7c2a1f9b3"
down_revision: Union[str, None] = "c1f2a9d8e7b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("momentum_20d", sa.Float))
    op.add_column("features_snapshot", sa.Column("momentum_60d", sa.Float))
    op.add_column("features_snapshot", sa.Column("momentum_120d", sa.Float))
    op.add_column("features_snapshot", sa.Column("short_term_reversal", sa.Float))
    op.add_column("features_snapshot", sa.Column("momentum_60d_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("momentum_120d_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("short_term_reversal_rank_market", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "short_term_reversal_rank_market")
    op.drop_column("features_snapshot", "momentum_120d_rank_market")
    op.drop_column("features_snapshot", "momentum_60d_rank_market")
    op.drop_column("features_snapshot", "short_term_reversal")
    op.drop_column("features_snapshot", "momentum_120d")
    op.drop_column("features_snapshot", "momentum_60d")
    op.drop_column("features_snapshot", "momentum_20d")
