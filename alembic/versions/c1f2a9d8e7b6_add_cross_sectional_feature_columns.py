"""add cross-sectional and missingness feature columns

Revision ID: c1f2a9d8e7b6
Revises: b5d8e72f1a3c
Create Date: 2026-03-12 21:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1f2a9d8e7b6"
down_revision: Union[str, None] = "b5d8e72f1a3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("sector_relative_return_1d", sa.Float))
    op.add_column("features_snapshot", sa.Column("sector_relative_return_5d", sa.Float))
    op.add_column("features_snapshot", sa.Column("momentum_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("volatility_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("rsi_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("volume_rank_market", sa.Float))
    op.add_column("features_snapshot", sa.Column("sector_momentum_rank", sa.Float))
    op.add_column("features_snapshot", sa.Column("volume_change_5d", sa.Float))
    op.add_column("features_snapshot", sa.Column("volume_zscore_20d", sa.Float))
    op.add_column("features_snapshot", sa.Column("volatility_expansion_5_20", sa.Float))
    op.add_column("features_snapshot", sa.Column("news_present_flag", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "news_present_flag")
    op.drop_column("features_snapshot", "volatility_expansion_5_20")
    op.drop_column("features_snapshot", "volume_zscore_20d")
    op.drop_column("features_snapshot", "volume_change_5d")
    op.drop_column("features_snapshot", "sector_momentum_rank")
    op.drop_column("features_snapshot", "volume_rank_market")
    op.drop_column("features_snapshot", "rsi_rank_market")
    op.drop_column("features_snapshot", "volatility_rank_market")
    op.drop_column("features_snapshot", "momentum_rank_market")
    op.drop_column("features_snapshot", "sector_relative_return_5d")
    op.drop_column("features_snapshot", "sector_relative_return_1d")
