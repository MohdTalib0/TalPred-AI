"""add OHLCV bar-shape flow feature columns

Adds three features computed from high/low intraday bar shape:
  volume_imbalance_proxy  - candle body direction × volume shock
  liquidity_shock_5d      - rolling 5d max of |volume_zscore_20d|
  vwap_deviation          - (close - typical_price) / typical_price

Revision ID: a2b8c4d6e1f3
Revises: f7a9b3c2e1d5
Create Date: 2026-03-14 12:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a2b8c4d6e1f3"
down_revision: Union[str, None] = "f7a9b3c2e1d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("volume_imbalance_proxy", sa.Float))
    op.add_column("features_snapshot", sa.Column("liquidity_shock_5d", sa.Float))
    op.add_column("features_snapshot", sa.Column("vwap_deviation", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "vwap_deviation")
    op.drop_column("features_snapshot", "liquidity_shock_5d")
    op.drop_column("features_snapshot", "volume_imbalance_proxy")
