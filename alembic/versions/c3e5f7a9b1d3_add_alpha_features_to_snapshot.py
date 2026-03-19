"""add alpha features to features_snapshot

Revision ID: c3e5f7a9b1d3
Revises: b2d4f6a8c0e2
Create Date: 2026-03-19 11:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c3e5f7a9b1d3"
down_revision: Union[str, None] = "b2d4f6a8c0e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_COLS = [
    "vol_adj_momentum_20d",
    "vol_adj_momentum_60d",
    "pct_from_52w_high",
    "idio_momentum_20d",
    "idio_momentum_60d",
    "vol_price_divergence",
    "vol_adj_momentum_20d_rank",
    "pct_from_52w_high_rank",
    "idio_momentum_20d_rank",
]


def upgrade() -> None:
    for col in _NEW_COLS:
        op.add_column("features_snapshot", sa.Column(col, sa.Float(), nullable=True))


def downgrade() -> None:
    for col in reversed(_NEW_COLS):
        op.drop_column("features_snapshot", col)
