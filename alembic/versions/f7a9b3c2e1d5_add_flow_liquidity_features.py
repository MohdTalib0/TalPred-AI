"""add flow-based liquidity feature columns

Adds three flow-based signal columns to features_snapshot:
  volume_acceleration  - 5d vs 20d volume change delta (sudden surge detection)
  signed_volume_proxy  - sign(return_1d) * volume_zscore_20d  (order-flow proxy)
  price_volume_trend   - momentum_5d * clipped(volume_change_5d) (trend confirmation)

Revision ID: f7a9b3c2e1d5
Revises: c1f2a9d8e7b6
Create Date: 2026-03-10 12:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f7a9b3c2e1d5"
down_revision: Union[str, None] = "d4e7c2a1f9b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("volume_acceleration", sa.Float))
    op.add_column("features_snapshot", sa.Column("signed_volume_proxy", sa.Float))
    op.add_column("features_snapshot", sa.Column("price_volume_trend", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "price_volume_trend")
    op.drop_column("features_snapshot", "signed_volume_proxy")
    op.drop_column("features_snapshot", "volume_acceleration")
