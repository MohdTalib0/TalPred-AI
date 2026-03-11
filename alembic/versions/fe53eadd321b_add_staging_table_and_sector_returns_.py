"""add staging table and sector returns daily

Revision ID: fe53eadd321b
Revises: 86d8f3762447
Create Date: 2026-03-11 19:50:51.724443
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'fe53eadd321b'
down_revision: Union[str, None] = '86d8f3762447'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_bars_staging (
            symbol TEXT,
            date DATE,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            adj_close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            source TEXT,
            event_time TIMESTAMPTZ,
            as_of_time TIMESTAMPTZ
        )
    """)

    op.create_table(
        "sector_returns_daily",
        sa.Column("sector", sa.String(100), primary_key=True),
        sa.Column("date", sa.Date, primary_key=True),
        sa.Column("sector_return_1d", sa.Float),
        sa.Column("sector_return_5d", sa.Float),
        sa.Column("sector_volatility_20d", sa.Float),
        sa.Column("avg_volume_20d", sa.Float),
        sa.Column("num_stocks", sa.Integer),
    )


def downgrade() -> None:
    op.drop_table("sector_returns_daily")
    op.execute("DROP TABLE IF EXISTS market_bars_staging")
