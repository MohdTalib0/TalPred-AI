"""add sentiment features and headline cache

Revision ID: a3c7e91d4f5b
Revises: fe53eadd321b
Create Date: 2026-03-11 23:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3c7e91d4f5b'
down_revision: Union[str, None] = 'fe53eadd321b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("news_sentiment_std", sa.Float))
    op.add_column("features_snapshot", sa.Column("news_positive_ratio", sa.Float))
    op.add_column("features_snapshot", sa.Column("news_negative_ratio", sa.Float))
    op.add_column("features_snapshot", sa.Column("news_volume", sa.Float))

    op.create_table(
        "headline_sentiment_cache",
        sa.Column("headline_hash", sa.String(64), primary_key=True),
        sa.Column("headline", sa.Text, nullable=False),
        sa.Column("sentiment", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("model", sa.String(100)),
        sa.Column("scored_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("headline_sentiment_cache")
    op.drop_column("features_snapshot", "news_volume")
    op.drop_column("features_snapshot", "news_negative_ratio")
    op.drop_column("features_snapshot", "news_positive_ratio")
    op.drop_column("features_snapshot", "news_sentiment_std")
