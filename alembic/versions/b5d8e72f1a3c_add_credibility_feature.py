"""add credibility feature to features_snapshot

Revision ID: b5d8e72f1a3c
Revises: a3c7e91d4f5b
Create Date: 2026-03-12 16:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b5d8e72f1a3c'
down_revision: Union[str, None] = 'a3c7e91d4f5b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("features_snapshot", sa.Column("news_credibility_avg", sa.Float))


def downgrade() -> None:
    op.drop_column("features_snapshot", "news_credibility_avg")
