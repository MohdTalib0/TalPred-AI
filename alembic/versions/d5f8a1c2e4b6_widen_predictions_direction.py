"""widen predictions.direction for outperform/underperform

Revision ID: d5f8a1c2e4b6
Revises: c3e5f7a9b1d3
Create Date: 2026-03-21 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d5f8a1c2e4b6"
down_revision: Union[str, None] = "c3e5f7a9b1d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "predictions",
        "direction",
        existing_type=sa.String(length=10),
        type_=sa.String(length=20),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "predictions",
        "direction",
        existing_type=sa.String(length=20),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
