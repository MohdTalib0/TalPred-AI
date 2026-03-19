"""add promoted_at to model_registry

Revision ID: b2d4f6a8c0e2
Revises: a7c3f92d8b15
Create Date: 2026-03-19 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b2d4f6a8c0e2"
down_revision: Union[str, None] = "a7c3f92d8b15"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_registry",
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_registry", "promoted_at")
