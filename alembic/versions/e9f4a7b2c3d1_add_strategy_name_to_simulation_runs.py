"""add strategy_name to simulation_runs

Revision ID: e9f4a7b2c3d1
Revises: a2b8c4d6e1f3
Create Date: 2026-03-17 20:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9f4a7b2c3d1"
down_revision: Union[str, None] = "a2b8c4d6e1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "simulation_runs",
        sa.Column("strategy_name", sa.String(50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("simulation_runs", "strategy_name")
