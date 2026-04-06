"""add performance indexes

Revision ID: f1a2b3c4d5e6
Revises: e9f4a7b2c3d1
Create Date: 2026-04-06 00:00:00.000000

Critical query paths covered:
  - predictions(symbol, target_date): API /predict and /predictions endpoints
  - predictions(as_of_time DESC): latest-prediction queries in monitoring
  - features_snapshot(symbol, target_session_date): batch_predict feature loading
  - features_snapshot(as_of_time DESC): data freshness checks in monitoring
  - model_registry(status, created_at DESC): production model lookup on every API call
  - symbols(is_active): active-universe filtering in every pipeline step
  - news_symbol_mapping(symbol): news-by-symbol lookups (PK is (event_id, symbol)
    so symbol-only searches were full scans)
"""
from typing import Sequence, Union

from alembic import op


revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "e9f4a7b2c3d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # predictions: primary API lookup pattern
    op.create_index(
        "ix_predictions_symbol_target_date",
        "predictions",
        ["symbol", "target_date"],
    )
    # predictions: monitoring and dashboard queries for recent predictions
    op.create_index(
        "ix_predictions_as_of_time",
        "predictions",
        ["as_of_time"],
        postgresql_ops={"as_of_time": "DESC"},
    )

    # features_snapshot: batch_predict loads latest snapshot per symbol
    op.create_index(
        "ix_features_snapshot_symbol_date",
        "features_snapshot",
        ["symbol", "target_session_date"],
    )
    # features_snapshot: data freshness / staleness monitoring queries
    op.create_index(
        "ix_features_snapshot_as_of_time",
        "features_snapshot",
        ["as_of_time"],
        postgresql_ops={"as_of_time": "DESC"},
    )

    # model_registry: production model lookup happens on every API call and
    # every batch predict run — without this it was a full table scan
    op.create_index(
        "ix_model_registry_status_created",
        "model_registry",
        ["status", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # symbols: almost every query filters by is_active = true
    op.create_index(
        "ix_symbols_is_active",
        "symbols",
        ["is_active"],
    )

    # news_symbol_mapping: PK is (event_id, symbol) so symbol-only lookups
    # (e.g. "get all news for AAPL") were full PK scans
    op.create_index(
        "ix_news_symbol_mapping_symbol",
        "news_symbol_mapping",
        ["symbol"],
    )


def downgrade() -> None:
    op.drop_index("ix_news_symbol_mapping_symbol", table_name="news_symbol_mapping")
    op.drop_index("ix_symbols_is_active", table_name="symbols")
    op.drop_index("ix_model_registry_status_created", table_name="model_registry")
    op.drop_index("ix_features_snapshot_as_of_time", table_name="features_snapshot")
    op.drop_index("ix_features_snapshot_symbol_date", table_name="features_snapshot")
    op.drop_index("ix_predictions_as_of_time", table_name="predictions")
    op.drop_index("ix_predictions_symbol_target_date", table_name="predictions")
