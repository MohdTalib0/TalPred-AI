"""Archive cold rows from Postgres to Supabase Storage.

Design goals:
1) Keep hot/live windows in Postgres for daily pipelines and API latency.
2) Move older monthly partitions to object storage as Parquet.
3) Verify row counts + file checksum before deleting archived rows.
4) Default to dry-run mode for safety.

Usage:
  # Preview only (no upload/delete)
  python -m scripts.archive_to_supabase_storage

  # Execute archive/delete for all tables
  python -m scripts.archive_to_supabase_storage --execute

  # Execute a small batch (first 2 months per table)
  python -m scripts.archive_to_supabase_storage --execute --max-months 2

Required env vars:
  DATABASE_URL
  REDIS_URL  (required indirectly by src.config)
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY

Optional env vars:
  SUPABASE_ARCHIVE_BUCKET (default: quant-archive)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db import SessionLocal, engine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("archive_storage")


@dataclass(frozen=True)
class TablePolicy:
    table: str
    date_column: str
    retention_years: int


TABLE_POLICIES: dict[str, TablePolicy] = {
    "market_bars_daily": TablePolicy("market_bars_daily", "date", 3),
    "features_snapshot": TablePolicy("features_snapshot", "target_session_date", 2),
    "predictions": TablePolicy("predictions", "target_date", 2),
    "paper_trades": TablePolicy("paper_trades", "date", 2),
    "simulation_runs": TablePolicy("simulation_runs", "end_date", 2),
    "sector_returns_daily": TablePolicy("sector_returns_daily", "date", 2),
}


def _safe_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe identifier: {name}")
    return name


def _cutoff_from_years(years: int) -> date:
    return date.today() - timedelta(days=365 * years)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _storage_upload(
    supabase_url: str,
    service_key: str,
    bucket: str,
    object_path: str,
    content: bytes,
    content_type: str,
) -> None:
    object_path_quoted = quote(object_path, safe="/")
    url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{object_path_quoted}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "x-upsert": "true",
        "content-type": content_type,
    }
    resp = requests.post(url, headers=headers, data=content, timeout=120)
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Upload failed [{resp.status_code}] for {bucket}/{object_path}: {resp.text}"
        )


def _list_month_partitions(
    db: Session,
    policy: TablePolicy,
    cutoff_date: date,
) -> list[tuple[date, int]]:
    table = _safe_ident(policy.table)
    col = _safe_ident(policy.date_column)
    q = text(f"""
        SELECT DATE_TRUNC('month', {col})::date AS month_start, COUNT(*) AS cnt
        FROM {table}
        WHERE {col} < :cutoff
        GROUP BY 1
        ORDER BY 1
    """)
    rows = db.execute(q, {"cutoff": cutoff_date}).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def _export_month_to_parquet(
    db: Session,
    policy: TablePolicy,
    month_start: date,
    cutoff_date: date,
    out_path: Path,
) -> int:
    table = _safe_ident(policy.table)
    col = _safe_ident(policy.date_column)
    month_end = (pd.Timestamp(month_start) + pd.offsets.MonthBegin(1)).date()

    q = text(f"""
        SELECT *
        FROM {table}
        WHERE {col} >= :month_start
          AND {col} < :month_end
          AND {col} < :cutoff
        ORDER BY {col}
    """)
    result = db.execute(
        q,
        {"month_start": month_start, "month_end": month_end, "cutoff": cutoff_date},
    )
    rows = result.fetchall()
    if not rows:
        return 0

    df = pd.DataFrame(rows, columns=list(result.keys()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def _delete_month_rows(
    db: Session,
    policy: TablePolicy,
    month_start: date,
    cutoff_date: date,
) -> int:
    table = _safe_ident(policy.table)
    col = _safe_ident(policy.date_column)
    month_end = (pd.Timestamp(month_start) + pd.offsets.MonthBegin(1)).date()

    q = text(f"""
        DELETE FROM {table}
        WHERE {col} >= :month_start
          AND {col} < :month_end
          AND {col} < :cutoff
    """)
    result = db.execute(
        q,
        {"month_start": month_start, "month_end": month_end, "cutoff": cutoff_date},
    )
    return int(result.rowcount or 0)


def _vacuum_table(table: str) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"VACUUM (ANALYZE) {_safe_ident(table)}"))


def run_archive(
    execute: bool,
    selected_tables: list[str] | None,
    max_months: int,
    tmp_dir: Path,
    bucket: str,
) -> dict:
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if execute:
        if not supabase_url:
            raise RuntimeError("SUPABASE_URL is required when --execute is set.")
        if not service_key:
            raise RuntimeError(
                "SUPABASE_SERVICE_ROLE_KEY is required when --execute is set."
            )

    table_names = selected_tables or list(TABLE_POLICIES.keys())
    policies = [TABLE_POLICIES[t] for t in table_names]

    summary: dict[str, dict] = {}
    db = SessionLocal()
    try:
        for policy in policies:
            cutoff = _cutoff_from_years(policy.retention_years)
            parts = _list_month_partitions(db, policy, cutoff)
            if max_months > 0:
                parts = parts[:max_months]

            logger.info(
                f"[{policy.table}] cutoff={cutoff} retention={policy.retention_years}y "
                f"months_to_process={len(parts)} execute={execute}"
            )

            table_result = {
                "cutoff_date": cutoff.isoformat(),
                "months_seen": len(parts),
                "months_archived": 0,
                "rows_archived": 0,
                "rows_deleted": 0,
            }

            for month_start, expected_rows in parts:
                y = month_start.year
                m = month_start.month
                run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                file_name = f"{policy.table}_{y:04d}{m:02d}_{run_id}.parquet"
                local_file = tmp_dir / policy.table / f"year={y:04d}" / f"month={m:02d}" / file_name
                object_path = f"{policy.table}/year={y:04d}/month={m:02d}/{file_name}"
                manifest_path = (
                    f"manifests/{policy.table}/year={y:04d}/month={m:02d}/"
                    f"{file_name.replace('.parquet', '.json')}"
                )

                exported_rows = _export_month_to_parquet(
                    db, policy, month_start, cutoff, local_file
                )
                if exported_rows == 0:
                    continue
                if exported_rows != expected_rows:
                    raise RuntimeError(
                        f"[{policy.table}] row mismatch for {month_start}: "
                        f"expected={expected_rows} exported={exported_rows}"
                    )

                checksum = _sha256_file(local_file)
                file_size = local_file.stat().st_size
                manifest = {
                    "table": policy.table,
                    "date_column": policy.date_column,
                    "month_start": month_start.isoformat(),
                    "cutoff_date": cutoff.isoformat(),
                    "rows": exported_rows,
                    "file_size_bytes": file_size,
                    "sha256": checksum,
                    "object_path": object_path,
                    "exported_at_utc": datetime.now(UTC).isoformat(),
                }

                if execute:
                    payload = local_file.read_bytes()
                    _storage_upload(
                        supabase_url=supabase_url,
                        service_key=service_key,
                        bucket=bucket,
                        object_path=object_path,
                        content=payload,
                        content_type="application/octet-stream",
                    )
                    _storage_upload(
                        supabase_url=supabase_url,
                        service_key=service_key,
                        bucket=bucket,
                        object_path=manifest_path,
                        content=json.dumps(manifest, ensure_ascii=True, indent=2).encode(
                            "utf-8"
                        ),
                        content_type="application/json",
                    )
                    deleted = _delete_month_rows(db, policy, month_start, cutoff)
                    db.commit()
                    if deleted != exported_rows:
                        raise RuntimeError(
                            f"[{policy.table}] delete mismatch for {month_start}: "
                            f"exported={exported_rows} deleted={deleted}"
                        )
                    table_result["rows_deleted"] += deleted
                else:
                    logger.info(
                        f"[DRY-RUN] {policy.table} {month_start}: "
                        f"rows={exported_rows} bytes={file_size} "
                        f"archive={bucket}/{object_path}"
                    )

                table_result["months_archived"] += 1
                table_result["rows_archived"] += exported_rows

            if execute and table_result["rows_deleted"] > 0:
                logger.info(f"[{policy.table}] running VACUUM (ANALYZE)")
                _vacuum_table(policy.table)

            summary[policy.table] = table_result

    finally:
        db.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive cold DB rows to Supabase Storage")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform uploads + deletes (default is dry-run preview).",
    )
    parser.add_argument(
        "--table",
        action="append",
        choices=list(TABLE_POLICIES.keys()),
        help="Limit to one or more tables (repeat flag).",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=0,
        help="Max months to process per table (0 = all).",
    )
    parser.add_argument(
        "--tmp-dir",
        default="artifacts/archive_tmp",
        help="Local temp directory for parquet staging.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("SUPABASE_ARCHIVE_BUCKET", "quant-archive"),
        help="Supabase Storage bucket for archives.",
    )
    args = parser.parse_args()

    summary = run_archive(
        execute=args.execute,
        selected_tables=args.table,
        max_months=max(0, args.max_months),
        tmp_dir=Path(args.tmp_dir),
        bucket=args.bucket,
    )
    logger.info("Archive summary:")
    logger.info(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
