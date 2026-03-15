# Archive + Retention Runbook

This project uses a permanent two-tier storage model:

- **Hot operational data (Postgres):** used by daily pipeline and API.
- **Cold archive (Supabase Storage):** monthly Parquet snapshots of older rows.

The archive job is implemented in:

- `scripts/archive_to_supabase_storage.py`
- `.github/workflows/monthly-archive.yml`

## Retention Policy (Hot DB)

- `market_bars_daily`: keep last 3 years
- `features_snapshot`: keep last 2 years
- `predictions`: keep last 2 years
- `paper_trades`: keep last 2 years
- `simulation_runs`: keep last 2 years
- `sector_returns_daily`: keep last 2 years

Rows older than the above windows are archived monthly to storage, then deleted from DB after verification.

## Required Secrets

Set these in GitHub repository secrets:

- `DATABASE_URL`
- `REDIS_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_ARCHIVE_BUCKET` (example: `quant-archive`)

## Safe Execution Pattern

1. **Dry-run first** (manual workflow dispatch):
   - `execute=false`
   - `max_months=2`
2. Review logs/summary.
3. Re-run with `execute=true`.
4. Once validated, let schedule run automatically.

## Archive Layout

Objects are written as:

- `{table}/year=YYYY/month=MM/{table}_YYYYMM_<run>.parquet`
- `manifests/{table}/year=YYYY/month=MM/{table}_YYYYMM_<run>.json`

Each manifest includes:

- table/date partition
- row count
- file size
- SHA256 checksum
- cutoff date
- UTC export timestamp

## Safety Guarantees

For each month partition:

1. Export rows to Parquet.
2. Validate exported row count equals SQL expected count.
3. Upload Parquet + manifest to storage.
4. Delete same partition rows in DB.
5. Validate deleted row count equals exported row count.
6. Run `VACUUM (ANALYZE)` for updated tables.

If any check fails, the job raises an error and stops.

## Manual Local Commands

Preview only:

```bash
python -m scripts.archive_to_supabase_storage
```

Execute all eligible months:

```bash
python -m scripts.archive_to_supabase_storage --execute
```

Execute small batch:

```bash
python -m scripts.archive_to_supabase_storage --execute --max-months 2
```
