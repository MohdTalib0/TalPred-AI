# TalPred AI

Production-style stock prediction and paper-trading monitoring system built with Python, FastAPI, PostgreSQL, Redis, GitHub Actions, and a React dashboard.

> Predictions are informational only and do not constitute financial advice.

## Reality Check

This repository is an **engineering-first research system**:

- strong pipeline automation, monitoring, and data lineage
- daily prediction and paper-trading simulation loop
- practical CI/CD and storage-retention strategy
- not a claim of guaranteed trading alpha

The current focus is operational reliability and 60-90 day paper-trading observation.

## What It Does

- Ingests daily market and macro data
- Generates features for active symbols
- Runs batch predictions and writes to DB + Redis
- Runs monitoring checks (freshness, quality, drift, pipeline health)
- Runs paper-trading monitor and DB simulation persistence
- Archives cold DB data to Supabase Storage monthly
- Exposes dashboard-ready data via Supabase Edge Function

## High-Level Architecture

- **Data Store (Hot):** PostgreSQL (Supabase)
- **Cache:** Redis
- **Cold Archive:** Supabase Storage (Parquet + manifests)
- **Pipelines:** Python scripts + GitHub Actions schedules
- **API:** FastAPI (`src/serving/api.py`)
- **Dashboard:** React (`dashboard-react`) + Supabase Edge Function (`supabase/functions/dashboard-data`)

## Repository Highlights

- `scripts/daily_pipeline.py` - 8-step EOD orchestrator
- `src/pipelines/batch_predict.py` - production model loading + feature-aligned inference
- `scripts/paper_trading_monitor.py` - paper-trading metrics and logs
- `scripts/archive_to_supabase_storage.py` - DB -> Storage archival
- `src/monitoring/checks.py` - monitoring and alert logic
- `dashboard-react/` - React ops dashboard
- `supabase/functions/dashboard-data/` - consolidated dashboard backend endpoint

## Daily Pipeline Steps

1. Calendar sync
2. Market ingestion
3. News ingestion (toggleable; currently disabled in daily workflow)
4. Macro ingestion
5. Feature generation (+ sector return persistence)
6. Batch prediction (+ Redis cache)
7. Monitoring checks
8. Paper-trading monitor + simulation persistence

## Scheduled Workflows

- **Daily EOD pipeline:** `.github/workflows/daily-pipeline.yml`
  - Cron: `30 1 * * 1-5` (01:30 UTC, Mon-Fri)
- **Monthly archive:** `.github/workflows/monthly-archive.yml`
  - Cron: `15 3 2 * *` (03:15 UTC, day 2 monthly)

## Local Setup

### 1) Environment

Copy and fill:

```bash
cp .env.example .env
```

Minimum required:

- `DATABASE_URL`
- `REDIS_URL`

### 2) Install dependencies

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Run API

```bash
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

### 4) Run daily pipeline manually

```bash
python -m scripts.daily_pipeline --step 1
# or run from prediction onward:
python -m scripts.daily_pipeline --step 6
```

## Dashboard Setup

The React dashboard reads from a Supabase Edge Function endpoint.

1. Deploy edge function:

```bash
supabase functions deploy dashboard-data --no-verify-jwt
```

2. Set dashboard env:

`dashboard-react/.env.local`

```env
VITE_DASHBOARD_EDGE_URL=https://<project-ref>.supabase.co/functions/v1/dashboard-data
```

3. Run dashboard:

```bash
cd dashboard-react
npm install
npm run dev
```

See `docs/DASHBOARD-NETLIFY.md` for Netlify deployment.

## Archival and Retention

Cold-data archival is implemented and scheduled.

- Script: `scripts/archive_to_supabase_storage.py`
- Runbook: `docs/ARCHIVE-RUNBOOK.md`
- Schedule: monthly workflow (manual dry-run + execute supported)

This keeps hot DB storage under control while preserving historical data in object storage.

## Key Runtime Controls (Env)

- `BATCH_PREDICT_EXPLANATIONS` - enable/disable SHAP explanations in daily step 6
- `PREDICT_MODEL_SOURCE`
  - `local_first` (fast CI path)
  - `mlflow_first` (canonical remote-first path)

## Known Limitations

- News ingestion is disabled in daily workflow (separate CI path)
- Some monitoring drift alerts can be noisy during regime shifts/weekend transitions
- Prediction outcome backfill into `predictions.realized_*` is not yet a dedicated pipeline step

## Documentation

- `PRD-Stock-Prediction-AI-v1.1.md`
- `ENG-SPEC-Stock-Prediction-AI-v1.1.md`
- `SPRINT-PLAN-v1.md`
- `docs/DASHBOARD-NETLIFY.md`
- `docs/ARCHIVE-RUNBOOK.md`

