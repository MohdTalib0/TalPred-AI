# Sprint Plan v1 - Stock Prediction AI Platform

## 0) Document Control
- Version: `1.0`
- Date: `2026-03-10`
- Inputs:
  - `PRD-Stock-Prediction-AI-v1.1.md`
  - `ENG-SPEC-Stock-Prediction-AI-v1.1.md`
- Planning Horizon: 8 weeks
- Cadence: 2-week sprints (4 sprints total)

## 1) Team Roles (Suggested)
- `Data Eng`: ingestion, curation, contracts
- `ML Eng`: features, modeling, evaluation, calibration
- `Backend Eng`: API, cache, serving, simulation endpoints
- `DevOps/Platform`: infra, CI/CD, monitoring, release
- `Product/QA`: acceptance tests, KPI validation, UAT

## 2) Delivery Strategy
- Deliver thin vertical slices each sprint.
- Enforce quality gates per sprint; no carry-over of critical quality debt.
- Keep v1 scope locked: EOD only, no execution automation.

## 3) Sprint Overview
- **Sprint 1 (Weeks 1-2):** Data foundations + market calendar + curated quality checks
- **Sprint 2 (Weeks 3-4):** Feature pipeline + baseline model + leakage-safe backtesting
- **Sprint 3 (Weeks 5-6):** MLflow/DVC lifecycle + serving API + Redis cache
- **Sprint 4 (Weeks 7-8):** Simulation, monitoring, hardening, launch readiness

## 4) Sprint 1 - Foundations (Weeks 1-2)

### Objective
Stand up reliable ingestion and curated data pipeline with market calendar correctness.

### Tickets
1. `DE-101` Create repository skeleton and environment bootstrap
   - Owner: DevOps
   - Deliverable: project structure, pyproject.toml (ruff + pytest config), Dockerfile, docker-compose.yml, .env.example, alembic setup
   - Includes: Supabase project creation, Upstash Redis provisioning, Dagshub repo setup, Prefect Cloud workspace, Oracle Cloud VM setup (or Render fallback), Grafana Cloud workspace
2. `DE-102` Implement `market_calendar` service
   - Owner: Data Eng
   - Deliverable: holidays, early closes, next trading date resolver
3. `DE-103` Build `market_bars_daily` ingestion (market provider)
   - Owner: Data Eng
   - Deliverable: raw + curated write path with idempotent upsert
4. `DE-104` Build `news_events` ingestion and normalization
   - Owner: Data Eng
   - Deliverable: timestamp normalization, duplicate suppression
5. `DE-105` Build `macro_series` ingestion with release lag fields
   - Owner: Data Eng
   - Deliverable: `release_time_utc`, `available_at_utc` captured
6. `DE-106` Data contract tests + quarantine flow
   - Owner: Data Eng
   - Deliverable: schema validation, outlier/null checks, quarantine table
7. `DE-107` Historical data backfill
   - Owner: Data Eng
   - Deliverable: 3-5 years of market history backfilled, calendar-aligned, and validated for missing sessions
8. `OP-101` CI pipeline bootstrap
   - Owner: DevOps
   - Deliverable: lint/test checks on pull requests

### Acceptance Criteria
- `market_calendar` correctly resolves 100% of test holiday/early-close cases.
- 7-day backfill completes for all connectors with no critical schema failures.
- Historical market backfill (3-5 years) completes with calendar alignment and no unresolved session gaps.
- Failed records routed to quarantine and alert emitted.
- CI must pass for all merged changes.

### Risks/Dependencies
- API rate limits on free providers (NewsAPI: 100 req/day, Finnhub: 60/min).
- Timestamp inconsistency across providers.
- Oracle Cloud free tier signup may be rejected; have Render fallback ready.
- NewsAPI free tier only has 1 month history; use GDELT for historical backfill.
- Supabase free tier has no TimescaleDB; vanilla Postgres is sufficient at v1 volume.

## 5) Sprint 2 - Features and Baseline ML (Weeks 3-4)

### Objective
Produce first leakage-safe feature set and validate baseline viability.

### Tickets
1. `ML-201` Implement feature generators (RSI, momentum, sector return, news sentiment)
   - Owner: ML Eng
2. `ML-202` Add regime label generation (`bull_low_vol`, `bull_high_vol`, `bear_low_vol`, `bear_high_vol`, `sideways`)
   - Owner: ML Eng
3. `ML-203` Build `features_snapshot` materialization (wide SQL schema)
   - Owner: ML Eng + Data Eng
4. `ML-204` Implement leakage-safe as-of join utilities
   - Owner: ML Eng
5. `ML-205` Train baseline XGBoost classifier (S&P100 gate experiment)
   - Owner: ML Eng
6. `ML-206` Walk-forward backtest pipeline
   - Owner: ML Eng
7. `ML-207` Calibration implementation (Platt/isotonic) + artifact persistence
   - Owner: ML Eng
8. `ML-208` Feature validation suite
   - Owner: ML Eng
   - Deliverable: automated checks for no future timestamps, no NaN leakage, and feature distribution sanity

### Acceptance Criteria
- Features generated for full training window with no point-in-time violations.
- Feature validation suite passes for all baseline feature groups.
- Baseline experiment executed with reproducible config and outputs.
- Viability gate met: directional accuracy approximately `52-54%+` in walk-forward on S&P100 baseline.
- If <50%, pause infra expansion and run feature redesign review.

### Risks/Dependencies
- Data quality from Sprint 1 must be stable.
- Feature drift thresholds may need initial tuning.

## 6) Sprint 3 - Serving + ML Lifecycle (Weeks 5-6)

### Objective
Operationalize model lifecycle and prediction serving path with cache-first API.

### Tickets
1. `ML-301` Integrate MLflow tracking standard (params, metrics, artifacts, tags)
   - Owner: ML Eng
2. `ML-302` Add DVC dataset versioning and lineage links
   - Owner: ML Eng + DevOps
3. `ML-303` Implement model promotion pipeline and gates
   - Owner: ML Eng
4. `BE-301` Implement batch prediction job for full universe
   - Owner: Backend Eng
5. `BE-302` Implement Redis prediction cache schema + TTL policy
   - Owner: Backend Eng
6. `BE-303` Implement `POST /predict` and `GET /predictions/{symbol}`
   - Owner: Backend Eng
7. `OP-301` Staging deployment (Oracle Cloud VM or Render fallback)
   - Owner: DevOps
8. `OP-302` API load testing
   - Owner: DevOps
   - Deliverable: load-test report proving cache-hit path target at burst load
9. `ML-304` Model reproducibility test
   - Owner: ML Eng
   - Deliverable: CI job retrains with same dataset version and verifies stable metrics within tolerance

### Acceptance Criteria
- Every model candidate has MLflow run + DVC dataset linkage.
- Promotion to staging blocked if leakage/KPI gates fail.
- API cache-hit p95 <= `250ms`.
- API sustains `100 requests/sec` burst with cache-hit p95 <= `250ms`.
- Batch predictions available for entire v1 symbol universe.
- Model reproducibility CI job passes for baseline candidate.

### Risks/Dependencies
- Redis sizing/TTL tuning required for stable cache hit rate.
- Registry and API versioning must stay compatible.

## 7) Sprint 4 - Simulation + Observability + Launch Readiness (Weeks 7-8)

### Objective
Finalize decision-support value and production reliability.

### Tickets
1. `BE-401` Implement simulation engine (confidence-weighted allocation)
   - Owner: Backend Eng + ML Eng
2. `BE-402` Implement `POST /simulation/run` and `GET /simulation/{run_id}`
   - Owner: Backend Eng
3. `BE-403` Add explainability outputs (`top_factors`) from SHAP artifacts
   - Owner: ML Eng + Backend Eng
4. `OP-401` Monitoring dashboards (pipeline, API, drift, model performance)
   - Owner: DevOps
5. `OP-402` Alert rules implementation
   - Owner: DevOps
6. `QA-401` End-to-end UAT and launch gate report
   - Owner: Product/QA

### Acceptance Criteria
- Simulation enforces:
  - `min_confidence_trade = 0.60`
  - `max_position = 5%`
- Dashboard/API expose Sharpe, drawdown, win rate, and returns.
- Alert thresholds active:
  - missing bars >1% (30d)
  - ingestion failures >3 (7d)
  - data latency >6h
  - PSI >0.25
- Launch checklist complete with rollback validated.

### Risks/Dependencies
- Explainability latency/size trade-offs for API payloads.
- Last-mile data drift tuning before go-live.

## 8) Cross-Sprint Non-Negotiables
- No random train/validation split for time-series tasks.
- Any suspiciously high performance triggers leakage audit.
- No model promotion without full lineage (MLflow + DVC).
- Calendar correctness is mandatory for all target date computations.
- Dataset retention policy must be enforced for DVC snapshots.

## 8.1 Dataset Snapshot Retention Policy (DVC)
- Retain all dataset snapshots tied to production model versions.
- Retain last 10 experiment dataset snapshots for active research.
- Archive or prune older experiment snapshots outside retention window.
- Retention policy changes require ML Eng + DevOps approval.

## 9) Definition of Done (Per Ticket)
- Code merged with passing CI.
- Unit/integration tests added and passing.
- Docs updated if schema/contract changed.
- Observable in staging (logs/metrics).
- Owner demo completed.

## 10) KPI Checkpoints
- End Sprint 2: baseline viability gate completed.
- End Sprint 3: serving SLO gate (API latency + batch coverage).
- End Sprint 4: launch gate (KPI + reliability + reproducibility).

## 11) Release Gate (Go/No-Go)
Go live only if all are true:
1. Pipeline success >= `99.5%` over last 30 staging days.
2. Data freshness SLA >= `99%`.
3. KPI thresholds from PRD met on locked holdout windows.
4. Rollback to previous model validated in staging.
5. Compliance disclaimer present in API/docs/UI.

## 12) Backlog and Expansion Path

### v1.2
- Expanded universe to 500 symbols.
- Additional feature packs and ensemble models.
- 4-hour horizon model exploration.

### v1.3
- Intraday signal exploration.
- Richer NLP/event feature set.
- Sector-specialized model candidates.

### v2
- Company relationship graph.
- Graph neural network model experiments.
- Broader scaling and model family upgrades.

## 13) Timeline Sanity Check
- Sprint 1: ingestion + calendar
- Sprint 2: features + baseline ML
- Sprint 3: serving API + lifecycle
- Sprint 4: simulation + monitoring
