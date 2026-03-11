# Engineering Spec v1.1 - Stock Prediction AI Platform

## 0) Document Control
- Product: `Market Intelligence Predictor`
- Spec Version: `1.1`
- Date: `2026-03-10`
- References: `PRD-Stock-Prediction-AI-v1.1.md`
- Status: Implementation-ready draft

## 1) Purpose
This document translates the PRD into buildable technical requirements for:
- data ingestion and storage
- feature engineering and model training
- model promotion and serving
- paper-trading simulation
- observability, security, and CI/CD quality gates

## 2) Scope

### In scope (v1)
- EOD prediction for next trading day (`t+1`) direction.
- Universe: top 200 US liquid equities.
- Data sources: market, news, macro, volatility.
- Batch predictions + on-demand API.
- Explainability output with calibrated confidence.
- Paper-trading simulation module.

### Out of scope (v1)
- Intraday horizons (4h/15m/5m).
- Auto trade execution.
- Sector-specialized models.

### Baseline Data Sizing Assumption (v1)
- Universe: ~200 stocks
- History: ~1,000 trading days
- Estimated rows in daily bars: ~200,000
- Estimated feature width: ~20 to 40 features per snapshot
- Capacity implication: this footprint is well within Supabase free-tier Postgres (500 MB) for v1.

## 3) System Architecture (Implementation View)

1. **Connectors Layer**
   - Market connector
   - News connector
   - Macro connector
   - Volatility connector

2. **Market Calendar Layer**
   - Trading session resolver using exchange calendars.
   - Handles holidays, early closes, next valid trading session.

3. **Storage Layers**
   - Raw (immutable source snapshots)
   - Curated (normalized, validated tables)
   - Feature snapshots (point-in-time materialized features)

4. **ML Layer**
   - Training pipelines
   - MLflow experiment tracking
   - Model evaluation reports
   - Model registry + promotion gates
   - Calibration artifact storage

5. **Serving Layer**
   - Batch prediction engine
   - Redis prediction cache
   - FastAPI inference service (cache first, fallback inference)

6. **App Layer**
   - Dashboard and paper-trading simulation views

7. **Ops Layer**
   - CI/CD
   - Monitoring and alerting
   - Incident runbooks

## 3.1) Timezone Convention (Global Standard)
- All timestamps in storage, pipelines, and APIs must be in UTC.
- Market session times (open/close) are stored as UTC in `market_calendar`.
- Display layer (dashboard) may convert to market-local time (US Eastern) for user convenience.
- No pipeline or feature logic may depend on local timezone; UTC is the single source of truth.

## 4) Technology Decisions

### Core
- Language: Python 3.11+
- API: FastAPI
- Data processing: pandas, numpy, pyarrow

### ML
- Modeling: XGBoost + scikit-learn calibration (Platt or isotonic)
- NLP/Sentiment: FinBERT (huggingface transformers, local CPU inference)
- Explainability: SHAP

### Infrastructure (Free-Tier Managed Services)
- Database: Supabase free tier (Postgres, 500 MB, direct connection string for pipelines)
- Cache: Upstash Redis free tier (256 MB, 10k commands/day, standard Redis protocol)
- Compute/API hosting: Oracle Cloud free tier VM (4 ARM cores, 24 GB RAM, always-free)
- Fallback compute: Render free tier with keep-warm health ping if Oracle unavailable
- Scheduling: Prefect Cloud free tier (hosted orchestrator, worker runs on Oracle VM)
- Experiment tracking: Dagshub free tier (hosted MLflow server)
- Data versioning: DVC with Dagshub storage (10 GB free)
- Dashboard hosting: Vercel free tier (Next.js/React)
- Monitoring: Grafana Cloud free tier (hosted Prometheus + Grafana, 10k metrics, 50 GB logs)

### Dev Tooling
- Testing: pytest + pytest-cov
- Linting/Formatting: ruff
- Database migrations: alembic
- Containerization: Docker (for local dev and Oracle VM deployment)

## 5) Repository Layout
```text
project-root/
  docs/
    PRD-Stock-Prediction-AI-v1.1.md
    ENG-SPEC-Stock-Prediction-AI-v1.1.md
    SPRINT-PLAN-v1.md
  src/
    connectors/
    calendar/
    pipelines/
    features/
    models/
    serving/
    simulation/
    monitoring/
  configs/
    symbols.yml
    thresholds.yml
    providers.yml
  alembic/              # database migrations
    versions/
    env.py
    alembic.ini
  data/
    raw/
    curated/
    features/
  dvc.yaml
  dvc.lock
  tests/
  Dockerfile
  docker-compose.yml    # local dev stack
  pyproject.toml        # dependencies, ruff config, pytest config
  requirements.txt
  .env.example
```

## 6) Data Model (Core Tables)

### 6.1 `symbols`
- `symbol` (PK)
- `company_name`
- `exchange`
- `sector`
- `market_cap`
- `avg_daily_volume_30d`
- `is_active`
- `effective_from`
- `effective_to`

### 6.2 `market_calendar`
- `exchange` (PK part)
- `session_date` (PK part)
- `open_time_utc`
- `close_time_utc`
- `early_close_flag` (bool)
- `is_holiday` (bool)
- `next_trading_date`

### 6.3 `market_bars_daily`
- `symbol` (PK part)
- `date` (PK part)
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`
- `source`
- `event_time`
- `as_of_time`

### 6.4 `news_events`
- `event_id` (PK)
- `headline`
- `source_name`
- `published_time`
- `ingested_time`
- `sentiment_score`
- `event_tags` (jsonb)
- `credibility_score`

### 6.4.1 `news_symbol_mapping`
- `event_id` (PK part, FK to `news_events`)
- `symbol` (PK part, FK to `symbols`)
- `relevance_score` (optional; strength of association)

This many-to-many table supports market-wide news (mapped to multiple symbols) and single-stock news equally. A news event with no symbol mappings is treated as market-wide.

### 6.5 `macro_series`
- `series_id` (PK part)
- `observation_date` (PK part)
- `value`
- `release_time_utc`
- `available_at_utc`
- `source`

### 6.6 `features_snapshot`
- `snapshot_id` (PK)
- `symbol`
- `as_of_time`
- `target_session_date`
- Wide SQL feature columns (RSI, momentum, sector_return, news_sentiment, etc.)
- `dataset_version` (DVC hash/tag)
- `regime_label`

### 6.7 `model_registry` (application view of MLflow)
- `model_version` (PK)
- `mlflow_run_id`
- `algorithm`
- `training_window_start`
- `training_window_end`
- `metrics` (jsonb)
- `status` (`candidate|staging|production|archived`)
- `created_at`

Model version naming standard:
- `model_version` format: `v{major}.{minor}.{patch}`
- Example usage:
  - `v1.0.0` baseline production model
  - `v1.1.0` feature or pipeline update
  - `v1.1.1` retraining/calibration refresh

### 6.8 `calibration_models`
- `model_version` (PK part)
- `calibration_type` (`platt|isotonic`)
- `training_window`
- `calibration_metrics` (jsonb)
- `artifact_uri`

### 6.9 `predictions`
- `prediction_id` (PK)
- `symbol`
- `as_of_time`
- `target_date`
- `direction`
- `probability_up`
- `confidence`
- `top_factors` (jsonb)
- `model_version`
- `feature_snapshot_id`
- `dataset_version` (optional; DVC hash/tag for audit lineage)
- `cache_ttl_seconds`
- `realized_direction` (nullable; backfilled after target date)
- `realized_return` (nullable; actual close-to-close return)
- `outcome_recorded_at` (nullable; timestamp when outcome was filled)

### 6.10 `simulation_runs`
- `run_id` (PK)
- `created_at`
- `start_date`
- `end_date`
- `starting_capital`
- `min_confidence_trade`
- `max_position`
- `transaction_cost_bps`
- `slippage_bps`
- `model_version`
- `status` (`running|completed|failed`)
- `result_metrics` (jsonb; Sharpe, drawdown, win rate, net return, turnover)

### 6.11 `paper_trades`
- `run_id` (PK part)
- `date` (PK part)
- `symbol` (PK part)
- `weight`
- `position_qty`
- `entry_price`
- `exit_price`
- `transaction_cost`
- `slippage_cost`
- `daily_pnl`

## 7) Data Contracts and Validation Rules

### Market data
- Required fields: OHLCV + date + symbol.
- Reject if `high < low` or any negative volume.
- Freshness SLA: available by `T+1 06:00 UTC`.

### News data
- Required: headline, timestamp, source.
- Duplicate suppression by normalized title + timestamp window.
- Sentiment score range must be `[-1, 1]`.

### Macro data
- Must include `release_time_utc` and `available_at_utc`.
- Any feature join must enforce `available_at_utc <= prediction_timestamp`.

### Global quality controls
- Null-rate checks per critical field.
- Outlier checks by robust z-score where applicable.
- Quarantine invalid batch and trigger alert.

## 8) Feature Engineering Spec

### 8.1 Core feature groups
- Technical: RSI, MACD, rolling returns, rolling volatility, momentum.
- Market-relative: sector return spread, benchmark-relative return.
- News: sentiment mean/decay over 24h and 7d.
- Macro: lagged macro indicators available before prediction time.
- Regime: classified using two signals:
  - Trend: S&P 500 200-day momentum (`bull` if > 0, `bear` if < 0)
  - Volatility: VIX level (`high_vol` if VIX > 20, `low_vol` if VIX <= 20)
  - Labels: `bull_low_vol`, `bull_high_vol`, `bear_low_vol`, `bear_high_vol`
  - `sideways`: optional label when 200-day momentum is near zero (absolute value < 2%)

### 8.2 Point-in-time safety
- All joins must be as-of joins.
- No record may use timestamp after prediction cut-off.
- Add automated leakage test in CI.

### 8.3 Feature drift monitoring
- PSI for each key feature between training baseline and live window.
- KS tests for distribution shift.
- Alert level:
  - Warning: PSI > 0.15
  - Critical: PSI > 0.25

## 9) Model Training and Evaluation

### 9.1 Training cadence
- Weekly retraining.
- Rolling 3-year training window.
- Compute fallback: if weekly retraining exceeds available resources on the v1 single-VM setup, fall back to biweekly cadence until infrastructure is scaled. Document any cadence changes in the model registry.

### 9.2 Model candidates
- Primary: XGBoost classifier.
- Optional secondary: return-range regressor.

### 9.3 Calibration and Confidence Definition
- Apply Platt or isotonic calibration on validation set.
- Persist calibration artifact to `calibration_models`.
- Definition of output fields:
  - `probability_up`: calibrated probability that the stock moves up on the target day.
  - `confidence`: `max(probability_up, 1 - probability_up)`. Represents model certainty in the predicted direction regardless of whether that direction is up or down.
  - `direction`: `up` if `probability_up >= 0.50`, otherwise `down`.
- The abstain threshold (`confidence < 0.60`) and allocation formula both use `confidence`, ensuring that high-conviction short (down) predictions are treated equally to high-conviction long (up) predictions.

### 9.4 Evaluation protocol
- Walk-forward validation only.
- Benchmarks:
  - naive directional baseline
  - market/sector baseline
- Metrics:
  - Accuracy, AUC, calibration error
  - Sharpe, max drawdown, turnover, net return in simulation

### 9.5 Performance Sanity Ranges (Leakage Guardrail)
- Realistic directional accuracy target band: 55 to 60 percent.
- Realistic Sharpe expectation in paper simulation: 1.0 to 1.5.
- Max drawdown expectation: below 15 percent.
- Any materially higher-than-expected out-of-sample performance should trigger mandatory leakage and data-availability audits before promotion.
- Hard guardrail: any sustained out-of-sample directional accuracy above 65 percent triggers immediate leakage audit before any promotion action.

## 10) MLflow and DVC Standards

### 10.1 MLflow required run metadata
- params: model hyperparameters, feature set version, training window
- metrics: train/val/test metrics, backtest metrics
- tags: git commit hash, dataset version, pipeline version
- artifacts: model binary, shap summary, evaluation report, calibration artifact

### 10.2 Promotion rule
Only models in MLflow registry can move to staging/production.

### 10.3 DVC required datasets
- curated training dataset snapshot
- feature snapshot dataset
- labels dataset

### 10.4 Lineage binding
Each promoted model must reference:
- `mlflow_run_id`
- DVC dataset version hash
- feature pipeline version

Prediction-level lineage should resolve through:
`prediction_id -> feature_snapshot_id -> dataset_version -> mlflow_run_id -> model_version`

## 11) Model Promotion Workflow

Lifecycle:
`training -> experiment -> evaluation -> registry -> staging -> production`

Promotion gates (all mandatory):
1. Leakage tests pass.
2. KPI thresholds pass.
3. Backtest validation approved.
4. Full lineage (MLflow + DVC) present.

Rollback:
- Keep previous production model active as fallback.
- One-click rollback to last healthy version.

## 12) Prediction Serving Spec

### 12.1 Batch prediction job
- Runs after market close once curated data and features are complete.
- Generates predictions for full universe and stores in `predictions`.
- Writes cache entries to Redis by `(symbol, target_date, model_version)`.

### 12.2 API behavior (`POST /predict`)
1. Validate symbol/date/session via `market_calendar`.
2. Check Redis cache.
3. If hit: return cached payload.
4. If miss: run on-demand inference (if feature snapshot available), store result, cache response.

### 12.3 Response payload
- direction
- probability_up
- confidence
- top_factors
- model_version
- feature_snapshot_id
- dataset_version (optional)
- prediction_id

### 12.4 Operational endpoints
- `GET /health` liveness and readiness check for load balancer and monitoring.
- `GET /symbols` returns current active universe with sector and liquidity metadata.
- `GET /model/info` returns current production model version, training window, and promotion timestamp.

### 12.5 Latency target
- p95 <= 250ms in cache-hit path.

## 13) Paper Trading Simulation Spec

### 13.1 Portfolio policy (locked)
- Confidence-weighted allocation:
  - `weight_i = confidence_i / sum(confidence_j)`
- Constraints:
  - `max_position = 5%`
  - `min_confidence_trade = 0.60`
- Optional filter: top-N signals/day (default candidate 20).

### 13.2 Execution assumptions
- Daily rebalance at defined session rule.
- Transaction costs + slippage configurable in run parameters.

### 13.3 Outputs
- Equity curve
- Sharpe
- Max drawdown
- Win rate
- Cumulative and net returns
- Turnover

## 14) Scheduler and Job Plan

### Daily jobs
1. `calendar_sync_job` (before all others)
2. `ingest_market_job`
3. `ingest_news_job`
4. `ingest_macro_job`
5. `curation_validation_job`
6. `feature_build_job`
7. `batch_predict_job`
8. `cache_update_job` (Redis refresh from batch predictions)
9. `simulation_update_job`
10. `monitoring_checks_job`

### Daily execution runtime target
- Expected daily EOD pipeline runtime: approximately 10-20 minutes under normal provider/API conditions.

### Weekly jobs
1. `retrain_job`
2. `evaluate_candidate_job`
3. `promotion_gate_job`
4. `drift_report_job`

## 15) Monitoring, Alerting, and SLOs

### Platform
- Use Grafana Cloud free tier (hosted Prometheus + Grafana).
- Push metrics from Oracle VM via Prometheus remote-write or Grafana Agent.
- No self-hosted Prometheus or Grafana containers needed.

### SLOs
- Pipeline success rate >= 99.5%
- Data freshness >= 99%
- API p95 <= 250ms

### Alert thresholds
- Missing market bars > 1% in last 30 days
- Ingestion failures > 3 in 7 days
- Data latency > 6 hours from SLA
- PSI > 0.25 critical drift

### Observability outputs
- Service dashboards (Grafana Cloud)
- Data quality dashboard (Grafana Cloud)
- Drift dashboard (Grafana Cloud)
- Model performance dashboard (Grafana Cloud)

## 16) Security and Compliance Controls
- Secrets stored in secret manager, not in repo.
- TLS for service communication.
- RBAC for model promotion endpoints.
- Audit logs for model promotions and prediction requests.
- Mandatory disclaimer in API/docs/UI:
  - "Predictions are informational only and do not constitute financial advice."

## 17) API Contracts (Detailed)

### `POST /predict`
Request:
```json
{
  "symbol": "AAPL",
  "as_of_date": "2026-03-10"
}
```

Response:
```json
{
  "prediction_id": "pred_20260310_AAPL_v1.1.1",
  "symbol": "AAPL",
  "target_date": "2026-03-11",
  "direction": "up",
  "probability_up": 0.67,
  "confidence": 0.67,
  "top_factors": [
    {"name": "news_sentiment_24h", "impact": 0.18},
    {"name": "sector_return_1d", "impact": 0.12}
  ],
  "model_version": "v1.1.1",
  "feature_snapshot_id": "fs_20260310_220000_aapl",
  "dataset_version": "dvc:3f9e4c1"
}
```

### `POST /simulation/run`
Request:
```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "starting_capital": 100000,
  "min_confidence_trade": 0.60,
  "max_position": 0.05,
  "transaction_cost_bps": 10,
  "slippage_bps": 5
}
```

## 18) CI/CD Quality Gates

Pull request must pass:
1. Unit tests
2. Data contract tests
3. Leakage test suite
4. Reproducibility checks for training pipeline
5. API schema contract tests

Release to staging must pass:
1. Backtest report generation
2. KPI threshold checks
3. Drift baseline snapshot creation

## 19) Test Plan

### Unit tests
- Feature calculators
- Calendar resolution logic
- Confidence weighting constraints

### Integration tests
- End-to-end pipeline from ingestion to prediction
- Cache hit/miss behavior in API
- MLflow + DVC lineage recording

### Regression tests
- Compare new model candidate against previous production in fixed holdout.

### Failure-mode tests
- Provider outage fallback
- Missing data batch quarantine
- Rollback to previous model version

## 20) Build Milestones

### M1 (Week 1-2)
- Market data ingestion + calendar correctness + curated store.

### M2 (Week 3-4)
- Feature pipeline + baseline model + leakage-safe backtest.

### M3 (Week 5)
- MLflow + DVC lineage + model promotion gates.

### M4 (Week 6)
- Batch predictions + Redis cache + prediction API.

### M5 (Week 7)
- Paper trading + explainability + dashboards.

### M6 (Week 8)
- Monitoring, alerts, hardening, and launch checklist.

## 21) Locked Engineering Decisions (v1)

### 21.1 Scheduler
- Use Prefect Cloud free tier for orchestration (hosted server, no self-hosted Prefect server needed).
- Run Prefect worker on the Oracle Cloud VM.
- Deployment pattern: flow-based ingestion/training/prediction pipelines.
- Rationale: zero-cost hosted orchestrator with lower setup overhead than self-hosted alternatives.
- Revisit Airflow only when scale and multi-team DAG governance requirements exceed Prefect fit.

### 21.2 Feature Store Shape
- Use wide SQL columns for `features_snapshot` in v1.
- Do not use JSONB payload as primary feature representation.
- Rationale:
  - faster analytical queries
  - simpler joins and training extraction
  - easier feature-level drift computation and monitoring

### 21.3 Hosting Strategy
- Free-tier managed services for database, cache, monitoring, and experiment tracking.
- Oracle Cloud free tier VM (4 ARM cores, 24 GB RAM) runs:
  - FastAPI prediction service
  - Prefect worker (connects to Prefect Cloud)
  - FinBERT sentiment inference (CPU batch)
  - Training/backtest jobs
- Managed services (zero self-hosted overhead):
  - Database: Supabase (Postgres via direct connection string)
  - Cache: Upstash Redis (standard protocol)
  - MLflow: Dagshub hosted
  - Monitoring: Grafana Cloud hosted
  - Dashboard: Vercel hosted
- Fallback: if Oracle Cloud signup fails, use Render free tier for API + run pipelines locally.
  - Render caveat: free tier spins down after inactivity; add keep-warm health ping every 5 minutes to maintain latency SLA.

### 21.4 Warehouse Strategy and Scaling Path
- v1 storage: Supabase Postgres (vanilla, no TimescaleDB).
- v1 data volume (~200k rows) is well within vanilla Postgres capability.
- v2 scale path: migrate to self-hosted PostgreSQL with TimescaleDB for larger universe.
- v3+ scale path: evaluate ClickHouse or Snowflake for higher-volume analytics workloads.

## 22) Immediate Execution Plan (M1 Kickoff)
- Create repository skeleton and configuration files.
- Implement `market_calendar` service (holidays, early closes, next-session resolution).
- Implement `market_bars_daily` ingestion pipeline.
- Implement curated data validation checks and quarantine path.
- Implement first feature generator pipeline for baseline features.

## 23) Baseline Experiment Gate (Before Full Platform Expansion)
- Run one constrained baseline experiment before scaling infra complexity:
  - Universe: S&P100
  - Features: RSI, momentum, sector return, news sentiment
  - Model: XGBoost
  - Target: next-day direction
- Decision gate:
  - If walk-forward directional accuracy is approximately 52-54 percent or better, continue full buildout.
  - If materially below 50 percent, revise feature design before expanding architecture scope.

## 24) Advanced Modeling Path (Post-v1)
- Company relationship graph is explicitly deferred to v2+.
- Candidate graph edges:
  - supplier relationships
  - competitor relationships
  - sector coupling
  - macro exposure coupling
- Candidate model family: graph neural networks integrated with time-series features.
- Adoption condition: only pursue after v1 baseline and operations are stable.
