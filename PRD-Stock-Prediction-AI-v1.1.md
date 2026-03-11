# PRD v1.1 - Stock Prediction AI Platform (Board-Ready)

## Document Control
- Product Name: `Market Intelligence Predictor` (working title)
- Version: `1.1`
- Date: `2026-03-10`
- Owner: Product + ML Engineering
- Status: Draft for execution approval

## 1) Executive Summary
Build a production-grade AI research platform that predicts next-day stock direction for a curated US equity universe, explains prediction drivers, and validates practical utility through paper trading.

The v1 product is intentionally scoped to EOD forecasting to maximize reliability, reproducibility, and signal quality while minimizing avoidable intraday complexity.

Success is defined by three pillars:
- Model quality: statistically robust out-of-sample performance
- Operational quality: reliable data/ML pipelines with strong observability
- Decision value: realistic paper-trading performance after costs/slippage

## 2) Problem Statement
Investors and analysts cannot consistently synthesize market prices, macro shifts, and high-volume news in a timely and systematic way. Existing workflows are fragmented and prone to bias.

The platform solves this by producing standardized, explainable, and reproducible predictive signals per stock each trading day.

## 3) Product Vision and Goals

### Vision
Deliver institutional-style, multi-source stock prediction capabilities in a transparent research product that can evolve into a broader quant intelligence platform.

### v1 Goals
- Predict next trading day direction (`up/down`) for the v1 universe.
- Provide confidence and top influencing factors for each prediction.
- Enable paper-trading simulation with realistic assumptions.
- Operate with production SLAs and full prediction traceability.

### Non-Goals (v1)
- Intraday prediction (4h, 15m, 5m)
- Automated trade execution
- Portfolio advisory recommendations
- Non-US or multi-asset coverage

## 4) Locked Strategic Decisions (v1)

### Horizon
- EOD only
- Prediction generated after market close for `t+1`

### Universe
- Top 200 US equities by liquidity
- Filters:
  - Market cap `> $10B`
  - Avg daily volume `> 1M shares`
  - Exchange: `NYSE` / `NASDAQ`
  - Broad sector representation across 11 sectors

### Data Strategy
- All-free-tier stack in v1:
  - Market: yfinance (EOD) + Finnhub (redundancy)
  - News: NewsAPI (daily) + GDELT (historical backfill)
  - Macro: FRED
  - Volatility proxy: CBOE VIX
  - NLP/Sentiment: FinBERT (local inference)
- Paid provider migration when reliability or SLA thresholds are breached.

### Infrastructure Strategy
- All infrastructure runs on free-tier managed services in v1:
  - Database: Supabase free tier (500 MB Postgres)
  - Cache: Upstash Redis free tier (256 MB, 10k commands/day)
  - Compute/API: Oracle Cloud free tier (4 ARM cores, 24 GB RAM always-free VM)
  - Fallback compute: Render free tier (with keep-warm ping) if Oracle unavailable
  - Scheduler: Prefect Cloud free tier (3 users, unlimited flows)
  - Experiment tracking: Dagshub free tier (hosted MLflow + 10 GB DVC storage)
  - Monitoring: Grafana Cloud free tier (10k metrics, 50 GB logs)
  - Dashboard hosting: Vercel free tier
- Total monthly cost: $0

### Compliance Posture
- Research/analytics only
- No execution, no investment advice
- Mandatory disclaimer in UI/API

### Paper Trading
- Included in v1 as required capability
- Daily rebalance simulation with costs/slippage and portfolio metrics

## 5) Users and Stakeholders

### Primary Users
- Advanced retail quants
- Independent analysts
- Internal research users

### Internal Stakeholders
- Product
- ML Engineering
- Data Engineering
- Platform/DevOps

## 6) User Stories
- As a user, I can request prediction for a stock and receive direction, confidence, and top factors.
- As a user, I can inspect why a prediction was made (feature/event contributions).
- As a user, I can run paper simulation and evaluate strategy-level outcomes.
- As an operator, I can monitor pipeline health, drift, and model performance degradation.
- As an ML engineer, I can reproduce any prediction from model/data lineage.

## 7) Functional Requirements

### FR-1 Data Ingestion
- Ingest daily OHLCV, corporate actions, benchmark/sector references.
- Ingest timestamped news/articles and compute sentiment/event signals.
- Ingest macro indicators (rates, inflation proxy, oil, volatility).
- Persist raw and curated data with `event_time` and `as_of_time`.
- For macro data, persist `release_time` and `available_at` to prevent release-lag leakage.
- Use `market_calendar` logic to resolve valid sessions, next trading day targets, holiday gaps, and early closes.

### FR-2 Data Quality and Contracts
- Schema validation for each source.
- Freshness, null-rate, and outlier checks.
- Source reliability scoring.
- Automatic quarantine for failed quality checks.
- Point-in-time availability checks ensure features only use records available before prediction timestamp.

### FR-3 Feature Engineering / Feature Store
- Point-in-time-safe joins only (no look-ahead).
- Rolling indicators and lagged features.
- Peer/sector-relative features.
- Market regime label features (`bull_low_vol`, `bull_high_vol`, `bear_low_vol`, `bear_high_vol`, `sideways`) derived from S&P 500 200-day momentum and VIX level.
- Versioned feature views and reproducible snapshots.

### FR-4 Modeling
- Baseline classifier (XGBoost) for direction.
- Optional return-range regressor.
- Walk-forward training/validation.
- Confidence calibration. Definitions:
  - `probability_up`: calibrated probability of upward movement.
  - `confidence`: `max(probability_up, 1 - probability_up)` representing certainty in predicted direction.
  - `direction`: `up` if `probability_up >= 0.50`, else `down`.
- Experiment tracking standard: MLflow for hyperparameters, dataset version, metrics, and model artifacts.
- Promotion control: only models registered and approved in MLflow Model Registry are eligible for staging/production.
- Persist calibration artifacts per promoted model version (Platt or isotonic) for reproducible confidence outputs.

### FR-5 Inference
- Daily batch inference for full universe.
- On-demand API inference for specific symbol/date.
- Return prediction payload including explainability and lineage IDs.

### FR-6 Explainability
- SHAP-based per-prediction feature contributions.
- Human-readable top drivers summary.

### FR-7 Paper Trading
- Configurable starting capital.
- Daily rebalance rules using predictions + confidence threshold.
- Transaction cost + slippage modeling.
- Equity curve and performance metrics dashboard.

### FR-8 Monitoring and Alerting
- Data quality monitors
- Service health, latency, failure rate
- Feature drift monitoring using PSI and KS tests
  - Alert threshold: PSI > 0.25 for high-severity drift
- Model drift and performance decay alerts
- Incident notification and runbook links

### FR-9 Data and Feature Versioning
- Use DVC for versioning datasets, feature snapshots, and training sets.
- Link DVC dataset versions to MLflow experiments and model registry entries for reproducibility.

## 8) Non-Functional Requirements
- Reliability: idempotent jobs, retries, dead-letter handling
- Availability: API uptime target >= 99.5% (v1)
- Latency: p95 API response <= 250ms
- Scalability: support 200 symbols with path to 500+
- Security: secrets manager, encrypted transport/storage, RBAC
- Auditability: full lineage from source record to prediction output
- Reproducibility: deterministic training runs with versioned artifacts
- Timezone standard: all timestamps stored and processed in UTC; display layer converts to market-local time

## 9) Success Metrics and KPIs

### ML KPIs
- Directional accuracy >= 58% (holdout, baseline-adjusted)
- AUC >= 0.62
- Calibration error <= 0.08

### Simulation KPIs
- Sharpe >= 1.2 (paper strategy)
- Max drawdown <= 15%
- Positive net returns after costs/slippage

### Product/Platform KPIs
- Pipeline success rate >= 99.5%
- Data freshness SLA met >= 99%
- Prediction API p95 <= 250ms

### Quality Gate Rule
Launch requires meeting minimum thresholds across all three KPI groups, not just model metrics.

### KPI Progression Note
- Viability gate (early baseline experiment): approximately 52-54% directional accuracy confirms the approach is sound.
- Launch KPI (production threshold): >= 58% directional accuracy after full feature set, calibration, and confidence filtering.
- The gap between viability and launch is expected to be closed by expanding from the baseline feature set to the full feature pipeline. If the gap does not close, revisit feature design before launch.

## 10) System Architecture (High-Level)
- Connectors Layer: market/news/macro fetchers (yfinance, NewsAPI, GDELT, FRED, Finnhub)
- Market Calendar Layer: trading-day calendar service (holidays, early closes, next-session resolution)
- NLP Layer: FinBERT sentiment inference (batch, CPU, on compute VM)
- Storage Layer: Supabase Postgres (raw, curated, features, predictions)
- Cache Layer: Upstash Redis (prediction cache)
- Feature Layer: point-in-time feature pipelines + store
- ML Layer: Dagshub-hosted MLflow experiment tracking, training, evaluation reports, model registry
- Serving Layer: batch scorer + FastAPI prediction API (Oracle Cloud VM)
- App Layer: dashboard on Vercel (predictions, factors, simulation)
- Scheduling Layer: Prefect Cloud (orchestrator) + Prefect worker (on compute VM)
- Ops Layer: Grafana Cloud (monitoring, alerts), CI/CD, rollback

## 11) Data Model (Core Entities)
- `symbols` (security metadata, sector, liquidity attributes)
- `market_calendar` (exchange, session_date, open_time, close_time, early_close_flag, next_trading_date)
- `market_bars_daily` (OHLCV + adjustments)
- `news_events` (article metadata, sentiment, event tags)
- `news_symbol_mapping` (many-to-many mapping of news events to affected symbols)
- `macro_series` (indicator, value, release timing)
- `features_snapshot` (feature vectors + as_of metadata)
- `model_registry` (model version, params, metrics, approval status)
- `calibration_models` (model_version, calibration_type, training_window, calibration_metrics, artifact_uri)
- `predictions` (direction, confidence, factors, version IDs, realized outcomes)
- `simulation_runs` (run configuration, parameters, result metrics)
- `paper_trades` (positions, fills, costs, PnL)

## 12) API Contract (v1)

### `POST /predict`
Input:
- `symbol`
- `as_of_date`

Output:
- `direction` (`up` or `down`)
- `probability_up`
- `confidence`
- `expected_return_range` (optional)
- `top_factors` (array)
- `model_version`
- `feature_snapshot_id`
- `prediction_id`

### `GET /predictions/{symbol}`
- Historical predictions + realized outcomes

### `POST /simulation/run`
- Config for start capital, cost assumptions, threshold, date range

### `GET /simulation/{run_id}`
- Equity curve + Sharpe + drawdown + win rate + turnover

### Operational Endpoints
- `GET /health` liveness/readiness check
- `GET /symbols` current active universe
- `GET /model/info` current production model version and metadata

## 13) Evaluation and Backtesting Protocol
- Strict walk-forward validation only
- No random shuffles across time
- Fixed transaction cost/slippage assumptions documented per run
- Benchmark against:
  - naive baseline
  - market/sector baseline
- Slice analysis by volatility and regime
- Leakage tests mandatory in CI before model promotion

### 13.1 Model Promotion Workflow (Mandatory)
- Lifecycle: training -> experiment -> evaluation -> registry -> staging -> production
- Promotion gate requires all of:
  - leakage tests pass
  - KPI thresholds pass
  - backtest validation signed off
  - artifact lineage complete (MLflow run + DVC dataset version)

## 14) Rollout Plan

### Phase 1 (Weeks 1-2): Foundations + Data
- Repo standards, CI/CD skeleton, environment setup
- Data contracts and schema definitions
- Market calendar, ingestion pipelines, historical backfill
- Curated data quality checks and quarantine

### Phase 2 (Weeks 3-4): Features + Baseline ML
- Feature pipeline and feature store v1
- Baseline model training and walk-forward backtest
- Viability gate experiment (S&P100 baseline)

### Phase 3 (Weeks 5-6): Serving + ML Lifecycle
- MLflow/DVC integration and model promotion gates
- Batch predictions + Redis cache + prediction API
- Staging deployment and load testing

### Phase 4 (Weeks 7-8): Simulation + Monitoring + Launch
- Simulation engine + explainability + performance dashboard
- Drift/data quality monitoring and alerting
- Shadow mode validation, launch gate review

## 15) Risks and Mitigations
- Leakage risk -> point-in-time joins, automated leakage tests
- Regime shifts -> rolling retraining + regime-aware KPI tracking
- Data outages/rate limits -> retries, caching, provider fallback strategy
- Overfitting -> model simplicity first, strict holdout discipline
- False certainty -> calibration + confidence thresholds + abstain mode

## 16) Compliance, Legal, and Messaging
- Product classified as analytics/research
- Required disclaimer in UI, API docs, and exportable reports:
  - "Predictions are informational only and do not constitute financial advice."
- No automated order execution in v1
- No portfolio suitability recommendations

## 17) Acceptance Criteria (Launch Gate)
- 30 consecutive days of successful staging pipeline runs
- KPI thresholds met on locked holdout windows
- API latency and reliability targets met under load test
- Drift and alerting tested with synthetic incidents
- Reproducibility test passes:
  - any `prediction_id` can be recreated from versioned data/model artifacts

## 18) Post-v1 Roadmap
- v2: 4-hour horizon model, expanded universe to 500
- v3: intraday horizon (15m/5m), expanded universe to 1000
- Add advanced multimodal models (TFT/transformers)
- Add company knowledge graph and relationship graph features (supplier, competitor, sector, macro exposure)
- Evaluate graph neural network models over company relationship graphs
- Add richer scenario simulation and portfolio constraints

## 19) Final Decisions Locked for Execution

### 19.1 Simulation Rebalancing Policy
- Use confidence-weighted allocation with caps.
- Allocation formula: `weight_i = confidence_i / sum(confidence_j)` across eligible positions.
- Guardrails:
  - `max_position = 5%`
  - `min_confidence_trade = 0.60`
- Rationale: captures model conviction, improves alpha concentration, and reduces low-quality noise trades compared to equal weight.

### 19.2 Confidence Threshold and Abstain Policy
- Default trade threshold: `0.60`
- Rule: abstain if `confidence < 0.60`.
- Optional secondary control for v1.1: trade only top-N predictions per day (default candidate: top 20).

### 19.3 Model Retraining Cadence
- Retrain weekly with a rolling 3-year training window.
- Rationale: balances adaptability to regime change with parameter stability and manageable compute cost.

### 19.4 Data Provider Upgrade Triggers
- Upgrade provider tier when any of the following occurs:
  - `>1%` missing market bars over last 30 days
  - `>3` ingestion failures in 7 days
  - API rate limits block ingestion jobs
  - data latency exceeds 6 hours from SLA target
- Target paid provider path:
  - Market: Polygon
  - News: RavenPack or Benzinga
  - Macro: FRED plus paid mirror/redundant source

### 19.5 KPI Threshold Scope
- Use global KPI thresholds for v1.
- Defer sector-specific thresholds and sector-specialized models to post-v1 once data depth and tuning capacity are sufficient.

## 20) Implementation Sequencing (Locked Build Order)
- Step 1: market data ingestion
- Step 2: feature engineering
- Step 3: baseline model training
- Step 4: backtesting and leakage validation
- Step 5: prediction API
- Step 6: explainability integration
- Step 7: dashboard
- Step 8: monitoring and alerting hardening

Never begin with dashboard-first development before data, modeling, and evaluation pipelines are stable.

## 21) Delivery Timeline and Performance Expectations

### 21.1 Realistic v1 Timeline (Execution Baseline)
- Data pipelines: 2 weeks
- Feature engineering: 1 to 2 weeks
- Baseline ML: 1 week
- Backtesting: 1 week
- API plus infrastructure: 1 week
- Dashboard: 1 week
- Total expected delivery: 6 to 8 weeks

### 21.2 Performance Expectation Guidance
- Realistic directional accuracy for strong models is often in the 55 to 60 percent range.
- Strategy quality should be evaluated with confidence filtering and risk controls, with target `Sharpe > 1` as a practical success threshold for v1.

### 21.3 First Milestone Viability Check
- Start with one baseline experiment before platform-scale expansion:
  - single model (XGBoost)
  - compact feature set (momentum, RSI, sector return, news sentiment)
  - next-day direction target
- Viability signal: if baseline reaches roughly 52 to 54 percent directional accuracy on strict walk-forward testing, continue expansion.

