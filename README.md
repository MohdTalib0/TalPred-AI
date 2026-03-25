# TalPred AI

Production-grade cross-sectional equity alpha system with automated daily predictions, paper trading, and monitoring. Built with Python, XGBoost, PostgreSQL, Redis, MLflow, GitHub Actions, and a React dashboard.

> Predictions are informational only and do not constitute financial advice.

## What This Actually Is

This is a **systematic quantitative strategy**, not just a project:

- Cross-sectional alpha model targeting market-relative returns (stock vs SPY)
- Walk-forward validated: **IC = 0.038, Sharpe = 2.12** (stride-5, non-overlapping)
- Validation IC = 0.085 at 5-day horizon (model's trained target)
- Regime-stable: signal **increases** during high-VIX periods (IC = 0.15 during VIX >= 30)
- Capacity tested: viable at **$5M-$10M** AUM
- Feature stability: 12/15 top features stable across all walk-forward folds
- **Multi-model support**: classifier, regressor, LambdaRank, and XGBoost+LightGBM ensemble
- **Fundamental features**: accruals [Sloan 1996], ROE trend, earnings momentum, operating leverage via SEC EDGAR + yfinance (free, no API key)
- **PCA statistical factor model** (30 factors, 252-day rolling) with factor exposure constraints
- **EWMA vol targeting** (half-life=10d) for faster regime adaptation
- **OOD detection**: Mahalanobis distance with Ledoit-Wolf shrinkage covariance
- **Deflated Sharpe Ratio** computed per Bailey & Lopez de Prado (2014) for multiple-testing adjustment
- **Fama-French 5-factor + Momentum attribution** to validate alpha vs factor beta
- Signal decay curve: IC measured at t+1 through t+60 with half-life estimation
- Production pipeline with daily ingestion, prediction, monitoring, and paper trading

## Model Performance (Walk-Forward Backtest)

| Metric | Stride=5 (true) | Notes |
|---|---|---|
| IC Mean (WF OOS) | **0.038** | Daily cross-sectional Spearman IC, fully out-of-sample |
| Validation IC | **0.085** | 5-day horizon IC on internal validation split |
| IC IR | **2.76** | IC / std(IC) — information ratio |
| Sharpe (net, NW) | **2.12** | After 10 bps costs, Newey-West adjusted |
| Deflated Sharpe | *computed* | Bailey & Lopez de Prado (2014), 35 effective trials |
| Max Drawdown | **-7.5%** | Manageable for systematic strategy |
| Decile Monotonicity | **0.94** | Top decile consistently beats bottom |
| Accuracy | **52.2%** | Expected for cross-sectional ranking |

### Capacity Profile

| Capital | Sharpe | Max DD | Notes |
|---|---|---|---|
| $100K | 2.32 | -7.0% | Retail scale |
| $1M | 2.12 | -7.5% | Sweet spot |
| $5M | 1.73 | -9.9% | Still strong |
| $10M | 1.13 | -14.1% | Viable |
| $50M | 0.15 | -21.0% | Alpha exhausted by slippage |

### Regime Resilience

| VIX Regime | IC | L/S Spread | Interpretation |
|---|---|---|---|
| Low Vol (<18) | +0.036 | +79 bps | Steady signal |
| Normal (18-25) | +0.007 | -7 bps | Flat — reversal alpha fills gap |
| Elevated (25-30) | +0.075 | +252 bps | Strong |
| Crisis (>=30) | +0.151 | +393 bps | Strongest — dispersions widen |

## Architecture

- **Data Store (Hot):** PostgreSQL (Supabase)
- **Cache:** Redis
- **Cold Archive:** Supabase Storage (Parquet + manifests)
- **Model Tracking:** MLflow (DagsHub)
- **Pipelines:** Python scripts + GitHub Actions schedules
- **API:** FastAPI (`src/serving/api.py`)
- **Dashboard:** React (`dashboard-react`) + Supabase Edge Function

## Alpha Features

The model uses a **cross-sectional alpha** feature profile — only features that vary across stocks on the same day:

**Core alpha features (stable across all folds):**
- `idio_momentum_20d` — stock momentum minus SPY momentum (CV = 0.07)
- `pct_from_52w_high` — proximity to 52-week high (CV = 0.09)
- `vol_adj_momentum_20d` — momentum scaled by volatility (CV = 0.06)
- `log_market_cap` — size factor (CV = 0.07)

**Cross-sectional transforms:**
- Rank features (momentum, volatility, RSI, volume ranks within market)
- Z-score normalised features (momentum_cs, sector_neutral variants)
- Sector-relative returns

**Excluded from production profile:**
- Market-level features (VIX, SP500 momentum) — these predict market direction, not cross-sectional alpha
- News sentiment features — insufficient coverage for reliable signal

## Strategy Framework

Four portfolio construction modes run daily in parallel:

- **MomentumLongShort** — confidence-weighted with multi-timeframe momentum alignment
- **SectorRotation** — sector-level feature ranking for sector selection, model confidence for stock picking
- **MeanReversion** — RSI/reversal contrarian filter, trades only extreme setups
- **MomentumReversal** — composite alpha blending momentum + short-term reversal with **regime-adaptive weighting**:

  | VIX | Reversal Weight | Momentum Weight |
  |---|---|---|
  | >= 30 (crisis) | 10% | 90% |
  | 25-30 (elevated) | 20% | 80% |
  | 18-25 (normal) | 30% | 70% |
  | < 18 (calm) | 45% | 55% |

  Weights interpolate smoothly between tiers (no step functions).

**Shared infrastructure:**
- **RiskManager** — position/sector/side limits, **EWMA vol targeting** (half-life=10d), VIX regime scaling, ADV filter, drawdown circuit breaker (-5% reduce, -10% halt), **PCA factor exposure constraints**
- **PortfolioConstructor** — **Almgren-Chriss market impact** (temporary + permanent components scaled by volatility and participation), short borrow costs, benchmark (SPY) tracking, **beta-neutrality hedge** (rolling 60-day beta estimation, SPY offset), **turnover-aware optimization** (15% shrinkage toward current weights), **partial fill modeling** (5% ADV participation cap clips unrealistic fills)
- **Multi-day hold simulation** — positions held between rebalances; costs charged only on turnover delta (not daily round-trips)
- **5-day rebalance stride** matching the model's 5-day prediction horizon
- **Newey-West adjusted Sharpe** (lag=4) for honest performance reporting with overlapping returns
- **Holm-Bonferroni** multiple testing correction on promotion gates (IC, monotonicity, Sharpe, AUC, drawdown)

## Daily Pipeline (9 Steps)

1. Calendar sync
2. Market ingestion (yfinance)
3. News ingestion (toggleable; currently disabled)
4. Macro ingestion (VIX, SP500)
5. Feature generation (+ sector return persistence)
6. Batch prediction (+ Redis cache) — direction labels: `outperform` / `underperform`; target date = feature date + 5 trading days
7. Outcome backfill — fills `realized_return` / `realized_direction` using actual 5-day market-relative returns
8. Monitoring checks (data quality, freshness, feature drift, **alpha quality**, **capacity indicators**, **regime stress**, **feature stability**)
9. Paper trading (legacy simulation + 4 strategy framework simulations)

## Repository Map

```
scripts/
  daily_pipeline.py          # 9-step EOD orchestrator
  retrain_market_relative.py # Retrain: 4-way split (train/WF/calibration/OOS holdout)
  research_factor_attribution.py  # FF5+MOM factor attribution (alpha vs factor beta)
  research_alpha_decay.py    # IC decay across prediction horizons
  prod_diagnostics.py        # Capacity + crash + feature stability tests
  paper_trading_monitor.py   # Paper-trading metrics
  archive_to_supabase_storage.py  # DB → Storage archival

src/
  pipelines/
    batch_predict.py         # Model loading + alpha feature computation + inference
    outcome_backfill.py      # 5-day realized return backfill for live IC
    ingest_market.py         # Market data ingestion + data quality checks
  connectors/
    market.py                # yfinance market data (hardened: retry, repair=True, staleness)
    alpha_vantage.py         # Free fallback + cross-source reconciliation
    sec_edgar.py             # SEC EDGAR XBRL (free, true PIT filing dates)
    yfinance_fundamentals.py # yfinance quarterly financials (zero-cost)
    macro.py                 # Macro data connector (VIX, etc.)
    news.py                  # News sentiment connector
    simfin.py                # SimFin API connector (optional, needs key)
  features/
    engine.py                # Technical + cross-sectional feature generation
    leakage.py               # Leakage-safe training dataset builder
    fundamentals.py          # Fundamental feature engineering (SUE, accruals)
  models/
    trainer.py               # Multi-mode training (classifier/regressor/ranker)
    ensemble.py              # XGBoost + LightGBM ensemble model
    backtest.py              # Walk-forward validation + DSR + signal decay
    factor_model.py          # PCA statistical factor model (30 factors)
    calibration.py           # Probability calibration
  strategies/
    momentum_long_short.py   # Momentum L/S strategy
    sector_rotation.py       # Sector rotation strategy
    mean_reversion.py        # Mean reversion strategy
    momentum_reversal.py     # Composite momentum+reversal with dynamic weights
    risk_manager.py          # Shared risk management
    portfolio.py             # Portfolio construction + beta hedge
    config.py                # All strategy configuration
  simulation/
    engine.py                # Simulation engine (legacy + framework)
  monitoring/
    checks.py                # Data quality, drift, alpha, capacity, regime stress, OOD detection
  ml/
    promotion.py             # Model registry + promotion gates
    tracking.py              # MLflow configuration

dashboard-react/             # React operations dashboard
supabase/functions/          # Edge function backend
```

## Scheduled Workflows

- **Daily EOD:** `.github/workflows/daily-pipeline.yml` — Cron: `30 1 * * 1-5` (01:30 UTC, Mon-Fri). **Does not** run the heavy fundamentals upsert in GitHub Actions (keeps the job within the 90-minute budget).
- **Weekly fundamentals:** `.github/workflows/fundamentals-pipeline.yml` — Cron: `0 3 * * 1` (Monday 03:00 UTC), **`--step 11` only** (fundamentals; step 10 in the CLI is 9b simulations), **6-hour** timeout. Sets `FUNDAMENTALS_INGEST_IN_CI=1`. **Run workflow** manually with **force_run** if you need a recovery run on a non-Monday.
- **Monthly Archive:** `.github/workflows/monthly-archive.yml` — Cron: `15 3 2 * *` (03:15 UTC, day 2)

For a **full fundamentals refresh** from your machine (same as CI): `FUNDAMENTALS_INGEST_IN_CI=1 python -m scripts.daily_pipeline --step 11` on a Monday (or set `PIPELINE_CALENDAR_TZ`). Use `--step 10` if you also want **9b DB simulations** in the same run.

### GitHub Actions + Supabase `DATABASE_URL`

You **do not** need to buy IPv6 or “Dedicated IPv4” from Supabase for GitHub CI. The direct host `db.<project>.supabase.co` is **IPv6-oriented** from GitHub’s runners — **changing the port to 5432 does not fix that** if the hostname is still `db.*.supabase.co`.

**What works on Actions (free):** use the **Transaction pooler** string from the dashboard — host like `aws-0-<region>.pooler.supabase.com`, port **6543**, user **`postgres.<project-ref>`** (not plain `postgres`).  
**Supabase:** Project Settings → Database → Connection string → **URI** → **Transaction pooler** → paste into the `DATABASE_URL` secret.

Locally you can keep using direct or pooler; for CI, use the pooler URI.

## Local Setup

### 1) Environment

```bash
cp .env.example .env
```

Minimum required: `DATABASE_URL`, `REDIS_URL`

### 2) Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 3) Run Pipeline

```bash
# Full pipeline
python -m scripts.daily_pipeline --step 1

# From prediction onward
python -m scripts.daily_pipeline --step 6
```

### 4) Retrain Model

```bash
# 5 seeds, cross-sectional alpha profile, 5-day horizon, non-overlapping samples
python -m scripts.retrain_market_relative --seeds 5 --feature-profile cross_sectional_alpha --target-horizon-days 5

# With auto-promotion to production
python -m scripts.retrain_market_relative --seeds 5 --allow-promote

# Override sample stride (default = target_horizon_days for non-overlapping targets)
python -m scripts.retrain_market_relative --seeds 5 --sample-stride 3
```

### 5) Retrain on Corrected Pipeline (7 years default)

```bash
# 7-year window covers COVID + 2022 rate shock, survivorship-bias-aware universe
python -m scripts.retrain_market_relative --seeds 5 --allow-promote

# LambdaRank pairwise ranking (typically 5-10% IC improvement)
python -m scripts.retrain_market_relative --model-mode ranker --years 5

# XGBoost + LightGBM ensemble (2-5% IC improvement, lower variance)
python -m scripts.retrain_market_relative --model-mode ensemble --years 5

# Regression mode with continuous return target
python -m scripts.retrain_market_relative --model-mode regressor --years 5

# Include fundamental features (requires SIMFIN_API_KEY)
python -m scripts.retrain_market_relative --include-fundamentals --feature-profile alpha_fundamental
```

### 6) Factor Attribution (Alpha vs Factor Beta)

```bash
# FF5 + Momentum regression on L/S equity curve
python -m scripts.research_factor_attribution --years 5

# Output: artifacts/research/factor_attribution.json + .png
```

### 7) Run Diagnostics

```bash
# Capacity test + crash simulation + feature stability
python -m scripts.prod_diagnostics
```

## Key Runtime Controls

| Variable | Values | Default |
|---|---|---|
| `PREDICT_MODEL_SOURCE` | `local_first`, `mlflow_first` | `mlflow_first` |
| `BATCH_PREDICT_EXPLANATIONS` | `0`, `1` | `1` |
| `STRATEGY_FRAMEWORK_ENABLED` | `0`, `1` | `1` |
| `SIM_FORCE_RERUN` | `0`, `1` (`1` = delete prior Step 9b runs for date + prod model, then re-run) | `0` |
| `PYTHONIOENCODING` | `utf-8` | (set for Windows) |

## Training Pipeline

The retrain script uses a strict **4-way chronological split**:

| Window | % | Purpose |
|---|---|---|
| Train + WF | 70% | Model training (internal 80/20 split) + walk-forward evaluation |
| Calibration | 15% | Isotonic calibration — never seen during training or early stopping |
| OOS Holdout | 15% | Truly unseen evaluation — never used for any model decisions |

Non-overlapping samples (stride=5) eliminate autocorrelated targets for honest effective sample counts.

## Known Limitations

- **Step 9b DB simulations**: `simulation_runs` are only written when there is an eligible `predictions.target_date` (model horizon, e.g. T+5) that is **on or before** the pipeline run date **and** `market_bars_daily` has that session. After promoting a new `model_version`, the first rows may all be **future** horizons — expect **~5 trading days** before simulations appear; logs include a diagnostic if skipped.
- **Single production model**: ensemble available but strategies currently share one alpha source
- **Fundamental coverage**: SEC EDGAR covers all US-listed filers; yfinance fills gaps; SimFin optional
- **7-year default backtest**: covers COVID crash + 2022 rate shock; extend with `--years 10` for deeper history
- **yfinance primary data**: hardened with retry/staleness/split checks; Alpha Vantage free-tier available as cross-validation source; no delisted stock data
- PCA factor model uses statistical factors (not named); interpretation requires mapping to economic factors
- Beta estimation uses Vasicek-shrunk 60-day window; may lag during sharp regime changes
- MeanReversion mode produces few trades when RSI extremes are rare
- Capacity ceiling ~$10M due to 200-stock universe
- Non-overlapping sample stride reduces effective training set size by ~5x (necessary for honest evaluation)
- Partial fill model clips at 5% ADV participation — ultra-illiquid names may still have adverse selection

## Documentation

- `PRD-Stock-Prediction-AI-v1.1.md`
- `ENG-SPEC-Stock-Prediction-AI-v1.1.md`
- `SPRINT-PLAN-v1.md`
- `docs/DASHBOARD-NETLIFY.md`
- `docs/ARCHIVE-RUNBOOK.md`
