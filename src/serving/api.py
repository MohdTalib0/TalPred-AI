"""TalPred AI Prediction API (BE-303).

Cache-first serving with DB fallback and on-demand inference.
Per ENG-SPEC 12: cache-hit p95 <= 250ms.
"""

import logging
from datetime import date

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.cache.redis_cache import get_cached_prediction, get_cache_stats, get_redis_client
from src.db import get_db
from src.models.schema import ModelRegistry, Prediction, Symbol

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Predictions are informational only and do not constitute financial advice."
)

app = FastAPI(
    title="TalPred AI - Market Intelligence Predictor",
    version="1.0.0",
    description=f"Stock prediction API with explainable signals. {DISCLAIMER}",
)


# ── Schemas ──────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    symbol: str
    as_of_date: date


class FactorImpact(BaseModel):
    name: str
    impact: float


class PredictResponse(BaseModel):
    prediction_id: str
    symbol: str
    target_date: date
    direction: str
    probability_up: float
    confidence: float
    top_factors: list[FactorImpact]
    model_version: str
    feature_snapshot_id: str
    dataset_version: str | None = None
    source: str = "db"


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_available: bool
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_version: str
    algorithm: str | None
    training_window_start: date | None
    training_window_end: date | None
    status: str
    metrics: dict | None = None


class SymbolResponse(BaseModel):
    symbol: str
    company_name: str
    exchange: str
    sector: str | None
    market_cap: float | None


class PredictionSummary(BaseModel):
    prediction_id: str
    target_date: str
    direction: str
    probability_up: float
    confidence: float
    realized_direction: str | None
    realized_return: float | None
    model_version: str


# ── Dependency ───────────────────────────────────────────────────────

_redis_client = None


def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = get_redis_client()
    return _redis_client


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check():
    r = get_redis()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cache_available=r is not None,
        disclaimer=DISCLAIMER,
    )


@app.get("/symbols", response_model=list[SymbolResponse])
def list_symbols(db: Session = Depends(get_db)):
    symbols = (
        db.query(Symbol)
        .filter(Symbol.is_active.is_(True))
        .order_by(Symbol.symbol)
        .all()
    )
    return [
        SymbolResponse(
            symbol=s.symbol,
            company_name=s.company_name,
            exchange=s.exchange,
            sector=s.sector,
            market_cap=s.market_cap,
        )
        for s in symbols
    ]


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(db: Session = Depends(get_db)):
    model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status == "production")
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="No production model found")
    return ModelInfoResponse(
        model_version=model.model_version,
        algorithm=model.algorithm,
        training_window_start=model.training_window_start,
        training_window_end=model.training_window_end,
        status=model.status,
        metrics=model.metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """Cache-first prediction endpoint.

    1. Check Redis cache
    2. If miss, check DB
    3. Return 404 if not found (on-demand inference deferred to batch job)
    """
    symbol_exists = db.query(Symbol).filter(
        Symbol.symbol == req.symbol, Symbol.is_active.is_(True)
    ).first()
    if not symbol_exists:
        raise HTTPException(status_code=404, detail=f"Symbol {req.symbol} not in universe")

    r = get_redis()
    prod_model = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status == "production")
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    model_version = prod_model.model_version if prod_model else ""

    if r and model_version:
        cached = get_cached_prediction(r, req.symbol, req.as_of_date, model_version)
        if cached:
            factors = cached.get("top_factors", [])
            return PredictResponse(
                prediction_id=cached["prediction_id"],
                symbol=cached["symbol"],
                target_date=cached.get("target_date", req.as_of_date),
                direction=cached["direction"],
                probability_up=cached["probability_up"],
                confidence=cached["confidence"],
                top_factors=[FactorImpact(**f) for f in factors],
                model_version=cached["model_version"],
                feature_snapshot_id=cached.get("feature_snapshot_id", ""),
                dataset_version=cached.get("dataset_version"),
                source="cache",
            )

    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.symbol == req.symbol,
            Prediction.target_date == req.as_of_date,
        )
        .order_by(Prediction.as_of_time.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction for {req.symbol} on {req.as_of_date}. "
            "Run the batch prediction job first.",
        )

    factors = prediction.top_factors or []
    return PredictResponse(
        prediction_id=prediction.prediction_id,
        symbol=prediction.symbol,
        target_date=prediction.target_date,
        direction=prediction.direction,
        probability_up=prediction.probability_up,
        confidence=prediction.confidence,
        top_factors=[FactorImpact(**f) for f in factors],
        model_version=prediction.model_version,
        feature_snapshot_id=prediction.feature_snapshot_id or "",
        dataset_version=prediction.dataset_version,
        source="db",
    )


@app.get("/predictions/{symbol}", response_model=list[PredictionSummary])
def prediction_history(
    symbol: str,
    limit: int = Query(30, ge=1, le=200),
    db: Session = Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .filter(Prediction.symbol == symbol)
        .order_by(Prediction.target_date.desc())
        .limit(limit)
        .all()
    )
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")

    return [
        PredictionSummary(
            prediction_id=p.prediction_id,
            target_date=p.target_date.isoformat(),
            direction=p.direction,
            probability_up=p.probability_up,
            confidence=p.confidence,
            realized_direction=p.realized_direction,
            realized_return=p.realized_return,
            model_version=p.model_version,
        )
        for p in predictions
    ]


@app.get("/predictions")
def latest_predictions(
    limit: int = Query(50, ge=1, le=500),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Get latest predictions across all symbols, optionally filtered by confidence."""
    query = db.query(Prediction).order_by(Prediction.as_of_time.desc())

    if min_confidence > 0:
        query = query.filter(Prediction.confidence >= min_confidence)

    predictions = query.limit(limit).all()
    return [
        {
            "prediction_id": p.prediction_id,
            "symbol": p.symbol,
            "target_date": p.target_date.isoformat(),
            "direction": p.direction,
            "probability_up": p.probability_up,
            "confidence": p.confidence,
            "model_version": p.model_version,
        }
        for p in predictions
    ]


@app.get("/cache/stats")
def cache_stats():
    r = get_redis()
    return get_cache_stats(r)


# ── Simulation Endpoints (BE-402) ────────────────────────────────────


class SimulationRequest(BaseModel):
    start_date: date
    end_date: date
    starting_capital: float = 100_000.0
    min_confidence_trade: float = 0.60
    max_position: float = 0.05
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    model_version: str | None = None
    enable_regime_guard: bool = True
    vix_calm_below: float = 12.0
    vix_elevated_below: float = 15.0
    vix_stressed_above: float = 20.0
    use_live_ic_guard: bool = False
    live_ic_lookback_days: int = 60
    live_ic_floor: float = 0.01
    low_ic_exposure_scale: float = 0.5


class SimulationMetrics(BaseModel):
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    total_trading_days: int
    total_transaction_costs: float
    final_equity: float
    net_profit: float


class SimulationResponse(BaseModel):
    run_id: str
    metrics: SimulationMetrics | None = None
    equity_curve: list[dict] | None = None
    n_trades: int = 0
    n_trading_days: int = 0
    error: str | None = None


class SimulationSummary(BaseModel):
    run_id: str
    start_date: date
    end_date: date
    starting_capital: float
    status: str
    result_metrics: dict | None = None


@app.post("/simulation/run", response_model=SimulationResponse)
def run_simulation_endpoint(req: SimulationRequest, db: Session = Depends(get_db)):
    """Run a paper-trading simulation over a date range."""
    from src.simulation.engine import run_simulation

    if req.end_date <= req.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    result = run_simulation(
        db,
        start_date=req.start_date,
        end_date=req.end_date,
        starting_capital=req.starting_capital,
        min_confidence_trade=req.min_confidence_trade,
        max_position=req.max_position,
        transaction_cost_bps=req.transaction_cost_bps,
        slippage_bps=req.slippage_bps,
        model_version=req.model_version,
        enable_regime_guard=req.enable_regime_guard,
        vix_calm_below=req.vix_calm_below,
        vix_elevated_below=req.vix_elevated_below,
        vix_stressed_above=req.vix_stressed_above,
        use_live_ic_guard=req.use_live_ic_guard,
        live_ic_lookback_days=req.live_ic_lookback_days,
        live_ic_floor=req.live_ic_floor,
        low_ic_exposure_scale=req.low_ic_exposure_scale,
    )

    if "error" in result:
        return SimulationResponse(run_id=result["run_id"], error=result["error"])

    return SimulationResponse(
        run_id=result["run_id"],
        metrics=SimulationMetrics(**result["metrics"]),
        equity_curve=result.get("equity_curve"),
        n_trades=result.get("n_trades", 0),
        n_trading_days=result.get("n_trading_days", 0),
    )


@app.get("/simulation/{run_id}", response_model=SimulationSummary)
def get_simulation(run_id: str, db: Session = Depends(get_db)):
    """Get simulation run details."""
    from src.models.schema import SimulationRun

    sim = db.query(SimulationRun).filter(SimulationRun.run_id == run_id).first()
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation {run_id} not found")

    return SimulationSummary(
        run_id=sim.run_id,
        start_date=sim.start_date,
        end_date=sim.end_date,
        starting_capital=sim.starting_capital,
        status=sim.status,
        result_metrics=sim.result_metrics,
    )


# ── Monitoring Endpoints (OP-401) ────────────────────────────────────


@app.get("/monitoring/status")
def monitoring_status(db: Session = Depends(get_db)):
    """Run all monitoring checks and return health report."""
    from src.monitoring.checks import run_all_checks
    return run_all_checks(db)


@app.get("/monitoring/data-quality")
def data_quality_check(db: Session = Depends(get_db)):
    from src.monitoring.checks import check_data_quality
    return check_data_quality(db)


@app.get("/monitoring/freshness")
def freshness_check(db: Session = Depends(get_db)):
    from src.monitoring.checks import check_data_freshness
    return check_data_freshness(db)


@app.get("/monitoring/drift")
def drift_check(db: Session = Depends(get_db)):
    from src.monitoring.checks import check_feature_drift
    return check_feature_drift(db)
