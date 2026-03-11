from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db import get_db
from src.models.schema import ModelRegistry, Prediction, Symbol

app = FastAPI(
    title="TalPred AI - Market Intelligence Predictor",
    version="0.1.0",
    description="Stock prediction API with explainable signals",
)


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


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    algorithm: str | None
    training_window_start: date | None
    training_window_end: date | None
    status: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/symbols")
def list_symbols(db: Session = Depends(get_db)):
    symbols = db.query(Symbol).filter(Symbol.is_active.is_(True)).all()
    return [
        {
            "symbol": s.symbol,
            "company_name": s.company_name,
            "exchange": s.exchange,
            "sector": s.sector,
            "market_cap": s.market_cap,
        }
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
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    prediction = (
        db.query(Prediction)
        .filter(Prediction.symbol == req.symbol, Prediction.target_date == req.as_of_date)
        .order_by(Prediction.as_of_time.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {req.symbol} on {req.as_of_date}",
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
    )


@app.get("/predictions/{symbol}")
def prediction_history(symbol: str, limit: int = 30, db: Session = Depends(get_db)):
    predictions = (
        db.query(Prediction)
        .filter(Prediction.symbol == symbol)
        .order_by(Prediction.target_date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "prediction_id": p.prediction_id,
            "target_date": p.target_date.isoformat(),
            "direction": p.direction,
            "probability_up": p.probability_up,
            "confidence": p.confidence,
            "realized_direction": p.realized_direction,
            "realized_return": p.realized_return,
            "model_version": p.model_version,
        }
        for p in predictions
    ]
