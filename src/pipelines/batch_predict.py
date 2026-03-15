"""Batch prediction pipeline (BE-301).

Runs after market close once features are generated:
1. Load production model from registry
2. Load latest feature snapshots for all symbols
3. Generate predictions with calibrated probabilities
4. Store in predictions table
5. Write to Redis cache
"""

import hashlib
import logging
import os
from datetime import UTC, date, datetime, timedelta

import mlflow
import pandas as pd
import shap
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.cache.redis_cache import cache_batch_predictions, get_redis_client
from src.ml.promotion import get_production_model
from src.ml.tracking import configure_mlflow
from src.models.schema import CalibrationModel, Prediction
from src.models.trainer import prepare_features

logger = logging.getLogger(__name__)

TOP_K_FACTORS = 5
LOCAL_PROD_MODEL_PATH = os.path.join("artifacts", "production_model", "model.json")


def _prediction_id(symbol: str, target_date: date, model_version: str) -> str:
    raw = f"{symbol}:{target_date.isoformat()}:{model_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def load_production_model(db: Session) -> tuple:
    """Load the production model, its metadata, and optional calibrator.

    Returns (model, model_registry_row, calibrated_model_or_None).
    """
    reg = get_production_model(db)
    if not reg:
        raise RuntimeError("No production model found in registry")
    # End SELECT transaction before long artifact download so pooled DB connections
    # don't go stale while MLflow fetches model files.
    db.rollback()

    configure_mlflow()
    model = None
    tried_model_uris = []
    for model_uri in (
        f"runs:/{reg.mlflow_run_id}/model",
        f"runs:/{reg.mlflow_run_id}/live_model",
        f"runs:/{reg.mlflow_run_id}",
    ):
        tried_model_uris.append(model_uri)
        try:
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Loaded production model artifact from {model_uri}")
            break
        except Exception:
            continue

    if model is None:
        # Final fallback to local artifact persisted by train/promote flow.
        if os.path.exists(LOCAL_PROD_MODEL_PATH):
            model = xgb.XGBClassifier()
            model.load_model(LOCAL_PROD_MODEL_PATH)
            logger.warning(
                "MLflow artifact load failed for %s. Falling back to local model at %s",
                tried_model_uris,
                LOCAL_PROD_MODEL_PATH,
            )
        else:
            raise RuntimeError(
                "Failed to load production model from MLflow artifacts "
                f"{tried_model_uris} and no local fallback found at {LOCAL_PROD_MODEL_PATH}"
            )

    calibrator = None
    cal_row = None
    try:
        cal_row = (
            db.query(CalibrationModel)
            .filter(CalibrationModel.model_version == reg.model_version)
            .first()
        )
    except Exception:
        # One retry after resetting failed transaction/connection state.
        db.rollback()
        try:
            cal_row = (
                db.query(CalibrationModel)
                .filter(CalibrationModel.model_version == reg.model_version)
                .first()
            )
        except Exception:
            logger.warning("Failed to query calibration model, using raw probabilities")
            cal_row = None
    if cal_row and cal_row.artifact_uri:
        try:
            calibrator = mlflow.sklearn.load_model(cal_row.artifact_uri)
        except Exception:
            logger.warning("Failed to load calibrator, using raw probabilities")

    logger.info(
        f"Loaded production model {reg.model_version} "
        f"(run_id={reg.mlflow_run_id}, calibrator={'yes' if calibrator else 'no'})"
    )
    return model, reg, calibrator


def load_latest_snapshots(db: Session, target_date: date | None = None) -> pd.DataFrame:
    """Load the most recent feature snapshot for each symbol."""
    if target_date:
        date_filter = "AND fs.target_session_date = :target_date"
        params = {"target_date": target_date}
    else:
        date_filter = """AND fs.target_session_date = (
            SELECT MAX(target_session_date) FROM features_snapshot
        )"""
        params = {}

    result = db.execute(text(f"""
        SELECT
            snapshot_id, symbol, target_session_date,
            rsi_14, momentum_5d, momentum_10d,
            rolling_return_5d, rolling_return_20d, rolling_volatility_20d,
            macd, macd_signal,
            sector_return_1d, sector_return_5d,
            benchmark_relative_return_1d,
            news_sentiment_24h, news_sentiment_7d,
            vix_level, sp500_momentum_200d, regime_label, dataset_version
        FROM features_snapshot fs
        WHERE 1=1 {date_filter}
        ORDER BY symbol
    """), params)

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    columns = [
        "snapshot_id", "symbol", "target_session_date",
        "rsi_14", "momentum_5d", "momentum_10d",
        "rolling_return_5d", "rolling_return_20d", "rolling_volatility_20d",
        "macd", "macd_signal",
        "sector_return_1d", "sector_return_5d",
        "benchmark_relative_return_1d",
        "news_sentiment_24h", "news_sentiment_7d",
        "vix_level", "sp500_momentum_200d", "regime_label", "dataset_version",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["target_session_date"] = pd.to_datetime(df["target_session_date"])
    return df


def resolve_next_trading_day(db: Session, feature_date: date) -> date:
    """Get next trading day after feature_date from market_calendar or heuristic."""
    result = db.execute(text("""
        SELECT MIN(session_date) FROM market_calendar
        WHERE session_date > :fd AND is_holiday = false
    """), {"fd": feature_date})
    row = result.fetchone()
    if row and row[0]:
        return row[0]

    next_day = feature_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def compute_shap_factors(explainer: shap.TreeExplainer, X_row: pd.DataFrame) -> list[dict]:
    """Compute top SHAP feature contributions for a single prediction."""
    try:
        shap_values = explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        feature_impacts = list(zip(X_row.columns, sv))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {"name": name, "impact": round(float(impact), 4)}
            for name, impact in feature_impacts[:TOP_K_FACTORS]
        ]
    except Exception:
        logger.warning("SHAP computation failed, returning empty factors")
        return []


def run_batch_predictions(
    db: Session,
    target_date: date | None = None,
    compute_explanations: bool = True,
) -> dict:
    """Execute batch predictions for all symbols.

    Returns summary dict with counts and any errors.
    """
    model, reg, calibrator = load_production_model(db)
    snapshots_df = load_latest_snapshots(db, target_date)

    if snapshots_df.empty:
        logger.warning("No feature snapshots found for batch prediction")
        return {"predictions": 0, "errors": 0}

    feature_date = snapshots_df["target_session_date"].iloc[0].date()
    prediction_target = resolve_next_trading_day(db, feature_date)
    logger.info(
        f"Batch predict: {len(snapshots_df)} symbols, "
        f"features_as_of={feature_date}, target={prediction_target}, model={reg.model_version}"
    )

    snapshots_df["direction"] = 0
    X, _, medians = prepare_features(snapshots_df)

    raw_probs = model.predict_proba(X)[:, 1]

    if calibrator:
        try:
            cal_probs = calibrator.predict_proba(X)[:, 1]
        except Exception:
            logger.warning("Calibrator failed, using raw probabilities")
            cal_probs = raw_probs
    else:
        cal_probs = raw_probs

    explainer = None
    if compute_explanations:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            logger.warning("Failed to create SHAP explainer, skipping explanations")

    now = datetime.now(UTC)
    predictions = []
    errors = 0

    for i, (_, row) in enumerate(snapshots_df.iterrows()):
        try:
            prob_up = float(cal_probs[i])
            confidence = max(prob_up, 1.0 - prob_up)
            direction = "up" if prob_up >= 0.5 else "down"

            factors = []
            if explainer is not None:
                factors = compute_shap_factors(explainer, X.iloc[[i]])

            pred_dict = {
                "prediction_id": _prediction_id(
                    row["symbol"], prediction_target, reg.model_version
                ),
                "symbol": row["symbol"],
                "as_of_time": now,
                "target_date": prediction_target,
                "direction": direction,
                "probability_up": prob_up,
                "confidence": confidence,
                "top_factors": factors,
                "model_version": reg.model_version,
                "feature_snapshot_id": row["snapshot_id"],
                "dataset_version": row.get("dataset_version"),
            }
            predictions.append(pred_dict)
        except Exception:
            logger.warning(f"Prediction failed for {row['symbol']}")
            errors += 1

    saved = _save_predictions(db, predictions)

    r = get_redis_client()
    cached = cache_batch_predictions(r, predictions)

    logger.info(
        f"Batch complete: {saved} saved, {cached} cached, {errors} errors"
    )
    return {
        "predictions": saved,
        "cached": cached,
        "errors": errors,
        "model_version": reg.model_version,
        "feature_date": str(feature_date),
        "target_date": str(prediction_target),
    }


def _save_predictions(db: Session, predictions: list[dict]) -> int:
    """Upsert predictions into the database."""
    count = 0
    for pred in predictions:
        existing = db.query(Prediction).filter(
            Prediction.prediction_id == pred["prediction_id"]
        ).first()

        if existing:
            for k, v in pred.items():
                setattr(existing, k, v)
        else:
            db.add(Prediction(**pred))
        count += 1

    db.commit()
    return count
