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
import pickle
from datetime import UTC, date, datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.cache.redis_cache import cache_batch_predictions, get_redis_client
from src.features.leakage import _add_cross_sectional_transforms
from src.ml.promotion import get_production_model
from src.ml.tracking import configure_mlflow
from src.models.schema import CalibrationModel, Prediction

logger = logging.getLogger(__name__)

TOP_K_FACTORS = 5
LOCAL_PROD_MODEL_PATH = os.path.join("artifacts", "production_model", "model.json")
LOCAL_PROD_MEDIANS_PATH = os.path.join(
    "artifacts", "production_model", "train_medians.pkl"
)


def _model_source_order() -> list[str]:
    """Resolve model loading strategy from env.

    PREDICT_MODEL_SOURCE values:
      - mlflow_first (default): try MLflow URIs, then local file
      - local_first: try local file first, then MLflow URIs
    """
    mode = os.getenv("PREDICT_MODEL_SOURCE", "mlflow_first").strip().lower()
    if mode not in {"mlflow_first", "local_first"}:
        logger.warning(
            "Unknown PREDICT_MODEL_SOURCE=%s, defaulting to mlflow_first", mode
        )
        mode = "mlflow_first"
    if mode == "local_first":
        return ["local", "mlflow_model", "mlflow_live_model", "mlflow_root"]
    return ["mlflow_model", "mlflow_live_model", "mlflow_root", "local"]


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
    tried_locations: list[str] = []
    load_plan = _model_source_order()
    logger.info("Production model load plan: %s", " -> ".join(load_plan))
    for source in load_plan:
        if source == "local":
            tried_locations.append(f"file:{LOCAL_PROD_MODEL_PATH}")
            if os.path.exists(LOCAL_PROD_MODEL_PATH):
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(LOCAL_PROD_MODEL_PATH)
                    logger.info(
                        "Loaded production model artifact from local file %s",
                        LOCAL_PROD_MODEL_PATH,
                    )
                    break
                except Exception:
                    continue
            continue

        if source == "mlflow_model":
            model_uri = f"runs:/{reg.mlflow_run_id}/model"
        elif source == "mlflow_live_model":
            model_uri = f"runs:/{reg.mlflow_run_id}/live_model"
        else:
            model_uri = f"runs:/{reg.mlflow_run_id}"

        tried_locations.append(model_uri)
        try:
            model = mlflow.xgboost.load_model(model_uri)
            logger.info("Loaded production model artifact from %s", model_uri)
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(
            "Failed to load production model from all configured sources: "
            f"{tried_locations}"
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

    result = db.execute(
        text(
            f"""
        SELECT fs.*, s.sector
        FROM features_snapshot fs
        LEFT JOIN symbols s ON s.symbol = fs.symbol
        WHERE 1=1 {date_filter}
        ORDER BY fs.symbol
    """
        ),
        params,
    )

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    columns = list(result.keys())
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


def compute_shap_factors_from_vector(
    feature_names: list[str],
    shap_vector: np.ndarray,
) -> list[dict]:
    """Compute top SHAP contributions from a precomputed SHAP vector."""
    try:
        feature_impacts = list(zip(feature_names, shap_vector))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {"name": name, "impact": round(float(impact), 4)}
            for name, impact in feature_impacts[:TOP_K_FACTORS]
        ]
    except Exception:
        logger.warning("SHAP computation failed, returning empty factors")
        return []


def _align_features_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """Align inference matrix columns to the model's expected feature schema.

    This prevents hard failures when a production model was trained with a
    different feature profile than the current default prepare_features output.
    Missing columns are filled with 0.0, extra columns are dropped.
    """
    expected = []
    try:
        booster = model.get_booster()
        expected = list(booster.feature_names or [])
    except Exception:
        expected = []

    if not expected:
        return X

    missing = [c for c in expected if c not in X.columns]
    extras = [c for c in X.columns if c not in expected]

    if missing:
        for c in missing:
            X[c] = 0.0
    if extras:
        X = X.drop(columns=extras, errors="ignore")

    X = X[expected]
    logger.info(
        "Aligned inference features to model schema: expected=%s, missing_filled=%s, extras_dropped=%s",
        len(expected),
        len(missing),
        len(extras),
    )
    return X


def _load_local_train_medians() -> pd.Series | None:
    if not os.path.exists(LOCAL_PROD_MEDIANS_PATH):
        return None
    try:
        with open(LOCAL_PROD_MEDIANS_PATH, "rb") as f:
            med = pickle.load(f)
        if isinstance(med, pd.Series):
            return med
        if isinstance(med, dict):
            return pd.Series(med)
    except Exception:
        logger.warning("Failed to load local train medians, using inference medians")
    return None


def _build_inference_matrix(
    db: Session,
    snapshots_df: pd.DataFrame,
    expected_cols: list[str] | None,
) -> pd.DataFrame:
    """Build model-aligned inference matrix with training-style transforms."""
    inf_df = snapshots_df.copy()
    feature_date = inf_df["target_session_date"].iloc[0].date()

    # Liquidity/size features are not persisted in features_snapshot, compute point-in-time values.
    needs_liquidity = any(
        c in (expected_cols or [])
        for c in (
            "log_market_cap",
            "market_cap_rank",
            "dollar_volume",
            "dollar_volume_rank_market",
            "turnover_ratio",
        )
    )
    if needs_liquidity:
        liq_rows = db.execute(
            text(
                """
            SELECT mb.symbol, mb.close, mb.volume, s.market_cap
            FROM market_bars_daily mb
            JOIN symbols s ON s.symbol = mb.symbol
            WHERE mb.date = :d
              AND s.is_active = true
            """
            ),
            {"d": feature_date},
        ).fetchall()
        if liq_rows:
            liq_df = pd.DataFrame(
                liq_rows, columns=["symbol", "close", "volume", "market_cap"]
            )
            liq_df["dollar_volume"] = liq_df["close"] * liq_df["volume"]
            liq_df["log_market_cap"] = liq_df["market_cap"].clip(lower=1).map(
                lambda v: float(np.log(v))
            )
            shares_float = (liq_df["market_cap"] / liq_df["close"].clip(lower=0.01)).clip(
                lower=1
            )
            liq_df["turnover_ratio"] = liq_df["volume"] / shares_float
            liq_df["market_cap_rank"] = liq_df["market_cap"].rank(pct=True)
            liq_df["dollar_volume_rank_market"] = liq_df["dollar_volume"].rank(pct=True)
            liq_cols = [
                "symbol",
                "dollar_volume",
                "log_market_cap",
                "turnover_ratio",
                "market_cap_rank",
                "dollar_volume_rank_market",
            ]
            inf_df = inf_df.merge(liq_df[liq_cols], on="symbol", how="left", suffixes=("", "_liq"))
            for c in liq_cols[1:]:
                if f"{c}_liq" in inf_df.columns:
                    if c in inf_df.columns:
                        inf_df[c] = inf_df[c].fillna(inf_df[f"{c}_liq"])
                        inf_df = inf_df.drop(columns=[f"{c}_liq"])
                    else:
                        inf_df = inf_df.rename(columns={f"{c}_liq": c})

    if "turnover_acceleration" not in inf_df.columns:
        inf_df["turnover_acceleration"] = 0.0
    else:
        inf_df["turnover_acceleration"] = inf_df["turnover_acceleration"].fillna(0.0)

    # Cross-sectional transforms expected by live model profile.
    transform_cols = [
        "momentum_20d",
        "momentum_60d",
        "volume_change_5d",
        "volume_zscore_20d",
        "rolling_volatility_20d",
        "turnover_ratio",
    ]
    usable_transform_cols = [c for c in transform_cols if c in inf_df.columns]
    if usable_transform_cols:
        inf_df = _add_cross_sectional_transforms(inf_df, usable_transform_cols)

    # Regime one-hot columns (e.g. regime_bull_high_vol).
    if "regime_label" in inf_df.columns:
        regime_dummies = pd.get_dummies(
            inf_df["regime_label"], prefix="regime", dtype=float
        )
        inf_df = pd.concat([inf_df, regime_dummies], axis=1)

    # Use model feature schema as source of truth.
    if expected_cols:
        for col in expected_cols:
            if col not in inf_df.columns:
                inf_df[col] = 0.0
        X = inf_df[expected_cols].copy()
    else:
        X = inf_df.copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    train_medians = _load_local_train_medians()
    if train_medians is not None:
        for col in X.columns:
            if col in train_medians.index and X[col].isna().any():
                X[col] = X[col].fillna(train_medians[col])
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X


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

    expected_cols = []
    try:
        booster = model.get_booster()
        expected_cols = list(booster.feature_names or [])
    except Exception:
        expected_cols = []
    X = _build_inference_matrix(db, snapshots_df, expected_cols=expected_cols)
    X = _align_features_to_model(model, X)
    if expected_cols and len(X.columns) != len(expected_cols):
        logger.error(
            "schema mismatch detected: expected_cols=%s, actual_cols=%s",
            len(expected_cols),
            len(X.columns),
        )
    if expected_cols and list(X.columns) != list(expected_cols):
        logger.error(
            "schema mismatch detected: column order/name mismatch "
            "(expected_head=%s, actual_head=%s)",
            expected_cols[:5],
            list(X.columns)[:5],
        )

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
    shap_matrix = None
    if compute_explanations:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[1])
            else:
                shap_matrix = np.asarray(shap_values)
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
            if explainer is not None and shap_matrix is not None and i < len(shap_matrix):
                factors = compute_shap_factors_from_vector(
                    list(X.columns),
                    np.asarray(shap_matrix[i]),
                )

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
