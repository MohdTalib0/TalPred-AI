"""Batch prediction pipeline (BE-301).

Runs after market close once features are generated:
1. Load production model from registry
2. Load latest feature snapshots for all symbols
3. Generate predictions with calibrated probabilities
4. Store in predictions table
5. Write to Redis cache
"""

import hashlib
import json
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
LOCAL_PROD_METADATA_PATH = os.path.join(
    "artifacts", "production_model", "metadata.json"
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


def resolve_nth_trading_day(db: Session, feature_date: date, n: int = 1) -> date:
    """Get the Nth trading day after feature_date.

    Tries market_calendar first; falls back to skipping weekends.
    """
    result = db.execute(
        text("""
            SELECT session_date FROM market_calendar
            WHERE session_date > :fd AND is_holiday = false
            ORDER BY session_date
            LIMIT :n
        """),
        {"fd": feature_date, "n": n},
    )
    rows = result.fetchall()
    if rows and len(rows) == n:
        return rows[-1][0]

    # Heuristic fallback: skip weekends
    day = feature_date
    count = 0
    while count < n:
        day += timedelta(days=1)
        if day.weekday() < 5:
            count += 1
    return day


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

    Missing columns are filled with training medians when available
    (preserving the distribution centre), falling back to 0.0 for
    one-hot / binary features.  Extra columns are dropped.
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
        train_medians = _load_local_train_medians()
        for c in missing:
            fill = 0.0
            if train_medians is not None and c in train_medians.index:
                fill = float(train_medians[c])
            X[c] = fill
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


def _load_model_metadata() -> dict:
    """Read production model metadata (target_mode, target_horizon_days, etc.)."""
    if not os.path.exists(LOCAL_PROD_METADATA_PATH):
        return {}
    try:
        with open(LOCAL_PROD_METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_target_mode() -> str:
    """Read target_mode from production model metadata (default: absolute)."""
    return _load_model_metadata().get("target_mode", "absolute")


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


def _compute_alpha_features(
    db: Session,
    inf_df: pd.DataFrame,
    feature_date: date,
    expected_cols: list[str] | None,
) -> pd.DataFrame:
    """Compute cross-sectional alpha features that aren't in the snapshot.

    These match the training-time features built in leakage.py / engine.py:
      vol_adj_momentum_20d/60d, pct_from_52w_high, idio_momentum_20d/60d,
      vol_price_divergence, and their _rank variants.
    """
    needed = {
        "vol_adj_momentum_20d",
        "vol_adj_momentum_60d",
        "pct_from_52w_high",
        "idio_momentum_20d",
        "idio_momentum_60d",
        "vol_price_divergence",
        "vol_adj_momentum_20d_rank",
        "pct_from_52w_high_rank",
        "idio_momentum_20d_rank",
    }
    if expected_cols and not needed.intersection(expected_cols):
        return inf_df

    # vol-adjusted momentum (only needs snapshot columns)
    if "momentum_20d" in inf_df.columns and "rolling_volatility_20d" in inf_df.columns:
        vol20 = inf_df["rolling_volatility_20d"].clip(lower=0.001)
        if "vol_adj_momentum_20d" not in inf_df.columns:
            inf_df["vol_adj_momentum_20d"] = inf_df["momentum_20d"] / vol20
        if "vol_adj_momentum_60d" not in inf_df.columns and "momentum_60d" in inf_df.columns:
            inf_df["vol_adj_momentum_60d"] = inf_df["momentum_60d"] / vol20

    # 52-week high proximity — query 252-day max close per symbol
    if "pct_from_52w_high" not in inf_df.columns:
        high52_rows = db.execute(
            text("""
                SELECT symbol, MAX(close) AS high_52w
                FROM market_bars_daily
                WHERE date >= :start AND date <= :end
                  AND symbol = ANY(:syms)
                GROUP BY symbol
            """),
            {
                "start": feature_date - timedelta(days=365),
                "end": feature_date,
                "syms": list(inf_df["symbol"].unique()),
            },
        ).fetchall()
        if high52_rows:
            h52 = pd.DataFrame(high52_rows, columns=["symbol", "high_52w"])
            inf_df = inf_df.merge(h52, on="symbol", how="left")
            # Use the snapshot's close or the latest market close for the ratio
            close_col = "close" if "close" in inf_df.columns else None
            if close_col is None:
                close_rows = db.execute(
                    text("""
                        SELECT symbol, close FROM market_bars_daily
                        WHERE date = :d AND symbol = ANY(:syms)
                    """),
                    {"d": feature_date, "syms": list(inf_df["symbol"].unique())},
                ).fetchall()
                if close_rows:
                    cl = pd.DataFrame(close_rows, columns=["symbol", "_close_latest"])
                    inf_df = inf_df.merge(cl, on="symbol", how="left")
                    close_col = "_close_latest"
            if close_col and "high_52w" in inf_df.columns:
                inf_df["pct_from_52w_high"] = (
                    inf_df[close_col] / inf_df["high_52w"].clip(lower=0.01)
                )
                inf_df = inf_df.drop(columns=["high_52w"], errors="ignore")
                if close_col == "_close_latest":
                    inf_df = inf_df.drop(columns=["_close_latest"], errors="ignore")

    # Idiosyncratic momentum — stock momentum minus SPY momentum
    needs_idio = (
        "idio_momentum_20d" not in inf_df.columns
        and "momentum_20d" in inf_df.columns
    )
    if needs_idio:
        spy_row = db.execute(
            text("""
                SELECT close FROM market_bars_daily
                WHERE symbol = 'SPY' AND date <= :d
                ORDER BY date DESC LIMIT 61
            """),
            {"d": feature_date},
        ).fetchall()
        if spy_row and len(spy_row) >= 21:
            spy_closes = [float(r[0]) for r in reversed(spy_row)]
            spy_mom20 = (spy_closes[-1] / spy_closes[-21]) - 1
            if len(spy_closes) >= 61:
                spy_mom60 = (spy_closes[-1] / spy_closes[-61]) - 1
            else:
                logger.warning(
                    "SPY history < 61 bars (%d), using 20d momentum as 60d proxy",
                    len(spy_closes),
                )
                spy_mom60 = spy_mom20
            inf_df["idio_momentum_20d"] = inf_df["momentum_20d"] - spy_mom20
            if "momentum_60d" in inf_df.columns:
                inf_df["idio_momentum_60d"] = inf_df["momentum_60d"] - spy_mom60

    # Volume-price divergence (cross-sectional rank difference per date,
    # matching training pipeline's groupby("target_session_date")).
    # Must use momentum_5d to match training (leakage.py), NOT momentum_20d.
    date_col = "target_session_date"
    if (
        "vol_price_divergence" not in inf_df.columns
        and "volume_change_5d" in inf_df.columns
    ):
        mom_col = "momentum_5d" if "momentum_5d" in inf_df.columns else "momentum_20d"
        inf_df["vol_price_divergence"] = (
            inf_df.groupby(date_col)["volume_change_5d"].rank(pct=True)
            - inf_df.groupby(date_col)[mom_col].rank(pct=True)
        )

    # Cross-sectional ranks for alpha features (per date, matching training)
    rank_map = {
        "vol_adj_momentum_20d_rank": "vol_adj_momentum_20d",
        "pct_from_52w_high_rank": "pct_from_52w_high",
        "idio_momentum_20d_rank": "idio_momentum_20d",
    }
    for rank_col, src_col in rank_map.items():
        if rank_col not in inf_df.columns and src_col in inf_df.columns:
            inf_df[rank_col] = inf_df.groupby(date_col)[src_col].rank(pct=True)

    return inf_df


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
        # Match training pipeline: use rolling 30-day average dollar volume as
        # size proxy instead of static symbols.market_cap (forward-looking bias).
        liq_rows = db.execute(
            text("""
                SELECT symbol, date, close, volume
                FROM market_bars_daily
                WHERE date BETWEEN (:d::date - INTERVAL '45 days') AND :d
                  AND symbol = ANY(:syms)
                ORDER BY symbol, date
            """),
            {
                "d": feature_date,
                "syms": list(inf_df["symbol"].unique()),
            },
        ).fetchall()
        if liq_rows:
            hist = pd.DataFrame(liq_rows, columns=["symbol", "date", "close", "volume"])
            hist["dollar_volume"] = hist["close"].abs() * hist["volume"].abs()
            hist = hist.sort_values(["symbol", "date"])

            # Rolling 30-day average dollar volume per symbol (size proxy)
            rolling_adv = (
                hist.groupby("symbol")["dollar_volume"]
                .apply(lambda x: x.rolling(30, min_periods=5).mean().iloc[-1])
            )
            rolling_adv = rolling_adv.clip(lower=1).rename("_pit_market_cap")

            # Latest-day values for turnover ratio
            latest = hist.groupby("symbol").tail(1).set_index("symbol")

            rolling_avg_vol = (
                hist.groupby("symbol")["volume"]
                .apply(lambda x: x.rolling(30, min_periods=5).mean().iloc[-1])
            )

            liq_df = pd.DataFrame({
                "dollar_volume": latest["dollar_volume"],
                "log_market_cap": np.log1p(rolling_adv),
                "market_cap_rank": rolling_adv.rank(pct=True),
                "dollar_volume_rank_market": latest["dollar_volume"].rank(pct=True),
                "turnover_ratio": latest["volume"] / rolling_avg_vol.clip(lower=1),
            })
            liq_df.index.name = "symbol"
            liq_df = liq_df.reset_index()

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

    # ── Alpha features not persisted in features_snapshot ──
    inf_df = _compute_alpha_features(db, inf_df, feature_date, expected_cols)

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
        X = X.fillna(train_medians)
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
    meta = _load_model_metadata()
    horizon = meta.get("target_horizon_days", 1)
    prediction_target = resolve_nth_trading_day(db, feature_date, n=horizon)
    logger.info(
        f"Batch predict: {len(snapshots_df)} symbols, "
        f"features_as_of={feature_date}, target={prediction_target} "
        f"(horizon={horizon}d), model={reg.model_version}"
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
        raise RuntimeError(
            f"Feature schema mismatch after alignment: "
            f"expected {len(expected_cols)} cols, got {len(X.columns)}. "
            f"Predictions would be corrupted — aborting."
        )
    if expected_cols and list(X.columns) != list(expected_cols):
        raise RuntimeError(
            f"Feature column order/name mismatch after alignment: "
            f"expected_head={expected_cols[:5]}, "
            f"actual_head={list(X.columns)[:5]}. "
            f"Predictions would be corrupted — aborting."
        )

    raw_probs = model.predict_proba(X)[:, 1]

    if calibrator:
        try:
            cal_probs = calibrator.predict_proba(X)[:, 1]
        except Exception:
            logger.warning("Calibrator failed, using raw probabilities", exc_info=True)
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

    target_mode = _load_target_mode()
    is_relative = target_mode in ("market_relative", "sector_relative")
    if is_relative:
        logger.info(f"Model target_mode={target_mode}: direction = outperform/underperform")

    now = datetime.now(UTC)
    predictions = []
    errors = 0

    for i, (_, row) in enumerate(snapshots_df.iterrows()):
        try:
            prob_up = float(cal_probs[i])
            confidence = max(prob_up, 1.0 - prob_up)
            if is_relative:
                direction = "outperform" if prob_up >= 0.5 else "underperform"
            else:
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
    """Bulk upsert predictions via INSERT ... ON CONFLICT DO UPDATE.

    Preserves realized_return / realized_direction / outcome_recorded_at
    that may have been written by the outcome backfill pipeline.
    """
    if not predictions:
        return 0

    upsert_cols = [
        "symbol", "as_of_time", "target_date", "direction",
        "probability_up", "confidence", "top_factors", "model_version",
        "feature_snapshot_id", "dataset_version",
    ]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in upsert_cols)

    stmt = text(f"""
        INSERT INTO predictions
            (prediction_id, {", ".join(upsert_cols)})
        VALUES
            (:prediction_id, :{", :".join(upsert_cols)})
        ON CONFLICT (prediction_id) DO UPDATE SET
            {set_clause}
    """)

    batch_size = 200
    for start in range(0, len(predictions), batch_size):
        batch = predictions[start : start + batch_size]
        for pred in batch:
            params = {"prediction_id": pred["prediction_id"]}
            for c in upsert_cols:
                val = pred.get(c)
                if c == "top_factors" and val is not None:
                    val = json.dumps(val)
                params[c] = val
            db.execute(stmt, params)
        db.commit()

    return len(predictions)
