"""Backfill alpha features into existing features_snapshot rows.

Computes:
  - vol_adj_momentum_20d/60d  (from existing snapshot columns)
  - pct_from_52w_high          (needs 252-day rolling max from market_bars_daily)
  - idio_momentum_20d/60d      (stock momentum minus SPY momentum)
  - vol_price_divergence        (cross-sectional rank difference per date)
  - vol_adj_momentum_20d_rank, pct_from_52w_high_rank, idio_momentum_20d_rank
"""

import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from src.db import get_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    db = next(get_db())

    # ── 1. Load all snapshots (only cols we need) ──
    logger.info("Loading snapshots...")
    rows = db.execute(text("""
        SELECT snapshot_id, symbol, target_session_date,
               momentum_20d, momentum_60d, momentum_5d,
               rolling_volatility_20d, volume_change_5d
        FROM features_snapshot
        ORDER BY target_session_date, symbol
    """)).fetchall()

    df = pd.DataFrame(rows, columns=[
        "snapshot_id", "symbol", "target_session_date",
        "momentum_20d", "momentum_60d", "momentum_5d",
        "rolling_volatility_20d", "volume_change_5d",
    ])
    logger.info(f"Loaded {len(df)} snapshots")

    # ── 2. Vol-adjusted momentum (from existing columns) ──
    vol20 = df["rolling_volatility_20d"].clip(lower=0.001)
    df["vol_adj_momentum_20d"] = df["momentum_20d"] / vol20
    df["vol_adj_momentum_60d"] = df["momentum_60d"] / vol20

    # ── 3. pct_from_52w_high (needs market data) ──
    logger.info("Loading market close prices for 52w high...")
    symbols = list(df["symbol"].unique())
    min_date = df["target_session_date"].min()

    market_rows = db.execute(text("""
        SELECT symbol, date, close
        FROM market_bars_daily
        WHERE symbol = ANY(:syms)
          AND date >= CAST(:start AS date) - INTERVAL '370 days'
        ORDER BY symbol, date
    """), {"syms": symbols, "start": min_date}).fetchall()

    mkt = pd.DataFrame(market_rows, columns=["symbol", "date", "close"])
    mkt["date"] = pd.to_datetime(mkt["date"])
    logger.info(f"Loaded {len(mkt)} market bars")

    mkt = mkt.sort_values(["symbol", "date"])
    mkt["high_52w"] = mkt.groupby("symbol")["close"].transform(
        lambda x: x.rolling(252, min_periods=60).max()
    )
    mkt["pct_from_52w_high"] = mkt["close"] / mkt["high_52w"].clip(lower=0.01)
    mkt["date_only"] = mkt["date"].dt.date

    h52_lookup = mkt.set_index(["symbol", "date_only"])["pct_from_52w_high"].to_dict()
    df["pct_from_52w_high"] = df.apply(
        lambda r: h52_lookup.get((r["symbol"], r["target_session_date"])), axis=1
    )
    logger.info(f"pct_from_52w_high: {df['pct_from_52w_high'].notna().sum()} non-null")

    # ── 4. Idiosyncratic momentum (stock - SPY) ──
    logger.info("Computing SPY momentum...")
    spy_rows = db.execute(text("""
        SELECT date, close FROM market_bars_daily
        WHERE symbol = 'SPY'
        ORDER BY date
    """)).fetchall()

    spy = pd.DataFrame(spy_rows, columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date")["close"]
    spy_mom20 = spy.pct_change(20)
    spy_mom60 = spy.pct_change(60)

    spy_20_map = {d.date(): v for d, v in spy_mom20.items() if pd.notna(v)}
    spy_60_map = {d.date(): v for d, v in spy_mom60.items() if pd.notna(v)}

    df["_spy_mom20"] = df["target_session_date"].map(spy_20_map)
    df["_spy_mom60"] = df["target_session_date"].map(spy_60_map)
    df["idio_momentum_20d"] = df["momentum_20d"] - df["_spy_mom20"]
    df["idio_momentum_60d"] = df["momentum_60d"] - df["_spy_mom60"]
    logger.info(f"idio_momentum_20d: {df['idio_momentum_20d'].notna().sum()} non-null")

    # ── 5. Vol-price divergence (cross-sectional per date) ──
    mom_col = "momentum_5d"
    df["vol_price_divergence"] = (
        df.groupby("target_session_date")["volume_change_5d"].rank(pct=True)
        - df.groupby("target_session_date")[mom_col].rank(pct=True)
    )

    # ── 6. Cross-sectional ranks ──
    for rank_col, src_col in [
        ("vol_adj_momentum_20d_rank", "vol_adj_momentum_20d"),
        ("pct_from_52w_high_rank", "pct_from_52w_high"),
        ("idio_momentum_20d_rank", "idio_momentum_20d"),
    ]:
        df[rank_col] = df.groupby("target_session_date")[src_col].rank(pct=True)

    # ── 7. Bulk UPDATE via UPDATE ... FROM (VALUES ...) ──
    update_cols = [
        "vol_adj_momentum_20d", "vol_adj_momentum_60d", "pct_from_52w_high",
        "idio_momentum_20d", "idio_momentum_60d", "vol_price_divergence",
        "vol_adj_momentum_20d_rank", "pct_from_52w_high_rank", "idio_momentum_20d_rank",
    ]

    total = len(df)
    batch_size = 500
    updated = 0

    def _val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NULL"
        if pd.isna(v):
            return "NULL"
        return str(float(v))

    logger.info(f"Updating {total} rows in batches of {batch_size}...")
    for start in range(0, total, batch_size):
        batch = df.iloc[start : start + batch_size]
        values_rows = []
        for _, row in batch.iterrows():
            vals = [f"'{row['snapshot_id']}'"]
            for c in update_cols:
                vals.append(_val(row[c]))
            values_rows.append(f"({', '.join(vals)})")

        col_aliases = ", ".join([f"v_{c}" for c in update_cols])
        set_clause = ", ".join(f"{c} = v.v_{c}::float" for c in update_cols)
        sql = f"""
            UPDATE features_snapshot AS fs SET {set_clause}
            FROM (VALUES {', '.join(values_rows)})
            AS v(v_snapshot_id, {col_aliases})
            WHERE fs.snapshot_id = v.v_snapshot_id
        """
        db.execute(text(sql))
        db.commit()
        updated += len(batch)
        if updated % 5000 == 0 or updated == total:
            logger.info(f"  Updated {updated}/{total} ({updated*100//total}%)")

    logger.info(f"Backfill complete: {updated} rows updated")

    # Verify
    r = db.execute(text("""
        SELECT COUNT(*) AS total,
               COUNT(idio_momentum_20d) AS has_idio,
               COUNT(pct_from_52w_high) AS has_52w,
               COUNT(vol_price_divergence) AS has_vpd
        FROM features_snapshot
    """)).fetchone()
    logger.info(f"Verification: total={r[0]}, idio={r[1]}, 52w_high={r[2]}, vpd={r[3]}")


if __name__ == "__main__":
    main()
