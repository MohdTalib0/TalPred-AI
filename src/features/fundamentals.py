"""Fundamental feature engineering from quarterly financial statements.

Data source priority (all free, no paid APIs required):
  1. SEC EDGAR XBRL (src/connectors/sec_edgar.py) — true PIT filing dates
  2. yfinance fundamentals (src/connectors/yfinance_fundamentals.py) — zero-cost
  3. SimFin (src/connectors/simfin.py) — if SIMFIN_API_KEY is set

Features computed:
  - Accruals ratio: (net_income - op_cash_flow) / total_assets [Sloan 1996]
  - ROE trend: Δ(net_income / equity) over 4 quarters
  - Earnings momentum: (EPS_q - EPS_{q-4}) / |EPS_{q-4}| [free PEAD proxy]
  - Revenue surprise: QoQ revenue growth deviation
  - Gross margin change: QoQ change in gross margin
  - Operating leverage: dOI/OI / (dRev/Rev)

Usage:
    from src.features.fundamentals import (
        load_fundamentals, compute_fundamental_features,
    )

    fundamentals = load_fundamentals(symbols, start_date, end_date)
    features_df = compute_fundamental_features(fundamentals)
"""

import logging
import os
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_fundamentals(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Load quarterly financial data using free sources.

    Source priority:
      1. SEC EDGAR (free, true PIT filing dates)
      2. yfinance (free, estimated PIT dates)
      3. SimFin (if SIMFIN_API_KEY is set)

    Returns dict with key 'unified' → DataFrame with columns:
        symbol, period_end_date, filed_at, net_income, total_revenue,
        total_assets, stockholders_equity, operating_cash_flow, eps_basic, ...

    Also includes legacy keys 'pl', 'bs', 'cf' for backward compat.
    """
    unified = pd.DataFrame()

    # Source 1: SEC EDGAR (best PIT reliability)
    try:
        from src.connectors.sec_edgar import fetch_fundamentals_edgar
        edgar_df = fetch_fundamentals_edgar(symbols)
        if not edgar_df.empty:
            logger.info(
                f"EDGAR: {len(edgar_df)} quarterly records for "
                f"{edgar_df['symbol'].nunique()} symbols"
            )
            unified = edgar_df
    except Exception as e:
        logger.warning(f"EDGAR fetch failed: {e}")

    # Source 2: yfinance fundamentals (fill gaps)
    if unified.empty or unified["symbol"].nunique() < len(symbols) * 0.5:
        try:
            from src.connectors.yfinance_fundamentals import fetch_fundamentals
            yf_df = fetch_fundamentals(symbols)
            if not yf_df.empty:
                if unified.empty:
                    unified = yf_df
                    logger.info(
                        f"yfinance fundamentals: {len(yf_df)} records for "
                        f"{yf_df['symbol'].nunique()} symbols"
                    )
                else:
                    covered = set(unified["symbol"].unique())
                    missing = [s for s in symbols if s not in covered]
                    if missing:
                        yf_fill = yf_df[yf_df["symbol"].isin(missing)]
                        if not yf_fill.empty:
                            unified = pd.concat([unified, yf_fill], ignore_index=True)
                            logger.info(
                                f"yfinance fundamentals: filled {yf_fill['symbol'].nunique()} "
                                f"symbols missing from EDGAR"
                            )
        except Exception as e:
            logger.warning(f"yfinance fundamentals fetch failed: {e}")

    # Source 3: SimFin (supplement with additional symbols if API key is set)
    if os.environ.get("SIMFIN_API_KEY"):
        covered = set(unified["symbol"].unique()) if not unified.empty else set()
        missing_syms = [s for s in symbols if s not in covered]

        if missing_syms or unified.empty:
            try:
                from src.connectors.simfin import fetch_statements

                fetch_syms = missing_syms if missing_syms else symbols
                result_legacy: dict[str, pd.DataFrame] = {}
                for stmt in ("pl", "bs", "cf"):
                    df = fetch_statements(fetch_syms, stmt, start_date, end_date)
                    if not df.empty:
                        for col in ["Report Date", "Publish Date", "Fiscal Year"]:
                            if col in df.columns:
                                if col in ("Report Date", "Publish Date"):
                                    df[col] = pd.to_datetime(df[col], errors="coerce")
                                elif col == "Fiscal Year":
                                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    result_legacy[stmt] = df if not df.empty else pd.DataFrame()

                has_simfin = any(not v.empty for v in result_legacy.values())
                if has_simfin:
                    if unified.empty:
                        logger.info(
                            f"SimFin: using as primary source for "
                            f"{len(fetch_syms)} symbols"
                        )
                        return result_legacy
                    else:
                        logger.info(
                            f"SimFin: supplementing {len(missing_syms)} symbols "
                            f"not covered by EDGAR/yfinance"
                        )
                        result_legacy["unified"] = unified
                        return result_legacy
            except Exception as e:
                logger.warning(f"SimFin fetch failed: {e}")

    if unified.empty:
        logger.warning("No fundamental data from any source")
        return {"unified": pd.DataFrame(), "pl": pd.DataFrame(), "bs": pd.DataFrame(), "cf": pd.DataFrame()}

    return {"unified": unified, "pl": pd.DataFrame(), "bs": pd.DataFrame(), "cf": pd.DataFrame()}


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_val(row: pd.Series, col: str) -> float | None:
    """Extract a float value from a row, returning None if missing."""
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _all_valid(*vals: float | None) -> bool:
    """Return True if all values are non-None."""
    return all(v is not None for v in vals)


def _finalize_features(features_list: list[dict]) -> pd.DataFrame:
    """Validate, winsorize, and return the feature DataFrame."""
    if not features_list:
        return pd.DataFrame()

    result = pd.DataFrame(features_list)
    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
    result = result.dropna(subset=["report_date"])
    result = result.sort_values(["symbol", "report_date"]).reset_index(drop=True)

    numeric_cols = [
        c for c in result.columns
        if c not in ("symbol", "report_date") and result[c].dtype in ("float64", "int64")
    ]
    for col in numeric_cols:
        p01 = result[col].quantile(0.01)
        p99 = result[col].quantile(0.99)
        result[col] = result[col].clip(p01, p99)

    logger.info(
        f"Computed {len(result)} fundamental feature rows, "
        f"{len(result['symbol'].unique())} symbols, "
        f"features: {[c for c in result.columns if c not in ('symbol', 'report_date')]}"
    )
    return result


def compute_fundamental_features(
    fundamentals: dict[str, pd.DataFrame],
    price_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute fundamental features from quarterly financial data.

    Handles two input formats:
      1. Unified format (from EDGAR/yfinance): dict with 'unified' key
      2. Legacy SimFin format: dict with 'pl', 'bs', 'cf' keys

    Returns DataFrame indexed by (symbol, report_date) with features:
      - accruals_ratio: (net_income - op_cash_flow) / total_assets
      - roe_trend: Δ(net_income / equity) over 4 quarters
      - earnings_momentum: (EPS_q - EPS_{q-4}) / |EPS_{q-4}|
      - revenue_surprise: QoQ revenue growth
      - gross_margin_change: QoQ gross margin delta
      - operating_leverage: dOI/OI / (dRev/Rev)
    """
    unified = fundamentals.get("unified", pd.DataFrame())
    has_legacy = any(
        not fundamentals.get(k, pd.DataFrame()).empty for k in ("pl", "bs", "cf")
    )

    if not unified.empty and has_legacy:
        # Both sources available: compute from each, merge
        unified_feats = _compute_from_unified(unified)
        legacy_feats = _compute_from_simfin_legacy(fundamentals)
        if unified_feats.empty:
            return legacy_feats
        if legacy_feats.empty:
            return unified_feats
        # Unified (EDGAR) symbols take priority; SimFin fills the rest
        edgar_syms = set(unified_feats["symbol"].unique())
        simfin_only = legacy_feats[~legacy_feats["symbol"].isin(edgar_syms)]
        if not simfin_only.empty:
            combined = pd.concat([unified_feats, simfin_only], ignore_index=True)
            logger.info(
                f"Combined fundamental features: "
                f"{len(edgar_syms)} symbols from EDGAR/yfinance, "
                f"{simfin_only['symbol'].nunique()} symbols from SimFin"
            )
            return combined
        return unified_feats

    if not unified.empty:
        return _compute_from_unified(unified)

    # Legacy SimFin path
    return _compute_from_simfin_legacy(fundamentals)


def _compute_from_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features from unified EDGAR/yfinance format.

    Each row has: symbol, period_end_date, filed_at, net_income,
    total_revenue, total_assets, stockholders_equity,
    operating_cash_flow, eps_basic, gross_profit, operating_income, ...
    """
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    pit_col = "filed_at" if "filed_at" in df.columns else "period_end_date"
    df = df.sort_values(["symbol", "period_end_date"]).reset_index(drop=True)

    features_list = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("period_end_date").reset_index(drop=True)
        if len(group) < 2:
            continue

        for idx in range(1, len(group)):
            row = group.iloc[idx]
            prev = group.iloc[idx - 1]

            feature_row = {
                "symbol": symbol,
                "report_date": row[pit_col],
            }

            ni = _safe_val(row, "net_income")
            ni_prev = _safe_val(prev, "net_income")
            rev = _safe_val(row, "total_revenue")
            rev_prev = _safe_val(prev, "total_revenue")
            ta = _safe_val(row, "total_assets")
            equity = _safe_val(row, "stockholders_equity")
            equity_prev = _safe_val(prev, "stockholders_equity")
            cfo = _safe_val(row, "operating_cash_flow")
            gp = _safe_val(row, "gross_profit")
            gp_prev = _safe_val(prev, "gross_profit")
            oi = _safe_val(row, "operating_income")
            oi_prev = _safe_val(prev, "operating_income")
            eps = _safe_val(row, "eps_basic")

            # Accruals ratio [Sloan 1996]
            if _all_valid(ni, cfo, ta) and ta != 0:
                feature_row["accruals_ratio"] = (ni - cfo) / abs(ta)

            # ROE trend: Δ(NI/equity) over last 4 quarters
            if idx >= 4:
                row_4q = group.iloc[idx - 4]
                ni_4q = _safe_val(row_4q, "net_income")
                eq_4q = _safe_val(row_4q, "stockholders_equity")
                if _all_valid(ni, equity, ni_4q, eq_4q) and equity != 0 and eq_4q != 0:
                    roe_now = ni / abs(equity)
                    roe_4q = ni_4q / abs(eq_4q)
                    feature_row["roe_trend"] = roe_now - roe_4q

            # Earnings momentum: (EPS_q - EPS_{q-4}) / |EPS_{q-4}|
            if idx >= 4:
                eps_4q = _safe_val(group.iloc[idx - 4], "eps_basic")
                if _all_valid(eps, eps_4q) and abs(eps_4q) > 0.01:
                    feature_row["earnings_momentum"] = (eps - eps_4q) / abs(eps_4q)

            # Revenue surprise (QoQ growth)
            if _all_valid(rev, rev_prev) and rev_prev != 0:
                feature_row["revenue_surprise"] = (rev - rev_prev) / abs(rev_prev)

            # Gross margin change
            if _all_valid(gp, gp_prev, rev, rev_prev) and rev != 0 and rev_prev != 0:
                gm_now = gp / rev
                gm_prev_val = gp_prev / rev_prev
                feature_row["gross_margin_change"] = gm_now - gm_prev_val

            # Operating leverage
            if _all_valid(oi, oi_prev, rev, rev_prev) and oi_prev != 0 and rev_prev != 0:
                oi_growth = (oi - oi_prev) / abs(oi_prev)
                rev_growth = (rev - rev_prev) / abs(rev_prev)
                if abs(rev_growth) > 1e-6:
                    feature_row["operating_leverage"] = oi_growth / rev_growth

            features_list.append(feature_row)

    return _finalize_features(features_list)


def _compute_from_simfin_legacy(
    fundamentals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute features from legacy SimFin pl/bs/cf format (backward compat)."""
    pl = fundamentals.get("pl", pd.DataFrame())
    bs = fundamentals.get("bs", pd.DataFrame())
    cf = fundamentals.get("cf", pd.DataFrame())

    if pl.empty:
        logger.warning("No income statement data — skipping fundamental features")
        return pd.DataFrame()

    features_list = []

    if not pl.empty and "symbol" in pl.columns:
        pl_cols = pl.columns.tolist()

        revenue_col = next(
            (c for c in ["Revenue", "Total Revenue", "Net Revenue"] if c in pl_cols),
            None,
        )
        net_income_col = next(
            (c for c in ["Net Income", "Net Income (Common)"] if c in pl_cols),
            None,
        )
        gross_profit_col = next(
            (c for c in ["Gross Profit"] if c in pl_cols), None,
        )
        operating_income_col = next(
            (c for c in ["Operating Income (Loss)", "Operating Income"] if c in pl_cols),
            None,
        )

        pit_col = "Publish Date" if "Publish Date" in pl_cols else "Report Date"
        pl_sorted = pl.sort_values(["symbol", pit_col])

        for symbol, group in pl_sorted.groupby("symbol"):
            group = group.sort_values(pit_col).reset_index(drop=True)
            if len(group) < 2:
                continue

            for idx in range(1, len(group)):
                row = group.iloc[idx]
                prev = group.iloc[idx - 1]

                feature_row = {
                    "symbol": symbol,
                    "report_date": row[pit_col],
                }

                if net_income_col:
                    eps_curr = _safe_float(pd.Series([row[net_income_col]])).iloc[0]
                    eps_prev = _safe_float(pd.Series([prev[net_income_col]])).iloc[0]
                    hist_eps = _safe_float(group[net_income_col].iloc[:idx + 1])
                    eps_std = hist_eps.std()
                    if pd.notna(eps_curr) and pd.notna(eps_prev) and eps_std > 0:
                        feature_row["sue"] = (eps_curr - eps_prev) / eps_std

                if revenue_col:
                    rev_curr = _safe_float(pd.Series([row[revenue_col]])).iloc[0]
                    rev_prev = _safe_float(pd.Series([prev[revenue_col]])).iloc[0]
                    if pd.notna(rev_curr) and pd.notna(rev_prev) and rev_prev != 0:
                        feature_row["revenue_surprise"] = (rev_curr - rev_prev) / abs(rev_prev)

                if gross_profit_col and revenue_col:
                    gp_curr = _safe_float(pd.Series([row[gross_profit_col]])).iloc[0]
                    gp_prev = _safe_float(pd.Series([prev[gross_profit_col]])).iloc[0]
                    r_curr = _safe_float(pd.Series([row[revenue_col]])).iloc[0]
                    r_prev = _safe_float(pd.Series([prev[revenue_col]])).iloc[0]
                    if all(pd.notna(v) and v != 0 for v in [r_curr, r_prev]):
                        gm_curr = gp_curr / r_curr if pd.notna(gp_curr) else np.nan
                        gm_prev_val = gp_prev / r_prev if pd.notna(gp_prev) else np.nan
                        if pd.notna(gm_curr) and pd.notna(gm_prev_val):
                            feature_row["gross_margin_change"] = gm_curr - gm_prev_val

                if operating_income_col and revenue_col:
                    oi_curr = _safe_float(pd.Series([row[operating_income_col]])).iloc[0]
                    oi_prev = _safe_float(pd.Series([prev[operating_income_col]])).iloc[0]
                    r_curr = _safe_float(pd.Series([row[revenue_col]])).iloc[0]
                    r_prev = _safe_float(pd.Series([prev[revenue_col]])).iloc[0]
                    if all(pd.notna(v) for v in [oi_curr, oi_prev, r_curr, r_prev]):
                        if oi_prev != 0 and r_prev != 0:
                            oi_growth = (oi_curr - oi_prev) / abs(oi_prev)
                            rev_growth = (r_curr - r_prev) / abs(r_prev)
                            if abs(rev_growth) > 1e-6:
                                feature_row["operating_leverage"] = oi_growth / rev_growth

                features_list.append(feature_row)

    # Accruals from BS + CF
    if not bs.empty and not cf.empty and "symbol" in bs.columns:
        bs_cols = bs.columns.tolist()
        cf_cols = cf.columns.tolist()

        total_assets_col = next(
            (c for c in ["Total Assets"] if c in bs_cols), None,
        )
        net_income_cf_col = next(
            (c for c in ["Net Income/Starting Line", "Net Income"] if c in cf_cols),
            None,
        )
        cfo_col = next(
            (c for c in [
                "Net Cash from Operating Activities",
                "Cash from Operating Activities",
            ] if c in cf_cols),
            None,
        )

        if total_assets_col and cfo_col and net_income_cf_col:
            pit_col_bs = "Publish Date" if "Publish Date" in bs_cols else "Report Date"
            pit_col_cf = "Publish Date" if "Publish Date" in cf_cols else "Report Date"

            bs_sorted = bs.sort_values(["symbol", pit_col_bs])
            cf_sorted = cf.sort_values(["symbol", pit_col_cf])

            for symbol in bs["symbol"].unique():
                bs_sym = bs_sorted[bs_sorted["symbol"] == symbol].reset_index(drop=True)
                cf_sym = cf_sorted[cf_sorted["symbol"] == symbol].reset_index(drop=True)

                if bs_sym.empty or cf_sym.empty:
                    continue

                for _, cf_row in cf_sym.iterrows():
                    cf_date = cf_row[pit_col_cf]
                    if pd.isna(cf_date):
                        continue

                    bs_match = bs_sym[bs_sym[pit_col_bs] <= cf_date]
                    if bs_match.empty:
                        continue
                    bs_row = bs_match.iloc[-1]

                    ta = _safe_float(pd.Series([bs_row[total_assets_col]])).iloc[0]
                    ni = _safe_float(pd.Series([cf_row[net_income_cf_col]])).iloc[0]
                    cfo = _safe_float(pd.Series([cf_row[cfo_col]])).iloc[0]

                    if all(pd.notna(v) for v in [ta, ni, cfo]) and ta != 0:
                        accrual = (ni - cfo) / abs(ta)
                        matched = [
                            f for f in features_list
                            if f["symbol"] == symbol
                            and f.get("report_date") is not None
                            and abs((pd.Timestamp(f["report_date"]) - pd.Timestamp(cf_date)).days) < 45
                        ]
                        if matched:
                            matched[-1]["accruals_ratio"] = accrual
                        else:
                            features_list.append({
                                "symbol": symbol,
                                "report_date": cf_date,
                                "accruals_ratio": accrual,
                            })

    return _finalize_features(features_list)


def merge_fundamental_features(
    features_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    date_col: str = "target_session_date",
) -> pd.DataFrame:
    """Point-in-time merge of fundamental features into the main feature DataFrame.

    For each (symbol, date), uses the most recent fundamental data
    available BEFORE that date (i.e., the latest published filing).
    """
    if fundamental_df.empty:
        logger.info("No fundamental data to merge")
        return features_df

    fund_cols = [
        c for c in fundamental_df.columns if c not in ("symbol", "report_date")
    ]
    if not fund_cols:
        return features_df

    result = features_df.copy()

    for symbol in result["symbol"].unique():
        sym_fund = fundamental_df[fundamental_df["symbol"] == symbol].sort_values("report_date")
        if sym_fund.empty:
            continue

        sym_mask = result["symbol"] == symbol
        sym_dates = result.loc[sym_mask, date_col]

        for col in fund_cols:
            values = []
            for d in sym_dates:
                available = sym_fund[sym_fund["report_date"] <= pd.Timestamp(d)]
                if available.empty:
                    values.append(np.nan)
                else:
                    values.append(available.iloc[-1][col])
            result.loc[sym_mask, col] = values

    n_filled = result[fund_cols].notna().sum().sum()
    logger.info(f"Merged {n_filled} fundamental feature values ({fund_cols})")
    return result
