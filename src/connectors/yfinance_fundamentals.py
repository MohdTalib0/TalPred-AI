"""Zero-cost fundamental data connector using yfinance.

Wraps ticker.quarterly_financials, .quarterly_balance_sheet, .quarterly_cashflow
to extract standardized quarterly financial statements.

Advantages over SimFin:
  - No API key required
  - No rate limits
  - Same data source already used for market bars

Limitations:
  - No true filing date (PIT) — uses period end + 45 days as conservative proxy
  - Coverage may be sparse for small-caps
  - Field names vary across stocks (normalized here)

For true PIT dates, use sec_edgar.py as primary source and this as fallback.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("artifacts") / "cache" / "yf_fundamentals"

_FIELD_MAP_PL = {
    "Net Income": "net_income",
    "Total Revenue": "total_revenue",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
    "Cost Of Revenue": "cost_of_revenue",
    "EBITDA": "ebitda",
    "Basic EPS": "eps_basic",
    "Diluted EPS": "eps_diluted",
}

_FIELD_MAP_BS = {
    "Total Assets": "total_assets",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    "Stockholders Equity": "stockholders_equity",
    "Total Debt": "total_debt",
    "Cash And Cash Equivalents": "cash",
}

_FIELD_MAP_CF = {
    "Operating Cash Flow": "operating_cash_flow",
    "Capital Expenditure": "capex",
    "Free Cash Flow": "free_cash_flow",
}


def _normalize_statement(
    df: pd.DataFrame,
    field_map: dict[str, str],
    symbol: str,
    statement_type: str,
) -> pd.DataFrame:
    """Convert yfinance's transposed format to normalized rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for col_date in df.columns:
        period_end = col_date.date() if hasattr(col_date, "date") else col_date
        record = {
            "symbol": symbol,
            "period_end_date": period_end,
            "filed_at": period_end + timedelta(days=45),
            "statement_type": statement_type,
        }
        for yf_name, our_name in field_map.items():
            val = df.at[yf_name, col_date] if yf_name in df.index else None
            if pd.notna(val):
                record[our_name] = float(val)

        rows.append(record)

    return pd.DataFrame(rows)


def fetch_fundamentals(
    symbols: list[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch quarterly financials for a list of symbols.

    Returns a DataFrame with columns:
        symbol, period_end_date, filed_at, statement_type,
        + normalized financial fields

    Filing date (filed_at) is estimated as period_end + 45 days.
    For true PIT dates, use SEC EDGAR connector.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for symbol in symbols:
        cache_file = CACHE_DIR / f"{symbol}.parquet"

        if use_cache and cache_file.exists():
            try:
                cached = pd.read_parquet(cache_file)
                cache_age = (
                    date.today() - pd.Timestamp(cache_file.stat().st_mtime, unit="s").date()
                ).days
                if cache_age < 7:
                    all_rows.append(cached)
                    continue
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)

            pl = _normalize_statement(
                ticker.quarterly_financials, _FIELD_MAP_PL, symbol, "income"
            )
            bs = _normalize_statement(
                ticker.quarterly_balance_sheet, _FIELD_MAP_BS, symbol, "balance_sheet"
            )
            cf = _normalize_statement(
                ticker.quarterly_cashflow, _FIELD_MAP_CF, symbol, "cash_flow"
            )

            combined = _merge_statements(pl, bs, cf, symbol)

            if not combined.empty:
                combined.to_parquet(cache_file, index=False)
                all_rows.append(combined)
                logger.debug(f"{symbol}: {len(combined)} quarterly records fetched")
            else:
                logger.debug(f"{symbol}: no fundamental data available")

        except Exception as e:
            logger.warning(f"{symbol}: yfinance fundamentals fetch failed: {e}")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def _merge_statements(
    pl: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Merge income, balance sheet, and cash flow into one row per quarter."""
    frames = []
    for df in (pl, bs, cf):
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]

    merged = frames[0]
    for other in frames[1:]:
        cols_to_use = [
            c for c in other.columns
            if c not in merged.columns or c in ("symbol", "period_end_date")
        ]
        if not cols_to_use:
            continue
        merge_cols = ["symbol", "period_end_date"]
        available_merge = [c for c in merge_cols if c in other.columns and c in merged.columns]
        if available_merge:
            merged = merged.merge(
                other[cols_to_use],
                on=available_merge,
                how="outer",
                suffixes=("", "_dup"),
            )

    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    return merged
