"""Market calendar service using pandas_market_calendars.

Resolves trading sessions, holidays, early closes, and next valid trading dates
for NYSE and NASDAQ exchanges. All times are in UTC.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal
from sqlalchemy.orm import Session

from src.models.schema import MarketCalendar


def get_exchange_calendar(exchange: str = "NYSE") -> mcal.MarketCalendar:
    return mcal.get_calendar(exchange)


def build_calendar_range(
    start_date: date,
    end_date: date,
    exchange: str = "NYSE",
) -> list[dict]:
    """Build calendar entries for a date range.

    Returns list of dicts ready to upsert into MarketCalendar table.
    """
    cal = get_exchange_calendar(exchange)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)

    early_closes = cal.early_closes(schedule).index
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")

    valid_sessions = set(schedule.index.date)
    entries = []

    for d in all_dates:
        d_date = d.date()
        is_holiday = d_date not in valid_sessions

        if is_holiday:
            entries.append(
                {
                    "exchange": exchange,
                    "session_date": d_date,
                    "open_time_utc": None,
                    "close_time_utc": None,
                    "early_close_flag": False,
                    "is_holiday": True,
                    "next_trading_date": None,
                }
            )
        else:
            row = schedule.loc[pd.Timestamp(d_date)]
            entries.append(
                {
                    "exchange": exchange,
                    "session_date": d_date,
                    "open_time_utc": row["market_open"].to_pydatetime(),
                    "close_time_utc": row["market_close"].to_pydatetime(),
                    "early_close_flag": d in early_closes,
                    "is_holiday": False,
                    "next_trading_date": None,
                }
            )

    _fill_next_trading_dates(entries)
    return entries


def _fill_next_trading_dates(entries: list[dict]) -> None:
    """Backfill next_trading_date for each entry."""
    trading_dates = sorted(
        [e["session_date"] for e in entries if not e["is_holiday"]]
    )

    for entry in entries:
        current = entry["session_date"]
        next_dates = [d for d in trading_dates if d > current]
        entry["next_trading_date"] = next_dates[0] if next_dates else None


def sync_calendar_to_db(
    db: Session,
    start_date: date,
    end_date: date,
    exchange: str = "NYSE",
) -> int:
    """Build and upsert calendar entries into the database.

    Returns count of rows upserted.
    """
    entries = build_calendar_range(start_date, end_date, exchange)
    count = 0

    for entry in entries:
        existing = (
            db.query(MarketCalendar)
            .filter(
                MarketCalendar.exchange == entry["exchange"],
                MarketCalendar.session_date == entry["session_date"],
            )
            .first()
        )

        if existing:
            for key, value in entry.items():
                setattr(existing, key, value)
        else:
            db.add(MarketCalendar(**entry))

        count += 1

    db.commit()
    return count


def get_next_trading_date(db: Session, from_date: date, exchange: str = "NYSE") -> date | None:
    """Get next valid trading date after from_date."""
    entry = (
        db.query(MarketCalendar)
        .filter(
            MarketCalendar.exchange == exchange,
            MarketCalendar.session_date > from_date,
            MarketCalendar.is_holiday.is_(False),
        )
        .order_by(MarketCalendar.session_date)
        .first()
    )
    return entry.session_date if entry else None


def is_trading_day(db: Session, check_date: date, exchange: str = "NYSE") -> bool:
    """Check if a date is a valid trading session."""
    entry = (
        db.query(MarketCalendar)
        .filter(
            MarketCalendar.exchange == exchange,
            MarketCalendar.session_date == check_date,
        )
        .first()
    )
    return entry is not None and not entry.is_holiday


def get_trading_days_between(
    db: Session, start: date, end: date, exchange: str = "NYSE"
) -> list[date]:
    """Get all trading days in a range."""
    entries = (
        db.query(MarketCalendar.session_date)
        .filter(
            MarketCalendar.exchange == exchange,
            MarketCalendar.session_date >= start,
            MarketCalendar.session_date <= end,
            MarketCalendar.is_holiday.is_(False),
        )
        .order_by(MarketCalendar.session_date)
        .all()
    )
    return [e.session_date for e in entries]
