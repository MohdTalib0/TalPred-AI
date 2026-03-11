from datetime import date

from src.calendar.service import build_calendar_range


def test_build_calendar_range_has_entries():
    entries = build_calendar_range(date(2024, 1, 1), date(2024, 1, 31), "NYSE")
    assert len(entries) > 0


def test_calendar_entries_have_required_fields():
    entries = build_calendar_range(date(2024, 1, 2), date(2024, 1, 5), "NYSE")
    required_fields = {
        "exchange",
        "session_date",
        "open_time_utc",
        "close_time_utc",
        "early_close_flag",
        "is_holiday",
        "next_trading_date",
    }
    for entry in entries:
        assert required_fields.issubset(entry.keys())


def test_holidays_detected():
    """Jan 1 2024 is New Year's Day (observed) and Jan 15 is MLK Day."""
    entries = build_calendar_range(date(2024, 1, 1), date(2024, 1, 31), "NYSE")
    holidays = [e for e in entries if e["is_holiday"]]
    holiday_dates = {e["session_date"] for e in holidays}
    assert date(2024, 1, 15) in holiday_dates  # MLK Day


def test_next_trading_date_populated():
    entries = build_calendar_range(date(2024, 1, 1), date(2024, 1, 31), "NYSE")
    trading_entries = [e for e in entries if not e["is_holiday"]]
    for entry in trading_entries[:-1]:
        assert entry["next_trading_date"] is not None
        assert entry["next_trading_date"] > entry["session_date"]


def test_no_weekend_entries():
    entries = build_calendar_range(date(2024, 1, 1), date(2024, 1, 31), "NYSE")
    for entry in entries:
        assert entry["session_date"].weekday() < 5  # Mon-Fri only
