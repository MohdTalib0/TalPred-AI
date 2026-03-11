from src.models.schema import (
    CalibrationModel,
    FeaturesSnapshot,
    MacroSeries,
    MarketBarsDaily,
    MarketCalendar,
    ModelRegistry,
    NewsEvent,
    NewsSymbolMapping,
    PaperTrade,
    Prediction,
    QuarantineRecord,
    SimulationRun,
    Symbol,
)


def test_all_tables_have_tablename():
    models = [
        Symbol,
        MarketCalendar,
        MarketBarsDaily,
        NewsEvent,
        NewsSymbolMapping,
        MacroSeries,
        FeaturesSnapshot,
        ModelRegistry,
        CalibrationModel,
        Prediction,
        SimulationRun,
        PaperTrade,
        QuarantineRecord,
    ]
    expected_tables = {
        "symbols",
        "market_calendar",
        "market_bars_daily",
        "news_events",
        "news_symbol_mapping",
        "macro_series",
        "features_snapshot",
        "model_registry",
        "calibration_models",
        "predictions",
        "simulation_runs",
        "paper_trades",
        "quarantine",
    }
    actual_tables = {m.__tablename__ for m in models}
    assert actual_tables == expected_tables
