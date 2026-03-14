from datetime import UTC, date, datetime


def _utcnow():
    return datetime.now(UTC)


from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.db import Base


class Symbol(Base):
    __tablename__ = "symbols"

    symbol = Column(String(20), primary_key=True)
    company_name = Column(String(255), nullable=False)
    exchange = Column(String(20), nullable=False)
    sector = Column(String(100))
    market_cap = Column(Float)
    avg_daily_volume_30d = Column(Float)
    is_active = Column(Boolean, default=True)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date)

    news_mappings = relationship("NewsSymbolMapping", back_populates="symbol_ref")


class MarketCalendar(Base):
    __tablename__ = "market_calendar"

    exchange = Column(String(20), primary_key=True)
    session_date = Column(Date, primary_key=True)
    open_time_utc = Column(DateTime(timezone=True))
    close_time_utc = Column(DateTime(timezone=True))
    early_close_flag = Column(Boolean, default=False)
    is_holiday = Column(Boolean, default=False)
    next_trading_date = Column(Date)


class MarketBarsDaily(Base):
    __tablename__ = "market_bars_daily"

    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    source = Column(String(50))
    event_time = Column(DateTime(timezone=True))
    as_of_time = Column(DateTime(timezone=True), default=_utcnow)


class NewsEvent(Base):
    __tablename__ = "news_events"

    event_id = Column(String(64), primary_key=True)
    headline = Column(Text, nullable=False)
    source_name = Column(String(100))
    published_time = Column(DateTime(timezone=True), nullable=False)
    ingested_time = Column(DateTime(timezone=True), default=_utcnow)
    sentiment_score = Column(Float)
    event_tags = Column(JSONB)
    credibility_score = Column(Float)

    symbol_mappings = relationship("NewsSymbolMapping", back_populates="news_event")


class NewsSymbolMapping(Base):
    __tablename__ = "news_symbol_mapping"

    event_id = Column(String(64), ForeignKey("news_events.event_id"), primary_key=True)
    symbol = Column(String(20), ForeignKey("symbols.symbol"), primary_key=True)
    relevance_score = Column(Float)

    news_event = relationship("NewsEvent", back_populates="symbol_mappings")
    symbol_ref = relationship("Symbol", back_populates="news_mappings")


class MacroSeries(Base):
    __tablename__ = "macro_series"

    series_id = Column(String(50), primary_key=True)
    observation_date = Column(Date, primary_key=True)
    value = Column(Float)
    release_time_utc = Column(DateTime(timezone=True))
    available_at_utc = Column(DateTime(timezone=True))
    source = Column(String(50))


class FeaturesSnapshot(Base):
    __tablename__ = "features_snapshot"

    snapshot_id = Column(String(128), primary_key=True)
    symbol = Column(String(20), nullable=False)
    as_of_time = Column(DateTime(timezone=True), nullable=False)
    target_session_date = Column(Date, nullable=False)

    rsi_14 = Column(Float)
    momentum_5d = Column(Float)
    momentum_10d = Column(Float)
    momentum_20d = Column(Float)
    momentum_60d = Column(Float)
    momentum_120d = Column(Float)
    rolling_return_5d = Column(Float)
    rolling_return_20d = Column(Float)
    rolling_volatility_20d = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    short_term_reversal = Column(Float)
    sector_return_1d = Column(Float)
    sector_return_5d = Column(Float)
    sector_relative_return_1d = Column(Float)
    sector_relative_return_5d = Column(Float)
    momentum_rank_market = Column(Float)
    momentum_60d_rank_market = Column(Float)
    momentum_120d_rank_market = Column(Float)
    short_term_reversal_rank_market = Column(Float)
    volatility_rank_market = Column(Float)
    rsi_rank_market = Column(Float)
    volume_rank_market = Column(Float)
    sector_momentum_rank = Column(Float)
    volume_change_5d = Column(Float)
    volume_zscore_20d = Column(Float)
    volatility_expansion_5_20 = Column(Float)
    volume_acceleration = Column(Float)
    signed_volume_proxy = Column(Float)
    price_volume_trend = Column(Float)
    volume_imbalance_proxy = Column(Float)
    liquidity_shock_5d = Column(Float)
    vwap_deviation = Column(Float)
    benchmark_relative_return_1d = Column(Float)
    news_sentiment_24h = Column(Float)
    news_sentiment_7d = Column(Float)
    news_sentiment_std = Column(Float)
    news_positive_ratio = Column(Float)
    news_negative_ratio = Column(Float)
    news_volume = Column(Float)
    news_credibility_avg = Column(Float)
    news_present_flag = Column(Float)
    vix_level = Column(Float)
    sp500_momentum_200d = Column(Float)

    regime_label = Column(String(30))
    dataset_version = Column(String(64))


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_version = Column(String(20), primary_key=True)
    mlflow_run_id = Column(String(64))
    algorithm = Column(String(50))
    training_window_start = Column(Date)
    training_window_end = Column(Date)
    metrics = Column(JSONB)
    status = Column(String(20), default="candidate")
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class CalibrationModel(Base):
    __tablename__ = "calibration_models"

    model_version = Column(String(20), primary_key=True)
    calibration_type = Column(String(20), primary_key=True)
    training_window = Column(String(100))
    calibration_metrics = Column(JSONB)
    artifact_uri = Column(String(500))


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String(128), primary_key=True)
    symbol = Column(String(20), nullable=False)
    as_of_time = Column(DateTime(timezone=True), nullable=False)
    target_date = Column(Date, nullable=False)
    direction = Column(String(10), nullable=False)
    probability_up = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    top_factors = Column(JSONB)
    model_version = Column(String(20))
    feature_snapshot_id = Column(String(128))
    dataset_version = Column(String(64))
    cache_ttl_seconds = Column(Integer, default=86400)
    realized_direction = Column(String(10))
    realized_return = Column(Float)
    outcome_recorded_at = Column(DateTime(timezone=True))


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    run_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    starting_capital = Column(Float, nullable=False)
    min_confidence_trade = Column(Float, default=0.60)
    max_position = Column(Float, default=0.05)
    transaction_cost_bps = Column(Float, default=10.0)
    slippage_bps = Column(Float, default=5.0)
    model_version = Column(String(20))
    status = Column(String(20), default="running")
    result_metrics = Column(JSONB)


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    run_id = Column(String(64), primary_key=True)
    date = Column(Date, primary_key=True)
    symbol = Column(String(20), primary_key=True)
    weight = Column(Float)
    position_qty = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    transaction_cost = Column(Float)
    slippage_cost = Column(Float)
    daily_pnl = Column(Float)


class QuarantineRecord(Base):
    __tablename__ = "quarantine"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    record_data = Column(JSONB, nullable=False)
    failure_reason = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
