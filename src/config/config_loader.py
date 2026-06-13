"""
Configuration loader for the trading bot.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import time


@dataclass
class DataSourcesConfig:
    news_api_key: str = ""
    finnhub_api_key: str = ""
    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    twitter_bearer_token: str = ""
    fred_api_key: str = ""
    alternative_api_key: str = ""


@dataclass
class BrokerConfig:
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    enabled_brokers: list = field(default_factory=lambda: ["alpaca"])


@dataclass
class TradingConfig:
    mode: str = "paper"
    watchlist: list = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    max_position_size: float = 0.1
    max_total_exposure: float = 0.8
    autonomous_trading: bool = True
    min_confidence_threshold: float = 0.75


@dataclass
class RiskConfig:
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.04
    max_daily_loss: float = 0.05
    max_weekly_loss: float = 0.10
    max_drawdown: float = 0.15
    max_open_positions: int = 10
    min_trade_size: float = 100
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: float = 0.03


@dataclass
class AIConfig:
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    pattern_model: str = "auto"
    weights_technical: float = 0.25
    weights_fundamental: float = 0.20
    weights_sentiment: float = 0.25
    weights_macro: float = 0.15
    weights_news: float = 0.15
    prediction_horizon: int = 60
    lookback_period: int = 1440
    min_prediction_confidence: float = 0.70
    high_confidence_threshold: float = 0.85


@dataclass
class DataConfig:
    update_price: int = 1
    update_news: int = 5
    update_sentiment: int = 15
    update_macro: int = 60
    retention_price: int = 365
    retention_news: int = 90
    retention_trades: int = 730
    enabled_sources: list = field(default_factory=lambda: ["yahoo_finance", "finnhub", "news_api"])


@dataclass
class NotificationConfig:
    enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    email_alerts: bool = False


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False


@dataclass
class FeaturesConfig:
    options_trading: bool = False
    crypto_trading: bool = False
    forex_trading: bool = False
    futures_trading: bool = False
    mean_reversion: bool = True
    momentum: bool = True
    breakout: bool = True
    pairs_trading: bool = False
    reinforcement_learning: bool = False
    transformer_models: bool = True
    ensemble_learning: bool = True


@dataclass
class Config:
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get("TRADE_CONFIG_PATH", "config/config.yaml")
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        return Config()
    
    with open(config_file, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        return Config()
    
    # Parse nested configurations
    config = Config()
    
    if "data_sources" in raw_config:
        ds = raw_config["data_sources"]
        config.data_sources = DataSourcesConfig(
            news_api_key=ds.get("news_api_key", ""),
            finnhub_api_key=ds.get("finnhub_api_key", ""),
            alpha_vantage_api_key=ds.get("alpha_vantage_api_key", ""),
            polygon_api_key=ds.get("polygon_api_key", ""),
            twitter_bearer_token=ds.get("twitter_bearer_token", ""),
            fred_api_key=ds.get("fred_api_key", ""),
            alternative_api_key=ds.get("alternative_api_key", ""),
        )
    
    if "broker" in raw_config:
        b = raw_config["broker"]
        alpaca = b.get("alpaca", {})
        config.broker = BrokerConfig(
            api_key=alpaca.get("api_key", ""),
            api_secret=alpaca.get("api_secret", ""),
            base_url=alpaca.get("base_url", "https://paper-api.alpaca.markets"),
            enabled_brokers=b.get("enabled_brokers", ["alpaca"]),
        )
    
    if "trading" in raw_config:
        t = raw_config["trading"]
        config.trading = TradingConfig(
            mode=t.get("mode", "paper"),
            watchlist=t.get("watchlist", ["AAPL", "GOOGL", "MSFT"]),
            start_time=t.get("trading_hours", {}).get("start", "09:30"),
            end_time=t.get("trading_hours", {}).get("end", "16:00"),
            timezone=t.get("trading_hours", {}).get("timezone", "America/New_York"),
            max_position_size=t.get("max_position_size", 0.1),
            max_total_exposure=t.get("max_total_exposure", 0.8),
            autonomous_trading=t.get("autonomous_trading", True),
            min_confidence_threshold=t.get("min_confidence_threshold", 0.75),
        )
    
    if "risk_management" in raw_config:
        r = raw_config["risk_management"]
        config.risk = RiskConfig(
            default_stop_loss=r.get("default_stop_loss", 0.02),
            default_take_profit=r.get("default_take_profit", 0.04),
            max_daily_loss=r.get("max_daily_loss", 0.05),
            max_weekly_loss=r.get("max_weekly_loss", 0.10),
            max_drawdown=r.get("max_drawdown", 0.15),
            max_open_positions=r.get("max_open_positions", 10),
            min_trade_size=r.get("min_trade_size", 100),
            enable_circuit_breaker=r.get("enable_circuit_breaker", True),
            circuit_breaker_threshold=r.get("circuit_breaker_threshold", 0.03),
        )
    
    if "ai" in raw_config:
        a = raw_config["ai"]
        weights = a.get("analysis_weights", {})
        config.ai = AIConfig(
            sentiment_model=a.get("sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english"),
            pattern_model=a.get("pattern_model", "auto"),
            weights_technical=weights.get("technical", 0.25),
            weights_fundamental=weights.get("fundamental", 0.20),
            weights_sentiment=weights.get("sentiment", 0.25),
            weights_macro=weights.get("macro", 0.15),
            weights_news=weights.get("news", 0.15),
            prediction_horizon=a.get("prediction_horizon", 60),
            lookback_period=a.get("lookback_period", 1440),
            min_prediction_confidence=a.get("min_prediction_confidence", 0.70),
            high_confidence_threshold=a.get("high_confidence_threshold", 0.85),
        )
    
    if "data" in raw_config:
        d = raw_config["data"]
        freqs = d.get("update_frequencies", {})
        ret = d.get("retention", {})
        config.data = DataConfig(
            update_price=freqs.get("price", 1),
            update_news=freqs.get("news", 5),
            update_sentiment=freqs.get("sentiment", 15),
            update_macro=freqs.get("macro", 60),
            retention_price=ret.get("price_data", 365),
            retention_news=ret.get("news_data", 90),
            retention_trades=ret.get("trade_history", 730),
            enabled_sources=d.get("enabled_sources", ["yahoo_finance", "finnhub", "news_api"]),
        )
    
    if "logging" in raw_config:
        l = raw_config["logging"]
        config.log_level = l.get("level", "INFO")
        config.log_file = l.get("log_file", "logs/trading_bot.log")
        notif = l.get("notifications", {})
        config.notifications = NotificationConfig(
            enabled=notif.get("enabled", True),
            telegram_bot_token=notif.get("telegram_bot_token", ""),
            telegram_chat_id=notif.get("telegram_chat_id", ""),
            email_alerts=notif.get("email_alerts", False),
        )
    
    if "dashboard" in raw_config:
        db = raw_config["dashboard"]
        config.dashboard = DashboardConfig(
            host=db.get("host", "0.0.0.0"),
            port=db.get("port", 8050),
            debug=db.get("debug", False),
        )
    
    if "features" in raw_config:
        f = raw_config["features"]
        config.features = FeaturesConfig(
            options_trading=f.get("options_trading", False),
            crypto_trading=f.get("crypto_trading", False),
            forex_trading=f.get("forex_trading", False),
            futures_trading=f.get("futures_trading", False),
            mean_reversion=f.get("mean_reversion", True),
            momentum=f.get("momentum", True),
            breakout=f.get("breakout", True),
            pairs_trading=f.get("pairs_trading", False),
            reinforcement_learning=f.get("reinforcement_learning", False),
            transformer_models=f.get("transformer_models", True),
            ensemble_learning=f.get("ensemble_learning", True),
        )
    
    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config