"""
COMPREHENSIVE TRADING EDUCATION & KNOWLEDGE BASE
Based on IG Trading Education and institutional trading practices
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


# ==========================================
# ASSET CLASSES & MARKETS
# ==========================================

class AssetClass(Enum):
    """Available asset classes for trading."""
    # Equity Markets
    STOCKS = "stocks"           # Individual company shares
    ETFs = "etfs"               # Exchange-Traded Funds
    INDICES = "indices"          # Stock market indices (S&P 500, FTSE 100)
    IPO = "ipo"                  # Initial Public Offerings
    
    # Forex Markets
    FOREX = "forex"             # Currency pairs (EUR/USD, GBP/USD)
    CRYPTO = "crypto"            # Cryptocurrencies (BTC, ETH)
    
    # Commodities
    COMMODITIES = "commodities"  # Physical goods
    METALS = "metals"            # Gold, Silver, Platinum
    ENERGY = "energy"           # Oil, Natural Gas, Electricity
    AGRICULTURE = "agriculture"  # Crops, Livestock
    SOFTS = "softs"              # Coffee, Cotton, Sugar
    
    # Fixed Income
    BONDS = "bonds"             # Government & Corporate bonds
    INTEREST_RATES = "interest_rates"
    
    # Derivatives
    OPTIONS = "options"         # Options contracts
    FUTURES = "futures"         # Futures contracts
    CFD = "cfd"                 # Contracts for Difference
    SWAPS = "swaps"              # Interest rate swaps


class Market(Enum):
    """Major trading markets."""
    # US Markets
    NYSE = "nyse"                # New York Stock Exchange
    NASDAQ = "nasdaq"             # NASDAQ Stock Market
    AMEX = "amex"                # American Stock Exchange
    
    # UK Markets
    LSE = "lse"                  # London Stock Exchange
    FTSE = "ftse"                # FTSE 100, 250
    
    # European Markets
    Euronext = "euronext"       # Paris, Amsterdam, Brussels
    XETRA = "xetra"             # German Exchange
    SIX = "six"                  # Swiss Exchange
    
    # Asian Markets
    TSE = "tse"                  # Tokyo Stock Exchange
    HKEX = "hkex"                # Hong Kong Exchange
    SGX = "sgx"                  # Singapore Exchange
    ASX = "asx"                  # Australian Exchange
    NSE = "nse"                  # National Stock Exchange of India
    
    # Forex
    GLOBAL_FOREX = "global_forex"
    
    # Crypto
    COINBASE = "coinbase"
    BINANCE = "binance"
    KRAKEN = "kraken"
    
    # Commodity Exchanges
    CME = "cme"                  # Chicago Mercantile Exchange
    CBOT = "cbot"                # Chicago Board of Trade
    NYMEX = "nymex"              # New York Mercantile Exchange
    LME = "lme"                 # London Metal Exchange


@dataclass
class AssetInfo:
    """Information about a tradeable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    market: Market
    currency: str = "USD"
    
    # Trading hours (UTC)
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"
    
    # Contract details (for derivatives)
    contract_size: float = 1.0
    tick_size: float = 0.01
    max_leverage: float = 1.0
    
    # Trading parameters
    min_trade_size: float = 0.0
    max_trade_size: float = float('inf')
    typical_spread: float = 0.0
    
    # Risk parameters
    volatility: float = 0.0
    beta: float = 1.0


# ==========================================
# ORDER TYPES
# ==========================================

class OrderType(Enum):
    """Order types based on IG Trading Education."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TRAILING_STOP_LIMIT = "trailing_stop_limit"
    MARKET_IF_TOUCHED = "mit"
    BOC = "boc"
    FOK = "fok"
    IOC = "ioc"
    GTC = "gtc"
    GTD = "gtd"
    DAY = "day"
    OCO = "oco"
    OTO = "oto"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Trading order representation."""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    filled_quantity: float = 0.0
    limit_price: float = 0.0
    stop_price: float = 0.0
    trigger_price: float = 0.0
    trailing_distance: float = 0.0
    time_in_force: str = "DAY"
    expiry_date: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = ""
    reason: str = ""
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]


# ==========================================
# POSITION MANAGEMENT
# ==========================================

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Trading position representation."""
    symbol: str = ""
    side: PositionSide = PositionSide.LONG
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_value: float = 0.0
    entry_commission: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    
    def update(self, current_price: float) -> None:
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.quantity
        
        cost_basis = self.quantity * self.avg_entry_price
        self.unrealized_pnl_percent = (self.unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        self.updated_at = datetime.now()
    
    def should_stop_loss(self, current_price: float = None) -> bool:
        if current_price is None:
            current_price = self.current_price
        if self.stop_loss <= 0:
            return False
        if self.side == PositionSide.LONG:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss


# ==========================================
# TRADING STRATEGIES
# ==========================================

class StrategyType(Enum):
    MOMENTUM = "momentum"
    TREND_LINE = "trend_line"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    SUPPORT_RESISTANCE = "support_resistance"
    OVERBOUGHT_OVERSOLD = "overbought_oversold"
    CANDLE_PATTERNS = "candle_patterns"
    CHART_PATTERNS = "chart_patterns"
    PAIRS_TRADING = "pairs_trading"
    ML_PREDICTION = "ml_prediction"
    AI_ANALYSIS = "ai_analysis"


@dataclass
class Strategy:
    """Trading strategy configuration."""
    name: str
    strategy_type: StrategyType
    enabled: bool = True
    position_sizing_method: str = "fixed"
    risk_per_trade: float = 0.02
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.04
    use_trailing_stop: bool = False
    trailing_stop_percent: float = 0.02
    max_concurrent_positions: int = 5
    max_daily_trades: int = 10


# ==========================================
# TRADING KNOWLEDGE BASE
# ==========================================

class TradingKnowledgeBase:
    """Complete trading education knowledge base."""
    
    ASSET_CLASSES = {
        "stocks": {
            "name": "Stocks/Shares",
            "description": "Ownership units in a company",
            "examples": ["AAPL", "GOOGL", "MSFT"],
            "hours": "9:30 AM - 4:00 PM ET",
        },
        "forex": {
            "name": "Foreign Exchange",
            "description": "Currency pairs trading",
            "examples": ["EUR/USD", "GBP/USD", "USD/JPY"],
            "hours": "24 hours",
        },
        "commodities": {
            "name": "Commodities",
            "description": "Raw materials and primary goods",
            "examples": ["Gold", "Oil", "Natural Gas"],
            "hours": "Varies by exchange",
        },
        "indices": {
            "name": "Stock Market Indices",
            "description": "Performance of a group of stocks",
            "examples": ["S&P 500", "NASDAQ", "FTSE 100"],
            "hours": "9:30 AM - 4:00 PM ET",
        },
        "crypto": {
            "name": "Cryptocurrencies",
            "description": "Digital assets",
            "examples": ["BTC", "ETH", "XRP"],
            "hours": "24/7",
        },
        "cfds": {
            "name": "Contracts for Difference",
            "description": "Trade price movements without ownership",
            "examples": ["SPY CFD", "EUR/USD CFD"],
            "leverage": "Up to 30:1",
        },
    }
    
    TRADING_GLOSSARY = {
        "LONG": "Buying expecting price to rise",
        "SHORT": "Selling expecting price to fall",
        "SPREAD": "Bid-ask price difference",
        "PIP": "Smallest price increment",
        "STOP_LOSS": "Limit losses at specific price",
        "TAKE_PROFIT": "Secure profits at target",
        "TRAILING_STOP": "Dynamic stop following price",
        "LEVERAGE": "Trade with borrowed capital",
        "MARGIN": "Collateral for leveraged positions",
        "SWAP": "Overnight interest charge",
        "LOT": "Standard trading unit",
        "LOT_SIZE": "Units per lot",
        "RISK_REWARD": "Potential profit vs loss ratio",
        "DRAWDOWN": "Peak-to-trough decline",
        "VOLATILITY": "Price fluctuation degree",
        "LIQUIDITY": "Ease of position entry/exit",
        "SLIPPAGE": "Execution price difference",
        "FILL": "Completed trade",
        "HEDGE": "Position to reduce risk",
        "DELTA": "Option price sensitivity",
        "GAMMA": "Delta rate of change",
        "THETA": "Time decay of option",
        "VEGA": "Volatility sensitivity",
    }
    
    RISK_REWARD_GUIDELINES = {
        "conservative": {"risk": 0.01, "reward": 0.02, "ratio": 2.0},
        "moderate": {"risk": 0.02, "reward": 0.06, "ratio": 3.0},
        "aggressive": {"risk": 0.03, "reward": 0.12, "ratio": 4.0},
    }
    
    POSITION_SIZING_METHODS = {
        "fixed_fractional": "Risk fixed % of portfolio per trade",
        "kelly_criterion": "Mathematical optimal sizing: f* = (bp - q) / b",
        "volatility_adjusted": "Position = Target Risk / (ATR x Multiplier)",
    }
    
    @classmethod
    def get_all_concepts(cls) -> Dict:
        return {
            "asset_classes": cls.ASSET_CLASSES,
            "glossary": cls.TRADING_GLOSSARY,
            "risk_reward": cls.RISK_REWARD_GUIDELINES,
            "position_sizing": cls.POSITION_SIZING_METHODS,
        }


TRADING_KB = TradingKnowledgeBase()