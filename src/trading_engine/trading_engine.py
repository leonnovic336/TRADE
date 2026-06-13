"""
Trading Engine - Order Execution and Broker Integration
Handles order placement, position management, and broker APIs.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import aiohttp

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill


@dataclass
class Order:
    """Order representation."""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    filled_quantity: float = 0.0
    limit_price: float = 0.0
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Execution
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    # Strategy info
    strategy_id: str = ""
    signal_id: str = ""
    
    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status)
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force)
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]


@dataclass
class Position:
    """Position representation."""
    symbol: str = ""
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    
    # Stop loss / take profit
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Timing
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Status
    is_long: bool = True
    
    def update(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        
        cost_basis = self.quantity * self.avg_entry_price
        self.unrealized_pnl = self.market_value - cost_basis
        
        if cost_basis > 0:
            self.unrealized_pnl_percent = (self.unrealized_pnl / cost_basis) * 100
        
        self.updated_at = datetime.now()
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0
    
    @property
    def should_stop_loss(self) -> bool:
        if self.stop_loss <= 0:
            return False
        if self.is_long:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    @property
    def should_take_profit(self) -> bool:
        if self.take_profit <= 0:
            return False
        if self.is_long:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit


@dataclass
class Trade:
    """Completed trade representation."""
    trade_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Trade metadata
    strategy_id: str = ""
    signal_id: str = ""
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class Portfolio:
    """Portfolio representation."""
    total_value: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Performance
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    
    # Limits
    max_position_size: float = 0.1
    max_total_exposure: float = 0.8
    
    # Timing
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_metrics(self):
        """Calculate portfolio metrics."""
        self.equity = sum(p.market_value for p in self.positions.values())
        self.total_value = self.cash + self.equity
        
        self.total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        for position in self.positions.values():
            position.update(position.current_price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self.positions
    
    def can_open_position(self, symbol: str, size: float, price: float) -> bool:
        """Check if new position can be opened."""
        position_value = size * price
        position_percent = position_value / max(self.total_value, 1)
        
        # Check position size limit
        if position_percent > self.max_position_size:
            return False
        
        # Check total exposure
        total_exposure = sum(p.market_value for p in self.positions.values()) / max(self.total_value, 1)
        if total_exposure + position_percent > self.max_total_exposure:
            return False
        
        # Check cash
        if position_value > self.cash:
            return False
        
        return True


class AlpacaBroker:
    """
    Alpaca trading platform integration.
    Supports both paper trading and live trading.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type": "application/json"
        }
        
        logger.info(f"Alpaca broker initialized with base URL: {base_url}")
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        url = f"{self.base_url}/v2/account"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Error getting account: {error}")
                    return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        url = f"{self.base_url}/v2/positions"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return []
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        url = f"{self.base_url}/v2/positions/{symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return None
    
    async def get_orders(
        self,
        status: str = "all",
        limit: int = 50,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get orders with optional filters."""
        params = f"status={status}&limit={limit}"
        if symbols:
            params += f"&symbols={','.join(symbols)}"
        
        url = f"{self.base_url}/v2/orders?{params}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return []
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Optional[Dict[str, Any]]:
        """Place a new order."""
        url = f"{self.base_url}/v2/orders"
        
        order_data = {
            "symbol": symbol,
            "side": side,
            "qty": str(quantity),
            "type": order_type,
            "time_in_force": time_in_force,
        }
        
        if limit_price:
            order_data["limit_price"] = str(limit_price)
        if stop_price:
            order_data["stop_price"] = str(stop_price)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=order_data, headers=self.headers) as resp:
                if resp.status in [200, 201]:
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Error placing order: {error}")
                    return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        url = f"{self.base_url}/v2/orders/{order_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=self.headers) as resp:
                return resp.status in [200, 204]
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        url = f"{self.base_url}/v2/orders"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=self.headers) as resp:
                return resp.status in [200, 204]
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bar data."""
        params = f"timeframe={timeframe}&limit={limit}"
        if start:
            params += f"&start={start}"
        if end:
            params += f"&end={end}"
        
        url = f"{self.base_url}/v2/stocks/{symbol}/bars?{params}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("bars", [])
                else:
                    return []


class TradingEngine:
    """
    Main trading engine that orchestrates order execution and portfolio management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize broker
        broker_config = config.get("broker", {})
        self.mode = config.get("trading", {}).get("mode", "paper")
        
        if "alpaca" in broker_config.get("enabled_brokers", []):
            alpaca_config = broker_config.get("alpaca", {})
            self.broker = AlpacaBroker(
                api_key=alpaca_config.get("api_key", ""),
                api_secret=alpaca_config.get("api_secret", ""),
                base_url=alpaca_config.get("base_url", "https://paper-api.alpaca.markets"),
            )
        else:
            self.broker = None
        
        # Portfolio state
        self.portfolio = Portfolio()
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Trading parameters
        self.max_positions = config.get("risk_management", {}).get("max_open_positions", 10)
        self.min_trade_size = config.get("risk_management", {}).get("min_trade_size", 100)
        self.autonomous = config.get("trading", {}).get("autonomous_trading", True)
        self.min_confidence = config.get("trading", {}).get("min_confidence_threshold", 0.75)
        
        # Watchlist
        self.watchlist = config.get("trading", {}).get("watchlist", [])
        
        # Trading hours
        self.trading_hours = config.get("trading", {}).get("trading_hours", {})
        
        # State
        self.is_trading = False
        self.circuit_breaker_triggered = False
        
        logger.info(f"Trading engine initialized in {self.mode} mode")
    
    async def start(self):
        """Start the trading engine."""
        logger.info("Starting trading engine...")
        
        # Sync with broker
        await self.sync_positions()
        await self.sync_account()
        
        self.is_trading = True
        logger.info("Trading engine started")
    
    async def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self.is_trading = False
        
        # Cancel pending orders
        await self.cancel_all_orders()
        
        logger.info("Trading engine stopped")
    
    async def sync_positions(self):
        """Sync positions with broker."""
        if not self.broker:
            return
        
        try:
            broker_positions = await self.broker.get_positions()
            
            # Clear and update local positions
            self.positions.clear()
            
            for bp in broker_positions:
                position = Position(
                    symbol=bp.get("symbol", ""),
                    quantity=float(bp.get("qty", 0)),
                    avg_entry_price=float(bp.get("avg_entry_price", 0)),
                    current_price=float(bp.get("current_price", 0)),
                    is_long=bp.get("side", "") == "long",
                )
                position.update(position.current_price)
                self.positions[position.symbol] = position
            
            logger.info(f"Synced {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    async def sync_account(self):
        """Sync account with broker."""
        if not self.broker:
            self.portfolio.total_value = 100000  # Default for backtesting
            self.portfolio.cash = 100000
            return
        
        try:
            account = await self.broker.get_account()
            
            self.portfolio.total_value = float(account.get("equity", 0))
            self.portfolio.cash = float(account.get("cash", 0))
            
            logger.info(f"Account synced: ${self.portfolio.cash:.2f} cash, ${self.portfolio.total_value:.2f} equity")
        except Exception as e:
            logger.error(f"Error syncing account: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        
        # Simple check for US market hours (Mon-Fri, 9:30 AM - 4:00 PM ET)
        if now.weekday() >= 5:  # Weekend
            return False
        
        start_hour, start_min = map(int, self.trading_hours.get("start", "09:30").split(":"))
        end_hour, end_min = map(int, self.trading_hours.get("end", "16:00").split(":"))
        
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        
        return start_minutes <= current_minutes <= end_minutes
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Order]:
        """Place a new order."""
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            logger.warning("Circuit breaker triggered - orders disabled")
            return None
        
        # Check trading hours
        if not self.is_market_open():
            logger.warning(f"Market closed - cannot place order for {symbol}")
            return None
        
        # Check position limits
        if len(self.positions) >= self.max_positions and side == OrderSide.BUY:
            logger.warning("Maximum positions reached")
            return None
        
        # Check minimum trade size
        estimated_value = quantity * (limit_price or 0)
        if estimated_value < self.min_trade_size:
            logger.warning(f"Trade size ${estimated_value:.2f} below minimum ${self.min_trade_size}")
            return None
        
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price or 0,
            stop_price=stop_price or 0,
            stop_loss=stop_loss or 0,
            take_profit=take_profit or 0,
        )
        
        # Place with broker
        if self.broker:
            broker_order = await self.broker.place_order(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                order_type=order_type.value,
                limit_price=limit_price,
                stop_price=stop_price,
            )
            
            if broker_order:
                order.order_id = broker_order.get("id", "")
                order.status = OrderStatus.SUBMITTED
            else:
                order.status = OrderStatus.REJECTED
                return order
        
        self.orders[order.order_id] = order
        logger.info(f"Order placed: {side.value.upper()} {quantity} {symbol}")
        
        return order
    
    async def execute_signal(self, signal) -> Optional[Order]:
        """
        Execute a trading signal from AI analysis.
        """
        if not signal:
            return None
        
        # Check confidence threshold
        if hasattr(signal, 'confidence') and signal.confidence < self.min_confidence:
            logger.info(f"Signal confidence {signal.confidence:.2%} below threshold {self.min_confidence:.2%}")
            return None
        
        # Get signal details
        symbol = signal.symbol if hasattr(signal, 'symbol') else None
        if not symbol:
            return None
        
        # Determine action
        signal_type = signal.signal.value if hasattr(signal, 'signal') else "hold"
        
        if signal_type == "hold":
            return None
        
        # Get current position
        current_position = self.positions.get(symbol)
        
        # Calculate position size
        if hasattr(signal, 'position_size_recommendation'):
            position_value = self.portfolio.total_value * signal.position_size_recommendation
        else:
            position_value = self.portfolio.total_value * 0.05  # Default 5%
        
        # Get current price
        current_price = signal.entry_price if hasattr(signal, 'entry_price') and signal.entry_price > 0 else 100
        quantity = position_value / current_price
        
        # Round to whole shares
        quantity = int(quantity)
        
        if quantity <= 0:
            logger.warning(f"Position size too small for {symbol}")
            return None
        
        # Determine order side and parameters
        if signal_type in ["buy", "strong_buy"]:
            side = OrderSide.BUY
            stop_loss = signal.stop_loss if hasattr(signal, 'stop_loss') and signal.stop_loss > 0 else current_price * 0.98
            take_profit = signal.take_profit if hasattr(signal, 'take_profit') and signal.take_profit > 0 else current_price * 1.04
        else:  # sell or strong_sell
            side = OrderSide.SELL
            stop_loss = signal.stop_loss if hasattr(signal, 'stop_loss') and signal.stop_loss > 0 else current_price * 1.02
            take_profit = signal.take_profit if hasattr(signal, 'take_profit') and signal.take_profit > 0 else current_price * 0.96
        
        # Place order
        order = await self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        if order:
            order.strategy_id = "ai_signal"
            order.signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            order.confidence = signal.confidence if hasattr(signal, 'confidence') else 0
        
        return order
    
    async def close_position(self, symbol: str, reason: str = "manual") -> Optional[Order]:
        """Close an existing position."""
        position = self.positions.get(symbol)
        
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None
        
        # Determine close side
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY
        
        order = await self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
        )
        
        if order:
            logger.info(f"Closing position {symbol}: {reason}")
        
        return order
    
    async def check_stop_losses(self):
        """Check and execute stop losses for all positions."""
        for symbol, position in list(self.positions.items()):
            if position.should_stop_loss:
                logger.info(f"Stop loss triggered for {symbol}")
                await self.close_position(symbol, reason="stop_loss")
            elif position.should_take_profit:
                logger.info(f"Take profit triggered for {symbol}")
                await self.close_position(symbol, reason="take_profit")
    
    async def sync_orders(self):
        """Sync order status with broker."""
        if not self.broker:
            return
        
        for order_id, order in list(self.orders.items()):
            if not order.is_active:
                continue
            
            try:
                orders = await self.broker.get_orders(status="all", symbols=[order.symbol])
                
                for broker_order in orders:
                    if broker_order.get("id") == order_id:
                        status = broker_order.get("status", "").lower()
                        
                        if status == "filled":
                            order.status = OrderStatus.FILLED
                            order.filled_quantity = float(broker_order.get("filled_qty", 0))
                            order.avg_fill_price = float(broker_order.get("filled_avg_price", 0))
                            order.filled_at = datetime.now()
                            
                            # Update position
                            await self.sync_positions()
                            
                        elif status == "partially_filled":
                            order.status = OrderStatus.PARTIAL
                            order.filled_quantity = float(broker_order.get("filled_qty", 0))
                            
                        elif status in ["cancelled", "rejected", "expired"]:
                            order.status = OrderStatus(status)
                            
                        order.updated_at = datetime.now()
                        
            except Exception as e:
                logger.error(f"Error syncing order {order_id}: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        
        if not order:
            return False
        
        if self.broker:
            success = await self.broker.cancel_order(order_id)
            if success:
                order.status = OrderStatus.CANCELLED
                return True
        
        return False
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if self.broker:
            return await self.broker.cancel_all_orders()
        return False
    
    async def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        await self.sync_account()
        await self.sync_positions()
        
        self.portfolio.calculate_metrics()
        return self.portfolio.total_value
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L."""
        return self.portfolio.daily_pnl
    
    def trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to pause trading."""
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        self.circuit_breaker_triggered = True
        
        # Cancel all pending orders
        if self.broker:
            asyncio.create_task(self.cancel_all_orders())
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker."""
        logger.info("Circuit breaker reset")
        self.circuit_breaker_triggered = False
    
    def get_trade_history(self, days: int = 30) -> List[Trade]:
        """Get trade history."""
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in self.trades if t.timestamp >= cutoff]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate trading report."""
        return {
            "timestamp": datetime.now(),
            "portfolio_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "positions": len(self.positions),
            "open_orders": len([o for o in self.orders.values() if o.is_active]),
            "total_trades": len(self.trades),
            "daily_pnl": self.portfolio.daily_pnl,
            "circuit_breaker": self.circuit_breaker_triggered,
            "is_trading": self.is_trading,
        }


def create_demo_engine() -> TradingEngine:
    """Create a demo trading engine without broker connection."""
    config = {
        "broker": {"enabled_brokers": []},
        "trading": {
            "mode": "demo",
            "watchlist": ["AAPL", "GOOGL", "MSFT"],
            "autonomous_trading": True,
            "min_confidence_threshold": 0.75,
        },
        "risk_management": {
            "max_open_positions": 10,
            "min_trade_size": 100,
        },
    }
    return TradingEngine(config)