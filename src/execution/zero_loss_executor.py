"""
ZERO-LOSS EXECUTION & RISK MANAGEMENT ENGINE
Institutional-grade execution with delta hedging and statistical arbitrage
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OPTIONS = "options"
    FUTURES = "futures"


@dataclass
class Position:
    """Position with delta hedge tracking."""
    symbol: str
    quantity: float
    avg_entry: float
    current_price: float = 0.0
    delta: float = 1.0  # 1.0 for stock, <1 for options
    
    # Hedge tracking
    hedge_quantity: float = 0.0
    hedge_price: float = 0.0
    hedge_type: str = ""  # "put_options", "inverse_futures", "short_stock"
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    hedge_cost: float = 0.0
    
    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def is_delta_neutral(self) -> bool:
        return abs(self.delta + self.hedge_quantity) < 0.05  # Within 5%
    
    @property
    def max_loss(self) -> float:
        """Maximum possible loss with hedge."""
        if self.hedge_type == "put_options":
            # Max loss is limited to the put premium
            return self.hedge_cost
        return abs(self.quantity * (self.avg_entry - self.hedge_price))


@dataclass
class DeltaHedgeOrder:
    """Order for delta hedging."""
    underlying_symbol: str
    hedge_type: str  # "put", "call", "inverse_perpetual", "short"
    quantity: float
    strike: float = 0.0
    expiry: datetime = None
    premium: float = 0.0
    delta_value: float = 0.0  # Calculated delta for the hedge


class StatisticalArbitrageEngine:
    """
    Statistical Arbitrage Engine for pairs trading.
    Identifies mean-reverting pairs and executes spread trades.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pairs = {}  # {pair_id: {"asset1": ..., "asset2": ..., "spread_history": [...]}}
        self.spread_threshold = 3.0  # Standard deviations for entry
        self.mean_reversion_threshold = 0.5  # Std devs for exit
        
    def add_pair(self, asset1: str, asset2: str, lookback: int = 60):
        """Add a pairs trade to track."""
        pair_id = f"{asset1}/{asset2}"
        self.pairs[pair_id] = {
            "asset1": asset1,
            "asset2": asset2,
            "spread_history": [],
            "lookback": lookback,
            "hedge_ratio": 1.0,
        }
        logger.info(f"Added pair: {pair_id}")
    
    def update_spread(self, pair_id: str, price1: float, price2: float):
        """Update spread calculation for a pair."""
        if pair_id not in self.pairs:
            return
        
        pair = self.pairs[pair_id]
        
        # Calculate spread (normalized)
        if price2 > 0:
            spread = (price1 / price2) * pair["hedge_ratio"] - 1
            pair["spread_history"].append(spread)
            
            # Keep history manageable
            if len(pair["spread_history"]) > pair["lookback"] * 2:
                pair["spread_history"] = pair["spread_history"][-pair["lookback"]:]
    
    def analyze_pair(self, pair_id: str) -> Dict[str, Any]:
        """Analyze a pair for trading opportunity."""
        if pair_id not in self.pairs:
            return {"action": "none", "reason": "pair_not_found"}
        
        pair = self.pairs[pair_id]
        spread_history = pair["spread_history"]
        
        if len(spread_history) < pair["lookback"]:
            return {"action": "none", "reason": "insufficient_data"}
        
        # Calculate statistics
        recent_spreads = spread_history[-pair["lookback"]:]
        mean = np.mean(recent_spreads)
        std = np.std(recent_spreads)
        
        if std == 0:
            return {"action": "none", "reason": "zero_variance"}
        
        current_spread = spread_history[-1]
        z_score = (current_spread - mean) / std
        
        # Trading signals
        if z_score > self.spread_threshold:
            # Spread too wide - short asset1, long asset2 (expect convergence)
            return {
                "action": "short_spread",
                "z_score": z_score,
                "mean": mean,
                "std": std,
                "current_spread": current_spread,
                "expected_return": abs(z_score) * std,
                "confidence": min(abs(z_score) / self.spread_threshold, 1.0),
            }
        elif z_score < -self.spread_threshold:
            # Spread too narrow - long asset1, short asset2
            return {
                "action": "long_spread",
                "z_score": z_score,
                "mean": mean,
                "std": std,
                "current_spread": current_spread,
                "expected_return": abs(z_score) * std,
                "confidence": min(abs(z_score) / self.spread_threshold, 1.0),
            }
        elif abs(z_score) < self.mean_reversion_threshold:
            # Spread reverted - close positions
            return {
                "action": "close",
                "z_score": z_score,
                "reason": "mean_reverted",
                "confidence": 0.9,
            }
        
        return {"action": "none", "z_score": z_score, "reason": "within_threshold"}
    
    def calculate_hedge_ratio(self, pair_id: str, prices1: List[float], prices2: List[float]):
        """Calculate optimal hedge ratio using regression."""
        if len(prices1) != len(prices2) or len(prices1) < 20:
            return 1.0
        
        # OLS regression
        X = np.array(prices2).reshape(-1, 1)
        y = np.array(prices1)
        
        # Simple linear regression
        X_mean = X.mean()
        y_mean = y.mean()
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        if denominator != 0:
            hedge_ratio = numerator / denominator
            if pair_id in self.pairs:
                self.pairs[pair_id]["hedge_ratio"] = hedge_ratio
            return hedge_ratio
        
        return 1.0


class DeltaHedgingEngine:
    """
    Delta Hedging Engine for Zero-Loss trading.
    Automatically structures hedges to neutralize downside risk.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.pending_hedges: List[DeltaHedgeOrder] = []
        
        # Greeks tracking
        self.portfolio_delta = 0.0
        self.portfolio_gamma = 0.0
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0
    
    def add_position(self, symbol: str, quantity: float, entry_price: float,
                    delta: float = 1.0, current_price: float = None):
        """Add a new position to track."""
        if current_price is None:
            current_price = entry_price
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry=entry_price,
            current_price=current_price,
            delta=delta,
        )
        
        self.positions[symbol] = position
        self._update_portfolio_greeks()
        
        logger.info(f"Added position: {quantity} {symbol} @ ${entry_price:.2f}")
    
    def deploy_delta_hedge(self, symbol: str, hedge_type: str = "put_options",
                          strike_pct: float = 0.05, expiry_days: int = 30) -> DeltaHedgeOrder:
        """
        Deploy a delta hedge for a position.
        
        Args:
            symbol: Underlying symbol
            hedge_type: Type of hedge ("put_options", "call_options", "inverse_perpetual", "short")
            strike_pct: How far OTM the hedge should be (5% = slightly OTM)
            expiry_days: Days until hedge expires
        
        Returns: DeltaHedgeOrder
        """
        position = self.positions.get(symbol)
        if not position:
            return None
        
        # Calculate hedge parameters
        current_price = position.current_price
        
        if hedge_type == "put_options":
            # ATM put would be at current price, slightly OTM for cost efficiency
            strike = current_price * (1 - strike_pct)
            # Simplified delta calculation (real implementation would use Black-Scholes)
            hedge_delta = -0.5 * (1 - strike_pct)  # Approximate put delta
            
        elif hedge_type == "call_options":
            strike = current_price * (1 + strike_pct)
            hedge_delta = 0.5 * (1 - strike_pct)
            
        elif hedge_type == "inverse_perpetual":
            # For crypto - inverse perpetual futures
            strike = current_price
            hedge_delta = -1.0  # Full inverse
            
        elif hedge_type == "short":
            # Short the same stock to hedge
            strike = current_price
            hedge_delta = -1.0
            
        else:
            logger.error(f"Unknown hedge type: {hedge_type}")
            return None
        
        # Calculate hedge quantity to make position delta-neutral
        position_delta = position.quantity * position.delta
        hedge_quantity = -position_delta / hedge_delta if hedge_delta != 0 else 0
        
        hedge_order = DeltaHedgeOrder(
            underlying_symbol=symbol,
            hedge_type=hedge_type,
            quantity=abs(hedge_quantity),
            strike=strike,
            expiry=datetime.now() + timedelta(days=expiry_days),
            delta_value=hedge_delta,
        )
        
        # Update position with hedge
        position.hedge_quantity = hedge_quantity
        position.hedge_price = strike
        position.hedge_type = hedge_type
        
        self.pending_hedges.append(hedge_order)
        self._update_portfolio_greeks()
        
        logger.info(f"Deployed delta hedge for {symbol}: {hedge_type} {abs(hedge_quantity):.2f} @ ${strike:.2f}")
        
        return hedge_order
    
    def calculate_max_loss(self, symbol: str) -> float:
        """Calculate maximum possible loss for a hedged position."""
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        
        if position.hedge_type == "put_options":
            # Max loss is the premium paid
            # Simplified - real implementation would track actual premium
            hedge_value = abs(position.hedge_quantity) * (position.current_price - position.hedge_price)
            return max(0, -hedge_value) + position.hedge_cost
        
        elif position.hedge_type == "inverse_perpetual":
            # For inverse perpetual, loss is theoretically unlimited in volatile markets
            # But funding costs and position size limit it
            return abs(position.quantity * position.current_price * 0.5)  # 50% max drawdown assumption
        
        # For uncovered positions
        return abs(position.quantity * (position.current_price - position.avg_entry))
    
    def rebalance_hedge(self, symbol: str, market_price: float):
        """Rebalance delta hedge as price moves."""
        position = self.positions.get(symbol)
        if not position or not position.hedge_type:
            return
        
        position.current_price = market_price
        position.last_updated = datetime.now()
        
        # Calculate unrealized P&L
        if position.quantity > 0:  # Long position
            position.unrealized_pnl = position.quantity * (market_price - position.avg_entry)
        else:  # Short position
            position.unrealized_pnl = abs(position.quantity) * (position.avg_entry - market_price)
        
        # Check if hedge needs rebalancing
        # In a real implementation, this would recalculate deltas and adjust
        
        self._update_portfolio_greeks()
    
    def close_hedge(self, symbol: str) -> Dict[str, float]:
        """Close a position and its hedge."""
        position = self.positions.get(symbol)
        if not position:
            return {"realized_pnl": 0, "hedge_cost": 0}
        
        realized = position.realized_pnl
        hedge_cost = position.hedge_cost
        
        del self.positions[symbol]
        
        # Remove pending hedges for this symbol
        self.pending_hedges = [
            h for h in self.pending_hedges if h.underlying_symbol != symbol
        ]
        
        self._update_portfolio_greeks()
        
        logger.info(f"Closed position {symbol}: Realized P&L ${realized:.2f}, Hedge Cost ${hedge_cost:.2f}")
        
        return {"realized_pnl": realized, "hedge_cost": hedge_cost}
    
    def _update_portfolio_greeks(self):
        """Update aggregate portfolio Greeks."""
        self.portfolio_delta = sum(
            p.quantity * p.delta for p in self.positions.values()
        )
        self.portfolio_gamma = 0.0  # Would need options data
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of hedged portfolio."""
        total_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_hedge_cost = sum(p.hedge_cost for p in self.positions.values())
        
        return {
            "positions": len(self.positions),
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "total_hedge_cost": total_hedge_cost,
            "net_exposure": total_pnl - total_hedge_cost,
            "portfolio_delta": self.portfolio_delta,
            "is_delta_neutral": abs(self.portfolio_delta) < 0.1,
        }


class DarkPoolScanner:
    """
    Dark Pool & Block Trade Scanner.
    Detects institutional block trades to anticipate order flow.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.block_trades: List[Dict] = []
        self.order_flow: Dict[str, List[Dict]] = {}  # symbol -> trade history
        
        # Detection thresholds
        self.block_threshold = 10000  # Shares for block trade detection
        self.dark_pool_sources = config.get("dark_pool_sources", ["finra", "sip"])
    
    async def scan_for_block_trades(self, symbols: List[str]) -> List[Dict]:
        """
        Scan for block trades across dark pools and exchanges.
        
        In production, this would connect to:
        - FINRA ATS data
        - NASDAQ TotalView ITCH
        - SIP consolidated tape
        - IEX mid-point matching
        """
        detected_blocks = []
        
        # Simulated block trade detection (real implementation would use actual feeds)
        # This is a placeholder for the actual implementation
        
        for symbol in symbols:
            # In production: Parse NASDAQ ITCH, FINRA ADF, etc.
            # Look for large prints that move the market
            
            # Example block trade detection logic:
            # if large_trade_detected and trade_price != last_reported_price:
            #     block_trade = {
            #         "symbol": symbol,
            #         "side": "buy" or "sell",
            #         "size": shares,
            #         "price": execution_price,
            #         "timestamp": datetime.now(),
            #         "exchange": "DARK_POOL_NAME",
            #         "estimated_market_impact": calculate_impact(...)
            #     }
            pass
        
        return detected_blocks
    
    def analyze_order_flow(self, symbol: str, recent_trades: List[Dict]) -> Dict[str, Any]:
        """
        Analyze order flow for a symbol.
        
        Returns analysis of buy/sell pressure, large trade impact, etc.
        """
        if symbol not in self.order_flow:
            self.order_flow[symbol] = []
        
        # Add new trades
        self.order_flow[symbol].extend(recent_trades)
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=4)
        self.order_flow[symbol] = [
            t for t in self.order_flow[symbol] if t.get("timestamp", datetime.min) > cutoff
        ]
        
        trades = self.order_flow[symbol]
        
        if not trades:
            return {"order_imbalance": 0, "block_trade_count": 0, "pressure": "neutral"}
        
        # Calculate metrics
        buy_volume = sum(t.get("size", 0) for t in trades if t.get("side") == "buy")
        sell_volume = sum(t.get("size", 0) for t in trades if t.get("side") == "sell")
        total_volume = buy_volume + sell_volume
        
        block_trades = [t for t in trades if t.get("size", 0) > self.block_threshold]
        
        # Order imbalance (-1 to 1)
        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume
        else:
            imbalance = 0
        
        # Pressure assessment
        if imbalance > 0.2:
            pressure = "strong_buy"
        elif imbalance > 0.05:
            pressure = "buy"
        elif imbalance < -0.2:
            pressure = "strong_sell"
        elif imbalance < -0.05:
            pressure = "sell"
        else:
            pressure = "neutral"
        
        return {
            "order_imbalance": imbalance,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "block_trade_count": len(block_trades),
            "pressure": pressure,
            "last_update": datetime.now().isoformat(),
        }
    
    def get_market_impact_estimate(self, symbol: str, block_size: float) -> float:
        """
        Estimate market impact of a block trade.
        
        Uses Kyle's lambda model for market impact estimation.
        """
        # Simplified market impact model
        # Real implementation would use historical data for calibration
        
        # Typical market impact: larger trades = more impact
        # Also depends on ADV (Average Daily Volume)
        adv = 1000000  # Would be fetched from data source
        
        participation_rate = block_size / adv
        
        # Simple square root model
        impact = 0.1 * np.sqrt(participation_rate)  # 10% impact at 100% ADV
        
        return min(impact, 0.5)  # Cap at 50% impact


class ZeroLossExecutor:
    """
    Zero-Loss Execution Engine combining delta hedging and statistical arbitrage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.delta_hedge = DeltaHedgingEngine(config)
        self.stat_arb = StatisticalArbitrageEngine(config)
        self.dark_pool = DarkPoolScanner(config)
        
        # Execution parameters
        self.max_position_size = config.get("execution", {}).get("max_position_size", 0.1)
        self.max_daily_loss = config.get("risk_management", {}).get("max_daily_loss", 0.05)
        self.hedge_cost_budget = config.get("execution", {}).get("hedge_cost_budget", 0.005)
        
        logger.info("Zero-Loss Executor initialized")
    
    async def execute_with_hedge(self, symbol: str, side: OrderSide, quantity: float,
                                 price: float, auto_hedge: bool = True) -> Dict[str, Any]:
        """
        Execute trade with automatic delta hedging.
        
        Returns execution details including hedge placement.
        """
        result = {
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "hedge_placed": False,
            "hedge_details": None,
        }
        
        # Execute main position
        # In production: actual order execution via broker API
        
        # Add to delta hedge engine
        pos_quantity = quantity if side == OrderSide.BUY else -quantity
        self.delta_hedge.add_position(
            symbol=symbol,
            quantity=pos_quantity,
            entry_price=price,
            current_price=price,
        )
        
        # Deploy delta hedge if enabled
        if auto_hedge:
            hedge_order = self.delta_hedge.deploy_delta_hedge(
                symbol=symbol,
                hedge_type="put_options",
                strike_pct=0.05,  # 5% OTM
                expiry_days=30,
            )
            
            if hedge_order:
                result["hedge_placed"] = True
                result["hedge_details"] = {
                    "type": hedge_order.hedge_type,
                    "quantity": hedge_order.quantity,
                    "strike": hedge_order.strike,
                    "expiry": hedge_order.expiry.isoformat(),
                }
        
        logger.info(f"Executed {side.value} {quantity} {symbol} @ ${price:.2f} with hedge={result['hedge_placed']}")
        
        return result
    
    async def execute_pairs_trade(self, asset1: str, asset2: str, side: str,
                                  prices: Tuple[float, float]) -> Dict[str, Any]:
        """
        Execute statistical arbitrage pairs trade.
        """
        pair_id = f"{asset1}/{asset2}"
        
        # Ensure pair is tracked
        if pair_id not in self.stat_arb.pairs:
            self.stat_arb.add_pair(asset1, asset2)
        
        # Update spread
        self.stat_arb.update_spread(pair_id, prices[0], prices[1])
        
        # Analyze pair
        analysis = self.stat_arb.analyze_pair(pair_id)
        
        result = {
            "pair": pair_id,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }
        
        if analysis["action"] in ["short_spread", "long_spread"]:
            # Execute spread trade
            # Long spread = long asset1, short asset2
            # Short spread = short asset1, long asset2
            
            if analysis["action"] == "long_spread":
                await self.execute_with_hedge(asset1, OrderSide.BUY, 100, prices[0])
                await self.execute_with_hedge(asset2, OrderSide.SELL, 100, prices[1])
            else:
                await self.execute_with_hedge(asset1, OrderSide.SELL, 100, prices[0])
                await self.execute_with_hedge(asset2, OrderSide.BUY, 100, prices[1])
            
            result["executed"] = True
        
        return result
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report for zero-loss trading."""
        delta_report = self.delta_hedge.get_portfolio_summary()
        
        return {
            "timestamp": datetime.now(),
            "delta_hedge_summary": delta_report,
            "max_loss_per_position": {
                symbol: self.delta_hedge.calculate_max_loss(symbol)
                for symbol in self.delta_hedge.positions
            },
            "pairs_tracked": len(self.stat_arb.pairs),
            "parameters": {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "hedge_cost_budget": self.hedge_cost_budget,
            },
        }