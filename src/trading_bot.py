"""
Main Trading Bot - Orchestrates all components
The ultimate AI-powered trading bot with autonomous trading capabilities.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from .config import load_config, get_config
from .data_sources.data_aggregator import DataAggregator, ComprehensiveTradeData
from .ai_analysis.ai_analyzer import AIAnalyzer, PredictionResult, TradeSignal
from .trading_engine.trading_engine import TradingEngine, OrderSide
from .risk_management.risk_manager import RiskManager, TradeRecommendation

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Current state of the trading bot."""
    is_running: bool = False
    is_paused: bool = False
    current_mode: str = "paper"  # paper, live
    last_update: datetime = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.last_update is None:
            self.last_update = datetime.now()


class TradingBot:
    """
    The Ultimate AI-Powered Trading Bot.
    
    Features:
    - Multi-source data aggregation (news, sentiment, macro, technical)
    - Advanced AI analysis (transformers, ML, pattern recognition)
    - Autonomous trading with configurable risk management
    - Real-time market monitoring and trade execution
    - Comprehensive risk controls and circuit breakers
    - Dashboard for monitoring and control
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.data_aggregator = DataAggregator(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        self.ai_analyzer = AIAnalyzer(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        self.trading_engine = TradingEngine(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        self.risk_manager = RiskManager(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        
        # State
        self.state = BotState()
        self.state.current_mode = self.config.trading.mode if hasattr(self.config, 'trading') else 'paper'
        
        # Watchlist
        self.watchlist = self.config.trading.watchlist if hasattr(self.config, 'trading') else ["AAPL", "GOOGL", "MSFT"]
        
        # Task handles
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Analysis cache
        self.analysis_cache: Dict[str, PredictionResult] = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.daily_summary: Dict[str, Any] = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading Bot initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())
    
    async def start(self):
        """Start the trading bot."""
        if self.state.is_running:
            logger.warning("Bot is already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING TRADING BOT")
        logger.info("=" * 60)
        
        # Start trading engine
        await self.trading_engine.start()
        
        self.state.is_running = True
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._market_monitor_loop()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._risk_monitor_loop()),
            asyncio.create_task(self._performance_loop()),
        ]
        
        logger.info("Trading Bot started successfully")
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.state.is_running:
            return
        
        logger.info("=" * 60)
        logger.info("🛑 STOPPING TRADING BOT")
        logger.info("=" * 60)
        
        self.state.is_running = False
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Stop trading engine
        await self.trading_engine.stop()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Trading Bot stopped")
    
    async def pause(self):
        """Pause the bot (stops new trades but keeps monitoring)."""
        self.state.is_paused = True
        logger.info("Bot paused - monitoring continues, trading halted")
    
    async def resume(self):
        """Resume the bot."""
        self.state.is_paused = False
        logger.info("Bot resumed - trading active")
    
    async def _market_monitor_loop(self):
        """Continuously monitor market conditions."""
        while self._running:
            try:
                # Check market hours
                if self.trading_engine.is_market_open():
                    if not hasattr(self, '_market_was_open') or not self._market_was_open:
                        logger.info("Market opened")
                        self._market_was_open = True
                else:
                    if hasattr(self, '_market_was_open') and self._market_was_open:
                        logger.info("Market closed")
                        self._market_was_open = False
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market monitor: {e}")
                await asyncio.sleep(60)
    
    async def _analysis_loop(self):
        """Continuously analyze all symbols in watchlist."""
        while self._running:
            try:
                for symbol in self.watchlist:
                    # Check cache
                    cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
                    
                    if cache_key in self.analysis_cache:
                        continue  # Already analyzed this minute
                    
                    # Fetch comprehensive data
                    data = await self.data_aggregator.fetch_all_data(symbol)
                    
                    # Run AI analysis
                    analysis = await self.ai_analyzer.analyze(
                        market_data=data.market_data,
                        news_data=data.news_data,
                        sentiment_data=data.sentiment_data,
                        macro_data=data.macro_data,
                        historical_prices=data.historical_prices,
                    )
                    
                    # Cache result
                    self.analysis_cache[symbol] = analysis
                    
                    # Log significant signals
                    if analysis.signal != TradeSignal.HOLD:
                        logger.info(f"📊 {symbol}: {analysis.signal.value} (confidence: {analysis.confidence:.1%})")
                    
                    # Small delay between symbols
                    await asyncio.sleep(2)
                
                # Sleep before next analysis cycle
                await asyncio.sleep(60)  # Analyze every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _trading_loop(self):
        """Execute trades based on AI signals."""
        while self._running:
            try:
                # Only trade during market hours
                if not self.trading_engine.is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Skip if paused
                if self.state.is_paused:
                    await asyncio.sleep(60)
                    continue
                
                # Check circuit breaker
                if self.trading_engine.circuit_breaker_triggered:
                    logger.warning("Circuit breaker active - skipping trading")
                    await asyncio.sleep(300)
                    continue
                
                # Analyze each symbol
                for symbol in self.watchlist:
                    analysis = self.analysis_cache.get(symbol)
                    
                    if not analysis:
                        continue
                    
                    # Check if signal warrants action
                    if analysis.signal == TradeSignal.HOLD:
                        continue
                    
                    # Check confidence threshold
                    if analysis.confidence < self.trading_engine.min_confidence:
                        continue
                    
                    # Get current position
                    current_position = self.trading_engine.positions.get(symbol)
                    
                    # Execute based on signal
                    if analysis.signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
                        if current_position is None:  # No existing position
                            await self._execute_buy(analysis)
                        elif analysis.signal == TradeSignal.STRONG_BUY:
                            # Scale in existing position
                            await self._execute_buy(analysis, scale_in=True)
                    
                    elif analysis.signal in [TradeSignal.SELL, TradeSignal.STRONG_SELL]:
                        if current_position is not None:
                            await self._execute_sell(analysis)
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
    
    async def _execute_buy(self, analysis: PredictionResult, scale_in: bool = False):
        """Execute a buy order."""
        symbol = analysis.symbol
        
        # Validate with risk manager
        recommendation = self.risk_manager.validate_and_size_trade(
            symbol=symbol,
            side="buy",
            quantity=100,  # Base quantity
            price=analysis.entry_price,
            portfolio_value=self.trading_engine.portfolio.total_value,
            cash=self.trading_engine.portfolio.cash,
            current_positions=[
                {"symbol": p.symbol, "market_value": p.market_value}
                for p in self.trading_engine.positions.values()
            ],
        )
        
        if not recommendation.is_approved:
            logger.info(f"Buy rejected for {symbol}: {recommendation.rejection_reasons}")
            return
        
        # Calculate quantity
        quantity = recommendation.adjusted_quantity
        
        if quantity <= 0:
            logger.info(f"Position size too small for {symbol}")
            return
        
        # Place order
        order = await self.trading_engine.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=1,  # Market order
            stop_loss=recommendation.stop_loss,
            take_profit=recommendation.take_profit,
        )
        
        if order:
            logger.info(f"✅ BUY ORDER PLACED: {symbol} - {quantity} shares @ ${analysis.entry_price:.2f}")
            logger.info(f"   Stop Loss: ${recommendation.stop_loss:.2f}, Take Profit: ${recommendation.take_profit:.2f}")
            logger.info(f"   Risk: ${recommendation.risk_amount:.2f} ({recommendation.risk_percent:.1%})")
            logger.info(f"   Confidence: {analysis.confidence:.1%}")
    
    async def _execute_sell(self, analysis: PredictionResult):
        """Execute a sell order."""
        symbol = analysis.symbol
        
        # Close position
        order = await self.trading_engine.close_position(symbol, reason="ai_signal")
        
        if order:
            logger.info(f"✅ SELL ORDER PLACED: {symbol} - closing position")
            logger.info(f"   Signal: {analysis.signal.value}")
            logger.info(f"   Confidence: {analysis.confidence:.1%}")
    
    async def _risk_monitor_loop(self):
        """Monitor and enforce risk controls."""
        while self._running:
            try:
                # Check stop losses
                await self.trading_engine.check_stop_losses()
                
                # Check loss limits
                loss_breach, message = self.risk_manager.check_loss_limits(
                    daily_pnl=self.risk_manager.daily_pnl,
                    weekly_pnl=self.risk_manager.weekly_pnl,
                    monthly_pnl=self.risk_manager.monthly_pnl,
                    portfolio_value=self.trading_engine.portfolio.total_value,
                )
                
                if loss_breach:
                    logger.warning(f"⚠️ LOSS LIMIT BREACH: {message}")
                    self.trading_engine.trigger_circuit_breaker(message)
                
                # Check circuit breaker condition
                cb_triggered, cb_message = self.risk_manager.check_circuit_breaker(
                    daily_pnl=self.risk_manager.daily_pnl,
                    portfolio_value=self.trading_engine.portfolio.total_value,
                )
                
                if cb_triggered and not self.trading_engine.circuit_breaker_triggered:
                    self.trading_engine.trigger_circuit_breaker(cb_message)
                
                # Update performance tracking
                portfolio_value = self.trading_engine.portfolio.total_value
                self.risk_manager.update_performance(
                    current_portfolio_value=portfolio_value,
                    daily_pnl=self.risk_manager.daily_pnl,
                    weekly_pnl=self.risk_manager.weekly_pnl,
                    monthly_pnl=self.risk_manager.monthly_pnl,
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(60)
    
    async def _performance_loop(self):
        """Track and report performance metrics."""
        while self._running:
            try:
                # Sync with broker
                await self.trading_engine.sync_account()
                await self.trading_engine.sync_positions()
                
                # Generate report
                report = self.trading_engine.generate_report()
                risk_report = self.risk_manager.get_risk_report()
                
                # Log periodic status
                if datetime.now().minute % 15 == 0:  # Every 15 minutes
                    logger.info("=" * 40)
                    logger.info("📈 PORTFOLIO STATUS")
                    logger.info("=" * 40)
                    logger.info(f"Total Value: ${report['portfolio_value']:.2f}")
                    logger.info(f"Cash: ${report['cash']:.2f}")
                    logger.info(f"Open Positions: {report['positions']}")
                    logger.info(f"Daily P&L: ${report['daily_pnl']:.2f}")
                    logger.info(f"Circuit Breaker: {'ACTIVE' if report['circuit_breaker'] else 'Inactive'}")
                    logger.info("=" * 40)
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance loop: {e}")
                await asyncio.sleep(300)
    
    async def analyze_symbol(self, symbol: str) -> PredictionResult:
        """Analyze a single symbol and return the prediction."""
        # Fetch data
        data = await self.data_aggregator.fetch_all_data(symbol)
        
        # Run analysis
        analysis = await self.ai_analyzer.analyze(
            market_data=data.market_data,
            news_data=data.news_data,
            sentiment_data=data.sentiment_data,
            macro_data=data.macro_data,
            historical_prices=data.historical_prices,
        )
        
        return analysis
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        await self.trading_engine.sync_account()
        await self.trading_engine.sync_positions()
        
        return {
            "total_value": self.trading_engine.portfolio.total_value,
            "cash": self.trading_engine.portfolio.cash,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "pnl": p.unrealized_pnl,
                    "pnl_percent": p.unrealized_pnl_percent,
                }
                for p in self.trading_engine.positions.values()
            ],
            "is_trading": self.state.is_running,
            "is_paused": self.state.is_paused,
            "mode": self.state.current_mode,
        }
    
    def get_analysis_cache(self) -> Dict[str, PredictionResult]:
        """Get cached analysis results."""
        return self.analysis_cache.copy()
    
    async def add_to_watchlist(self, symbol: str):
        """Add a symbol to the watchlist."""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            logger.info(f"Added {symbol} to watchlist")
    
    async def remove_from_watchlist(self, symbol: str):
        """Remove a symbol from the watchlist."""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            logger.info(f"Removed {symbol} from watchlist")
    
    async def manual_trade(self, symbol: str, side: str, quantity: float):
        """Execute a manual trade."""
        if not self.state.is_running:
            logger.warning("Bot not running")
            return None
        
        if self.state.is_paused:
            logger.warning("Bot is paused")
            return None
        
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        order = await self.trading_engine.place_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
        )
        
        return order


# Convenience function to create and run the bot
async def run_bot(config_path: Optional[str] = None):
    """Run the trading bot."""
    bot = TradingBot(config_path)
    
    try:
        await bot.start()
        
        # Keep running until interrupted
        while bot.state.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Trading Bot")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--mode", "-m", choices=["paper", "live"], help="Trading mode")
    parser.add_argument("--watchlist", "-w", nargs="+", help="Symbols to trade")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/trading_bot.log"),
        ]
    )
    
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run bot
    asyncio.run(run_bot(args.config))


if __name__ == "__main__":
    main()