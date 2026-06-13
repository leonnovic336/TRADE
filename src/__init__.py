"""
OMNI-TRADE: The Ultimate AI-Powered Trading Bot
Institutional-grade multi-modal AI trading system.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import load_config, get_config
from .data_sources.data_aggregator import DataAggregator
from .data_sources.advanced_data_ingester import (
    AdvancedDataAggregator,
    ClimateDataIngester,
    GeopoliticalIngester,
    DarkPoolIngester,
    HighFrequencyDataIngester,
    DataPoisoningDetector,
)
from .advanced_ai.multi_modal_engine import (
    MultiModalAIEngine,
    MultiModalSignal,
    VPINCalculator,
    HawkesProcessAnalyzer,
)
from .execution.zero_loss_executor import (
    ZeroLossExecutor,
    DeltaHedgingEngine,
    StatisticalArbitrageEngine,
    DarkPoolScanner,
)
from .trading_engine.trading_engine import TradingEngine, OrderSide
from .risk_management.risk_manager import RiskManager
from .storage.storage_manager import StorageManager, AuditLogger
from .security.security import APICredentialManager, NetworkSecurity
from .monitoring.monitoring import (
    MetricsCollector,
    PrometheusExporter,
    HealthChecker,
    KillSwitch,
)

logger = logging.getLogger(__name__)


@dataclass
class OmniTradeState:
    """Complete system state."""
    is_running: bool = False
    is_paused: bool = False
    mode: str = "paper"
    last_update: datetime = None
    errors: List[str = None]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.last_update is None:
            self.last_update = datetime.now()


class OmniTradeBot:
    """
    OMNI-TRADE AI: Institutional-Grade Trading System
    
    Features:
    - Multi-modal AI (NLP, Audio, Video, Time-Series, RL)
    - Zero-Loss Delta Hedging
    - Statistical Arbitrage
    - Dark Pool Detection
    - Climate/Geopolitical Analysis
    - Adversarial Defense
    - High-Frequency Data
    - WORM Audit Logging
    - Prometheus Monitoring
    - Kill Switch
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        
        logger.info("=" * 60)
        logger.info("🚀 OMNI-TRADE AI INITIALIZING...")
        logger.info("=" * 60)
        
        # Core components
        self.data_aggregator = DataAggregator(config_dict)
        self.advanced_data = AdvancedDataAggregator(config_dict)
        self.ai_engine = MultiModalAIEngine(config_dict)
        self.trading_engine = TradingEngine(config_dict)
        self.risk_manager = RiskManager(config_dict)
        
        # Advanced execution
        self.zero_loss_executor = ZeroLossExecutor(config_dict)
        self.stat_arb = StatisticalArbitrageEngine(config_dict)
        self.dark_pool_scanner = DarkPoolScanner(config_dict)
        
        # Storage & security
        self.storage = StorageManager(config_dict)
        self.audit = AuditLogger(config_dict.get("audit", {}).get("path", "logs/audit.db"))
        self.credentials = APICredentialManager(config_dict)
        self.network_security = NetworkSecurity(config_dict)
        
        # Monitoring
        self.metrics = MetricsCollector()
        self.prometheus = PrometheusExporter()
        self.health = HealthChecker()
        self.kill_switch = KillSwitch()
        
        # AI analysis components
        self.vpin = VPINCalculator()
        self.hawkes = HawkesProcessAnalyzer()
        self.poisoning_detector = DataPoisoningDetector(config_dict)
        
        # State
        self.state = OmniTradeState()
        self.state.mode = self.config.trading.mode if hasattr(self.config, 'trading') else 'paper'
        
        # Watchlist
        self.watchlist = self.config.trading.watchlist if hasattr(self.config, 'trading') else ["AAPL", "GOOGL", "MSFT"]
        
        # Analysis cache
        self.analysis_cache: Dict[str, MultiModalSignal] = {}
        self.cache_expiry = 300
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("All components initialized")
    
    async def initialize(self):
        """Initialize all systems."""
        logger.info("Initializing AI models...")
        await self.ai_engine.initialize()
        
        logger.info("Initializing storage...")
        await self.storage.initialize()
        
        logger.info("Initializing monitoring...")
        await self.prometheus.start()
        
        # Register health checks
        self.health.register_check("ai_engine", lambda: self.ai_engine._initialized)
        self.health.register_check("trading_engine", lambda: self.trading_engine is not None)
        self.health.register_check("kill_switch", lambda: not self.kill_switch.is_triggered())
        
        logger.info("Initialization complete")
    
    async def start(self):
        """Start the trading bot."""
        if self.state.is_running:
            logger.warning("Bot is already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING OMNI-TRADE AI")
        logger.info("=" * 60)
        
        # Initialize systems
        await self.initialize()
        
        # Start trading engine
        await self.trading_engine.start()
        
        self.state.is_running = True
        self._running = True
        
        # Log startup
        self.audit.log_decision("SYSTEM_START", {
            "mode": self.state.mode,
            "watchlist": self.watchlist,
        })
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._market_monitor_loop()),
            asyncio.create_task(self._data_ingestion_loop()),
            asyncio.create_task(self._ai_analysis_loop()),
            asyncio.create_task(self._execution_loop()),
            asyncio.create_task(self._risk_monitor_loop()),
            asyncio.create_task(self._monitoring_loop()),
        ]
        
        logger.info("OMNI-TRADE AI started successfully")
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.state.is_running:
            return
        
        logger.info("=" * 60)
        logger.info("🛑 STOPPING OMNI-TRADE AI")
        logger.info("=" * 60)
        
        self.state.is_running = False
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Stop trading
        await self.trading_engine.stop()
        
        # Wait for tasks
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Log shutdown
        self.audit.log_decision("SYSTEM_STOP", {
            "reason": "manual_shutdown",
        })
        
        logger.info("OMNI-TRADE AI stopped")
    
    async def pause(self):
        """Pause trading (keep monitoring)."""
        self.state.is_paused = True
        self.audit.log_decision("SYSTEM_PAUSE", {})
        logger.info("OMNI-TRADE AI paused - monitoring continues")
    
    async def resume(self):
        """Resume trading."""
        self.state.is_paused = False
        self.audit.log_decision("SYSTEM_RESUME", {})
        logger.info("OMNI-TRADE AI resumed")
    
    async def trigger_kill_switch(self, reason: str):
        """Trigger emergency kill switch."""
        self.kill_switch.trigger(reason)
        self.audit.log_decision("KILL_SWITCH_TRIGGERED", {"reason": reason})
        
        # Cancel all orders
        await self.trading_engine.cancel_all_orders()
        
        # Pause system
        self.state.is_paused = True
        
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
    
    # ==========================================
    # DATA INGESTION LOOPS
    # ==========================================
    
    async def _market_monitor_loop(self):
        """Monitor market conditions."""
        while self._running:
            try:
                # Check market hours
                market_open = self.trading_engine.is_market_open()
                
                # Update metrics
                self.metrics.update_portfolio(
                    self.trading_engine.portfolio.total_value,
                    self.risk_manager.current_drawdown,
                )
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market monitor: {e}")
                await asyncio.sleep(60)
    
    async def _data_ingestion_loop(self):
        """Continuously ingest data from all sources."""
        while self._running:
            try:
                for symbol in self.watchlist:
                    # Fetch comprehensive data
                    data = await self.advanced_data.fetch_comprehensive_data(symbol)
                    
                    # Store to ClickHouse/Redis
                    await self.storage.store_tick({
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "data": data,
                    })
                    
                    await asyncio.sleep(1)
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data ingestion: {e}")
                await asyncio.sleep(60)
    
    async def _ai_analysis_loop(self):
        """Run AI analysis on all symbols."""
        while self._running:
            try:
                for symbol in self.watchlist:
                    # Get market data
                    market_data = await self.data_aggregator.fetch_all_data(symbol)
                    
                    # Run multi-modal AI analysis
                    analysis = await self.ai_engine.analyze(
                        text_data=" ".join([n.headline for n in market_data.news_data[:5]]),
                        market_data=self._prepare_market_features(market_data),
                        economic_data=self._prepare_economic_features(market_data),
                        climate_data=self._prepare_climate_features(),
                        order_book=market_data.market_data.__dict__ if market_data.market_data else None,
                    )
                    
                    self.analysis_cache[symbol] = analysis
                    
                    # Update metrics
                    self.metrics.record_signal(analysis.final_signal, symbol)
                    self.metrics.update_ai_confidence(
                        analysis.final_signal,
                        symbol,
                        analysis.final_confidence
                    )
                    
                    # Check VPIN
                    if hasattr(market_data.market_data, 'volume'):
                        vpin = self.vpin.update(
                            market_data.market_data.current_price,
                            market_data.market_data.volume,
                            market_data.market_data.bid if hasattr(market_data.market_data, 'bid') else market_data.market_data.current_price,
                            market_data.market_data.ask if hasattr(market_data.market_data, 'ask') else market_data.market_data.current_price,
                        )
                        self.metrics.update_vpin(symbol, vpin)
                    
                    # Log to audit
                    self.audit.log_signal({
                        "symbol": symbol,
                        "signal": analysis.final_signal,
                        "confidence": analysis.final_confidence,
                        "risk_score": analysis.risk_score,
                    })
                    
                    await asyncio.sleep(2)
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AI analysis: {e}")
                await asyncio.sleep(60)
    
    async def _execution_loop(self):
        """Execute trades based on AI signals."""
        while self._running:
            try:
                if not self.trading_engine.is_market_open() or self.state.is_paused:
                    await asyncio.sleep(60)
                    continue
                
                if self.kill_switch.is_triggered():
                    await asyncio.sleep(300)
                    continue
                
                for symbol in self.watchlist:
                    analysis = self.analysis_cache.get(symbol)
                    
                    if not analysis:
                        continue
                    
                    # High confidence signal required
                    if analysis.final_confidence < 0.75:
                        continue
                    
                    # Execute based on signal
                    if analysis.final_signal == 1:  # Buy
                        await self._execute_buy(symbol, analysis)
                    elif analysis.final_signal == 2:  # Sell
                        await self._execute_sell(symbol, analysis)
                
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution: {e}")
                await asyncio.sleep(60)
    
    async def _execute_buy(self, symbol: str, analysis: MultiModalSignal):
        """Execute buy order with delta hedge."""
        if self.trading_engine.positions.get(symbol):
            return  # Already have position
        
        # Validate with risk manager
        rec = self.risk_manager.validate_and_size_trade(
            symbol=symbol,
            side="buy",
            quantity=100,
            price=analysis.raw_predictions.get("entry_price", 100),
            portfolio_value=self.trading_engine.portfolio.total_value,
            cash=self.trading_engine.portfolio.cash,
            current_positions=[
                {"symbol": p.symbol, "market_value": p.market_value}
                for p in self.trading_engine.positions.values()
            ],
        )
        
        if not rec.is_approved:
            logger.info(f"Buy rejected for {symbol}: {rec.rejection_reasons}")
            return
        
        # Execute with zero-loss hedge
        result = await self.zero_loss_executor.execute_with_hedge(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=rec.adjusted_quantity,
            price=rec.entry_price,
            auto_hedge=True,
        )
        
        # Record metrics
        self.metrics.record_trade(symbol, "buy", "success")
        
        # Log trade
        self.storage.log_trade(result)
        self.audit.log_trade(result)
        
        logger.info(f"✅ BUY {symbol}: {result['quantity']} shares @ ${result['price']:.2f}")
        if result.get("hedge_placed"):
            logger.info(f"   Delta hedge placed: {result['hedge_details']}")
    
    async def _execute_sell(self, symbol: str, analysis: MultiModalSignal):
        """Execute sell order."""
        if not self.trading_engine.positions.get(symbol):
            return  # No position to sell
        
        result = await self.trading_engine.close_position(symbol, "ai_signal")
        
        if result:
            self.metrics.record_trade(symbol, "sell", "success")
            self.storage.log_trade({"symbol": symbol, "side": "sell"})
            self.audit.log_trade({"symbol": symbol, "side": "sell"})
            
            logger.info(f"✅ SELL {symbol}: closing position")
    
    async def _risk_monitor_loop(self):
        """Monitor and enforce risk controls."""
        while self._running:
            try:
                # Check stop losses
                await self.trading_engine.check_stop_losses()
                
                # Check loss limits
                breach, msg = self.risk_manager.check_loss_limits(
                    self.risk_manager.daily_pnl,
                    self.risk_manager.weekly_pnl,
                    self.risk_manager.monthly_pnl,
                    self.trading_engine.portfolio.total_value,
                )
                
                if breach:
                    logger.warning(f"⚠️ LOSS LIMIT BREACH: {msg}")
                    await self.trigger_kill_switch(msg)
                
                # Check circuit breaker
                cb, cb_msg = self.risk_manager.check_circuit_breaker(
                    self.risk_manager.daily_pnl,
                    self.trading_engine.portfolio.total_value,
                )
                
                if cb and not self.trading_engine.circuit_breaker_triggered:
                    await self.trigger_kill_switch(cb_msg)
                
                # Update metrics
                self.metrics.update_positions(len(self.trading_engine.positions))
                
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Monitor system health and export metrics."""
        while self._running:
            try:
                # Run health checks
                health_status = await self.health.run_all_checks()
                
                # Export Prometheus metrics
                metrics = await self.prometheus.get_metrics()
                
                # Periodic status report
                if datetime.now().minute % 15 == 0:
                    logger.info("=" * 40)
                    logger.info("📊 OMNI-TRADE STATUS")
                    logger.info("=" * 40)
                    logger.info(f"Portfolio: ${self.trading_engine.portfolio.total_value:,.2f}")
                    logger.info(f"Positions: {len(self.trading_engine.positions)}")
                    logger.info(f"Daily P&L: ${self.risk_manager.daily_pnl:,.2f}")
                    logger.info(f"Health: {health_status.get('status', 'unknown')}")
                    logger.info(f"Kill Switch: {'TRIGGERED' if self.kill_switch.is_triggered() else 'Inactive'}")
                    logger.info("=" * 40)
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(60)
    
    # ==========================================
    # FEATURE PREPARATION
    # ==========================================
    
    def _prepare_market_features(self, data) -> Any:
        """Prepare market features for AI model."""
        import numpy as np
        
        if not data.historical_prices.empty:
            returns = data.historical_prices['Close'].pct_change().dropna().values[-50:]
            if len(returns) < 50:
                returns = np.pad(returns, (50 - len(returns), 0))
            return returns
        return np.zeros(50)
    
    def _prepare_economic_features(self, data) -> Any:
        """Prepare economic features for AI model."""
        import numpy as np
        
        # Simple placeholder - would include GDP, CPI, rates, etc.
        features = np.zeros(50)
        features[0] = data.macro_data.federal_funds_rate if data.macro_data else 0.05
        features[1] = data.macro_data.inflation_rate if data.macro_data else 0.03
        features[2] = data.macro_data.vix if data.macro_data else 20
        return features
    
    def _prepare_climate_features(self) -> Any:
        """Prepare climate features for AI model."""
        import numpy as np
        
        # Placeholder for climate data
        features = np.zeros(50)
        return features
    
    # ==========================================
    # PUBLIC API
    # ==========================================
    
    async def analyze_symbol(self, symbol: str) -> MultiModalSignal:
        """Analyze a single symbol."""
        data = await self.data_aggregator.fetch_all_data(symbol)
        
        analysis = await self.ai_engine.analyze(
            text_data=" ".join([n.headline for n in data.news_data[:5]]),
            market_data=self._prepare_market_features(data),
        )
        
        return analysis
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        await self.trading_engine.sync_account()
        await self.trading_engine.sync_positions()
        
        return {
            "total_value": self.trading_engine.portfolio.total_value,
            "cash": self.trading_engine.portfolio.cash,
            "positions": len(self.trading_engine.positions),
            "daily_pnl": self.risk_manager.daily_pnl,
            "drawdown": self.risk_manager.current_drawdown,
            "is_running": self.state.is_running,
            "is_paused": self.state.is_paused,
            "mode": self.state.mode,
            "health": self.health.get_health_status(),
            "kill_switch": self.kill_switch.get_status(),
        }
    
    def get_metrics_prometheus(self) -> str:
        """Get Prometheus metrics."""
        return self.prometheus.collector.registry.export_prometheus()


async def run_omnit_trade(config_path: Optional[str] = None):
    """Run the OMNI-TRADE bot."""
    bot = OmniTradeBot(config_path)
    
    try:
        await bot.start()
        while bot.state.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await bot.stop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OMNI-TRADE AI Trading Bot")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", choices=["paper", "live"], help="Trading mode")
    parser.add_argument("--analyze", help="Analyze a symbol and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/omnitrade.log"),
        ]
    )
    
    import os
    os.makedirs("logs", exist_ok=True)
    
    if args.analyze:
        # Analyze mode
        async def analyze():
            bot = OmniTradeBot(args.config)
            await bot.initialize()
            result = await bot.analyze_symbol(args.analyze.upper())
            
            print("\n" + "=" * 60)
            print(f"📊 OMNI-TRADE ANALYSIS: {args.analyze.upper()}")
            print("=" * 60)
            print(f"Signal: {result.final_signal} (0=hold, 1=buy, 2=sell)")
            print(f"Confidence: {result.final_confidence:.1%}")
            print(f"Risk Score: {result.risk_score:.1%}")
            print(f"Text Sentiment: {result.text_sentiment:.2f}")
            print(f"Audio Sentiment: {result.audio_sentiment:.2f}")
            print(f"TFT Prediction: {result.tft_prediction:.2f}")
            print(f"RL Action: {result.rl_action}")
            print("\nFactor Breakdown:")
            for k, v in result.factors.items():
                print(f"  {k}: {v:.2f}")
            print("=" * 60)
        
        asyncio.run(analyze())
    else:
        asyncio.run(run_omnit_trade(args.config))


if __name__ == "__main__":
    main()