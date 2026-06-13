#!/usr/bin/env python3
"""
TRADE - The Ultimate AI-Powered Trading Bot

A comprehensive trading system with:
- Multi-source data aggregation (news, sentiment, macro, technical)
- Advanced AI analysis (transformers, ML, pattern recognition)
- Autonomous trading with configurable risk management
- Real-time market monitoring and trade execution
- Comprehensive risk controls and circuit breakers
- Web dashboard for monitoring and control

Usage:
    python -m src.main --config config/config.yaml
    
    # Quick start with demo mode
    python -m src.main --demo
    
    # Run dashboard only
    python -m src.main --dashboard
    
    # Analyze a symbol
    python -m src.main --analyze AAPL
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading_bot import TradingBot, run_bot
from src.dashboard import run_dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading_bot.log"),
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="TRADE - AI-Powered Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start trading bot with configuration
  python -m src.main --config config/config.yaml

  # Start in demo mode (no real trading)
  python -m src.main --demo

  # Run only the dashboard
  python -m src.main --dashboard

  # Analyze a symbol without trading
  python -m src.main --analyze AAPL

  # Start in live trading mode
  python -m src.main --config config/config.yaml --mode live

Environment Variables:
  TRADE_CONFIG_PATH    Path to configuration file
  ALPACA_API_KEY       Alpaca API key
  ALPACA_API_SECRET    Alpaca API secret
  NEWS_API_KEY         NewsAPI key
  FINNHUB_API_KEY      Finnhub API key
  FRED_API_KEY         FRED API key
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default="config/config.yaml"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["paper", "live"],
        help="Trading mode (overrides config)"
    )
    
    parser.add_argument(
        "--watchlist", "-w",
        nargs="+",
        help="Symbols to trade (overrides config)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (no real trading)"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run only the dashboard"
    )
    
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze a single symbol and exit"
    )
    
    parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8050,
        help="Dashboard port (default: 8050)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Handle different modes
    if args.dashboard:
        # Run dashboard only
        logger.info("Starting TRADE Dashboard...")
        bot = TradingBot(args.config)
        
        # Start bot in background for dashboard data
        asyncio.run(bot.start())
        
        # Run dashboard
        try:
            run_dashboard(bot, host=args.dashboard_host, port=args.dashboard_port)
        finally:
            asyncio.run(bot.stop())
    
    elif args.analyze:
        # Analyze single symbol
        logger.info(f"Analyzing {args.analyze}...")
        bot = TradingBot(args.config)
        
        async def analyze_and_exit():
            analysis = await bot.analyze_symbol(args.analyze.upper())
            
            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS RESULTS FOR {args.analyze.upper()}")
            print("=" * 60)
            print(f"\nSignal: {analysis.signal.value}")
            print(f"Confidence: {analysis.confidence:.1%}")
            print(f"Overall Score: {analysis.overall_score:.1%}")
            
            print("\n📈 Score Breakdown:")
            print(f"  Technical: {analysis.technical_score:.1%}")
            print(f"  Fundamental: {analysis.fundamental_score:.1%}")
            print(f"  Sentiment: {analysis.sentiment_score:.1%}")
            print(f"  Macro: {analysis.macro_score:.1%}")
            print(f"  News: {analysis.news_score:.1%}")
            
            if analysis.detected_patterns:
                print(f"\n🔍 Detected Patterns: {', '.join(analysis.detected_patterns)}")
                print(f"   Pattern Confidence: {analysis.pattern_confidence:.1%}")
            
            print("\n💰 Trade Recommendations:")
            print(f"  Entry Price: ${analysis.entry_price:.2f}")
            print(f"  Stop Loss: ${analysis.stop_loss:.2f}")
            print(f"  Take Profit: ${analysis.take_profit:.2f}")
            print(f"  Position Size: {analysis.position_size_recommendation:.1%}")
            
            print("\n⚠️ Risk Assessment:")
            print(f"  Risk Score: {analysis.risk_score:.1%}")
            print(f"  Confidence: {analysis.confidence:.1%}")
            
            if analysis.warnings:
                print("\n⚡ Warnings:")
                for warning in analysis.warnings:
                    print(f"  - {warning}")
            
            print("\n📝 Reasoning:")
            print(f"  {analysis.reasoning}")
            
            print("\n" + "=" * 60)
            
            return analysis
        
        try:
            asyncio.run(analyze_and_exit())
        except KeyboardInterrupt:
            pass
    
    else:
        # Run full trading bot
        if args.demo:
            logger.info("Running in DEMO mode (no real trading)")
            # Override config for demo mode
            import yaml
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            config["trading"] = config.get("trading", {})
            config["trading"]["mode"] = "demo"
            config["trading"]["autonomous_trading"] = True
            config["broker"] = {"enabled_brokers": []}
            
            # Save temp config
            temp_config = Path("config/demo_config.yaml")
            temp_config.parent.mkdir(exist_ok=True)
            with open(temp_config, "w") as f:
                yaml.dump(config, f)
            args.config = str(temp_config)
        
        if args.mode:
            logger.info(f"Setting trading mode to: {args.mode}")
            # Could override config here
        
        try:
            asyncio.run(run_bot(args.config))
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()