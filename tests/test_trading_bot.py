"""
Tests for the trading bot.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.config import load_config, Config
from src.data_sources import DataAggregator, MarketData, NewsData, SentimentData
from src.ai_analysis import AIAnalyzer, PredictionResult, TradeSignal
from src.trading_engine import TradingEngine, OrderSide, OrderType
from src.risk_management import RiskManager, RiskLimits, TradeRecommendation


class TestConfig:
    """Test configuration loading."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = load_config("nonexistent.yaml")
        assert config is not None
        assert config.trading.mode == "paper"
    
    def test_config_trading_params(self):
        """Test trading parameters in config."""
        config = Config()
        assert config.trading.max_position_size == 0.1
        assert config.trading.autonomous_trading == True


class TestDataAggregator:
    """Test data aggregator."""
    
    @pytest.mark.asyncio
    async def test_fetch_market_data(self):
        """Test fetching market data."""
        config = {"data_sources": {}, "data": {"enabled_sources": ["yahoo_finance"]}}
        aggregator = DataAggregator(config)
        
        # Mock yfinance
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.info = {
                "currentPrice": 150.0,
                "open": 149.0,
                "dayHigh": 151.0,
                "dayLow": 148.0,
                "volume": 1000000,
                "marketCap": 1000000000,
            }
            mock_instance.history.return_value = Mock()
            mock_ticker.return_value = mock_instance
            
            data = await aggregator.fetch_market_data("AAPL")
            
            assert data.symbol == "AAPL"
            assert data.current_price == 150.0


class TestAIAnalyzer:
    """Test AI analyzer."""
    
    def test_generate_signal(self):
        """Test signal generation."""
        config = {
            "ai": {
                "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
                "analysis_weights": {
                    "technical": 0.25,
                    "sentiment": 0.25,
                    "macro": 0.25,
                    "news": 0.25,
                },
                "min_prediction_confidence": 0.70,
                "high_confidence_threshold": 0.85,
            }
        }
        
        analyzer = AIAnalyzer(config)
        
        # Create mock result
        result = PredictionResult(symbol="AAPL")
        result.overall_score = 0.75
        result.confidence = 0.90
        
        # Test signal generation
        signal, strength = analyzer._generate_signal(result)
        
        assert signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY, TradeSignal.HOLD]
        assert 0 <= strength <= 1


class TestRiskManager:
    """Test risk manager."""
    
    def test_validate_trade(self):
        """Test trade validation."""
        limits = RiskLimits()
        manager = RiskManager({"risk_management": {}})
        
        # Test valid trade
        recommendation = manager.validate_and_size_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
            portfolio_value=100000,
            cash=50000,
            current_positions=[],
        )
        
        assert recommendation.is_approved
        assert recommendation.stop_loss > 0
        assert recommendation.take_profit > recommendation.stop_loss
    
    def test_reject_large_position(self):
        """Test rejection of oversized positions."""
        manager = RiskManager({"risk_management": {}})
        
        # Position would be 50% of portfolio (exceeds 10% limit)
        recommendation = manager.validate_and_size_trade(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            price=150.0,
            portfolio_value=100000,
            cash=200000,
            current_positions=[],
        )
        
        assert not recommendation.is_approved
        assert len(recommendation.rejection_reasons) > 0
    
    def test_check_loss_limits(self):
        """Test loss limit checking."""
        manager = RiskManager({
            "risk_management": {
                "max_daily_loss": 0.05,
                "max_weekly_loss": 0.10,
                "max_monthly_loss": 0.20,
            }
        })
        
        # Test within limits
        breach, message = manager.check_loss_limits(
            daily_pnl=-300,
            weekly_pnl=-500,
            monthly_pnl=-1000,
            portfolio_value=100000,
        )
        
        assert not breach
        
        # Test breach
        breach, message = manager.check_loss_limits(
            daily_pnl=-6000,
            weekly_pnl=-500,
            monthly_pnl=-1000,
            portfolio_value=100000,
        )
        
        assert breach
        assert "Daily loss" in message


class TestTradingEngine:
    """Test trading engine."""
    
    @pytest.mark.asyncio
    async def test_place_order(self):
        """Test order placement."""
        config = {
            "broker": {"enabled_brokers": []},
            "trading": {
                "mode": "paper",
                "autonomous_trading": True,
                "min_confidence_threshold": 0.75,
            },
            "risk_management": {
                "max_open_positions": 10,
                "min_trade_size": 100,
            },
        }
        
        engine = TradingEngine(config)
        engine.portfolio.total_value = 100000
        engine.portfolio.cash = 100000
        
        order = await engine.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
        )
        
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.status.value in ["pending", "submitted"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker."""
        config = {
            "broker": {"enabled_brokers": []},
            "trading": {"mode": "paper"},
            "risk_management": {"circuit_breaker_threshold": 0.03},
        }
        
        engine = TradingEngine(config)
        
        # Trigger circuit breaker
        engine.trigger_circuit_breaker("Test trigger")
        
        assert engine.circuit_breaker_triggered
        
        # Order should be blocked
        order = await engine.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
        )
        
        assert order is None


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        # This would be a full integration test
        # Skipping for unit test suite
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])