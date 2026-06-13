# TRADE - AI-Powered Trading Bot

<div align="center">

![TRADE Logo](https://img.shields.io/badge/TRADE-AI%20Trading%20Bot-667eea?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**The Ultimate AI-Powered Trading Bot** - A comprehensive system that aggregates data from multiple sources, uses advanced AI for analysis, and can autonomously execute trades.

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Configuration](#configuration) • [Documentation](#documentation)

</div>

---

## 🎯 Overview

TRADE is an advanced AI-powered trading bot that combines:

- **Multi-source data aggregation** from news, social media, financial APIs, and economic indicators
- **Advanced AI analysis** using transformer models, machine learning, and pattern recognition
- **Autonomous trading** with configurable risk management and circuit breakers
- **Real-time monitoring** with a beautiful web dashboard
- **Comprehensive risk controls** to protect your capital

## ✨ Features

### 📊 Data Sources
- **Financial Data**: Yahoo Finance, Alpha Vantage, Polygon.io, Finnhub
- **News & Sentiment**: NewsAPI, Finnhub News, Twitter/X
- **Economic Data**: FRED (Federal Reserve Economic Data)
- **Market Indices**: S&P 500, NASDAQ, Dow Jones, VIX
- **Commodities**: Gold, Oil, Silver, Bitcoin
- **Political & Climate News**: Affecting markets and specific trades

### 🤖 AI Analysis
- **Sentiment Analysis**: Transformer-based NLP for news and social media
- **Pattern Recognition**: Technical patterns, candlestick patterns, volume analysis
- **Price Prediction**: Ensemble ML models combining multiple approaches
- **Risk Assessment**: Comprehensive risk scoring and factor analysis
- **Confidence Scoring**: Model confidence for every prediction

### 📈 Trading Features
- **Autonomous Trading**: No intervention needed - AI decides and executes
- **Multiple Brokers**: Alpaca, Interactive Brokers, and more
- **Order Types**: Market, Limit, Stop, Trailing Stop
- **Position Management**: Dynamic sizing, stop losses, take profits
- **Paper Trading**: Test strategies without risking real money

### 🛡️ Risk Management
- **Stop Loss / Take Profit**: Automatic protection for every trade
- **Position Sizing**: Kelly Criterion, Equal Weight, Risk Parity
- **Loss Limits**: Daily, weekly, monthly drawdown limits
- **Circuit Breakers**: Automatic pause when limits are breached
- **Correlation Limits**: Diversification enforcement

### 🎨 Dashboard
- **Real-time Monitoring**: Portfolio value, positions, P&L
- **Signal Visualization**: AI analysis scores and confidence
- **Trade History**: Complete audit trail
- **Control Panel**: Start, pause, stop trading
- **Watchlist Management**: Add/remove symbols easily

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- API keys for data sources (optional but recommended)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/leonnovic336/TRADE.git
cd TRADE

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys Setup

Get free API keys from:
- [NewsAPI](https://newsapi.org) - News data
- [Finnhub](https://finnhub.io) - Market data & news
- [Alpha Vantage](https://alphavantage.co) - Stock data
- [FRED](https://fred.stlouisfed.org) - Economic data
- [Alpaca](https://alpaca.markets) - Broker API

Create a `config/config.yaml` file and add your API keys.

## 📖 Quick Start

### Demo Mode (No Real Trading)

```bash
python -m src.main --demo
```

### Analyze a Symbol

```bash
python -m src.main --analyze AAPL
```

### Start Trading Bot

```bash
# Paper trading mode
python -m src.main --config config/config.yaml

# Live trading mode
python -m src.main --config config/config.yaml --mode live
```

### Run Dashboard Only

```bash
python -m src.main --dashboard
```

## ⚙️ Configuration

### Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | paper | Trading mode: paper or live |
| `autonomous_trading` | true | Enable automatic trade execution |
| `min_confidence_threshold` | 0.75 | Minimum AI confidence to trade |
| `max_position_size` | 0.10 | Max % of portfolio per position |
| `max_total_exposure` | 0.80 | Max % of portfolio exposed |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_stop_loss` | 0.02 | Default stop loss (2%) |
| `default_take_profit` | 0.04 | Default take profit (4%) |
| `max_daily_loss` | 0.05 | Daily loss limit (5%) |
| `max_drawdown` | 0.15 | Maximum drawdown (15%) |
| `circuit_breaker_threshold` | 0.03 | Auto-pause at 3% daily loss |

## ⚠️ Important Disclaimers

1. **No Guarantee of Profit**: Trading involves substantial risk. Past performance does not guarantee future results.
2. **Start with Paper Trading**: Always test strategies in paper mode first.
3. **Risk Management**: Never invest more than you can afford to lose.
4. **Human Oversight**: While the bot can trade autonomously, regular monitoring is recommended.

## 📄 License

MIT License - see LICENSE file for details.

---

<div align="center">

**Built with ❤️ for traders who want an edge**

</div>
