"""
Data Aggregator - Comprehensive market data from multiple sources.
Collects: News, Social Sentiment, Stock Data, Economic Data, Political/Climate News,
Analyst Views, Historical Patterns, and Any External Factors.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class DataSource(Enum):
    YAHOO_FINANCE = "yahoo_finance"
    FINNHUB = "finnhub"
    NEWS_API = "news_api"
    FRED = "fred"
    TWITTER = "twitter"
    TRADINGECONOMICS = "tradingeconomics"
    SEC_GOV = "sec_gov"
    WALKA = "wolframalpha"
    CUSTOM = "custom"


@dataclass
class MarketData:
    """Comprehensive market data structure."""
    symbol: str
    timestamp: datetime
    
    # Price data
    current_price: float = 0.0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: int = 0
    market_cap: float = 0.0
    
    # Technical indicators
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    
    # Pattern detection
    detected_patterns: List[str] = field(default_factory=list)
    pattern_confidence: float = 0.0
    
    # Volatility metrics
    historical_volatility: float = 0.0
    implied_volatility: float = 0.0
    atr: float = 0.0


@dataclass
class NewsData:
    """News and sentiment data structure."""
    headline: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_label: str = "neutral"
    relevance_score: float = 0.0  # 0 to 1
    category: str = "general"
    
    # Detailed categorization
    impact_type: str = ""  # positive, negative, neutral
    affected_sectors: List[str] = field(default_factory=list)
    affected_symbols: List[str] = field(default_factory=list)
    
    # Analysis fields
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class SentimentData:
    """Social media and sentiment analysis data."""
    platform: str
    symbol: str
    timestamp: datetime
    
    # Overall sentiment
    overall_sentiment: float = 0.0  # -1 to 1
    sentiment_strength: float = 0.0  # 0 to 1
    
    # Volume metrics
    mentions: int = 0
    bullish_mentions: int = 0
    bearish_mentions: int = 0
    neutral_mentions: int = 0
    
    # Trend data
    sentiment_trend: str = "stable"  # rising, falling, stable
    trend_strength: float = 0.0
    
    # Key themes
    trending_topics: List[str] = field(default_factory=list)
    key_influencers: List[str] = field(default_factory=list)


@dataclass
class EconomicData:
    """Economic indicator data."""
    indicator_name: str
    value: float
    unit: str
    timestamp: datetime
    previous_value: float = 0.0
    change_percent: float = 0.0
    
    # Classification
    category: str = ""  # GDP, Inflation, Employment, etc.
    impact_on_markets: str = "neutral"  # positive, negative, neutral
    affected_sectors: List[str] = field(default_factory=list)
    
    # Forecast data
    forecast_value: float = 0.0
    forecast_confidence: float = 0.0


@dataclass
class AnalystData:
    """Analyst ratings and price targets."""
    symbol: str
    timestamp: datetime
    
    # Ratings
    rating: str = ""  # Buy, Hold, Sell, etc.
    rating_scale: str = ""  # 1-5, Strong Buy to Strong Sell
    rating_count: int = 0
    
    # Price targets
    price_target_low: float = 0.0
    price_target_average: float = 0.0
    price_target_high: float = 0.0
    current_price: float = 0.0
    
    # Upside/downside
    upside_percent: float = 0.0
    
    # Analyst details
    firms_covering: List[str] = field(default_factory=list)
    recent_changes: List[Dict] = field(default_factory=list)


@dataclass
class MacroData:
    """Macroeconomic data affecting trades."""
    # Market indices
    sp500: float = 0.0
    nasdaq: float = 0.0
    dow_jones: float = 0.0
    vix: float = 0.0
    
    # Interest rates
    federal_funds_rate: float = 0.0
    treasury_10y: float = 0.0
    treasury_2y: float = 0.0
    mortgage_rate: float = 0.0
    
    # Currency
    dollar_index: float = 0.0
    eur_usd: float = 0.0
    usd_jpy: float = 0.0
    
    # Commodities
    gold: float = 0.0
    oil: float = 0.0
    silver: float = 0.0
    
    # Crypto
    bitcoin: float = 0.0
    
    # Economic indicators
    gdp_growth: float = 0.0
    inflation_rate: float = 0.0
    unemployment_rate: float = 0.0
    consumer_confidence: float = 0.0
    
    # Geopolitical risk
    geopolitical_risk_index: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PoliticalClimateData:
    """Political and climate-related market data."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Political events
    political_events: List[Dict] = field(default_factory=list)
    election_impact_score: float = 0.0
    regulation_changes: List[str] = field(default_factory=list)
    
    # Climate/Weather
    weather_events: List[Dict] = field(default_factory=list)
    commodity_impact: Dict[str, float] = field(default_factory=dict)
    energy_demand_indicator: float = 0.0
    
    # Trade policies
    tariff_updates: List[Dict] = field(default_factory=list)
    trade_deal_status: Dict[str, str] = field(default_factory=dict)
    
    # Central bank policy
    central_bank_statements: List[str] = field(default_factory=list)
    monetary_policy_stance: str = "neutral"
    
    # Global events
    global_events: List[Dict] = field(default_factory=list)


@dataclass
class ComprehensiveTradeData:
    """Complete data package for a trade decision."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core data
    market_data: Optional[MarketData] = None
    news_data: List[NewsData] = field(default_factory=list)
    sentiment_data: Optional[SentimentData] = None
    economic_data: List[EconomicData] = field(default_factory=list)
    analyst_data: Optional[AnalystData] = None
    macro_data: Optional[MacroData] = None
    political_climate_data: Optional[PoliticalClimateData] = None
    
    # Historical data
    historical_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_news: List[NewsData] = field(default_factory=list)
    historical_patterns: List[Dict] = field(default_factory=list)
    
    # Computed scores
    overall_score: float = 0.0
    confidence_score: float = 0.0
    risk_score: float = 0.0
    
    # Recommendation
    recommendation: str = "hold"  # buy, sell, hold
    target_entry: float = 0.0
    target_exit: float = 0.0
    stop_loss: float = 0.0
    
    # Additional context
    data_sources_used: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0  # 0 to 1


class DataAggregator:
    """
    Aggregates data from multiple sources for comprehensive market analysis.
    
    Sources:
    - Yahoo Finance: Real-time prices, historical data
    - Finnhub: News, company data, market news
    - NewsAPI: General financial news
    - FRED: Economic indicators
    - Twitter/X: Social sentiment
    - SEC EDGAR: Regulatory filings
    - Custom APIs: Additional data sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = config.get("data", {}).get("enabled_sources", ["yahoo_finance"])
        self.api_keys = config.get("data_sources", {})
        
        # Rate limiting
        self.request_history: Dict[str, List[datetime]] = {}
        self.rate_limits = {
            "news_api": 100,  # requests per day
            "finnhub": 60,    # requests per minute
            "alpha_vantage": 5,  # requests per minute
            "twitter": 450,  # requests per 15 min
        }
        
        # Cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = {
            "price": 60,       # 1 minute
            "news": 300,       # 5 minutes
            "sentiment": 900,  # 15 minutes
            "macro": 3600,     # 1 hour
        }
        
        logger.info(f"DataAggregator initialized with sources: {self.data_sources}")
    
    async def fetch_all_data(self, symbol: str, lookback_days: int = 30) -> ComprehensiveTradeData:
        """
        Fetch comprehensive data for a symbol from all enabled sources.
        """
        logger.info(f"Fetching comprehensive data for {symbol}")
        
        tasks = [
            self.fetch_market_data(symbol),
            self.fetch_news(symbol),
            self.fetch_sentiment(symbol),
            self.fetch_economic_data(),
            self.fetch_analyst_data(symbol),
            self.fetch_macro_data(),
            self.fetch_political_climate_data(symbol),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = ComprehensiveTradeData(symbol=symbol)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data type {i}: {result}")
                continue
            
            if isinstance(result, MarketData):
                data.market_data = result
            elif isinstance(result, list) and result and isinstance(result[0], NewsData):
                data.news_data = result
            elif isinstance(result, SentimentData):
                data.sentiment_data = result
            elif isinstance(result, list) and result and isinstance(result[0], EconomicData):
                data.economic_data = result
            elif isinstance(result, AnalystData):
                data.analyst_data = result
            elif isinstance(result, MacroData):
                data.macro_data = result
            elif isinstance(result, PoliticalClimateData):
                data.political_climate_data = result
        
        # Fetch historical data
        data.historical_prices = await self.fetch_historical_prices(symbol, lookback_days)
        
        # Analyze patterns
        data.historical_patterns = self.analyze_historical_patterns(data.historical_prices)
        
        # Calculate overall scores
        data = self.calculate_overall_scores(data)
        
        return data
    
    async def fetch_market_data(self, symbol: str) -> MarketData:
        """Fetch real-time market data from Yahoo Finance."""
        if not self._should_fetch("price", symbol):
            return self.cache.get(f"market_{symbol}", MarketData(symbol=symbol, timestamp=datetime.now()))
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
            
            data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                open_price=info.get("open", 0),
                high_price=info.get("dayHigh", 0),
                low_price=info.get("dayLow", 0),
                close_price=info.get("previousClose", 0),
                volume=info.get("volume", 0),
                market_cap=info.get("marketCap", 0),
                rsi=self._calculate_rsi(hist),
                sma_20=self._calculate_sma(hist, 20),
                sma_50=self._calculate_sma(hist, 50),
                sma_200=self._calculate_sma(hist, 200),
                bollinger_upper=self._calculate_bollinger_upper(hist),
                bollinger_lower=self._calculate_bollinger_lower(hist),
                atr=self._calculate_atr(hist),
            )
            
            data.detected_patterns = self._detect_patterns(hist)
            
            self._cache_data(f"market_{symbol}", data, "price")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return MarketData(symbol=symbol, timestamp=datetime.now())
    
    async def fetch_news(self, symbol: str) -> List[NewsData]:
        """Fetch news from multiple sources."""
        news_items = []
        
        # Fetch from Finnhub
        if "finnhub" in self.data_sources:
            try:
                finnhub_news = await self._fetch_finnhub_news(symbol)
                news_items.extend(finnhub_news)
            except Exception as e:
                logger.error(f"Error fetching Finnhub news: {e}")
        
        # Fetch from NewsAPI
        if "news_api" in self.data_sources:
            try:
                newsapi_news = await self._fetch_newsapi(symbol)
                news_items.extend(newsapi_news)
            except Exception as e:
                logger.error(f"Error fetching NewsAPI: {e}")
        
        # Deduplicate and sort by date
        seen = set()
        unique_news = []
        for news in news_items:
            if news.headline not in seen:
                seen.add(news.headline)
                unique_news.append(news)
        
        unique_news.sort(key=lambda x: x.published_at, reverse=True)
        
        return unique_news[:50]  # Keep top 50 most recent
    
    async def _fetch_finnhub_news(self, symbol: str) -> List[NewsData]:
        """Fetch news from Finnhub API."""
        api_key = self.api_keys.get("finnhub_api_key", "")
        if not api_key:
            return []
        
        url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    articles = await resp.json()
                    return [
                        NewsData(
                            headline=a.get("headline", ""),
                            source=a.get("source", "Finnhub"),
                            url=a.get("url", ""),
                            published_at=datetime.fromtimestamp(a.get("datetime", 0)),
                            summary=a.get("summary", ""),
                        )
                        for a in articles[:20]
                        if symbol.lower() in (a.get("headline", "") + a.get("summary", "")).lower()
                    ]
        return []
    
    async def _fetch_newsapi(self, symbol: str) -> List[NewsData]:
        """Fetch news from NewsAPI."""
        api_key = self.api_keys.get("news_api_key", "")
        if not api_key:
            return []
        
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&apiKey={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    articles = data.get("articles", [])
                    return [
                        NewsData(
                            headline=a.get("title", ""),
                            source=a.get("source", {}).get("name", "NewsAPI"),
                            url=a.get("url", ""),
                            published_at=datetime.fromisoformat(a.get("publishedAt", "").replace("Z", "+00:00")),
                            summary=a.get("description", ""),
                        )
                        for a in articles[:20]
                    ]
        return []
    
    async def fetch_sentiment(self, symbol: str) -> SentimentData:
        """Fetch social media sentiment data."""
        sentiment = SentimentData(
            platform="aggregated",
            symbol=symbol,
            timestamp=datetime.now(),
        )
        
        # Fetch from Twitter if available
        if "twitter" in self.data_sources:
            try:
                twitter_sentiment = await self._fetch_twitter_sentiment(symbol)
                sentiment.overall_sentiment = twitter_sentiment.get("sentiment", 0)
                sentiment.mentions = twitter_sentiment.get("mentions", 0)
                sentiment.trending_topics = twitter_sentiment.get("trending_topics", [])
            except Exception as e:
                logger.error(f"Error fetching Twitter sentiment: {e}")
        
        return sentiment
    
    async def _fetch_twitter_sentiment(self, symbol: str) -> Dict:
        """Fetch sentiment from Twitter/X API."""
        bearer_token = self.api_keys.get("twitter_bearer_token", "")
        if not bearer_token:
            return {}
        
        url = f"https://api.twitter.com/2/tweets/search/recent?query=${symbol}"
        
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {bearer_token}"}
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tweets = data.get("data", [])
                    # Simple sentiment analysis based on keyword presence
                    bullish_words = ["moon", "bull", "buy", "long", "pump", "🚀", "diamond hands"]
                    bearish_words = ["dump", "bear", "sell", "short", "crash", "💩", "paper hands"]
                    
                    bullish = sum(1 for t in tweets for w in bullish_words if w.lower() in str(t).lower())
                    bearish = sum(1 for t in tweets for w in bearish_words if w.lower() in str(t).lower())
                    
                    sentiment_score = (bullish - bearish) / max(bullish + bearish, 1)
                    
                    return {
                        "sentiment": sentiment_score,
                        "mentions": len(tweets),
                        "bullish_count": bullish,
                        "bearish_count": bearish,
                        "trending_topics": [],
                    }
        return {}
    
    async def fetch_economic_data(self) -> List[EconomicData]:
        """Fetch macroeconomic indicators from FRED."""
        economic_data = []
        
        fred_api_key = self.api_keys.get("fred_api_key", "")
        
        indicators = [
            ("GDP", "US GDP Growth Rate"),
            ("CPIAUCSL", "Inflation Rate (CPI)"),
            ("UNRATE", "Unemployment Rate"),
            ("FEDFUNDS", "Federal Funds Rate"),
            ("DGS10", "10-Year Treasury Rate"),
            ("PCE", "Consumer Spending"),
            ("CONSCONF", "Consumer Confidence"),
            ("ISMNONMAN", "Manufacturing PMI"),
            ("HOUST", "Housing Starts"),
            ("INDPRO", "Industrial Production"),
        ]
        
        if fred_api_key:
            for series_id, name in indicators:
                try:
                    data = await self._fetch_fred_series(fred_api_key, series_id, name)
                    if data:
                        economic_data.append(data)
                except Exception as e:
                    logger.error(f"Error fetching FRED data for {series_id}: {e}")
        
        return economic_data
    
    async def _fetch_fred_series(self, api_key: str, series_id: str, name: str) -> Optional[EconomicData]:
        """Fetch a single FRED series."""
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    observations = data.get("observations", [])
                    if len(observations) >= 2:
                        latest = float(observations[-1]["value"])
                        previous = float(observations[-2]["value"])
                        change = ((latest - previous) / previous * 100) if previous != 0 else 0
                        
                        return EconomicData(
                            indicator_name=name,
                            value=latest,
                            unit=self._get_fred_unit(series_id),
                            timestamp=datetime.now(),
                            previous_value=previous,
                            change_percent=change,
                            category=self._get_fred_category(series_id),
                        )
        return None
    
    def _get_fred_unit(self, series_id: str) -> str:
        """Get the unit for a FRED series."""
        units = {
            "GDP": "%",
            "CPIAUCSL": "Index",
            "UNRATE": "%",
            "FEDFUNDS": "%",
            "DGS10": "%",
            "PCE": "Billion $",
            "CONSCONF": "Index",
            "ISMNONMAN": "Index",
            "HOUST": "Thousands",
            "INDPRO": "Index",
        }
        return units.get(series_id, "")
    
    def _get_fred_category(self, series_id: str) -> str:
        """Get the category for a FRED series."""
        categories = {
            "GDP": "GDP",
            "CPIAUCSL": "Inflation",
            "UNRATE": "Employment",
            "FEDFUNDS": "Interest Rates",
            "DGS10": "Interest Rates",
            "PCE": "Consumer",
            "CONSCONF": "Consumer",
            "ISMNONMAN": "Manufacturing",
            "HOUST": "Housing",
            "INDPRO": "Manufacturing",
        }
        return categories.get(series_id, "General")
    
    async def fetch_analyst_data(self, symbol: str) -> AnalystData:
        """Fetch analyst ratings and price targets."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            recommendations = info.get("recommendationKey", "none")
            
            target_low = info.get("targetLowPrice", 0)
            target_high = info.get("targetHighPrice", 0)
            target_avg = info.get("targetMeanPrice", 0)
            current = info.get("currentPrice", info.get("regularMarketPrice", 0))
            
            upside = ((target_avg - current) / current * 100) if current > 0 else 0
            
            return AnalystData(
                symbol=symbol,
                timestamp=datetime.now(),
                rating=recommendations,
                price_target_low=target_low,
                price_target_average=target_avg,
                price_target_high=target_high,
                current_price=current,
                upside_percent=upside,
                firms_covering=info.get("major_news_holders", []),
            )
        except Exception as e:
            logger.error(f"Error fetching analyst data for {symbol}: {e}")
            return AnalystData(symbol=symbol, timestamp=datetime.now())
    
    async def fetch_macro_data(self) -> MacroData:
        """Fetch comprehensive macro market data."""
        macro = MacroData(timestamp=datetime.now())
        
        try:
            # Major indices
            indices = {
                "^GSPC": "sp500",
                "^IXIC": "nasdaq",
                "^DJI": "dow_jones",
                "^VIX": "vix",
            }
            
            for symbol, field in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if len(hist) > 0:
                        setattr(macro, field, hist["Close"].iloc[-1])
                except Exception:
                    pass
            
            # Treasury rates from FRED
            fred_api_key = self.api_keys.get("fred_api_key", "")
            if fred_api_key:
                try:
                    rates = await self._fetch_multiple_fred(fred_api_key, ["DGS10", "DGS2", "FEDFUNDS"])
                    macro.treasury_10y = rates.get("DGS10", 0)
                    macro.treasury_2y = rates.get("DGS2", 0)
                    macro.federal_funds_rate = rates.get("FEDFUNDS", 0)
                except Exception:
                    pass
            
            # Commodities
            commodities = {
                "GC=F": "gold",
                "CL=F": "oil",
                "SI=F": "silver",
            }
            
            for symbol, field in commodities.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if len(hist) > 0:
                        setattr(macro, field, hist["Close"].iloc[-1])
                except Exception:
                    pass
            
            # Crypto
            try:
                btc = yf.Ticker("BTC-USD")
                hist = btc.history(period="1d")
                if len(hist) > 0:
                    macro.bitcoin = hist["Close"].iloc[-1]
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
        
        return macro
    
    async def _fetch_multiple_fred(self, api_key: str, series_ids: List[str]) -> Dict[str, float]:
        """Fetch multiple FRED series at once."""
        results = {}
        
        for series_id in series_ids:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&limit=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        observations = data.get("observations", [])
                        if observations:
                            results[series_id] = float(observations[-1]["value"])
        
        return results
    
    async def fetch_political_climate_data(self, symbol: str) -> PoliticalClimateData:
        """Fetch political and climate-related market data."""
        data = PoliticalClimateData(timestamp=datetime.now())
        
        # Fetch news related to politics and climate
        if "news_api" in self.data_sources:
            try:
                api_key = self.api_keys.get("news_api_key", "")
                if api_key:
                    # Fetch political news
                    political_url = f"https://newsapi.org/v2/everything?q=stock+market+politics&sortBy=publishedAt&apiKey={api_key}"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(political_url, timeout=10) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                articles = result.get("articles", [])[:10]
                                data.political_events = [
                                    {
                                        "title": a.get("title", ""),
                                        "source": a.get("source", {}).get("name", ""),
                                        "url": a.get("url", ""),
                                    }
                                    for a in articles
                                ]
                    
                    # Fetch climate/weather news
                    climate_url = f"https://newsapi.org/v2/everything?q=climate+weather+impact+stocks&sortBy=publishedAt&apiKey={api_key}"
                    
                    async with session.get(climate_url, timeout=10) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            articles = result.get("articles", [])[:10]
                            data.weather_events = [
                                {
                                    "title": a.get("title", ""),
                                    "source": a.get("source", {}).get("name", ""),
                                    "url": a.get("url", ""),
                                }
                                for a in articles
                            ]
            except Exception as e:
                logger.error(f"Error fetching political/climate data: {e}")
        
        return data
    
    async def fetch_historical_prices(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical price data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_historical_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Analyze historical price patterns."""
        patterns = []
        
        if len(price_data) < 50:
            return patterns
        
        # Detect common patterns
        patterns.extend(self._detect_double_bottom(price_data))
        patterns.extend(self._detect_head_shoulders(price_data))
        patterns.extend(self._detect_triangle_patterns(price_data))
        patterns.extend(self._detect_support_resistance(price_data))
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Detect double bottom pattern."""
        # Simplified detection - more complex implementations available
        return []
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern."""
        return []
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns."""
        return []
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> List[Dict]:
        """Detect support and resistance levels."""
        return []
    
    def calculate_overall_scores(self, data: ComprehensiveTradeData) -> ComprehensiveTradeData:
        """Calculate overall scores from all data sources."""
        # Calculate market data score
        if data.market_data:
            # Technical score based on RSI, moving averages
            technical_score = 0.5
            if data.market_data.rsi < 30:
                technical_score = 0.8  # Oversold - potential buy
            elif data.market_data.rsi > 70:
                technical_score = 0.2  # Overbought - potential sell
            
            data.market_data.pattern_confidence = technical_score
        
        # Calculate overall sentiment score
        sentiment_scores = []
        
        if data.sentiment_data:
            sentiment_scores.append(data.sentiment_data.overall_sentiment)
        
        # News sentiment
        if data.news_data:
            news_sentiment = sum(n.sentiment_score for n in data.news_data) / len(data.news_data)
            sentiment_scores.append(news_sentiment)
        
        if sentiment_scores:
            data.overall_score = sum(sentiment_scores) / len(sentiment_scores)
        
        # Confidence score based on data quality
        data.data_quality_score = min(len(data.news_data) / 10, 1.0)
        data.confidence_score = data.data_quality_score
        
        return data
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(df) < period + 1:
            return 50.0
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def _calculate_sma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(df) < period:
            return 0.0
        return df["Close"].rolling(window=period).mean().iloc[-1]
    
    def _calculate_bollinger_upper(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> float:
        """Calculate Bollinger Bands upper band."""
        if len(df) < period:
            return 0.0
        sma = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        return (sma + std * std_dev).iloc[-1]
    
    def _calculate_bollinger_lower(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> float:
        """Calculate Bollinger Bands lower band."""
        if len(df) < period:
            return 0.0
        sma = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        return (sma - std * std_dev).iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return 0.0
        
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift())
        low_close = abs(df["Low"] - df["Close"].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.0
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect price patterns."""
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        # Simple pattern detection examples
        # Add more sophisticated pattern detection
        
        return patterns
    
    def _should_fetch(self, data_type: str, key: str) -> bool:
        """Check if data should be fetched based on cache TTL."""
        cache_key = f"{data_type}_{key}"
        
        if cache_key not in self.cache:
            return True
        
        cached_time = self.cache[cache_key].get("timestamp")
        if cached_time is None:
            return True
        
        ttl = self.cache_ttl.get(data_type, 60)
        if (datetime.now() - cached_time).total_seconds() > ttl:
            return True
        
        return False
    
    def _cache_data(self, key: str, data: Any, data_type: str) -> None:
        """Cache data with timestamp."""
        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now(),
            "type": data_type,
        }
    
    async def get_market_sentiment_index(self) -> float:
        """Get overall market sentiment (fear/greed indicator)."""
        # Composite of VIX, momentum, breadth indicators
        sentiment = 50.0  # Neutral
        
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1d")
            if len(vix_hist) > 0:
                vix_value = vix_hist["Close"].iloc[-1]
                # VIX > 20 indicates fear, < 15 indicates greed
                if vix_value > 30:
                    sentiment = 20  # Extreme fear
                elif vix_value > 20:
                    sentiment = 35  # Fear
                elif vix_value < 15:
                    sentiment = 70  # Greed
                elif vix_value < 10:
                    sentiment = 85  # Extreme greed
        except Exception:
            pass
        
        return sentiment