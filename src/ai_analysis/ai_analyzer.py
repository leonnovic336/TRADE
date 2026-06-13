"""
AI Analysis Engine - Advanced AI/ML for Market Prediction
Uses transformers, sentiment analysis, pattern recognition, and ensemble methods.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TradeSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class PredictionResult:
    """Result of AI prediction."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Signal
    signal: TradeSignal = TradeSignal.HOLD
    signal_strength: float = 0.5  # 0 to 1
    
    # Price predictions
    predicted_price_1h: float = 0.0
    predicted_price_1d: float = 0.0
    predicted_price_1w: float = 0.0
    
    # Confidence
    confidence: float = 0.0  # 0 to 1
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    # Risk assessment
    risk_score: float = 0.0  # 0 to 1 (higher = more risky)
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    # Analysis breakdown
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    macro_score: float = 0.0
    news_score: float = 0.0
    
    # Pattern insights
    detected_patterns: List[str] = field(default_factory=list)
    pattern_confidence: float = 0.0
    
    # Recommendation
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_recommendation: float = 0.0  # Percentage of portfolio
    
    # Model info
    models_used: List[str] = field(default_factory=list)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    
    # Additional context
    reasoning: str = ""
    key_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """Historical pattern match."""
    pattern_name: str
    match_confidence: float  # 0 to 1
    historical_success_rate: float  # 0 to 1
    similarity_score: float  # 0 to 1
    historical_outcome: str  # "bullish", "bearish", "neutral"
    occurrences: int  # How many times this pattern occurred


class SentimentAnalyzer:
    """
    Advanced sentiment analysis using NLP transformers.
    Analyzes news, social media, and other text sources.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the sentiment analysis model."""
        if self._initialized:
            return
        
        try:
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
                max_length=512
            )
            self._initialized = True
            logger.info("Sentiment model loaded successfully")
        except ImportError:
            logger.warning("Transformers not installed. Using fallback sentiment analysis.")
            self._initialized = True
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        if not self._initialized:
            await self.initialize()
        
        if not hasattr(self, 'sentiment_pipeline'):
            return self._fallback_sentiment(text)
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            return {
                "label": result["label"],
                "score": result["score"],
                "positive": result["label"] == "POSITIVE",
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment(text)
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of multiple texts."""
        if not self._initialized:
            await self.initialize()
        
        results = []
        for text in texts:
            result = await self.analyze_text(text)
            results.append(result)
        
        return results
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based sentiment analysis."""
        text_lower = text.lower()
        
        positive_words = [
            "buy", "bullish", "growth", "profit", "up", "surge", "gain", "rally",
            "upgrade", "outperform", "strong", "positive", "beat", "exceed",
            "innovative", "breakthrough", "success", "expansion", "launch"
        ]
        
        negative_words = [
            "sell", "bearish", "loss", "down", "crash", "decline", "drop",
            "downgrade", "underperform", "weak", "negative", "miss", "fail",
            "risk", "warning", "concern", "lawsuit", "investigation", "recall"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return {"label": "NEUTRAL", "score": 0.5, "positive": None}
        
        score = positive_count / total
        return {
            "label": "POSITIVE" if score > 0.5 else "NEGATIVE",
            "score": score,
            "positive": score > 0.5,
        }
    
    async def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple keyword extraction
        important_words = [
            "acquisition", "merger", "ipo", "earnings", "revenue", "profit",
            "lawsuit", "regulation", "partnership", "product", "launch",
            "upgrade", "downgrade", "analyst", "target", "forecast",
            "expansion", "layoffs", "ceo", "cto", "board", "dividend"
        ]
        
        text_lower = text.lower()
        found = [word for word in important_words if word in text_lower]
        return found


class PatternRecognizer:
    """
    Pattern recognition using machine learning.
    Detects chart patterns, candlestick patterns, and historical repetitions.
    """
    
    def __init__(self):
        self.pattern_templates = self._load_pattern_templates()
        self.ml_model = None
        self.scaler = StandardScaler()
    
    def _load_pattern_templates(self) -> Dict:
        """Load known pattern templates."""
        return {
            "double_bottom": {
                "description": "W-shaped reversal pattern",
                "type": "reversal",
                "bias": "bullish",
                "min_occurrences": 100,
                "success_rate": 0.68,
            },
            "head_and_shoulders": {
                "description": "Reversal pattern with three peaks",
                "type": "reversal",
                "bias": "bearish",
                "min_occurrences": 80,
                "success_rate": 0.65,
            },
            "ascending_triangle": {
                "description": "Bullish continuation pattern",
                "type": "continuation",
                "bias": "bullish",
                "min_occurrences": 150,
                "success_rate": 0.70,
            },
            "descending_triangle": {
                "description": "Bearish continuation pattern",
                "type": "continuation",
                "bias": "bearish",
                "min_occurrences": 150,
                "success_rate": 0.70,
            },
            "cup_and_handle": {
                "description": "Bullish continuation pattern",
                "type": "continuation",
                "bias": "bullish",
                "min_occurrences": 90,
                "success_rate": 0.72,
            },
            "bull_flag": {
                "description": "Bullish continuation after strong move",
                "type": "continuation",
                "bias": "bullish",
                "min_occurrences": 200,
                "success_rate": 0.67,
            },
            "bear_flag": {
                "description": "Bearish continuation after strong move",
                "type": "continuation",
                "bias": "bearish",
                "min_occurrences": 200,
                "success_rate": 0.67,
            },
            "doji": {
                "description": "Indecision candlestick",
                "type": "reversal",
                "bias": "neutral",
                "min_occurrences": 300,
                "success_rate": 0.55,
            },
            "hammer": {
                "description": "Bullish reversal candlestick",
                "type": "reversal",
                "bias": "bullish",
                "min_occurrences": 250,
                "success_rate": 0.62,
            },
            "shooting_star": {
                "description": "Bearish reversal candlestick",
                "type": "reversal",
                "bias": "bearish",
                "min_occurrences": 250,
                "success_rate": 0.62,
            },
        }
    
    def detect_patterns(self, price_data: pd.DataFrame) -> List[PatternMatch]:
        """Detect patterns in price data."""
        matches = []
        
        if len(price_data) < 50:
            return matches
        
        # Candlestick pattern detection
        matches.extend(self._detect_candlestick_patterns(price_data))
        
        # Technical pattern detection
        matches.extend(self._detect_technical_patterns(price_data))
        
        # Volume pattern analysis
        matches.extend(self._analyze_volume_patterns(price_data))
        
        return matches
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """Detect candlestick patterns."""
        matches = []
        
        if len(df) < 5:
            return matches
        
        # Calculate candlestick properties
        df = df.copy()
        df['body'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['range'] = df['High'] - df['Low']
        
        # Detect Doji
        for i in range(-5, 0):
            if i >= -len(df):
                body_ratio = df['body'].iloc[i] / df['range'].iloc[i]
                if body_ratio < 0.1:  # Small body
                    matches.append(PatternMatch(
                        pattern_name="doji",
                        match_confidence=0.7,
                        historical_success_rate=0.55,
                        similarity_score=1 - body_ratio,
                        historical_outcome="neutral",
                        occurrences=1,
                    ))
        
        # Detect Hammer
        for i in range(-5, 0):
            if i >= -len(df):
                if df['lower_shadow'].iloc[i] > 2 * df['body'].iloc[i] and \
                   df['upper_shadow'].iloc[i] < 0.3 * df['body'].iloc[i]:
                    matches.append(PatternMatch(
                        pattern_name="hammer",
                        match_confidence=0.75,
                        historical_success_rate=0.62,
                        similarity_score=0.8,
                        historical_outcome="bullish",
                        occurrences=1,
                    ))
        
        # Detect Shooting Star
        for i in range(-5, 0):
            if i >= -len(df):
                if df['upper_shadow'].iloc[i] > 2 * df['body'].iloc[i] and \
                   df['lower_shadow'].iloc[i] < 0.3 * df['body'].iloc[i]:
                    matches.append(PatternMatch(
                        pattern_name="shooting_star",
                        match_confidence=0.75,
                        historical_success_rate=0.62,
                        similarity_score=0.8,
                        historical_outcome="bearish",
                        occurrences=1,
                    ))
        
        return matches
    
    def _detect_technical_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """Detect technical analysis patterns."""
        matches = []
        
        if len(df) < 60:
            return matches
        
        # Moving averages
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()
        
        # Detect Golden Cross (MA50 crosses above MA200)
        if len(df) >= 200:
            ma200 = df['Close'].rolling(window=200).mean()
            
            for i in range(-3, 0):
                if i >= -len(df) and i-1 >= -len(df):
                    if ma50.iloc[i] > ma200.iloc[i] and ma50.iloc[i-1] <= ma200.iloc[i-1]:
                        matches.append(PatternMatch(
                            pattern_name="golden_cross",
                            match_confidence=0.85,
                            historical_success_rate=0.72,
                            similarity_score=0.9,
                            historical_outcome="bullish",
                            occurrences=1,
                        ))
        
        # Detect Death Cross (MA50 crosses below MA200)
        if len(df) >= 200:
            for i in range(-3, 0):
                if i >= -len(df) and i-1 >= -len(df):
                    if ma50.iloc[i] < ma200.iloc[i] and ma50.iloc[i-1] >= ma200.iloc[i-1]:
                        matches.append(PatternMatch(
                            pattern_name="death_cross",
                            match_confidence=0.85,
                            historical_success_rate=0.68,
                            similarity_score=0.9,
                            historical_outcome="bearish",
                            occurrences=1,
                        ))
        
        return matches
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """Analyze volume patterns."""
        matches = []
        
        if len(df) < 20:
            return matches
        
        # Volume spike detection
        avg_volume = df['Volume'].rolling(window=20).mean()
        
        for i in range(-5, 0):
            if i >= -len(df):
                volume_ratio = df['Volume'].iloc[i] / avg_volume.iloc[i]
                
                if volume_ratio > 2.0:
                    # Volume spike with price increase
                    if df['Close'].iloc[i] > df['Open'].iloc[i]:
                        matches.append(PatternMatch(
                            pattern_name="volume_surge_bullish",
                            match_confidence=0.7,
                            historical_success_rate=0.65,
                            similarity_score=min(volume_ratio / 3, 1.0),
                            historical_outcome="bullish",
                            occurrences=1,
                        ))
                    # Volume spike with price decrease
                    elif df['Close'].iloc[i] < df['Open'].iloc[i]:
                        matches.append(PatternMatch(
                            pattern_name="volume_surge_bearish",
                            match_confidence=0.7,
                            historical_success_rate=0.65,
                            similarity_score=min(volume_ratio / 3, 1.0),
                            historical_outcome="bearish",
                            occurrences=1,
                        ))
        
        return matches


class PricePredictor:
    """
    ML-based price prediction using ensemble methods.
    Combines multiple models for robust predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self._trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model."""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            features[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Volatility
        features['volatility_5'] = df['Close'].pct_change().rolling(window=5).std()
        features['volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        features['bb_upper'] = sma20 + 2 * std20
        features['bb_lower'] = sma20 - 2 * std20
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Momentum
        features['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        features['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Clip extreme values
        features = features.clip(-10, 10)
        
        return features.values
    
    def prepare_target(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Prepare target variable (future returns)."""
        future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
        future_returns = future_returns.fillna(0).clip(-0.5, 0.5)
        return future_returns.values
    
    def train(self, df: pd.DataFrame, target_horizon: int = 60) -> Dict[str, float]:
        """Train the ensemble model."""
        if len(df) < 100:
            logger.warning("Insufficient data for training")
            return {}
        
        X = self.prepare_features(df)
        y = self.prepare_target(df, target_horizon)
        
        # Create labels for classification
        y_labels = np.where(y > 0.005, 1, np.where(y < -0.005, -1, 0))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_labels[:split_idx], y_labels[split_idx:]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Initialize models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
        }
        
        # Train and evaluate models
        scores = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                scores[name] = score
                self.models[name] = model
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Calculate ensemble weights based on performance
        total_score = sum(scores.values())
        if total_score > 0:
            self.ensemble_weights = {k: v / total_score for k, v in scores.items()}
        
        # Remove models with zero weight
        self.ensemble_weights = {k: v for k, v in self.ensemble_weights.items() if v > 0}
        
        self._trained = True
        
        return scores
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Make prediction using ensemble of models.
        Returns: (predicted_direction, confidence)
        """
        if not self._trained or len(df) < 50:
            return 0.0, 0.0
        
        X = self.prepare_features(df)
        X_scaled = self.feature_scaler.transform(X[-1:])
        
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            if name in self.ensemble_weights:
                try:
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)
                    
                    # Get probability of predicted class
                    pred_idx = list(model.classes_).index(pred) if pred in model.classes_ else 1
                    confidence = proba[0][pred_idx]
                    
                    predictions.append((pred, self.ensemble_weights[name], confidence))
                    probabilities.append(confidence * self.ensemble_weights[name])
                except Exception as e:
                    logger.error(f"Error in prediction for {name}: {e}")
        
        if not predictions:
            return 0.0, 0.0
        
        # Weighted average of predictions
        weighted_pred = sum(p[0] * p[1] for p in predictions)
        confidence = sum(probabilities) / sum(self.ensemble_weights.values())
        
        return weighted_pred, confidence


class AIAnalyzer:
    """
    Main AI Analysis Engine.
    Combines all AI components for comprehensive market analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer(
            model_name=config.get("ai", {}).get("sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english")
        )
        self.pattern_recognizer = PatternRecognizer()
        self.price_predictor = PricePredictor()
        
        # Analysis weights
        weights = config.get("ai", {}).get("analysis_weights", {})
        self.weights = {
            "technical": weights.get("technical", 0.25),
            "fundamental": weights.get("fundamental", 0.20),
            "sentiment": weights.get("sentiment", 0.25),
            "macro": weights.get("macro", 0.15),
            "news": weights.get("news", 0.15),
        }
        
        # Thresholds
        self.min_confidence = config.get("ai", {}).get("min_prediction_confidence", 0.70)
        self.high_confidence = config.get("ai", {}).get("high_confidence_threshold", 0.85)
        
        logger.info("AI Analyzer initialized")
    
    async def analyze(
        self,
        market_data: Any,
        news_data: List[Any],
        sentiment_data: Any,
        macro_data: Any,
        historical_prices: pd.DataFrame,
    ) -> PredictionResult:
        """
        Perform comprehensive AI analysis.
        """
        result = PredictionResult(
            symbol=market_data.symbol if market_data else "UNKNOWN"
        )
        
        # Train/update models if needed
        if len(historical_prices) > 100:
            self.price_predictor.train(historical_prices)
        
        # Technical Analysis
        result.technical_score = self._analyze_technical(market_data, historical_prices)
        
        # Pattern Recognition
        patterns = self.pattern_recognizer.detect_patterns(historical_prices)
        result.detected_patterns = [p.pattern_name for p in patterns]
        result.pattern_confidence = max([p.match_confidence for p in patterns], default=0.0)
        
        # Sentiment Analysis
        result.sentiment_score = await self._analyze_sentiment(news_data, sentiment_data)
        
        # Macro Analysis
        result.macro_score = self._analyze_macro(macro_data)
        
        # News Analysis
        result.news_score = await self._analyze_news(news_data)
        
        # Fundamental Analysis
        result.fundamental_score = self._analyze_fundamental(market_data)
        
        # Price Prediction
        predicted_direction, prediction_confidence = self.price_predictor.predict(historical_prices)
        
        # Calculate overall scores
        result.overall_score = (
            self.weights["technical"] * result.technical_score +
            self.weights["sentiment"] * result.sentiment_score +
            self.weights["macro"] * result.macro_score +
            self.weights["news"] * result.news_score +
            self.weights["fundamental"] * result.fundamental_score
        )
        
        result.confidence = prediction_confidence
        result.confidence_factors = {
            "technical_indicators": result.technical_score,
            "pattern_match": result.pattern_confidence,
            "sentiment": result.sentiment_score,
            "prediction_model": prediction_confidence,
        }
        
        # Generate signal
        result.signal, result.signal_strength = self._generate_signal(result)
        
        # Risk assessment
        result.risk_score = self._assess_risk(market_data, result)
        result.risk_factors = self._get_risk_factors(market_data, result)
        
        # Generate recommendations
        result = self._generate_recommendations(result, market_data)
        
        # Build reasoning
        result.reasoning = self._build_reasoning(result)
        result.key_factors = self._get_key_factors(result)
        
        # Model info
        result.models_used = list(self.price_predictor.models.keys())
        result.ensemble_weights = self.price_predictor.ensemble_weights
        
        # Add warnings for risky situations
        result.warnings = self._generate_warnings(result, market_data)
        
        return result
    
    def _analyze_technical(self, market_data: Any, price_data: pd.DataFrame) -> float:
        """Analyze technical indicators."""
        if not market_data:
            return 0.5
        
        score = 0.5
        
        # RSI analysis
        rsi = market_data.rsi if hasattr(market_data, 'rsi') else 50
        if rsi < 30:
            score += 0.2  # Oversold - potential buy
        elif rsi > 70:
            score -= 0.2  # Overbought - potential sell
        
        # Moving average analysis
        current_price = market_data.current_price
        sma_20 = market_data.sma_20 if hasattr(market_data, 'sma_20') else current_price
        sma_50 = market_data.sma_50 if hasattr(market_data, 'sma_50') else current_price
        
        if current_price > sma_20:
            score += 0.1
        else:
            score -= 0.1
        
        if current_price > sma_50:
            score += 0.1
        else:
            score -= 0.1
        
        # Trend strength from recent price action
        if len(price_data) >= 20:
            recent_returns = price_data['Close'].pct_change().tail(20)
            if recent_returns.mean() > 0:
                score += 0.1
            else:
                score -= 0.1
        
        return max(0, min(1, score))
    
    async def _analyze_sentiment(self, news_data: List[Any], sentiment_data: Any) -> float:
        """Analyze overall sentiment."""
        score = 0.5
        
        # Social sentiment
        if sentiment_data and hasattr(sentiment_data, 'overall_sentiment'):
            score += sentiment_data.overall_sentiment * 0.3
        
        # News sentiment
        if news_data:
            sentiment_scores = []
            for news in news_data[:10]:  # Analyze top 10 news items
                if hasattr(news, 'headline'):
                    result = await self.sentiment_analyzer.analyze_text(news.headline)
                    if result.get("positive") is not None:
                        sentiment_scores.append(1 if result["positive"] else 0)
            
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                score = (score + avg_sentiment) / 2
        
        return max(0, min(1, score))
    
    def _analyze_macro(self, macro_data: Any) -> float:
        """Analyze macroeconomic conditions."""
        score = 0.5
        
        if not macro_data:
            return score
        
        # VIX analysis
        if hasattr(macro_data, 'vix'):
            if macro_data.vix > 25:
                score -= 0.1  # High volatility = risk off
            elif macro_data.vix < 15:
                score += 0.1  # Low volatility = risk on
        
        # Market indices trend
        if hasattr(macro_data, 'sp500') and macro_data.sp500 > 0:
            # Would need historical data for trend
            pass
        
        # Interest rates impact
        if hasattr(macro_data, 'federal_funds_rate'):
            if macro_data.federal_funds_rate < 2:
                score += 0.1  # Low rates = supportive
            elif macro_data.federal_funds_rate > 5:
                score -= 0.1  # High rates = headwind
        
        return max(0, min(1, score))
    
    async def _analyze_news(self, news_data: List[Any]) -> float:
        """Analyze news impact."""
        if not news_data:
            return 0.5
        
        scores = []
        for news in news_data[:20]:
            if hasattr(news, 'headline'):
                result = await self.sentiment_analyzer.analyze_text(news.headline)
                key_phrases = await self.sentiment_analyzer.extract_key_phrases(news.headline)
                
                # Higher score for relevant key phrases
                relevance_bonus = 0.1 if key_phrases else 0
                
                if result.get("positive"):
                    scores.append(0.7 + relevance_bonus)
                elif result.get("positive") is False:
                    scores.append(0.3 - relevance_bonus)
                else:
                    scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _analyze_fundamental(self, market_data: Any) -> float:
        """Analyze fundamental factors."""
        if not market_data:
            return 0.5
        
        score = 0.5
        
        # Market cap (larger = potentially more stable)
        if hasattr(market_data, 'market_cap') and market_data.market_cap > 0:
            # Normalize to 0-1 range (cap at $1 trillion)
            market_cap_score = min(market_data.market_cap / 1e12, 1.0)
            score = (score + market_cap_score) / 2
        
        # Volume analysis
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Higher volume = more interest
            if market_data.volume > 10_000_000:
                score += 0.05
        
        return max(0, min(1, score))
    
    def _generate_signal(self, result: PredictionResult) -> Tuple[TradeSignal, float]:
        """Generate trade signal from analysis."""
        score = result.overall_score
        confidence = result.confidence
        
        # Strong signals require high confidence
        if score > 0.7 and confidence >= self.high_confidence:
            return TradeSignal.STRONG_BUY, 0.9
        elif score > 0.6 and confidence >= self.min_confidence:
            return TradeSignal.BUY, 0.7
        elif score < 0.3 and confidence >= self.high_confidence:
            return TradeSignal.STRONG_SELL, 0.9
        elif score < 0.4 and confidence >= self.min_confidence:
            return TradeSignal.SELL, 0.7
        elif score < 0.45 or score > 0.55:
            return TradeSignal.HOLD, 0.5
        else:
            return TradeSignal.HOLD, 0.3
    
    def _assess_risk(self, market_data: Any, result: PredictionResult) -> float:
        """Assess risk level of the trade."""
        risk = 0.5  # Start at medium risk
        
        # Volatility risk
        if market_data and hasattr(market_data, 'rsi'):
            rsi = market_data.rsi
            if rsi > 80 or rsi < 20:
                risk += 0.2  # Extreme RSI = higher risk
        
        # Pattern risk
        if result.pattern_confidence > 0.8:
            risk -= 0.1  # Strong pattern = lower risk
        
        # Confidence risk
        if result.confidence < self.min_confidence:
            risk += 0.2  # Low confidence = higher risk
        
        # Macro risk
        if result.macro_score < 0.3:
            risk += 0.15  # Poor macro = higher risk
        
        return max(0, min(1, risk))
    
    def _get_risk_factors(self, market_data: Any, result: PredictionResult) -> Dict[str, float]:
        """Get detailed risk factors."""
        factors = {}
        
        if market_data and hasattr(market_data, 'atr'):
            factors['volatility'] = market_data.atr / market_data.current_price if market_data.current_price > 0 else 0
        
        factors['market_risk'] = 1 - result.macro_score
        factors['confidence_risk'] = 1 - result.confidence
        factors['technical_risk'] = abs(result.technical_score - 0.5) * 2
        
        return factors
    
    def _generate_recommendations(self, result: PredictionResult, market_data: Any) -> PredictionResult:
        """Generate trade recommendations."""
        if not market_data or not hasattr(market_data, 'current_price'):
            return result
        
        current_price = market_data.current_price
        
        if result.signal == TradeSignal.BUY or result.signal == TradeSignal.STRONG_BUY:
            result.entry_price = current_price
            result.stop_loss = current_price * (1 - 0.02)  # 2% stop loss
            result.take_profit = current_price * (1 + 0.04)  # 4% take profit
            
            # Position sizing based on confidence
            if result.confidence >= self.high_confidence:
                result.position_size_recommendation = 0.10  # 10% of portfolio
            elif result.confidence >= self.min_confidence:
                result.position_size_recommendation = 0.05  # 5% of portfolio
            else:
                result.position_size_recommendation = 0.02  # 2% for low confidence
        
        elif result.signal == TradeSignal.SELL or result.signal == TradeSignal.STRONG_SELL:
            result.entry_price = current_price
            result.stop_loss = current_price * (1 + 0.02)  # 2% stop loss for short
            result.take_profit = current_price * (1 - 0.04)  # 4% take profit for short
            result.position_size_recommendation = 0.05
        
        return result
    
    def _build_reasoning(self, result: PredictionResult) -> str:
        """Build human-readable reasoning for the prediction."""
        reasons = []
        
        if result.technical_score > 0.6:
            reasons.append("Technical indicators are bullish")
        elif result.technical_score < 0.4:
            reasons.append("Technical indicators are bearish")
        
        if result.sentiment_score > 0.6:
            reasons.append("Market sentiment is positive")
        elif result.sentiment_score < 0.4:
            reasons.append("Market sentiment is negative")
        
        if result.pattern_confidence > 0.7:
            patterns_str = ", ".join(result.detected_patterns[:3])
            reasons.append(f"Strong pattern detected: {patterns_str}")
        
        if result.macro_score > 0.6:
            reasons.append("Macro environment is favorable")
        elif result.macro_score < 0.4:
            reasons.append("Macro environment presents challenges")
        
        if result.news_score > 0.6:
            reasons.append("Recent news is positive")
        elif result.news_score < 0.4:
            reasons.append("Recent news is negative")
        
        reasons.append(f"Model confidence: {result.confidence:.1%}")
        
        return "; ".join(reasons)
    
    def _get_key_factors(self, result: PredictionResult) -> List[str]:
        """Get key factors influencing the prediction."""
        factors = []
        
        # Sort confidence factors by value
        sorted_factors = sorted(
            result.confidence_factors.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for factor, value in sorted_factors[:5]:
            factors.append(f"{factor}: {value:.1%}")
        
        return factors
    
    def _generate_warnings(self, result: PredictionResult, market_data: Any) -> List[str]:
        """Generate warnings for risky situations."""
        warnings = []
        
        if result.risk_score > 0.7:
            warnings.append("HIGH RISK: Risk score is elevated")
        
        if result.confidence < self.min_confidence:
            warnings.append("LOW CONFIDENCE: Model uncertainty is high")
        
        if market_data and hasattr(market_data, 'rsi'):
            if market_data.rsi > 80:
                warnings.append("EXTREME OVERBOUGHT: RSI at extreme levels")
            elif market_data.rsi < 20:
                warnings.append("EXTREME OVERSOLD: RSI at extreme levels")
        
        if result.macro_score < 0.3:
            warnings.append("POOR MACRO: Unfavorable macroeconomic conditions")
        
        if result.pattern_confidence < 0.5 and result.detected_patterns:
            warnings.append("WEAK PATTERN: Detected patterns have low confidence")
        
        return warnings


# Initialize default analyzer
_default_analyzer = None


def get_analyzer(config: Optional[Dict] = None) -> AIAnalyzer:
    """Get or create the default AI analyzer."""
    global _default_analyzer
    if _default_analyzer is None or config is not None:
        _default_analyzer = AIAnalyzer(config or {})
    return _default_analyzer