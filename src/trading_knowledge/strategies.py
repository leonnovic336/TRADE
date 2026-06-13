"""
COMPREHENSIVE TRADING STRATEGIES ENGINE
Implements all major trading strategies based on trading education
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ==========================================
# TECHNICAL INDICATORS
# ==========================================

class TechnicalIndicator(Enum):
    """Technical analysis indicators."""
    SMA = "sma"                    # Simple Moving Average
    EMA = "ema"                    # Exponential Moving Average
    RSI = "rsi"                   # Relative Strength Index
    MACD = "macd"                  # Moving Average Convergence Divergence
    BOLLINGER = "bollinger"         # Bollinger Bands
    ATR = "atr"                   # Average True Range
    ADX = "adx"                   # Average Directional Index
    STOCHASTIC = "stochastic"      # Stochastic Oscillator
    CCI = "cci"                   # Commodity Channel Index
    WILLIAMS_R = "williams_r"     # Williams %R
    OBV = "obv"                   # On-Balance Volume
    VWAP = "vwap"                 # Volume Weighted Average Price
    FIBONACCI = "fibonacci"        # Fibonacci Retracements
    PIVOT = "pivot"               # Pivot Points


class TrendDirection(Enum):
    """Trend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis results."""
    # Trend indicators
    trend: TrendDirection = TrendDirection.NEUTRAL
    trend_strength: float = 0.0  # 0-1
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    
    # Momentum indicators
    rsi: float = 50.0
    rsi_signal: str = "neutral"
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Volatility indicators
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    
    # Oscillators
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    cci: float = 0.0
    williams_r: float = -50.0
    
    # Volume
    obv: float = 0.0
    vwap: float = 0.0
    volume_ratio: float = 1.0
    
    # Support/Resistance
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    pivot_point: float = 0.0
    
    # Pattern detection
    detected_patterns: List[str] = field(default_factory=list)
    pattern_confidence: float = 0.0


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis engine.
    Calculates all major technical indicators.
    """
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalAnalyzer.calculate_atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def calculate_williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap
    
    @staticmethod
    def find_support_resistance(
        data: pd.Series,
        window: int = 20,
        threshold: float = 0.001
    ) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels."""
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(data) - window):
            is_support = True
            is_resistance = True
            
            current = data.iloc[i]
            
            # Check if it's a local minimum (support)
            for j in range(i - window, i + window + 1):
                if j != i and j < len(data):
                    if data.iloc[j] < current:
                        is_support = False
                        break
            
            # Check if it's a local maximum (resistance)
            for j in range(i - window, i + window + 1):
                if j != i and j < len(data):
                    if data.iloc[j] > current:
                        is_resistance = False
                        break
            
            if is_support:
                support_levels.append(current)
            if is_resistance:
                resistance_levels.append(current)
        
        # Remove duplicates within threshold
        support_levels = sorted(set([
            round(s, 2) for s in support_levels
            if not any(abs(s - other) / s < threshold for other in support_levels if other != s)
        ]))
        resistance_levels = sorted(set([
            round(r, 2) for r in resistance_levels
            if not any(abs(r - other) / r < threshold for other in resistance_levels if other != r)
        ]))
        
        return support_levels, resistance_levels
    
    def analyze(self, df: pd.DataFrame) -> TechnicalAnalysis:
        """Perform complete technical analysis."""
        analysis = TechnicalAnalysis()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=close.index)
        
        # Moving averages
        analysis.sma_20 = float(close.tail(20).mean())
        analysis.sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else analysis.sma_20
        analysis.sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else analysis.sma_50
        analysis.ema_12 = float(close.ewm(span=12).mean().iloc[-1])
        analysis.ema_26 = float(close.ewm(span=26).mean().iloc[-1])
        
        # Trend detection
        current_price = float(close.iloc[-1])
        if current_price > analysis.sma_20 > analysis.sma_50 > analysis.sma_200:
            analysis.trend = TrendDirection.BULLISH
            analysis.trend_strength = 0.8
        elif current_price < analysis.sma_20 < analysis.sma_50 < analysis.sma_200:
            analysis.trend = TrendDirection.BEARISH
            analysis.trend_strength = 0.8
        elif current_price > analysis.sma_20:
            analysis.trend = TrendDirection.BULLISH
            analysis.trend_strength = 0.5
        else:
            analysis.trend = TrendDirection.BEARISH
            analysis.trend_strength = 0.5
        
        # RSI
        analysis.rsi = float(self.calculate_rsi(close).iloc[-1])
        if analysis.rsi > 70:
            analysis.rsi_signal = "overbought"
        elif analysis.rsi < 30:
            analysis.rsi_signal = "oversold"
        else:
            analysis.rsi_signal = "neutral"
        
        # MACD
        macd, signal, hist = self.calculate_macd(close)
        analysis.macd = float(macd.iloc[-1])
        analysis.macd_signal = float(signal.iloc[-1])
        analysis.macd_histogram = float(hist.iloc[-1])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        analysis.bollinger_upper = float(bb_upper.iloc[-1])
        analysis.bollinger_middle = float(bb_middle.iloc[-1])
        analysis.bollinger_lower = float(bb_lower.iloc[-1])
        
        # ATR
        analysis.atr = float(self.calculate_atr(high, low, close).iloc[-1])
        analysis.volatility = float(close.pct_change().std() * np.sqrt(252))
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
        analysis.stochastic_k = float(stoch_k.iloc[-1])
        analysis.stochastic_d = float(stoch_d.iloc[-1])
        
        # CCI
        analysis.cci = float(self.calculate_cci(high, low, close).iloc[-1])
        
        # Williams %R
        analysis.williams_r = float(self.calculate_williams_r(high, low, close).iloc[-1])
        
        # Volume indicators
        analysis.obv = float(self.calculate_obv(close, volume).iloc[-1])
        analysis.vwap = float(self.calculate_vwap(high, low, close, volume).iloc[-1])
        analysis.volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean()) if len(volume) >= 20 else 1.0
        
        # Support/Resistance
        support, resistance = self.find_support_resistance(close)
        analysis.support_levels = support[-5:] if len(support) > 5 else support
        analysis.resistance_levels = resistance[-5:] if len(resistance) > 5 else resistance
        
        # Pivot points
        pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        analysis.pivot_point = float(pivot)
        
        return analysis


# ==========================================
# PATTERN RECOGNITION
# ==========================================

class CandlestickPattern(Enum):
    """Candlestick pattern types."""
    # Reversal Patterns
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    DOJI = "doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    HAMMER_INVERTED = "hammer_inverted"
    
    # Continuation Patterns
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    MARUBOZU = "marubozu"
    SPINNING_TOP = "spinning_top"
    
    # Doji patterns
    LONG_LEGGED_DOJI = "long_legged_doji"


class ChartPattern(Enum):
    """Chart pattern types."""
    # Reversal Patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_BOTTOM = "rounding_bottom"
    
    # Continuation Patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PENNANT_BULL = "pennant_bull"
    PENNANT_BEAR = "pennant_bear"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    
    # Other
    CUP_AND_HANDLE = "cup_and_handle"
    RECTANGLE = "rectangle"
    CHANNEL = "channel"


class PatternRecognizer:
    """
    Pattern recognition for candlestick and chart patterns.
    """
    
    @staticmethod
    def get_candle_color(open_price: float, close_price: float) -> str:
        """Determine if candle is bullish or bearish."""
        return "bullish" if close_price >= open_price else "bearish"
    
    @staticmethod
    def is_hammer(open_price: float, high: float, low: float, close_price: float) -> bool:
        """Detect Hammer pattern (bullish reversal)."""
        body = abs(close_price - open_price)
        lower_shadow = min(open_price, close_price) - low
        upper_shadow = high - max(open_price, close_price)
        
        # Small body at top
        is_small_body = body < 0.1 * (high - low)
        # Long lower shadow (2-3x body)
        is_long_lower = lower_shadow > 2 * body
        # Small or no upper shadow
        is_small_upper = upper_shadow < 0.1 * body
        
        return is_small_body and is_long_lower and is_small_upper
    
    @staticmethod
    def is_engulfing_bullish(
        prev_open: float, prev_close: float,
        curr_open: float, curr_close: float
    ) -> bool:
        """Detect Bullish Engulfing pattern."""
        # Previous candle is bearish
        prev_bearish = prev_close < prev_open
        # Current candle is bullish
        curr_bullish = curr_close > curr_open
        
        # Current opens below previous close
        curr_opens_below = curr_open < prev_close
        # Current closes above previous open
        curr_closes_above = curr_close > prev_open
        
        return prev_bearish and curr_bullish and curr_opens_below and curr_closes_above
    
    @staticmethod
    def is_engulfing_bearish(
        prev_open: float, prev_close: float,
        curr_open: float, curr_close: float
    ) -> bool:
        """Detect Bearish Engulfing pattern."""
        # Previous candle is bullish
        prev_bullish = prev_close > prev_open
        # Current candle is bearish
        curr_bearish = curr_close < curr_open
        
        # Current opens above previous close
        curr_opens_above = curr_open > prev_close
        # Current closes below previous open
        curr_closes_below = curr_close < prev_open
        
        return prev_bullish and curr_bearish and curr_opens_above and curr_closes_below
    
    @staticmethod
    def is_doji(open_price: float, close_price: float, high: float, low: float) -> bool:
        """Detect Doji pattern (indecision)."""
        body = abs(close_price - open_price)
        range_size = high - low
        
        # Very small body relative to range
        return body < 0.1 * range_size
    
    @staticmethod
    def is_morning_star(
        prev_open: float, prev_close: float,
        curr_open: float, curr_close: float,
        middle_open: float, middle_close: float
    ) -> bool:
        """Detect Morning Star pattern (bullish reversal)."""
        # First candle: long bearish
        first_long_bearish = prev_close < prev_open and \
                            (prev_close - prev_open) / (prev_open - prev_open) < -0.6
        
        # Middle candle: small body with gap down
        middle_gap = middle_open < prev_close
        
        # Third candle: long bullish closing above middle
        third_long_bullish = curr_close > curr_open and \
                           (curr_close - curr_open) / (curr_open - curr_open) > 0.6
        third_recovery = curr_close > (prev_open + prev_close) / 2
        
        return first_long_bearish and middle_gap and third_long_bullish and third_recovery
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """Detect all candlestick patterns in recent candles."""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        close = df['Close'].values
        open_prices = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        
        for i in range(2, len(df)):
            # Hammer
            if self.is_hammer(open_prices[i], high[i], low[i], close[i]):
                patterns.append((CandlestickPattern.HAMMER.value, 0.7))
            
            # Engulfing patterns
            if self.is_engulfing_bullish(
                open_prices[i-1], close[i-1],
                open_prices[i], close[i]
            ):
                patterns.append((CandlestickPattern.ENGULFING_BULLISH.value, 0.75))
            
            if self.is_engulfing_bearish(
                open_prices[i-1], close[i-1],
                open_prices[i], close[i]
            ):
                patterns.append((CandlestickPattern.ENGULFING_BEARISH.value, 0.75))
            
            # Doji
            if self.is_doji(open_prices[i], close[i], high[i], low[i]):
                patterns.append((CandlestickPattern.DOJI.value, 0.6))
            
            # Morning Star
            if i >= 2:
                if self.is_morning_star(
                    open_prices[i-2], close[i-2],
                    open_prices[i], close[i],
                    open_prices[i-1], close[i-1]
                ):
                    patterns.append((CandlestickPattern.MORNING_STAR.value, 0.8))
        
        return patterns


# ==========================================
# STRATEGY IMPLEMENTATIONS
# ==========================================

@dataclass
class StrategySignal:
    """Signal generated by a trading strategy."""
    strategy_name: str
    signal_type: str  # "buy", "sell", "hold"
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: List[str] = field(default_factory=list)
    indicators: Dict[str, float] = field(default_factory=dict)


class MomentumStrategy:
    """Momentum trading strategy - trade in direction of strong trends."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "Momentum"
    
    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        """Generate momentum-based signal."""
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(df)
        
        # Momentum indicators
        rsi = analysis.rsi
        macd_hist = analysis.macd_histogram
        trend = analysis.trend
        
        # Entry and exit prices
        current_price = float(df['Close'].iloc[-1])
        atr = analysis.atr
        
        if trend == TrendDirection.BULLISH and rsi > 50 and macd_hist > 0:
            # Bullish momentum
            signal_type = "buy"
            strength = min(0.5 + (rsi - 50) / 100 + analysis.trend_strength / 2, 1.0)
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
            reasoning = [
                f"Trend: {trend.value} (strength: {analysis.trend_strength:.1%})",
                f"RSI: {rsi:.1f} (momentum confirming)",
                f"MACD histogram: {macd_hist:.4f} (positive)",
            ]
        
        elif trend == TrendDirection.BEARISH and rsi < 50 and macd_hist < 0:
            # Bearish momentum
            signal_type = "sell"
            strength = min(0.5 + (50 - rsi) / 100 + analysis.trend_strength / 2, 1.0)
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
            reasoning = [
                f"Trend: {trend.value} (strength: {analysis.trend_strength:.1%})",
                f"RSI: {rsi:.1f} (negative momentum)",
                f"MACD histogram: {macd_hist:.4f} (negative)",
            ]
        
        else:
            # No clear momentum
            signal_type = "hold"
            strength = 0.5
            stop_loss = current_price - atr
            take_profit = current_price + atr
            reasoning = ["No clear momentum signal"]
        
        confidence = strength * analysis.pattern_confidence if analysis.pattern_confidence > 0 else strength
        
        return StrategySignal(
            strategy_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            indicators={
                "rsi": rsi,
                "macd_histogram": macd_hist,
                "trend_strength": analysis.trend_strength,
                "atr": atr,
            }
        )


class MeanReversionStrategy:
    """Mean reversion strategy - fade extreme moves."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "Mean Reversion"
    
    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        """Generate mean reversion signal."""
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(df)
        
        current_price = float(df['Close'].iloc[-1])
        bb_upper = analysis.bollinger_upper
        bb_lower = analysis.bollinger_lower
        bb_middle = analysis.bollinger_middle
        
        # Calculate position within bands
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # RSI signals
        rsi = analysis.rsi
        atr = analysis.atr
        
        if bb_position < 0.2 and rsi < 30:
            # Oversold - expect bounce
            signal_type = "buy"
            strength = 0.5 + (0.2 - bb_position)
            stop_loss = bb_lower
            take_profit = bb_middle
            reasoning = [
                f"Price below lower Bollinger Band",
                f"RSI oversold: {rsi:.1f}",
                f"Expected reversion to mean",
            ]
        
        elif bb_position > 0.8 and rsi > 70:
            # Overbought - expect pullback
            signal_type = "sell"
            strength = 0.5 + (bb_position - 0.8)
            stop_loss = bb_upper
            take_profit = bb_middle
            reasoning = [
                f"Price above upper Bollinger Band",
                f"RSI overbought: {rsi:.1f}",
                f"Expected reversion to mean",
            ]
        
        else:
            # Near mean - no signal
            signal_type = "hold"
            strength = 0.5
            stop_loss = current_price - atr
            take_profit = current_price + atr
            reasoning = ["Price near mean"]
        
        return StrategySignal(
            strategy_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            indicators={
                "bb_position": bb_position,
                "rsi": rsi,
                "bollinger_width": bb_upper - bb_lower,
            }
        )


class BreakoutStrategy:
    """Breakout strategy - trade breakouts from consolidation."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "Breakout"
        self.lookback = config.get("lookback", 20) if config else 20
    
    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        """Generate breakout signal."""
        if len(df) < self.lookback:
            return StrategySignal(
                strategy_name=self.name,
                signal_type="hold",
                strength=0.5,
                confidence=0.3,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                reasoning=["Insufficient data"]
            )
        
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate resistance and support from lookback period
        lookback_data = df.tail(self.lookback)
        resistance = float(lookback_data['High'].max())
        support = float(lookback_data['Low'].min())
        
        # Calculate breakout threshold
        range_size = resistance - support
        threshold = 0.001  # 0.1% threshold
        
        atr = float(TechnicalAnalyzer.calculate_atr(
            lookback_data['High'],
            lookback_data['Low'],
            lookback_data['Close']
        ).iloc[-1])
        
        analyzer = TechnicalAnalyzer()
        patterns = PatternRecognizer().detect_candlestick_patterns(lookback_data)
        
        # Bullish breakout
        if current_price > resistance * (1 + threshold):
            signal_type = "buy"
            strength = min(abs(current_price - resistance) / resistance * 5, 1.0)
            stop_loss = support
            take_profit = current_price + 2 * range_size
            reasoning = [
                f"Breakout above resistance: ${resistance:.2f}",
                f"Range size: ${range_size:.2f}",
                f"Volume confirm: {len(patterns) > 0}",
            ]
        
        # Bearish breakdown
        elif current_price < support * (1 - threshold):
            signal_type = "sell"
            strength = min(abs(support - current_price) / support * 5, 1.0)
            stop_loss = resistance
            take_profit = current_price - 2 * range_size
            reasoning = [
                f"Breakdown below support: ${support:.2f}",
                f"Range size: ${range_size:.2f}",
            ]
        
        # Consolidation - no signal
        else:
            signal_type = "hold"
            strength = 0.5
            stop_loss = current_price - atr
            take_profit = current_price + atr
            reasoning = [
                f"Inside range: ${support:.2f} - ${resistance:.2f}",
                f"No breakout yet",
            ]
        
        return StrategySignal(
            strategy_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            indicators={
                "resistance": resistance,
                "support": support,
                "range_size": range_size,
            }
        )


class ScalpingStrategy:
    """Scalping strategy - trade on small price movements."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "Scalping"
        self.fast_ema = config.get("fast_ema", 9) if config else 9
        self.slow_ema = config.get("slow_ema", 21) if config else 21
    
    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        """Generate scalping signal."""
        if len(df) < max(self.fast_ema, self.slow_ema):
            return StrategySignal(
                strategy_name=self.name,
                signal_type="hold",
                strength=0.5,
                confidence=0.3,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                reasoning=["Insufficient data"]
            )
        
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate EMAs
        fast_ema = float(df['Close'].ewm(span=self.fast_ema).mean().iloc[-1])
        slow_ema = float(df['Close'].ewm(span=self.slow_ema).mean().iloc[-1])
        
        # Calculate VWAP for reference
        analyzer = TechnicalAnalyzer()
        vwap = float(analyzer.calculate_vwap(
            df['High'], df['Low'], df['Close'], df['Volume']
        ).iloc[-1])
        
        # ATR for stops
        atr = float(analyzer.calculate_atr(
            df['High'], df['Low'], df['Close']
        ).iloc[-1])
        
        # Scalping signals based on EMA crossover
        if fast_ema > slow_ema and current_price > vwap:
            signal_type = "buy"
            strength = min((fast_ema - slow_ema) / slow_ema * 100, 1.0)
            stop_loss = current_price - 0.5 * atr
            take_profit = current_price + 1.0 * atr
            reasoning = [
                f"Fast EMA ({self.fast_ema}): ${fast_ema:.2f}",
                f"Slow EMA ({self.slow_ema}): ${slow_ema:.2f}",
                f"Price above VWAP: ${vwap:.2f}",
            ]
        
        elif fast_ema < slow_ema and current_price < vwap:
            signal_type = "sell"
            strength = min((slow_ema - fast_ema) / slow_ema * 100, 1.0)
            stop_loss = current_price + 0.5 * atr
            take_profit = current_price - 1.0 * atr
            reasoning = [
                f"Fast EMA ({self.fast_ema}): ${fast_ema:.2f}",
                f"Slow EMA ({self.slow_ema}): ${slow_ema:.2f}",
                f"Price below VWAP: ${vwap:.2f}",
            ]
        
        else:
            signal_type = "hold"
            strength = 0.5
            stop_loss = current_price - 0.5 * atr
            take_profit = current_price + 0.5 * atr
            reasoning = ["No clear scalping signal"]
        
        return StrategySignal(
            strategy_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            indicators={
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "vwap": vwap,
                "ema_diff": fast_ema - slow_ema,
            }
        )


class SwingTradingStrategy:
    """Swing trading strategy - capture multi-day swings."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = "Swing Trading"
    
    def analyze(self, df: pd.DataFrame) -> StrategySignal:
        """Generate swing trading signal."""
        analyzer = TechnicalAnalyzer()
        analysis = analyzer.analyze(df)
        
        current_price = float(df['Close'].iloc[-1])
        atr = analysis.atr
        
        # Use daily data for swing trades
        # Look for RSI extremes with trend confirmation
        
        rsi = analysis.rsi
        adx = float(TechnicalAnalyzer.calculate_adx(
            df['High'], df['Low'], df['Close']
        ).iloc[-1])
        
        if analysis.trend == TrendDirection.BULLISH and rsi < 40:
            # Pullback in uptrend - buy opportunity
            signal_type = "buy"
            strength = 0.6 + analysis.trend_strength * 0.3
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 4 * atr
            reasoning = [
                f"Uptrend confirmed (SMA confirmation)",
                f"RSI pullback: {rsi:.1f} (potential entry)",
                f"ADX: {adx:.1f} (trend strength)",
            ]
        
        elif analysis.trend == TrendDirection.BULLISH and rsi > 70:
            # Overbought in uptrend - hold profits
            signal_type = "hold"
            strength = 0.5
            stop_loss = analysis.sma_20 * 0.98
            take_profit = current_price + 2 * atr
            reasoning = [
                f"Uptrend but overbought",
                f"RSI: {rsi:.1f} (wait for pullback)",
            ]
        
        elif analysis.trend == TrendDirection.BEARISH and rsi > 60:
            # Pullback in downtrend - sell opportunity
            signal_type = "sell"
            strength = 0.6 + analysis.trend_strength * 0.3
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 4 * atr
            reasoning = [
                f"Downtrend confirmed",
                f"RSI bounce: {rsi:.1f} (potential short)",
                f"ADX: {adx:.1f} (trend strength)",
            ]
        
        else:
            signal_type = "hold"
            strength = 0.5
            stop_loss = current_price - atr
            take_profit = current_price + atr
            reasoning = ["No clear swing setup"]
        
        return StrategySignal(
            strategy_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            indicators={
                "rsi": rsi,
                "adx": adx,
                "trend": analysis.trend.value,
                "trend_strength": analysis.trend_strength,
            }
        )


# ==========================================
# STRATEGY MANAGER
# ==========================================

class StrategyManager:
    """
    Manages all trading strategies and generates combined signals.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize all strategies
        self.strategies = {
            "momentum": MomentumStrategy(config),
            "mean_reversion": MeanReversionStrategy(config),
            "breakout": BreakoutStrategy(config),
            "scalping": ScalpingStrategy(config),
            "swing": SwingTradingStrategy(config),
        }
        
        # Strategy weights for combining signals
        self.weights = self.config.get("strategy_weights", {
            "momentum": 0.3,
            "mean_reversion": 0.2,
            "breakout": 0.2,
            "scalping": 0.15,
            "swing": 0.15,
        })
    
    def get_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Get combined signal from all strategies."""
        signals = {}
        weights = {}
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(df)
                signals[name] = signal
                weights[name] = self.weights.get(name, 0.2)
            except Exception as e:
                logger.error(f"Error in strategy {name}: {e}")
        
        if not signals:
            return StrategySignal(
                strategy_name="combined",
                signal_type="hold",
                strength=0.5,
                confidence=0.3,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                reasoning=["No strategies generated signals"]
            )
        
        # Combine signals
        buy_signals = [s for s in signals.values() if s.signal_type == "buy"]
        sell_signals = [s for s in signals.values() if s.signal_type == "sell"]
        
        # Calculate weighted scores
        buy_score = sum(s.strength * weights[name] for name, s in signals.items() if s.signal_type == "buy")
        sell_score = sum(s.strength * weights[name] for name, s in signals.items() if s.signal_type == "sell")
        
        total_weight = sum(weights.values())
        buy_score /= total_weight
        sell_score /= total_weight
        
        # Determine final signal
        if buy_score > 0.6:
            final_signal = "buy"
            strength = buy_score
            # Use average of buy signals for entry/stop/target
            avg_entry = np.mean([s.entry_price for s in buy_signals])
            avg_stop = np.mean([s.stop_loss for s in buy_signals])
            avg_target = np.mean([s.take_profit for s in buy_signals])
        elif sell_score > 0.6:
            final_signal = "sell"
            strength = sell_score
            avg_entry = np.mean([s.entry_price for s in sell_signals])
            avg_stop = np.mean([s.stop_loss for s in sell_signals])
            avg_target = np.mean([s.take_profit for s in sell_signals])
        else:
            final_signal = "hold"
            strength = 0.5
            current_price = float(df['Close'].iloc[-1])
            avg_entry = current_price
            avg_stop = current_price
            avg_target = current_price
        
        # Aggregate reasoning
        all_reasons = []
        for name, signal in signals.items():
            if signal.signal_type != "hold":
                all_reasons.append(f"{name}: {signal.signal_type} ({signal.strength:.1%})")
        
        confidence = strength * (len([s for s in signals.values() if s.signal_type != "hold"]) / max(len(signals), 1))
        
        return StrategySignal(
            strategy_name="combined",
            signal_type=final_signal,
            strength=strength,
            confidence=confidence,
            entry_price=avg_entry,
            stop_loss=avg_stop,
            take_profit=avg_target,
            reasoning=all_reasons,
        )