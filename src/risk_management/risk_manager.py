"""
Risk Management System
Comprehensive risk controls, position sizing, and loss prevention.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Position metrics
    position_size: float = 0.0
    position_percent: float = 0.0
    portfolio_concentration: float = 0.0
    
    # Loss metrics
    max_loss_tolerance: float = 0.0
    current_loss: float = 0.0
    loss_percent: float = 0.0
    days_since_loss: int = 0
    
    # Volatility metrics
    portfolio_volatility: float = 0.0
    position_volatility: float = 0.0
    correlation_risk: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_tolerance: float = 0.0
    
    # Exposure metrics
    total_exposure: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    single_stock_exposure: float = 0.0
    
    # Score
    overall_risk_score: float = 0.0  # 0-100
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Configured risk limits."""
    # Position limits
    max_position_size: float = 0.10  # 10% of portfolio
    max_total_exposure: float = 0.80  # 80% max exposure
    max_single_stock: float = 0.20  # 20% max in single stock
    min_position_size: float = 0.01  # 1% min position
    
    # Loss limits
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_weekly_loss: float = 0.10  # 10% weekly loss limit
    max_monthly_loss: float = 0.20  # 20% monthly loss limit
    max_total_drawdown: float = 0.15  # 15% max drawdown
    
    # Position count
    max_open_positions: int = 10
    min_cash_reserve: float = 0.10  # 10% cash reserve
    
    # Stop loss defaults
    default_stop_loss: float = 0.02  # 2%
    default_take_profit: float = 0.04  # 4%
    trailing_stop_enabled: bool = True
    trailing_stop_percent: float = 0.015  # 1.5%
    
    # Circuit breaker
    circuit_breaker_threshold: float = 0.03  # 3% daily loss triggers pause
    circuit_breaker_cooldown: int = 3600  # 1 hour cooldown
    
    # Volatility limits
    max_portfolio_volatility: float = 0.25  # 25% annual volatility max
    max_position_volatility: float = 0.50  # 50% annual volatility max
    volatility_adjustment: bool = True  # Reduce position size in high volatility
    
    # Correlation limits
    max_correlation: float = 0.70  # Max correlation between positions
    diversification_bonus: float = 0.05  # Bonus for diversification


@dataclass
class TradeRecommendation:
    """Validated trade recommendation with risk adjustments."""
    symbol: str
    side: str  # buy, sell
    original_quantity: float
    adjusted_quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Risk metrics
    risk_amount: float
    risk_percent: float
    position_size: float
    position_percent: float
    
    # Validation
    is_approved: bool
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Score
    risk_score: float = 0.0
    confidence: float = 0.0


class PositionSizer:
    """
    Dynamic position sizing based on risk parameters.
    """
    
    def __init__(self, risk_limits: RiskLimits, volatility_target: float = 0.15):
        self.risk_limits = risk_limits
        self.volatility_target = volatility_target
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.02,
        current_volatility: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate optimal position size.
        
        Returns: (quantity, risk_amount, position_value)
        """
        # Base risk amount
        risk_amount = portfolio_value * risk_percent
        
        # Price difference for stop loss
        if entry_price <= 0 or stop_loss <= 0:
            return 0, 0, 0
        
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            return 0, 0, 0
        
        # Calculate raw quantity
        raw_quantity = risk_amount / price_risk
        position_value = raw_quantity * entry_price
        
        # Apply volatility adjustment
        if current_volatility > 0 and self.risk_limits.volatility_adjustment:
            volatility_ratio = self.volatility_target / max(current_volatility, 0.01)
            volatility_multiplier = min(volatility_ratio, 1.0)
            raw_quantity *= volatility_multiplier
            position_value *= volatility_multiplier
        
        # Apply position size limit
        max_position_value = portfolio_value * self.risk_limits.max_position_size
        position_value = min(position_value, max_position_value)
        
        # Round down to whole shares
        quantity = int(raw_quantity)
        
        # Final validation
        if quantity <= 0:
            return 0, 0, 0
        
        final_position_value = quantity * entry_price
        final_risk_amount = quantity * abs(entry_price - stop_loss)
        
        return quantity, final_risk_amount, final_position_value
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.
        Returns the recommended fraction of portfolio to risk.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply Kelly fraction (typically use 1/4 Kelly for safety)
        return max(0, min(kelly * fraction, self.risk_limits.max_position_size))
    
    def calculate_equal_weight(
        self,
        portfolio_value: float,
        num_positions: int,
        exclude_symbols: List[str] = None
    ) -> float:
        """Calculate equal weight position size."""
        max_positions = self.risk_limits.max_open_positions
        effective_positions = min(num_positions + 1, max_positions)
        
        available_capital = portfolio_value * self.risk_limits.max_total_exposure
        
        weight = 1.0 / effective_positions
        position_value = available_capital * weight
        
        return position_value
    
    def calculate_risk_parity(
        self,
        portfolio_value: float,
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """
        Calculate risk-parity weighted positions.
        Each position contributes equally to portfolio risk.
        """
        symbols = list(volatilities.keys())
        if not symbols:
            return {}
        
        # Inverse volatility weights
        inv_vol = {s: 1.0 / max(v, 0.01) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        weights = {s: inv_vol[s] / total_inv_vol for s in symbols}
        
        # Convert to position values
        exposure = portfolio_value * self.risk_limits.max_total_exposure
        positions = {s: weights[s] * exposure for s in symbols}
        
        return positions


class StopLossManager:
    """
    Advanced stop loss management.
    """
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool,
        atr: float = 0,
        volatility: float = 0,
        max_loss_percent: float = 0.02
    ) -> float:
        """
        Calculate optimal stop loss level.
        """
        # Base stop loss using max loss percent
        base_stop = entry_price * (1 - max_loss_percent) if is_long else entry_price * (1 + max_loss_percent)
        
        # ATR-based stop
        if atr > 0:
            atr_stop = entry_price - (2 * atr) if is_long else entry_price + (2 * atr)
            # Use the wider stop for more protection
            stop = min(base_stop, atr_stop) if is_long else max(base_stop, atr_stop)
        else:
            stop = base_stop
        
        # Volatility-based adjustment
        if volatility > 0 and volatility < 1:
            vol_multiplier = 1 + (volatility * 0.5)
            stop = entry_price * (1 - max_loss_percent * vol_multiplier) if is_long else \
                   entry_price * (1 + max_loss_percent * vol_multiplier)
        
        return stop
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0,
        is_long: bool = True
    ) -> float:
        """
        Calculate take profit level based on risk-reward ratio.
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        is_long: bool,
        trail_percent: float = 0.015
    ) -> float:
        """
        Calculate trailing stop level.
        """
        if is_long:
            return highest_price * (1 - trail_percent)
        else:
            return highest_price * (1 + trail_percent)
    
    def should_adjust_stop(
        self,
        current_price: float,
        current_stop: float,
        is_long: bool,
        profit_percent: float = 0.02,
        min_profit_lock: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Determine if stop loss should be moved (to lock in profits).
        """
        if profit_percent < min_profit_lock:
            return False, current_stop
        
        # Move stop to breakeven after 1% profit
        if profit_percent >= min_profit_lock and profit_percent < 2 * min_profit_lock:
            new_stop = current_stop  # Keep initial stop
            return False, new_stop
        
        # Move stop to lock in partial profit
        if is_long:
            new_stop = max(current_stop, current_price * (1 - self.risk_limits.default_stop_loss))
        else:
            new_stop = min(current_stop, current_price * (1 + self.risk_limits.default_stop_loss))
        
        moved = abs(new_stop - current_stop) > 0.001
        return moved, new_stop


class RiskCalculator:
    """
    Comprehensive risk calculation and validation.
    """
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]], portfolio_value: float) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        metrics = RiskMetrics()
        
        if not positions or portfolio_value <= 0:
            return metrics
        
        # Position exposure
        total_exposure = sum(p.get('market_value', 0) for p in positions)
        metrics.total_exposure = total_exposure / portfolio_value
        
        # Concentration risk
        for position in positions:
            symbol = position.get('symbol', '')
            value = position.get('market_value', 0)
            percent = value / portfolio_value
            
            if percent > metrics.single_stock_exposure:
                metrics.single_stock_exposure = percent
            
            # Sector exposure (would need sector mapping)
        
        # Calculate volatility (simplified)
        if len(positions) > 1:
            returns = [p.get('daily_return', 0) for p in positions]
            metrics.portfolio_volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Drawdown
        metrics.max_drawdown_tolerance = self.risk_limits.max_total_drawdown
        
        # Risk score (0-100 scale)
        metrics.overall_risk_score = self._calculate_risk_score(metrics)
        metrics.risk_level = self._get_risk_level(metrics.overall_risk_score)
        
        return metrics
    
    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0
        
        # Concentration risk (0-30 points)
        if metrics.single_stock_exposure > 0.20:
            score += 30 * (metrics.single_stock_exposure / 0.30)
        elif metrics.single_stock_exposure > 0.10:
            score += 15
        
        # Exposure risk (0-25 points)
        if metrics.total_exposure > 0.80:
            score += 25 * ((metrics.total_exposure - 0.80) / 0.20)
        elif metrics.total_exposure > 0.60:
            score += 10
        
        # Volatility risk (0-25 points)
        if metrics.portfolio_volatility > 0.25:
            score += 25
        elif metrics.portfolio_volatility > 0.15:
            score += 15
        
        # Drawdown risk (0-20 points)
        if metrics.current_drawdown > 0.10:
            score += 20
        elif metrics.current_drawdown > 0.05:
            score += 10
        
        return min(score, 100)
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if score < 20:
            return RiskLevel.VERY_LOW
        elif score < 40:
            return RiskLevel.LOW
        elif score < 60:
            return RiskLevel.MEDIUM
        elif score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def validate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        cash: float,
        current_positions: List[Dict[str, Any]]
    ) -> TradeRecommendation:
        """Validate a trade against all risk rules."""
        position_value = quantity * price
        position_percent = position_value / portfolio_value
        
        recommendation = TradeRecommendation(
            symbol=symbol,
            side=side,
            original_quantity=quantity,
            adjusted_quantity=quantity,
            entry_price=price,
            stop_loss=0,
            take_profit=0,
            risk_reward_ratio=0,
            risk_amount=0,
            risk_percent=0,
            position_size=position_value,
            position_percent=position_percent,
            is_approved=True,
        )
        
        # Rule 1: Check position size limit
        if position_percent > self.risk_limits.max_position_size:
            recommendation.is_approved = False
            recommendation.rejection_reasons.append(
                f"Position size {position_percent:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}"
            )
        
        # Rule 2: Check total exposure
        current_exposure = sum(p.get('market_value', 0) for p in current_positions)
        new_exposure = (current_exposure + position_value) / portfolio_value
        if new_exposure > self.risk_limits.max_total_exposure:
            recommendation.is_approved = False
            recommendation.rejection_reasons.append(
                f"Total exposure {new_exposure:.1%} would exceed limit {self.risk_limits.max_total_exposure:.1%}"
            )
        
        # Rule 3: Check cash available for buy orders
        if side.lower() == "buy" and position_value > cash:
            recommendation.is_approved = False
            recommendation.rejection_reasons.append(
                f"Insufficient cash: need ${position_value:.2f}, have ${cash:.2f}"
            )
        
        # Rule 4: Check minimum position size
        if position_percent < self.risk_limits.min_position_size:
            recommendation.is_approved = False
            recommendation.rejection_reasons.append(
                f"Position size {position_percent:.1%} below minimum {self.risk_limits.min_position_size:.1%}"
            )
        
        # Rule 5: Check position count
        if side.lower() == "buy" and len(current_positions) >= self.risk_limits.max_open_positions:
            recommendation.is_approved = False
            recommendation.rejection_reasons.append(
                f"Maximum positions ({self.risk_limits.max_open_positions}) reached"
            )
        
        # Rule 6: Check for duplicate position (scale-in only)
        for pos in current_positions:
            if pos.get('symbol') == symbol and side.lower() == "buy":
                recommendation.warnings.append(
                    f"Adding to existing position in {symbol}"
                )
        
        # Calculate risk metrics
        recommendation.risk_amount = position_value * self.risk_limits.default_stop_loss
        recommendation.risk_percent = self.risk_limits.default_stop_loss
        recommendation.risk_reward_ratio = self.risk_limits.default_take_profit / self.risk_limits.default_stop_loss
        
        # Calculate stop loss and take profit
        if recommendation.is_approved:
            recommendation.stop_loss = price * (1 - self.risk_limits.default_stop_loss)
            recommendation.take_profit = price * (1 + self.risk_limits.default_take_profit)
        
        return recommendation
    
    def check_loss_limits(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Check if loss limits have been breached."""
        daily_loss_percent = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        weekly_loss_percent = abs(weekly_pnl) / portfolio_value if weekly_pnl < 0 else 0
        monthly_loss_percent = abs(monthly_pnl) / portfolio_value if monthly_pnl < 0 else 0
        
        if daily_loss_percent >= self.risk_limits.max_daily_loss:
            return True, f"Daily loss limit breached: {daily_loss_percent:.1%} vs {self.risk_limits.max_daily_loss:.1%}"
        
        if weekly_loss_percent >= self.risk_limits.max_weekly_loss:
            return True, f"Weekly loss limit breached: {weekly_loss_percent:.1%} vs {self.risk_limits.max_weekly_loss:.1%}"
        
        if monthly_loss_percent >= self.risk_limits.max_monthly_loss:
            return True, f"Monthly loss limit breached: {monthly_loss_percent:.1%} vs {self.risk_limits.max_monthly_loss:.1%}"
        
        return False, ""
    
    def calculate_var(
        self,
        positions: List[Dict[str, Any]],
        portfolio_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        """
        if not positions or portfolio_value <= 0:
            return 0
        
        # Simplified VaR calculation
        returns = [p.get('daily_return', 0) for p in positions]
        weights = [p.get('market_value', 0) / portfolio_value for p in positions]
        
        # Portfolio return volatility
        portfolio_return_std = np.sqrt(sum(w**2 * r**2 for w, r in zip(weights, returns)) + 
                                        2 * sum(w1 * w2 * r1 * r2 * 0.5 
                                               for i, w1 in enumerate(weights) 
                                               for j, w2 in enumerate(weights) 
                                               if i < j
                                               for r1 in [returns[i]] 
                                               for r2 in [returns[j]]))
        
        # Z-score for confidence level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z = z_scores.get(confidence_level, 1.65)
        
        var = portfolio_value * portfolio_return_std * np.sqrt(time_horizon) * z
        
        return var


class RiskManager:
    """
    Main risk management orchestrator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize risk limits from config
        risk_config = config.get("risk_management", {})
        
        self.limits = RiskLimits(
            max_position_size=risk_config.get("max_position_size", 0.10),
            max_total_exposure=risk_config.get("max_total_exposure", 0.80),
            max_single_stock=risk_config.get("max_single_stock", 0.20),
            min_position_size=risk_config.get("min_position_size", 0.01),
            max_daily_loss=risk_config.get("max_daily_loss", 0.05),
            max_weekly_loss=risk_config.get("max_weekly_loss", 0.10),
            max_monthly_loss=risk_config.get("max_monthly_loss", 0.20),
            max_total_drawdown=risk_config.get("max_drawdown", 0.15),
            max_open_positions=risk_config.get("max_open_positions", 10),
            min_cash_reserve=risk_config.get("min_cash_reserve", 0.10),
            default_stop_loss=risk_config.get("default_stop_loss", 0.02),
            default_take_profit=risk_config.get("default_take_profit", 0.04),
            trailing_stop_enabled=risk_config.get("trailing_stop_enabled", True),
            trailing_stop_percent=risk_config.get("trailing_stop_percent", 0.015),
            circuit_breaker_threshold=risk_config.get("circuit_breaker_threshold", 0.03),
            circuit_breaker_cooldown=risk_config.get("circuit_breaker_cooldown", 3600),
            max_portfolio_volatility=risk_config.get("max_portfolio_volatility", 0.25),
            max_position_volatility=risk_config.get("max_position_volatility", 0.50),
            volatility_adjustment=risk_config.get("volatility_adjustment", True),
        )
        
        # Initialize components
        self.position_sizer = PositionSizer(self.limits)
        self.stop_loss_manager = StopLossManager(self.limits)
        self.risk_calculator = RiskCalculator(self.limits)
        
        # State
        self.last_circuit_breaker_time: Optional[datetime] = None
        self.circuit_breaker_active = False
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        
        logger.info("Risk manager initialized")
    
    def validate_and_size_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        cash: float,
        current_positions: List[Dict[str, Any]],
        atr: float = 0,
        volatility: float = 0,
    ) -> TradeRecommendation:
        """Validate trade and calculate optimal position size."""
        # First validate
        recommendation = self.risk_calculator.validate_trade(
            symbol, side, quantity, price, portfolio_value, cash, current_positions
        )
        
        if not recommendation.is_approved:
            return recommendation
        
        # Calculate optimal position size
        if side.lower() == "buy":
            adjusted_qty, risk_amount, position_value = self.position_sizer.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=price,
                stop_loss=price * (1 - self.limits.default_stop_loss),
                risk_percent=self.limits.default_stop_loss,
                current_volatility=volatility,
            )
            
            if adjusted_qty < quantity:
                recommendation.warnings.append(
                    f"Position size reduced from {quantity} to {adjusted_qty} shares"
                )
            
            recommendation.adjusted_quantity = adjusted_qty
            recommendation.position_size = position_value
            recommendation.risk_amount = risk_amount
            
            # Calculate adjusted stop loss and take profit
            recommendation.stop_loss = price * (1 - self.limits.default_stop_loss)
            recommendation.take_profit = price * (1 + self.limits.default_take_profit)
            recommendation.risk_reward_ratio = self.limits.default_take_profit / self.limits.default_stop_loss
        
        # Calculate risk score
        recommendation.risk_score = self._calculate_trade_risk_score(
            recommendation, portfolio_value
        )
        
        return recommendation
    
    def _calculate_trade_risk_score(
        self,
        recommendation: TradeRecommendation,
        portfolio_value: float
    ) -> float:
        """Calculate risk score for a specific trade."""
        score = 0
        
        # Position size risk (0-30)
        if recommendation.position_percent > 0.10:
            score += 30
        elif recommendation.position_percent > 0.05:
            score += 15
        
        # Risk-reward ratio (0-30)
        if recommendation.risk_reward_ratio < 1.5:
            score += 30
        elif recommendation.risk_reward_ratio < 2.0:
            score += 15
        
        # Risk percent (0-20)
        if recommendation.risk_percent > 0.03:
            score += 20
        elif recommendation.risk_percent > 0.02:
            score += 10
        
        # Cash utilization (0-20)
        if recommendation.side.lower() == "buy":
            position_ratio = recommendation.position_size / max(portfolio_value, 1)
            if position_ratio > 0.20:
                score += 20
            elif position_ratio > 0.10:
                score += 10
        
        return min(score, 100)
    
    def check_circuit_breaker(self, daily_pnl: float, portfolio_value: float) -> Tuple[bool, str]:
        """Check if circuit breaker should be triggered."""
        if self.circuit_breaker_active:
            # Check cooldown
            if self.last_circuit_breaker_time:
                elapsed = (datetime.now() - self.last_circuit_breaker_time).total_seconds()
                if elapsed < self.limits.circuit_breaker_cooldown:
                    return True, f"Circuit breaker cooling down ({int(elapsed)}s elapsed)"
                else:
                    self.circuit_breaker_active = False
        
        loss_percent = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        
        if loss_percent >= self.limits.circuit_breaker_threshold:
            self.circuit_breaker_active = True
            self.last_circuit_breaker_time = datetime.now()
            return True, f"Circuit breaker triggered: {loss_percent:.1%} daily loss"
        
        return False, ""
    
    def update_performance(
        self,
        current_portfolio_value: float,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float
    ):
        """Update performance tracking metrics."""
        self.daily_pnl = daily_pnl
        self.weekly_pnl = weekly_pnl
        self.monthly_pnl = monthly_pnl
        
        # Update peak value and drawdown
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            "timestamp": datetime.now(),
            "limits": {
                "max_position_size": f"{self.limits.max_position_size:.1%}",
                "max_total_exposure": f"{self.limits.max_total_exposure:.1%}",
                "max_daily_loss": f"{self.limits.max_daily_loss:.1%}",
                "max_drawdown": f"{self.limits.max_total_drawdown:.1%}",
            },
            "current_metrics": {
                "daily_pnl": self.daily_pnl,
                "weekly_pnl": self.weekly_pnl,
                "monthly_pnl": self.monthly_pnl,
                "peak_value": self.peak_portfolio_value,
                "current_drawdown": f"{self.current_drawdown:.1%}",
            },
            "circuit_breaker": {
                "active": self.circuit_breaker_active,
                "threshold": f"{self.limits.circuit_breaker_threshold:.1%}",
            },
            "stop_loss": {
                "default": f"{self.limits.default_stop_loss:.1%}",
                "take_profit": f"{self.limits.default_take_profit:.1%}",
                "trailing_enabled": self.limits.trailing_stop_enabled,
            },
        }
    
    def reset_daily(self):
        """Reset daily tracking metrics."""
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0