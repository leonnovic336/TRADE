"""
MONITORING, TELEMETRY & OBSERVABILITY
Prometheus metrics, alerting, and real-time dashboards
"""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


# ==========================================
# METRICS DEFINITIONS
# ==========================================

@dataclass
class Metric:
    """Metric definition."""
    name: str
    description: str
    metric_type: str  # "counter", "gauge", "histogram", "summary"
    labels: List[str] = field(default_factory=list)
    
    # For counters
    value: float = 0.0
    
    # For gauges
    min_value: float = float('inf')
    max_value: float = float('-inf')


class MetricsRegistry:
    """Central metrics registry."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
    
    def register(self, metric: Metric):
        """Register a new metric."""
        with self._lock:
            self.metrics[metric.name] = metric
    
    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        metric = self.get(name)
        if metric and metric.metric_type == "counter":
            with self._lock:
                metric.value += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        metric = self.get(name)
        if metric and metric.metric_type == "gauge":
            with self._lock:
                metric.value = value
                metric.min_value = min(metric.min_value, value)
                metric.max_value = max(metric.max_value, value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a value for histogram."""
        metric = self.get(name)
        if metric and metric.metric_type == "histogram":
            with self._lock:
                metric.value = value  # Simplified
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for metric in self.metrics.values():
                # HELP line
                lines.append(f"# HELP {metric.name} {metric.description}")
                
                # TYPE line
                lines.append(f"# TYPE {metric.name} {metric.metric_type}")
                
                # Value line
                if metric.labels:
                    label_str = "{" + ",".join(f'{l}=""' for l in metric.labels) + "}"
                    lines.append(f"{metric.name}{label_str} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")
                
                lines.append("")
        
        return "\n".join(lines)


# Global registry
REGISTRY = MetricsRegistry()

# Pre-register common metrics
TRADES_EXECUTED = Metric(
    name="omnitrade_trades_total",
    description="Total number of trades executed",
    metric_type="counter",
    labels=["symbol", "side", "result"]
)
REGISTRY.register(TRADES_EXECUTED)

LATENCY_HISTOGRAM = Metric(
    name="omnitrade_execution_latency_ms",
    description="Trade execution latency in milliseconds",
    metric_type="histogram",
    labels=["operation"]
)
REGISTRY.register(LATENCY_HISTOGRAM)

PORTFOLIO_VALUE = Metric(
    name="omnitrade_portfolio_value_dollars",
    description="Current portfolio value in dollars",
    metric_type="gauge"
)
REGISTRY.register(PORTFOLIO_VALUE)

PORTFOLIO_DRAWDOWN = Metric(
    name="omnitrade_current_drawdown_pct",
    description="Current portfolio drawdown percentage",
    metric_type="gauge"
)
REGISTRY.register(PORTFOLIO_DRAWDOWN)

AI_CONFIDENCE = Metric(
    name="omnitrade_ai_confidence",
    description="AI prediction confidence score",
    metric_type="gauge",
    labels=["signal_type", "symbol"]
)
REGISTRY.register(AI_CONFIDENCE)

VPIN_SCORE = Metric(
    name="omnitrade_vpin_score",
    description="Volume-synchronized probability of informed trading",
    metric_type="gauge",
    labels=["symbol"]
)
REGISTRY.register(VPIN_SCORE)

DAILY_PNL = Metric(
    name="omnitrade_daily_pnl_dollars",
    description="Daily profit/loss in dollars",
    metric_type="gauge"
)
REGISTRY.register(DAILY_PNL)

OPEN_POSITIONS = Metric(
    name="omnitrade_open_positions",
    description="Number of currently open positions",
    metric_type="gauge"
)
REGISTRY.register(OPEN_POSITIONS)

HEDGE_COST = Metric(
    name="omnitrade_hedge_costs_total",
    description="Total hedge costs incurred",
    metric_type="counter"
)
REGISTRY.register(HEDGE_COST)

SIGNALS_GENERATED = Metric(
    name="omnitrade_signals_total",
    description="Total AI signals generated",
    metric_type="counter",
    labels=["signal_type", "symbol"]
)
REGISTRY.register(SIGNALS_GENERATED)


# ==========================================
# METRICS COLLECTOR
# ==========================================

class MetricsCollector:
    """
    Collects and aggregates metrics from the trading system.
    """
    
    def __init__(self):
        self.registry = REGISTRY
        self.start_time = time.time()
        
        # Alert thresholds
        self.alert_thresholds = {
            "drawdown_pct": 2.0,  # Alert if drawdown > 2%
            "latency_ms": 100,    # Alert if latency > 100ms
            "vpin_score": 0.6,    # Alert if VPIN > 0.6
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
    
    def record_trade(self, symbol: str, side: str, result: str = "success"):
        """Record a trade execution."""
        self.registry.increment("omnitrade_trades_total", labels={
            "symbol": symbol,
            "side": side,
            "result": result
        })
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record execution latency."""
        self.registry.observe_histogram(
            "omnitrade_execution_latency_ms",
            latency_ms,
            labels={"operation": operation}
        )
        
        # Check for alerts
        if latency_ms > self.alert_thresholds["latency_ms"]:
            self._trigger_alert("HIGH_LATENCY", {
                "operation": operation,
                "latency_ms": latency_ms,
                "threshold": self.alert_thresholds["latency_ms"]
            })
    
    def update_portfolio(self, value: float, drawdown: float):
        """Update portfolio metrics."""
        self.registry.set_gauge("omnitrade_portfolio_value_dollars", value)
        self.registry.set_gauge("omnitrade_current_drawdown_pct", drawdown)
        
        # Check for drawdown alert
        if drawdown > self.alert_thresholds["drawdown_pct"]:
            self._trigger_alert("HIGH_DRAWDOWN", {
                "drawdown_pct": drawdown,
                "threshold": self.alert_thresholds["drawdown_pct"]
            })
    
    def update_ai_confidence(self, signal_type: str, symbol: str, confidence: float):
        """Update AI confidence metric."""
        self.registry.set_gauge("omnitrade_ai_confidence", confidence, labels={
            "signal_type": signal_type,
            "symbol": symbol
        })
    
    def update_vpin(self, symbol: str, vpin_score: float):
        """Update VPIN score."""
        self.registry.set_gauge("omnitrade_vpin_score", vpin_score, labels={
            "symbol": symbol
        })
        
        # Check for VPIN alert
        if vpin_score > self.alert_thresholds["vpin_score"]:
            self._trigger_alert("HIGH_VPIN", {
                "symbol": symbol,
                "vpin_score": vpin_score,
                "threshold": self.alert_thresholds["vpin_score"]
            })
    
    def update_positions(self, count: int):
        """Update open positions count."""
        self.registry.set_gauge("omnitrade_open_positions", count)
    
    def record_signal(self, signal_type: str, symbol: str):
        """Record an AI signal."""
        self.registry.increment("omnitrade_signals_total", labels={
            "signal_type": signal_type,
            "symbol": symbol
        })
    
    def record_hedge_cost(self, cost: float):
        """Record hedge cost."""
        self.registry.increment("omnitrade_hedge_costs_total", cost)
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L."""
        self.registry.set_gauge("omnitrade_daily_pnl_dollars", pnl)
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler (e.g., PagerDuty, email)."""
        self.alert_handlers.append(handler)
    
    def _trigger_alert(self, alert_type: str, details: Dict):
        """Trigger an alert through all handlers."""
        alert = {
            "type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        
        logger.warning(f"ALERT: {alert_type} - {details}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "uptime_seconds": time.time() - self.start_time,
            "portfolio_value": self.registry.get("omnitrade_portfolio_value_dollars").value,
            "drawdown_pct": self.registry.get("omnitrade_current_drawdown_pct").value,
            "open_positions": self.registry.get("omnitrade_open_positions").value,
            "daily_pnl": self.registry.get("omnitrade_daily_pnl_dollars").value,
            "total_trades": self.registry.get("omnitrade_trades_total").value,
            "total_signals": self.registry.get("omnitrade_signals_total").value,
        }


# ==========================================
# PROMETHEUS EXPORTER
# ==========================================

class PrometheusExporter:
    """
    Prometheus metrics exporter for scraping.
    """
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.collector = MetricsCollector()
        self._server = None
    
    async def start(self):
        """Start the Prometheus metrics server."""
        # In production, use aiohttp or prometheus_client
        # from prometheus_client import start_http_server
        # start_http_server(self.port)
        logger.info(f"Prometheus exporter configured on port {self.port}")
    
    async def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return self.collector.registry.export_prometheus()


# ==========================================
# GRAFANA DASHBOARD CONFIGURATION
# ==========================================

GRAFANA_DASHBOARD_JSON = """
{
  "dashboard": {
    "title": "OMNI-TRADE AI Dashboard",
    "panels": [
      {
        "title": "Portfolio Value",
        "type": "stat",
        "targets": [{"expr": "omnitrade_portfolio_value_dollars"}]
      },
      {
        "title": "Daily P&L",
        "type": "stat",
        "targets": [{"expr": "omnitrade_daily_pnl_dollars"}]
      },
      {
        "title": "Drawdown",
        "type": "gauge",
        "targets": [{"expr": "omnitrade_current_drawdown_pct"}]
      },
      {
        "title": "Execution Latency",
        "type": "histogram",
        "targets": [{"expr": "rate(omnitrade_execution_latency_ms_sum[5m]) / rate(omnitrade_execution_latency_ms_count[5m])"}]
      },
      {
        "title": "AI Confidence",
        "type": "timeseries",
        "targets": [{"expr": "omnitrade_ai_confidence"}]
      },
      {
        "title": "VPIN Score",
        "type": "gauge",
        "targets": [{"expr": "omnitrade_vpin_score"}]
      },
      {
        "title": "Trades per Minute",
        "type": "timeseries",
        "targets": [{"expr": "rate(omnitrade_trades_total[1m])"}]
      },
      {
        "title": "Signal Distribution",
        "type": "piechart",
        "targets": [{"expr": "sum by(signal_type) (omnitrade_signals_total)"}]
      }
    ]
  }
}
"""


# ==========================================
# LOGGING & TRACING
# ==========================================

class StructuredLogger:
    """
    Structured logging for observability.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def add_context(self, **kwargs):
        """Add context to all future log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear log context."""
        self.context = {}
    
    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} | {context_str}"
        return message
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message), extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message), extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message), extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message), extra=kwargs)


# ==========================================
# HEALTH CHECKS
# ==========================================

class HealthChecker:
    """
    System health monitoring.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_check_time: Dict[str, datetime] = {}
        self.last_check_result: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_fn: Callable):
        """Register a health check function."""
        self.checks[name] = check_fn
    
    async def run_check(self, name: str) -> bool:
        """Run a specific health check."""
        if name not in self.checks:
            return False
        
        try:
            result = await self.checks[name]() if asyncio.iscoroutinefunction(self.checks[name]) else self.checks[name]()
            self.last_check_result[name] = result
            self.last_check_time[name] = datetime.now()
            return result
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            self.last_check_result[name] = False
            self.last_check_time[name] = datetime.now()
            return False
    
    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        for name in self.checks:
            results[name] = await self.run_check(name)
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        checks_run = len(self.last_check_result) > 0
        
        if not checks_run:
            return {"status": "unknown", "checks": {}}
        
        all_healthy = all(self.last_check_result.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": {
                name: {
                    "healthy": result,
                    "last_check": self.last_check_time.get(name).isoformat() if name in self.last_check_time else None
                }
                for name, result in self.last_check_result.items()
            }
        }


# ==========================================
# KILL SWITCH
# ==========================================

class KillSwitch:
    """
    Emergency kill switch for catastrophic failures.
    """
    
    def __init__(self):
        self._triggered = False
        self._trigger_time: Optional[datetime] = None
        self._trigger_reason = ""
    
    def trigger(self, reason: str):
        """Trigger the kill switch."""
        self._triggered = True
        self._trigger_time = datetime.now()
        self._trigger_reason = reason
        
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
    
    def reset(self):
        """Reset the kill switch."""
        self._triggered = False
        self._trigger_time = None
        self._trigger_reason = ""
        logger.info("Kill switch reset")
    
    def is_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self._triggered
    
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status."""
        return {
            "triggered": self._triggered,
            "trigger_time": self._trigger_time.isoformat() if self._trigger_time else None,
            "reason": self._trigger_reason,
        }


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_prometheus_exporter: Optional[PrometheusExporter] = None
_health_checker: Optional[HealthChecker] = None
_kill_switch: Optional[KillSwitch] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_prometheus_exporter(port: int = 9090) -> PrometheusExporter:
    """Get the Prometheus exporter."""
    global _prometheus_exporter
    if _prometheus_exporter is None:
        _prometheus_exporter = PrometheusExporter(port)
    return _prometheus_exporter


def get_health_checker() -> HealthChecker:
    """Get the health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def get_kill_switch() -> KillSwitch:
    """Get the kill switch."""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = KillSwitch()
    return _kill_switch