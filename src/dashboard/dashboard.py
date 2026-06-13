"""
Dashboard - Web-based monitoring and control interface.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Global dash app
app = dash.Dash(__name__)
app.title = "TRADE - AI Trading Bot Dashboard"


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    title: str = "TRADE Bot Dashboard"


class Dashboard:
    """
    Web dashboard for monitoring and controlling the trading bot.
    """
    
    def __init__(self, bot, config: Optional[DashboardConfig] = None):
        self.bot = bot
        self.config = config or DashboardConfig()
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 TRADE - AI Trading Bot", className="header-title"),
                html.Div([
                    html.Span("Status: ", className="status-label"),
                    html.Span(id="bot-status", className="status-value"),
                ], className="status-indicator"),
            ], className="header"),
            
            # Main content
            html.Div([
                # Left column - Portfolio Overview
                html.Div([
                    html.H2("📊 Portfolio Overview"),
                    html.Div(id="portfolio-summary", className="card"),
                    html.Div(id="positions-list", className="card"),
                ], className="column left"),
                
                # Middle column - Charts
                html.Div([
                    html.H2("📈 Market Analysis"),
                    dcc.Graph(id="portfolio-chart"),
                    html.H2("🔍 Signal Analysis"),
                    dcc.Graph(id="signals-chart"),
                ], className="column middle"),
                
                # Right column - Controls
                html.Div([
                    html.H2("⚙️ Controls"),
                    html.Div([
                        html.Button("▶️ Start", id="btn-start", n_clicks=0, className="btn btn-success"),
                        html.Button("⏸️ Pause", id="btn-pause", n_clicks=0, className="btn btn-warning"),
                        html.Button("⏹️ Stop", id="btn-stop", n_clicks=0, className="btn btn-danger"),
                    ], className="button-group"),
                    
                    html.H2("📋 Watchlist"),
                    html.Div(id="watchlist-display", className="card"),
                    
                    html.H2("📜 Trade History"),
                    html.Div(id="trade-history", className="card scrollable"),
                ], className="column right"),
            ], className="main-content"),
            
            # Analysis cache
            html.Div(id="analysis-cache", style={"display": "none"}),
            
            # Interval for auto-refresh
            dcc.Interval(
                id="refresh-interval",
                interval=5000,  # 5 seconds
                n_intervals=0
            ),
        ], className="container")
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @app.callback(
            [Output("bot-status", "children"),
             Output("portfolio-summary", "children"),
             Output("positions-list", "children"),
             Output("watchlist-display", "children"),
             Output("trade-history", "children"),
             Output("portfolio-chart", "figure"),
             Output("signals-chart", "figure")],
            [Input("refresh-interval", "n_intervals"),
             Input("btn-start", "n_clicks"),
             Input("btn-pause", "n_clicks"),
             Input("btn-stop", "n_clicks")]
        )
        def update_dashboard(n_intervals, n_start, n_pause, n_stop):
            """Update all dashboard components."""
            ctx = dash.callback_context
            
            if ctx.triggered:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
                
                if button_id == "btn-start":
                    asyncio.create_task(self.bot.start())
                elif button_id == "btn-pause":
                    asyncio.create_task(self.bot.pause())
                elif button_id == "btn-stop":
                    asyncio.create_task(self.bot.stop())
            
            # Get current status
            status = "🟢 Running" if self.bot.state.is_running else "🔴 Stopped"
            if self.bot.state.is_paused:
                status = "🟡 Paused"
            
            # Get portfolio status
            portfolio = asyncio.run(self.bot.get_portfolio_status())
            
            # Portfolio summary
            summary = html.Div([
                html.Div([
                    html.Span("Total Value: ", className="metric-label"),
                    html.Span(f"${portfolio['total_value']:,.2f}", className="metric-value"),
                ]),
                html.Div([
                    html.Span("Cash: ", className="metric-label"),
                    html.Span(f"${portfolio['cash']:,.2f}", className="metric-value"),
                ]),
                html.Div([
                    html.Span("Mode: ", className="metric-label"),
                    html.Span(portfolio['mode'].upper(), className="metric-value"),
                ]),
            ])
            
            # Positions list
            positions_html = html.Ul([
                html.Li([
                    html.Strong(p['symbol']),
                    f" - {p['quantity']} shares",
                    f" | P&L: ${p['pnl']:.2f} ({p['pnl_percent']:.1f}%)"
                ])
                for p in portfolio['positions']
            ]) if portfolio['positions'] else html.P("No open positions")
            
            # Watchlist
            watchlist_html = html.Ul([
                html.Li(symbol) for symbol in self.bot.watchlist
            ])
            
            # Trade history
            trades = self.bot.trade_history[-10:] if self.bot.trade_history else []
            history_html = html.Ul([
                html.Li([
                    html.Span(t.get('side', '').upper(), className=f"trade-side {t.get('side', '')}"),
                    f" {t.get('symbol', '')} - {t.get('quantity', 0)} @ ${t.get('price', 0):.2f}"
                ])
                for t in trades
            ]) if trades else html.P("No recent trades")
            
            # Portfolio chart (placeholder)
            portfolio_fig = go.Figure()
            portfolio_fig.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[portfolio['total_value']],
                mode='lines+markers',
                name='Portfolio Value'
            ))
            portfolio_fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                height=300
            )
            
            # Signals chart
            signals_fig = go.Figure()
            analysis_cache = self.bot.get_analysis_cache()
            
            symbols = list(analysis_cache.keys())
            scores = [a.overall_score for a in analysis_cache.values()]
            confidences = [a.confidence for a in analysis_cache.values()]
            
            signals_fig.add_trace(go.Bar(
                x=symbols,
                y=scores,
                name="Score",
                marker_color=['green' if s > 0.6 else 'red' if s < 0.4 else 'gray' for s in scores]
            ))
            signals_fig.add_trace(go.Scatter(
                x=symbols,
                y=confidences,
                name="Confidence",
                yaxis='y2',
                mode='markers'
            ))
            signals_fig.update_layout(
                title="AI Analysis Signals",
                xaxis_title="Symbol",
                yaxis_title="Score",
                yaxis2=dict(title="Confidence", overlaying='y', side='right'),
                height=300
            )
            
            return status, summary, positions_html, watchlist_html, history_html, portfolio_fig, signals_fig
    
    def run(self):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard on {self.config.host}:{self.config.port}")
        app.run_server(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug
        )


# CSS styles
app.css.append_css({
    "external_url": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
})

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <title>TRADE - AI Trading Bot</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
                color: #ffffff;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                margin-bottom: 20px;
            }
            
            .header-title {
                font-size: 28px;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .status-indicator {
                padding: 8px 16px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                font-size: 14px;
            }
            
            .main-content {
                display: grid;
                grid-template-columns: 300px 1fr 300px;
                gap: 20px;
            }
            
            .column {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 20px;
            }
            
            .card h2 {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #a0a0a0;
            }
            
            .metric-label {
                color: #a0a0a0;
                font-size: 14px;
            }
            
            .metric-value {
                font-weight: 600;
                font-size: 18px;
                color: #667eea;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-success {
                background: #10b981;
                color: white;
            }
            
            .btn-warning {
                background: #f59e0b;
                color: white;
            }
            
            .btn-danger {
                background: #ef4444;
                color: white;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            
            .scrollable {
                max-height: 300px;
                overflow-y: auto;
            }
            
            ul {
                list-style: none;
            }
            
            li {
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .trade-side {
                font-weight: 600;
                padding: 2px 8px;
                border-radius: 4px;
            }
            
            .trade-side.buy {
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
            }
            
            .trade-side.sell {
                background: rgba(239, 68, 68, 0.2);
                color: #ef4444;
            }
            
            @media (max-width: 1200px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def create_dashboard(bot, config: Optional[DashboardConfig] = None) -> Dashboard:
    """Create a dashboard for the trading bot."""
    return Dashboard(bot, config)


def run_dashboard(bot, host: str = "0.0.0.0", port: int = 8050):
    """Run the dashboard server."""
    dashboard = create_dashboard(bot, DashboardConfig(host=host, port=port))
    dashboard.run()