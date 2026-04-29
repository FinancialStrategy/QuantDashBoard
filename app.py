# =============================================================================
# QFA_Dashboard_FinTECH_RENDER
# Render-ready Panel app: Yahoo-only data + TA-Lib if available + QuantStats + PyPortfolioOpt
# =============================================================================

import os
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

try:
    from pypfopt import expected_returns, risk_models, EfficientFrontier, HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

# =============================================================================
# 3) App configuration
# =============================================================================

APP_NAME = "QFA_Dashboard_FinTECH"
OUTPUT_DIR = os.environ.get("QFA_OUTPUT_DIR", "QFA_Dashboard_FinTECH_reports")
HTML_OUTPUT = os.path.join(OUTPUT_DIR, f"{APP_NAME}.html")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045
DEFAULT_START = datetime(2018, 1, 1)
DEFAULT_END = datetime.now()

# =============================================================================
# 4) Institutional universe
# =============================================================================

UNIVERSE = {
    "Equity ETF": {
        "United States": {
            "SPY": "SPDR S&P 500 ETF",
            "QQQ": "Invesco Nasdaq 100 ETF",
            "IWM": "iShares Russell 2000 ETF",
            "DIA": "SPDR Dow Jones Industrial Average ETF",
            "VTI": "Vanguard Total Stock Market ETF",
            "RSP": "Invesco S&P 500 Equal Weight ETF",
            "MGK": "Vanguard Mega Cap Growth ETF",
            "IJR": "iShares Core S&P Small-Cap ETF",
        },
        "Europe": {
            "VGK": "Vanguard FTSE Europe ETF",
            "FEZ": "SPDR Euro STOXX 50 ETF",
            "EWG": "iShares MSCI Germany ETF",
            "EWQ": "iShares MSCI France ETF",
            "EWI": "iShares MSCI Italy ETF",
            "EWP": "iShares MSCI Spain ETF",
        },
        "Emerging Markets": {
            "VWO": "Vanguard Emerging Markets ETF",
            "EEM": "iShares MSCI Emerging Markets ETF",
            "FXI": "iShares China Large-Cap ETF",
            "EWZ": "iShares MSCI Brazil ETF",
            "INDA": "iShares MSCI India ETF",
        },
    },
    "Sector ETF": {
        "United States": {
            "XLB": "Materials Select Sector SPDR",
            "XLC": "Communication Services Select Sector SPDR",
            "XLE": "Energy Select Sector SPDR",
            "XLF": "Financials Select Sector SPDR",
            "XLI": "Industrials Select Sector SPDR",
            "XLK": "Technology Select Sector SPDR",
            "XLP": "Consumer Staples Select Sector SPDR",
            "XLRE": "Real Estate Select Sector SPDR",
            "XLU": "Utilities Select Sector SPDR",
            "XLV": "Health Care Select Sector SPDR",
            "XLY": "Consumer Discretionary Select Sector SPDR",
        },
    },
    "Fixed Income": {
        "United States": {
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "HYG": "iShares High Yield Corporate Bond ETF",
            "LQD": "iShares Investment Grade Corporate Bond ETF",
        },
    },
    "Commodities": {
        "Global": {
            "GLD": "SPDR Gold Shares",
            "SLV": "iShares Silver Trust",
            "USO": "United States Oil Fund",
            "UNG": "United States Natural Gas Fund",
            "DBC": "Invesco DB Commodity Index Tracking Fund",
            "DBA": "Invesco DB Agriculture Fund",
        },
    },
    "European Equities": {
        "Germany": {
            "SAP.DE": "SAP",
            "SIE.DE": "Siemens",
            "DTE.DE": "Deutsche Telekom",
            "BAS.DE": "BASF",
            "BMW.DE": "BMW",
            "VOW3.DE": "Volkswagen",
            "BAYN.DE": "Bayer",
            "ALV.DE": "Allianz",
            "DBK.DE": "Deutsche Bank",
            "IFX.DE": "Infineon",
        },
        "France": {
            "AIR.PA": "Airbus",
            "SU.PA": "Schneider Electric",
            "DG.PA": "Vinci",
            "AI.PA": "Air Liquide",
            "EN.PA": "Bouygues",
            "HO.PA": "Thales",
            "RNO.PA": "Renault",
            "SGO.PA": "Saint-Gobain",
            "ML.PA": "Michelin",
            "SAF.PA": "Safran",
        },
    },
}

# Very important:
# S&P 500 benchmark is ^GSPC, not SPY.
# SPY can still exist as a selectable instrument, but it is not the benchmark proxy.
BENCHMARKS = {
    "S&P 500 Index": "^GSPC",
    "Nasdaq 100 Index": "^NDX",
    "Dow Jones Industrial Average": "^DJI",
    "Russell 2000 Index": "^RUT",
    "US Aggregate Bond ETF": "BND",
    "Gold ETF": "GLD",
    "Emerging Markets ETF": "EEM",
    "Euro STOXX 50 ETF": "FEZ",
}

def flatten_universe() -> pd.DataFrame:
    rows = []
    for asset_class, regions in UNIVERSE.items():
        for region, instruments in regions.items():
            for ticker, name in instruments.items():
                rows.append({
                    "Asset Class": asset_class,
                    "Region": region,
                    "Ticker": ticker,
                    "Name": name,
                })
    return pd.DataFrame(rows)

UNIVERSE_DF = flatten_universe()

def get_regions(asset_class: str):
    return list(UNIVERSE.get(asset_class, {}).keys())

def get_tickers(asset_class: str, region: str):
    return list(UNIVERSE.get(asset_class, {}).get(region, {}).keys())

def get_name(ticker: str):
    row = UNIVERSE_DF[UNIVERSE_DF["Ticker"] == ticker]
    return ticker if row.empty else row.iloc[0]["Name"]

def pct_fmt(x, digits=2):
    try:
        if pd.isna(x):
            return "N/A"
        return f"{x * 100:.{digits}f}%"
    except Exception:
        return "N/A"

def num_fmt(x, digits=2):
    try:
        if pd.isna(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"

def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")

# =============================================================================
# 5) Yahoo data layer: original matched data only
# =============================================================================

_DATA_CACHE = {}

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch real OHLCV data from Yahoo Finance only.
    No synthetic data.
    No fallback generated prices.
    """
    cache_key = (ticker, start, end)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if df is None or df.empty:
            _DATA_CACHE[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if any(col not in df.columns for col in required):
            _DATA_CACHE[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        # Do not manufacture missing prices.
        if df.empty:
            _DATA_CACHE[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        _DATA_CACHE[cache_key] = df.copy()
        return df

    except Exception:
        _DATA_CACHE[cache_key] = pd.DataFrame()
        return pd.DataFrame()

def fetch_prices(tickers, start: str, end: str):
    """
    Fetch close prices for multiple tickers.
    Returns:
    - matched original close prices only
    - excluded tickers list
    Note: .dropna() creates the common matched date window.
    """
    series = {}
    excluded = []

    for t in tickers:
        df = fetch_data(t, start, end)
        if df.empty or "Close" not in df.columns:
            excluded.append(t)
            continue
        s = df["Close"].dropna()
        if len(s) < 30:
            excluded.append(t)
            continue
        series[t] = s

    if not series:
        return pd.DataFrame(), excluded

    prices = pd.DataFrame(series).dropna(how="any")
    if prices.empty:
        return pd.DataFrame(), tickers

    return prices, excluded

def matched_asset_benchmark_returns(asset_ticker: str, benchmark_ticker: str, start: str, end: str):
    """
    Returns matched returns for asset and benchmark using original Yahoo observations only.
    No forward-fill.
    No synthetic series.
    No forced SPY benchmark.
    """
    asset_df = fetch_data(asset_ticker, start, end)
    bench_df = fetch_data(benchmark_ticker, start, end)

    if asset_df.empty or bench_df.empty:
        return pd.DataFrame(), asset_df, bench_df

    closes = pd.concat(
        [
            asset_df["Close"].rename("asset_close"),
            bench_df["Close"].rename("benchmark_close"),
        ],
        axis=1,
    ).dropna(how="any")

    if closes.empty or len(closes) < 30:
        return pd.DataFrame(), asset_df, bench_df

    returns = closes.pct_change().dropna()
    returns = returns.rename(
        columns={
            "asset_close": "asset",
            "benchmark_close": "benchmark",
        }
    )
    return returns, asset_df, bench_df

# =============================================================================
# 6) Indicators: TA-Lib if available, formula fallback only for indicators
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to Yahoo OHLCV.
    Price data remains Yahoo-only.
    Indicator fallback is only mathematical calculation if TA-Lib is unavailable.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Log_Return"] = np.log(d["Close"] / d["Close"].shift(1))
    d["Cumulative_Return"] = (1 + d["Return"].fillna(0)).cumprod() - 1
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["Vol_21D"] = d["Return"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    d["Vol_63D"] = d["Return"].rolling(63).std() * np.sqrt(TRADING_DAYS)

    wealth = (1 + d["Return"].fillna(0)).cumprod()
    d["Drawdown"] = wealth / wealth.cummax() - 1

    if TALIB_AVAILABLE:
        try:
            close = d["Close"].astype(float).values
            high = d["High"].astype(float).values
            low = d["Low"].astype(float).values

            d["RSI"] = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(
                close,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9,
            )
            d["MACD"] = macd
            d["MACD_signal"] = signal
            d["MACD_hist"] = hist

            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
            )
            d["BB_upper"] = upper
            d["BB_mid"] = middle
            d["BB_lower"] = lower
            d["ATR"] = talib.ATR(high, low, close, timeperiod=14)

            slowk, slowd = talib.STOCH(
                high,
                low,
                close,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3,
            )
            d["Stoch_K"] = slowk
            d["Stoch_D"] = slowd
            return d
        except Exception:
            pass

    # Indicator fallback only. Never price fallback.
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    d["BB_mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift()).abs()
    tr3 = (d["Low"] - d["Close"].shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR"] = true_range.rolling(14).mean()

    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * (d["Close"] - low14) / (high14 - low14)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    return d

# =============================================================================
# 7) Risk metrics
# =============================================================================

def calc_risk_metrics(returns, rf=RISK_FREE_RATE) -> dict:
    r = pd.Series(returns).dropna()

    if len(r) < 30:
        return {
            "Ann Return": np.nan,
            "Ann Vol": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max Drawdown": np.nan,
            "VaR 95": np.nan,
            "CVaR 95": np.nan,
            "Skew": np.nan,
            "Kurtosis": np.nan,
            "Win Rate": np.nan,
        }

    ann_return = (1 + r.mean()) ** TRADING_DAYS - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_return - rf) / downside if downside and downside > 0 else np.nan

    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()

    return {
        "Ann Return": ann_return,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": drawdown.min(),
        "VaR 95": var95,
        "CVaR 95": cvar95,
        "Skew": r.skew(),
        "Kurtosis": r.kurtosis(),
        "Win Rate": (r > 0).mean(),
    }

# =============================================================================
# 8) UI components
# =============================================================================

def kpi_html(metrics, ticker, name, asset_class, region, benchmark_label, benchmark_ticker, matched_obs):
    cards = [
        ("Ticker", ticker, "neutral"),
        ("Instrument", name, "neutral"),
        ("Asset Class", asset_class, "neutral"),
        ("Region", region, "neutral"),
        ("Benchmark", f"{benchmark_label} ({benchmark_ticker})", "neutral"),
        ("Matched Obs.", f"{matched_obs:,}", "neutral"),
        ("Annual Return", pct_fmt(metrics["Ann Return"]), "positive" if metrics["Ann Return"] > 0 else "negative"),
        ("Annual Volatility", pct_fmt(metrics["Ann Vol"]), "neutral"),
        ("Sharpe Ratio", num_fmt(metrics["Sharpe"], 2), "positive" if metrics["Sharpe"] > 1 else "neutral"),
        ("Sortino Ratio", num_fmt(metrics["Sortino"], 2), "positive" if metrics["Sortino"] > 1 else "neutral"),
        ("Max Drawdown", pct_fmt(metrics["Max Drawdown"]), "negative"),
        ("VaR 95", pct_fmt(metrics["VaR 95"]), "negative"),
        ("CVaR 95", pct_fmt(metrics["CVaR 95"]), "negative"),
        ("Win Rate", pct_fmt(metrics["Win Rate"]), "positive" if metrics["Win Rate"] > 0.5 else "neutral"),
        ("Data Policy", "Yahoo matched only", "neutral"),
        ("SPY Benchmark", "Never hard-coded", "neutral"),
    ]

    styles = {
        "positive": ("#e9f7ef", "#2e7d32"),
        "negative": ("#fdecea", "#b71c1c"),
        "neutral": ("#f6f8fb", "#d7dee8"),
    }

    html = """
    <div style="display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px;margin-bottom:14px;">
    """
    for label, value, tone in cards:
        bg, border = styles[tone]
        html += f"""
        <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,.04);">
            <div style="font-size:12px;color:#64748b;font-weight:700;">{label}</div>
            <div style="font-size:18px;color:#111827;font-weight:800;margin-top:5px;">{value}</div>
        </div>
        """
    html += "</div>"
    return html

def institutional_layout(fig, title, height=760):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=20)),
        template="plotly_white",
        height=height,
        autosize=True,
        margin=dict(l=55, r=35, t=85, b=55),
        font=dict(family="Arial", size=12, color="#1f2937"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(210,220,230,.45)", zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(210,220,230,.45)", zeroline=False)
    return fig

# =============================================================================
# 9) Charts
# =============================================================================

def chart_price_technical(df, ticker):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Yahoo Finance data available for selected ticker/date range.", showarrow=False)
        return institutional_layout(fig, f"{ticker} | Price & Technicals", 650)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.48, 0.18, 0.18, 0.16],
        subplot_titles=[
            "OHLC, Bollinger Bands, MA50 and MA200",
            "RSI 14",
            "MACD",
            "Stochastic Oscillator",
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#2e7d32",
            decreasing_line_color="#b71c1c",
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )

    for col, name, width, dash in [
        ("BB_upper", "BB Upper", 1.2, "dash"),
        ("BB_mid", "BB Mid", 1.1, "dot"),
        ("BB_lower", "BB Lower", 1.2, "dash"),
        ("MA50", "MA50", 1.6, "solid"),
        ("MA200", "MA200", 1.8, "solid"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=name,
                    mode="lines",
                    line=dict(width=width, dash=dash),
                    hovertemplate="%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines", hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="gray")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="gray")
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines", hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", mode="lines", hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Histogram", opacity=0.45, hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="Stoch %K", mode="lines", hovertemplate="%{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], name="Stoch %D", mode="lines", hovertemplate="%{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_hline(y=80, row=4, col=1, line_dash="dash", line_color="gray")
    fig.add_hline(y=20, row=4, col=1, line_dash="dash", line_color="gray")

    source = "TA-Lib" if TALIB_AVAILABLE else "Formula indicator mode"
    return institutional_layout(fig, f"{ticker} | Price & Technical Dashboard | {source}", 900)

def chart_risk(df, ticker):
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Yahoo Finance data available.", showarrow=False)
        return institutional_layout(fig, f"{ticker} | Risk Diagnostics", 650)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Cumulative Return",
            "Rolling Volatility 21D / 63D",
            "Drawdown",
            "Daily Return Distribution",
        ],
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Cumulative_Return"] * 100, name="Cumulative Return %", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_21D"] * 100, name="Vol 21D %", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_63D"] * 100, name="Vol 63D %", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"] * 100, name="Drawdown %", fill="tozeroy", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Histogram(x=df["Return"].dropna() * 100, name="Daily Returns %", nbinsx=70, opacity=0.75, hovertemplate="%{x:.2f}%<extra></extra>"), row=4, col=1)

    return institutional_layout(fig, f"{ticker} | Institutional Risk Diagnostics", 900)

def chart_benchmark_relative(returns, ticker, benchmark_label, benchmark_ticker):
    if returns.empty:
        fig = go.Figure()
        fig.add_annotation(text="No matched Yahoo observations for selected asset and benchmark.", showarrow=False)
        return institutional_layout(fig, "Benchmark Relative Analytics", 650)

    cum = (1 + returns).cumprod() - 1
    active = returns["asset"] - returns["benchmark"]
    active_cum = (1 + active).cumprod() - 1
    rolling_ir = (active.rolling(63).mean() * TRADING_DAYS) / (active.rolling(63).std() * np.sqrt(TRADING_DAYS))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Cumulative Return Comparison",
            "Active Cumulative Return",
            "Rolling Information Ratio 63D",
        ],
    )

    fig.add_trace(go.Scatter(x=cum.index, y=cum["asset"] * 100, name=ticker, mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum.index, y=cum["benchmark"] * 100, name=f"{benchmark_label} ({benchmark_ticker})", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_cum.index, y=active_cum * 100, name="Active Return", mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_ir.index, y=rolling_ir, name="Rolling IR", mode="lines", hovertemplate="%{y:.2f}<extra></extra>"), row=3, col=1)

    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
    fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")

    return institutional_layout(fig, f"{ticker} vs {benchmark_label} ({benchmark_ticker}) | Matched Yahoo Data", 850)

def chart_universe_board(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices, excluded = fetch_prices(tickers, start, end)

    if prices.empty:
        fig = go.Figure()
        fig.add_annotation(text="No matched Yahoo Finance data for selected universe.", showarrow=False)
        return institutional_layout(fig, "Investment Universe Board", 650)

    returns = prices.pct_change().dropna()
    rows = []

    for t in returns.columns:
        m = calc_risk_metrics(returns[t])
        total_return = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        rows.append({
            "Ticker": t,
            "Name": get_name(t),
            "Total Return": total_return,
            "Ann Return": m["Ann Return"],
            "Ann Vol": m["Ann Vol"],
            "Sharpe": m["Sharpe"],
            "Max DD": m["Max Drawdown"],
            "VaR 95": m["VaR 95"],
        })

    board = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            "Universe Sharpe Ranking",
            "Return versus Volatility Map",
        ],
    )

    fig.add_trace(go.Bar(x=board["Ticker"], y=board["Sharpe"], name="Sharpe Ratio", hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=board["Ann Vol"] * 100,
            y=board["Ann Return"] * 100,
            mode="markers+text",
            text=board["Ticker"],
            textposition="top center",
            marker=dict(size=np.clip((board["Sharpe"].fillna(0).abs() + 0.5) * 14, 8, 34)),
            name="Risk/Return",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    note = f"Excluded due to missing/unmatched Yahoo data: {', '.join(sorted(set(excluded)))}" if excluded else "All selected instruments matched."
    fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=-0.10, showarrow=False, font=dict(size=11, color="#64748b"))
    return institutional_layout(fig, f"{asset_class} | {region} | Original Matched Yahoo Universe Board", 850)

# =============================================================================
# 10) QuantStats
# =============================================================================

def make_quantstats_report(ticker, benchmark_label, benchmark_ticker, start, end):
    if not QUANTSTATS_AVAILABLE:
        return pn.pane.Markdown("### QuantStats not available. Install with `pip install quantstats`.")

    returns, _, _ = matched_asset_benchmark_returns(ticker, benchmark_ticker, start, end)

    if returns.empty:
        return pn.pane.Markdown(
            f"""
            ### QuantStats report cannot be generated

            No matched Yahoo observations for:
            - Asset: **{ticker}**
            - Benchmark: **{benchmark_label} ({benchmark_ticker})**

            **Policy:** no synthetic data and no fallback price series.
            """
        )

    out_name = (
        f"{APP_NAME}_QuantStats_"
        f"{safe_filename(ticker)}_vs_"
        f"{safe_filename(benchmark_label)}_"
        f"{safe_filename(benchmark_ticker)}.html"
    )
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        asset_returns = returns["asset"].copy()
        bench_returns = returns["benchmark"].copy()
        asset_returns.name = ticker
        bench_returns.name = f"{benchmark_label} ({benchmark_ticker})"

        try:
            qs.reports.html(
                asset_returns,
                benchmark=bench_returns,
                output=out_path,
                title=f"{ticker} vs {benchmark_label} ({benchmark_ticker}) | QFA FinTECH Tearsheet",
                rf=RISK_FREE_RATE,
            )
        except TypeError:
            # Older QuantStats versions may not expose rf in reports.html.
            # In that case, generation continues, while all dashboard risk metrics
            # still use the fixed 4.50% risk-free rate.
            qs.reports.html(
                asset_returns,
                benchmark=bench_returns,
                output=out_path,
                title=f"{ticker} vs {benchmark_label} ({benchmark_ticker}) | QFA FinTECH Tearsheet",
            )

        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        info_html = f"""
        <div style='background:#f8fafc;border:1px solid #dbe3ef;border-radius:14px;padding:14px;margin-bottom:10px;'>
            <b>QuantStats Tearsheet Updated</b><br>
            Asset: <b>{ticker}</b><br>
            Benchmark: <b>{benchmark_label}</b> | Yahoo symbol: <b>{benchmark_ticker}</b><br>
            Matched observations: <b>{len(returns):,}</b><br>
            Output: <code>{out_path}</code><br>
            Policy: <b>No hard-coded SPY benchmark. No synthetic data.</b>
        </div>
        """

        return pn.Column(
            pn.pane.HTML(info_html, sizing_mode="stretch_width"),
            pn.pane.HTML(html, sizing_mode="stretch_width", height=950),
            sizing_mode="stretch_width",
        )

    except Exception as e:
        return pn.pane.Markdown(
            f"""
            ### QuantStats generation failed

            ```text
            {e}
            ```
            """
        )


# =============================================================================
# 10B) Selected ETF batch HTML report generator
# =============================================================================

def generate_selected_etf_batch_report(ticker, benchmark_label, benchmark_ticker, start, end):
    """
    Button-triggered Python batch generator.

    It uses the current sidebar combo-box selections and creates a standalone
    HTML report for the selected ETF/instrument.

    Strict policy:
    - Yahoo Finance original matched observations only.
    - No synthetic prices.
    - No fallback price series.
    - Benchmark is selected from the sidebar; SPY is never forced.
    - Dashboard risk metrics use fixed RF = 4.50%.
    """
    asset_df = fetch_data(ticker, start, end)
    if asset_df.empty:
        raise ValueError(f"No original Yahoo Finance OHLCV data for selected instrument: {ticker}")

    asset_df = add_indicators(asset_df)
    metrics = calc_risk_metrics(asset_df["Return"], rf=RISK_FREE_RATE)

    matched_returns, _, _ = matched_asset_benchmark_returns(ticker, benchmark_ticker, start, end)
    matched_obs = len(matched_returns) if not matched_returns.empty else 0

    price_fig = chart_price_technical(asset_df, ticker)
    risk_fig = chart_risk(asset_df, ticker)
    rel_fig = chart_benchmark_relative(matched_returns, ticker, benchmark_label, benchmark_ticker)

    qs_info = ""
    qs_html = ""

    if QUANTSTATS_AVAILABLE and not matched_returns.empty:
        qs_name = (
            f"{APP_NAME}_Batch_Tearsheet_"
            f"{safe_filename(ticker)}_vs_"
            f"{safe_filename(benchmark_label)}_"
            f"{safe_filename(benchmark_ticker)}.html"
        )
        qs_path = os.path.join(OUTPUT_DIR, qs_name)

        try:
            asset_returns = matched_returns["asset"].copy()
            bench_returns = matched_returns["benchmark"].copy()
            asset_returns.name = ticker
            bench_returns.name = f"{benchmark_label} ({benchmark_ticker})"

            try:
                qs.reports.html(
                    asset_returns,
                    benchmark=bench_returns,
                    output=qs_path,
                    title=f"{ticker} vs {benchmark_label} ({benchmark_ticker}) | QFA FinTECH Tearsheet",
                    rf=RISK_FREE_RATE,
                )
            except TypeError:
                qs.reports.html(
                    asset_returns,
                    benchmark=bench_returns,
                    output=qs_path,
                    title=f"{ticker} vs {benchmark_label} ({benchmark_ticker}) | QFA FinTECH Tearsheet",
                )

            with open(qs_path, "r", encoding="utf-8", errors="ignore") as f:
                qs_html = f.read()

            qs_info = f"<p><b>Tearsheet file:</b> <code>{qs_path}</code></p>"

        except Exception as e:
            qs_info = f"<p><b>Tearsheet generation warning:</b> {str(e)}</p>"
    else:
        qs_info = "<p><b>Tearsheet:</b> QuantStats unavailable or no matched benchmark observations.</p>"

    kpi_section = kpi_html(
        metrics=metrics,
        ticker=ticker,
        name=get_name(ticker),
        asset_class="Selected from dashboard sidebar",
        region="Selected from dashboard sidebar",
        benchmark_label=benchmark_label,
        benchmark_ticker=benchmark_ticker,
        matched_obs=matched_obs,
    )

    out_name = (
        f"{APP_NAME}_Selected_Report_"
        f"{safe_filename(ticker)}_vs_"
        f"{safe_filename(benchmark_label)}_"
        f"{safe_filename(benchmark_ticker)}.html"
    )
    out_path = os.path.join(OUTPUT_DIR, out_name)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{ticker} | QFA Dashboard FinTECH Batch Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f8fafc;
                color: #111827;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 1480px;
                margin: 0 auto;
                padding: 24px;
            }}
            .hero {{
                background: linear-gradient(90deg,#0f172a,#1e293b);
                color: white;
                border-radius: 18px;
                padding: 24px 28px;
                margin-bottom: 18px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.12);
            }}
            .hero h1 {{
                margin: 0;
                font-size: 30px;
            }}
            .hero p {{
                color: #cbd5e1;
                margin-bottom: 0;
            }}
            .section {{
                background: white;
                border: 1px solid #dbe3ef;
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 18px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }}
            code {{
                background: #eef2f7;
                padding: 2px 5px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero">
                <h1>QFA Dashboard FinTECH — Selected ETF Batch Report</h1>
                <p>
                    Instrument: <b>{ticker}</b> — {get_name(ticker)} |
                    Benchmark: <b>{benchmark_label}</b> ({benchmark_ticker}) |
                    Risk-free rate: <b>4.50%</b>
                </p>
                <p>
                    Data policy: Yahoo Finance original matched observations only.
                    No synthetic prices. No fallback price series. SPY is never forced as benchmark.
                </p>
            </div>

            <div class="section">
                <h2>Executive KPI Dashboard</h2>
                {kpi_section}
                {qs_info}
            </div>

            <div class="section">
                <h2>Price & TA-Lib Technicals</h2>
                {price_fig.to_html(full_html=False, include_plotlyjs="cdn")}
            </div>

            <div class="section">
                <h2>Institutional Risk Metrics</h2>
                {risk_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="section">
                <h2>Benchmark Relative Analytics</h2>
                {rel_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="section">
                <h2>Tearsheet</h2>
                {qs_html if qs_html else "<p>Tearsheet output not embedded.</p>"}
            </div>
        </div>
    </body>
    </html>
    """

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


# =============================================================================
# 10C) PyPortfolioOpt institutional optimizer
# =============================================================================

def optimize_universe_pyportfolioopt(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices, excluded = fetch_prices(tickers, start, end)

    if prices.empty or prices.shape[1] < 2:
        return None, None, f"Not enough matched Yahoo Finance price series for optimization. Excluded: {', '.join(excluded) if excluded else 'None'}"

    if not PYPFOPT_AVAILABLE:
        return prices, None, "PyPortfolioOpt is not available in this environment. Check requirements.txt and Render build logs."

    try:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
        try:
            cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            cov_method = "Ledoit-Wolf shrinkage covariance"
        except Exception:
            cov = risk_models.sample_cov(prices, frequency=TRADING_DAYS)
            cov_method = "Sample covariance fallback"

        results = []
        weights_frames = []

        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            clean_w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            results.append({"Strategy": "Max Sharpe", "Expected Return": perf[0], "Volatility": perf[1], "Sharpe": perf[2], "Method": cov_method})
            for t, w in clean_w.items():
                if abs(w) > 1e-5:
                    weights_frames.append({"Strategy": "Max Sharpe", "Ticker": t, "Name": get_name(t), "Weight": w})
        except Exception as e:
            results.append({"Strategy": "Max Sharpe", "Expected Return": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Method": f"Failed: {e}"})

        try:
            ef2 = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef2.min_volatility()
            clean_w2 = ef2.clean_weights()
            perf2 = ef2.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            results.append({"Strategy": "Minimum Volatility", "Expected Return": perf2[0], "Volatility": perf2[1], "Sharpe": perf2[2], "Method": cov_method})
            for t, w in clean_w2.items():
                if abs(w) > 1e-5:
                    weights_frames.append({"Strategy": "Minimum Volatility", "Ticker": t, "Name": get_name(t), "Weight": w})
        except Exception as e:
            results.append({"Strategy": "Minimum Volatility", "Expected Return": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Method": f"Failed: {e}"})

        try:
            returns = prices.pct_change().dropna()
            hrp = HRPOpt(returns)
            hrp.optimize()
            clean_hrp = hrp.clean_weights()
            perf3 = hrp.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            results.append({"Strategy": "HRP", "Expected Return": perf3[0], "Volatility": perf3[1], "Sharpe": perf3[2], "Method": "Hierarchical Risk Parity"})
            for t, w in clean_hrp.items():
                if abs(w) > 1e-5:
                    weights_frames.append({"Strategy": "HRP", "Ticker": t, "Name": get_name(t), "Weight": w})
        except Exception as e:
            results.append({"Strategy": "HRP", "Expected Return": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Method": f"Failed: {e}"})

        payload = {"results": pd.DataFrame(results), "weights": pd.DataFrame(weights_frames), "prices": prices, "excluded": excluded, "cov_method": cov_method}
        return prices, payload, None
    except Exception as e:
        return prices, None, f"PyPortfolioOpt optimization failed: {e}"

def chart_optimization_summary(results_df):
    if results_df is None or results_df.empty:
        fig = go.Figure(); fig.add_annotation(text="No optimizer result available.", showarrow=False)
        return institutional_layout(fig, "PyPortfolioOpt Optimization", 650)
    plot_df = results_df.dropna(subset=["Expected Return", "Volatility"]).copy()
    if plot_df.empty:
        fig = go.Figure(); fig.add_annotation(text="Optimization strategies failed. Check constraints / Yahoo data coverage.", showarrow=False)
        return institutional_layout(fig, "PyPortfolioOpt Optimization", 650)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14, subplot_titles=["Risk/Return Strategy Map", "Sharpe Ranking"])
    fig.add_trace(go.Scatter(x=plot_df["Volatility"]*100, y=plot_df["Expected Return"]*100, mode="markers+text", text=plot_df["Strategy"], textposition="top center", marker=dict(size=np.clip((plot_df["Sharpe"].fillna(0).abs()+0.5)*18, 12, 40)), name="Optimized Strategies", hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df["Strategy"], y=plot_df["Sharpe"], name="Sharpe", hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title="Annualized Volatility (%)", row=1, col=1); fig.update_yaxes(title="Expected Return (%)", row=1, col=1); fig.update_yaxes(title="Sharpe", row=2, col=1)
    return institutional_layout(fig, "PyPortfolioOpt | Portfolio Construction Diagnostics", 780)

def chart_weight_allocation(weights_df):
    if weights_df is None or weights_df.empty:
        fig = go.Figure(); fig.add_annotation(text="No optimized weights available.", showarrow=False)
        return institutional_layout(fig, "Optimized Weight Allocation", 650)
    fig = go.Figure()
    for strat in weights_df["Strategy"].unique():
        sub = weights_df[weights_df["Strategy"] == strat].sort_values("Weight", ascending=True)
        fig.add_trace(go.Bar(x=sub["Weight"]*100, y=sub["Ticker"], orientation="h", name=strat, hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group", xaxis_title="Weight (%)", yaxis_title="Ticker")
    return institutional_layout(fig, "Optimized Portfolio Weights", 780)

# =============================================================================
# 11) Robust widget-driven reactive dashboard
# =============================================================================

class QFADashboardFinTECH:
    """
    Explicit widget-driven Panel app.
    This avoids Colab issues with param.Selector + pn.Param not propagating updates.
    """

    def __init__(self):
        first_asset_class = list(UNIVERSE.keys())[0]
        first_region = get_regions(first_asset_class)[0]
        first_ticker = get_tickers(first_asset_class, first_region)[1] if len(get_tickers(first_asset_class, first_region)) > 1 else get_tickers(first_asset_class, first_region)[0]

        self.asset_class = pn.widgets.Select(
            name="Asset Class",
            options=list(UNIVERSE.keys()),
            value=first_asset_class,
            sizing_mode="stretch_width",
        )
        self.region = pn.widgets.Select(
            name="Region",
            options=get_regions(first_asset_class),
            value=first_region,
            sizing_mode="stretch_width",
        )
        self.ticker = pn.widgets.Select(
            name="Instrument",
            options=get_tickers(first_asset_class, first_region),
            value=first_ticker,
            sizing_mode="stretch_width",
        )
        self.benchmark = pn.widgets.Select(
            name="Benchmark",
            options=list(BENCHMARKS.keys()),
            value="Nasdaq 100 Index",
            sizing_mode="stretch_width",
        )
        self.start_date = pn.widgets.DatePicker(
            name="Start Date",
            value=DEFAULT_START,
            sizing_mode="stretch_width",
        )
        self.end_date = pn.widgets.DatePicker(
            name="End Date",
            value=DEFAULT_END,
            sizing_mode="stretch_width",
        )

        self.generate_report_button = pn.widgets.Button(
            name="Generate Selected ETF HTML Report",
            button_type="primary",
            sizing_mode="stretch_width",
        )

        self.batch_status = pn.pane.Markdown(
            "Batch report generator is ready.",
            sizing_mode="stretch_width",
        )

        # Persistent KPI pane. We update this object explicitly whenever a selector
        # changes. This avoids stale KPI cards on Render/Bokeh websocket sessions.
        self.kpi_pane = pn.pane.HTML(
            "<h3>Loading Executive KPI Dashboard...</h3>",
            sizing_mode="stretch_width",
        )

        self.asset_class.param.watch(self._on_asset_class_change, "value")
        self.region.param.watch(self._on_region_change, "value")
        self.ticker.param.watch(self._on_input_change, "value")
        self.benchmark.param.watch(self._on_input_change, "value")
        self.start_date.param.watch(self._on_input_change, "value")
        self.end_date.param.watch(self._on_input_change, "value")
        self.generate_report_button.on_click(self._on_generate_report_click)

        self._refresh_kpi()

    def _on_asset_class_change(self, event):
        regions = get_regions(event.new)
        self.region.options = regions
        self.region.value = regions[0]

        tickers = get_tickers(event.new, self.region.value)
        self.ticker.options = tickers
        self.ticker.value = tickers[0]
        self._refresh_kpi()

    def _on_region_change(self, event):
        tickers = get_tickers(self.asset_class.value, event.new)
        self.ticker.options = tickers
        self.ticker.value = tickers[0]
        self._refresh_kpi()

    def _on_input_change(self, event):
        self._refresh_kpi()

    def _refresh_kpi(self):
        try:
            pane = self.render_kpi(
                self.ticker.value,
                self.asset_class.value,
                self.region.value,
                self.benchmark.value,
                self.start_date.value,
                self.end_date.value,
            )
            self.kpi_pane.object = pane.object
        except Exception as exc:
            self.kpi_pane.object = f"""
            <div style='padding:16px;border:1px solid #b91c1c;border-radius:12px;background:#fff5f5;'>
                <h3 style='margin-top:0;color:#991b1b;'>KPI refresh failed</h3>
                <pre style='white-space:pre-wrap;'>{str(exc)}</pre>
            </div>
            """

    @staticmethod
    def _dates(start_date, end_date):
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def render_kpi(self, ticker, asset_class, region, benchmark, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        benchmark_ticker = BENCHMARKS[benchmark]
        returns, asset_df, _ = matched_asset_benchmark_returns(ticker, benchmark_ticker, start, end)

        if asset_df.empty:
            return pn.pane.HTML("<h3>No Yahoo Finance data available for selected ticker.</h3>")

        asset_df = add_indicators(asset_df)
        metrics = calc_risk_metrics(asset_df["Return"])
        matched_obs = len(returns) if not returns.empty else len(asset_df)

        return pn.pane.HTML(
            kpi_html(
                metrics,
                ticker,
                get_name(ticker),
                asset_class,
                region,
                benchmark,
                benchmark_ticker,
                matched_obs,
            ),
            sizing_mode="stretch_width",
        )

    def render_price_technical(self, ticker, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        df = fetch_data(ticker, start, end)
        if not df.empty:
            df = add_indicators(df)
        return pn.pane.Plotly(chart_price_technical(df, ticker), sizing_mode="stretch_width")

    def render_risk_metrics(self, ticker, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        df = fetch_data(ticker, start, end)
        if not df.empty:
            df = add_indicators(df)
        return pn.pane.Plotly(chart_risk(df, ticker), sizing_mode="stretch_width")

    def render_benchmark_relative(self, ticker, benchmark, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        benchmark_ticker = BENCHMARKS[benchmark]
        returns, _, _ = matched_asset_benchmark_returns(ticker, benchmark_ticker, start, end)
        return pn.pane.Plotly(
            chart_benchmark_relative(returns, ticker, benchmark, benchmark_ticker),
            sizing_mode="stretch_width",
        )

    def render_universe_board(self, asset_class, region, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        return pn.pane.Plotly(
            chart_universe_board(asset_class, region, start, end),
            sizing_mode="stretch_width",
        )

    def render_risk_table(self, asset_class, region, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        prices, excluded = fetch_prices(get_tickers(asset_class, region), start, end)

        if prices.empty:
            return pn.pane.Markdown("### No matched Yahoo Finance data for selected universe.")

        returns = prices.pct_change().dropna()
        rows = []
        for t in returns.columns:
            m = calc_risk_metrics(returns[t])
            rows.append({
                "Ticker": t,
                "Name": get_name(t),
                "Ann Return": pct_fmt(m["Ann Return"]),
                "Ann Vol": pct_fmt(m["Ann Vol"]),
                "Sharpe": num_fmt(m["Sharpe"], 2),
                "Sortino": num_fmt(m["Sortino"], 2),
                "Max Drawdown": pct_fmt(m["Max Drawdown"]),
                "VaR 95": pct_fmt(m["VaR 95"]),
                "CVaR 95": pct_fmt(m["CVaR 95"]),
                "Win Rate": pct_fmt(m["Win Rate"]),
            })

        note = f"Excluded: {', '.join(sorted(set(excluded)))}" if excluded else "All tickers matched."

        return pn.Column(
            pn.pane.Markdown(f"### Risk Metrics Table\n**Data policy:** original matched Yahoo prices only. {note}"),
            pn.widgets.Tabulator(
                pd.DataFrame(rows),
                pagination="remote",
                page_size=12,
                sizing_mode="stretch_width",
                height=430,
            ),
            sizing_mode="stretch_width",
        )

    def render_optimization(self, asset_class, region, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        prices, payload, err = optimize_universe_pyportfolioopt(asset_class, region, start, end)
        if err:
            return pn.Column(pn.pane.Markdown(f"### PyPortfolioOpt Optimizer Warning\n{err}"), sizing_mode="stretch_width")
        results = payload["results"]
        weights = payload["weights"]
        excluded = payload.get("excluded", [])
        note = f"Excluded due to missing/unmatched Yahoo data: {', '.join(excluded)}" if excluded else "All selected instruments matched."
        display_results = results.copy()
        for c in ["Expected Return", "Volatility"]:
            display_results[c] = display_results[c].map(lambda x: pct_fmt(x) if pd.notna(x) else "N/A")
        display_results["Sharpe"] = display_results["Sharpe"].map(lambda x: num_fmt(x, 2) if pd.notna(x) else "N/A")
        display_weights = weights.copy()
        if not display_weights.empty:
            display_weights["Weight"] = display_weights["Weight"].map(lambda x: pct_fmt(x) if pd.notna(x) else "N/A")
        return pn.Column(
            pn.pane.Markdown(f"""
### PyPortfolioOpt Portfolio Optimizer
**Universe:** {asset_class} / {region}  
**Risk-Free Rate:** 4.50% fixed  
**Data policy:** original matched Yahoo close prices only; no synthetic or fallback prices.  
**{note}**
"""),
            pn.pane.Plotly(chart_optimization_summary(results), sizing_mode="stretch_width"),
            pn.pane.Plotly(chart_weight_allocation(weights), sizing_mode="stretch_width"),
            pn.widgets.Tabulator(display_results, pagination="remote", page_size=10, sizing_mode="stretch_width", height=260),
            pn.widgets.Tabulator(display_weights, pagination="remote", page_size=20, sizing_mode="stretch_width", height=360),
            sizing_mode="stretch_width",
        )

    def render_quantstats(self, ticker, benchmark, start_date, end_date):
        start, end = self._dates(start_date, end_date)
        benchmark_ticker = BENCHMARKS[benchmark]
        return make_quantstats_report(ticker, benchmark, benchmark_ticker, start, end)

    def _on_generate_report_click(self, event):
        """
        Sidebar button callback.
        Generates a standalone selected ETF/instrument HTML report from current combo-box values.
        """
        try:
            start, end = self._dates(self.start_date.value, self.end_date.value)
            benchmark_ticker = BENCHMARKS[self.benchmark.value]

            out_path = generate_selected_etf_batch_report(
                ticker=self.ticker.value,
                benchmark_label=self.benchmark.value,
                benchmark_ticker=benchmark_ticker,
                start=start,
                end=end,
            )

            self.batch_status.object = (
                f"✅ Batch HTML report generated successfully.  \n"
                f"`{out_path}`"
            )

        except Exception as e:
            self.batch_status.object = (
                "❌ Batch report generation failed.  \n"
                f"`{str(e)}`"
            )

    def sidebar(self):
        return pn.Column(
            pn.pane.Markdown(
                """
                # QFA Prime  
                ## FinTECH Institutional Platform
                """
            ),
            pn.pane.Markdown("### Investment Universe"),
            self.asset_class,
            self.region,
            self.ticker,
            pn.pane.Markdown("### Benchmark Selection"),
            self.benchmark,
            pn.pane.Markdown("### Analysis Period"),
            self.start_date,
            self.end_date,
            pn.pane.Markdown("### Selected ETF Batch Report"),
            self.generate_report_button,
            self.batch_status,
            pn.pane.Markdown(
                f"""
                ---
                **TA-Lib:** {'Active' if TALIB_AVAILABLE else 'Indicator formula mode'}  
                **QuantStats:** {'Active' if QUANTSTATS_AVAILABLE else 'Not Available'}  
                **PyPortfolioOpt:** {'Active' if PYPFOPT_AVAILABLE else 'Not Available'}  
                **Risk-Free Rate:** 4.50% fixed  
                **Data:** Yahoo Finance original matched data only  
                **Synthetic/Fallback Prices:** Never  
                **S&P Benchmark Symbol:** ^GSPC  
                **SPY Hard-Code Benchmark:** Never
                """
            ),
            width=350,
            sizing_mode="fixed",
            styles={
                "background": "#f4f7fb",
                "padding": "20px",
                "border-right": "1px solid #d8e1ec",
                "height": "100vh",
                "overflow-y": "auto",
            },
        )

    def view(self):
        header = pn.pane.HTML(
            """
            <div style="background:linear-gradient(90deg,#0f172a,#1e293b);padding:22px 26px;border-radius:16px;margin-bottom:16px;color:white;box-shadow:0 4px 14px rgba(0,0,0,0.12);">
                <div style="font-size:29px;font-weight:850;">QFA Dashboard FinTECH</div>
                <div style="font-size:14px;color:#cbd5e1;margin-top:6px;">
                    Reactive institutional analytics with strict Yahoo Finance matched data,
                    benchmark-sensitive QuantStats tearsheets, and no synthetic price fallback.
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        )

        tabs = pn.Tabs(
            (
                "Executive KPI Dashboard",
                self.kpi_pane,
            ),
            (
                "Price & TA-Lib Technicals",
                pn.bind(self.render_price_technical, self.ticker, self.start_date, self.end_date),
            ),
            (
                "Institutional Risk Metrics",
                pn.bind(self.render_risk_metrics, self.ticker, self.start_date, self.end_date),
            ),
            (
                "Benchmark Relative Analytics",
                pn.bind(self.render_benchmark_relative, self.ticker, self.benchmark, self.start_date, self.end_date),
            ),
            (
                "Investment Universe Board",
                pn.bind(self.render_universe_board, self.asset_class, self.region, self.start_date, self.end_date),
            ),
            (
                "Risk Metrics Table",
                pn.bind(self.render_risk_table, self.asset_class, self.region, self.start_date, self.end_date),
            ),
            (
                "PyPortfolioOpt Optimizer",
                pn.bind(self.render_optimization, self.asset_class, self.region, self.start_date, self.end_date),
            ),
            (
                "Tearsheet",
                pn.bind(self.render_quantstats, self.ticker, self.benchmark, self.start_date, self.end_date),
            ),
            dynamic=False,
            sizing_mode="stretch_width",
        )

        main = pn.Column(
            header,
            tabs,
            sizing_mode="stretch_width",
            styles={"padding": "18px", "background": "#ffffff"},
        )

        return pn.Row(self.sidebar(), main, sizing_mode="stretch_width")


# =============================================================================
# Render / Panel server entry point
# =============================================================================

dashboard = QFADashboardFinTECH()
app = dashboard.view()
app.servable(title="QFA Dashboard FinTECH")
