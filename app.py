
# =============================================================================
# QFA Hedge Fund Dashboard - Render Ready
# Fully reactive Panel architecture: pn.widgets + pn.bind only
# Yahoo-only data policy | QuantStats Tearsheet | PyPortfolioOpt | TA-Lib optional
# =============================================================================

import os
import io
import time
import math
import json
import warnings
import traceback
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matplotlib / QuantStats font hygiene for Render Linux
import matplotlib
matplotlib.use("Agg")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    qs = None
    QUANTSTATS_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    expected_returns = None
    risk_models = None
    EfficientFrontier = None
    HRPOpt = None
    PYPFOPT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Panel Extension
# -----------------------------------------------------------------------------
pn.extension("plotly", "tabulator", sizing_mode="stretch_width", notifications=True)


# -----------------------------------------------------------------------------
# Global Institutional Config
# -----------------------------------------------------------------------------
APP_TITLE = "QFA Hedge Fund Dashboard"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045  # fixed 4.5%, later can be replaced by 13-week T-Bill
MIN_OBS = 90
CACHE_TTL_SECONDS = 900

# No Arial. Keep a cross-platform stack.
FONT_STACK = "DejaVu Sans, Liberation Sans, Segoe UI, Helvetica, sans-serif"


# -----------------------------------------------------------------------------
# Investment Universe
# -----------------------------------------------------------------------------
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
            "VUG": "Vanguard Growth ETF",
            "VTV": "Vanguard Value ETF",
        },
        "Europe": {
            "VGK": "Vanguard FTSE Europe ETF",
            "FEZ": "SPDR EURO STOXX 50 ETF",
            "EWG": "iShares MSCI Germany ETF",
            "EWQ": "iShares MSCI France ETF",
            "EWI": "iShares MSCI Italy ETF",
            "EWP": "iShares MSCI Spain ETF",
        },
        "Emerging Markets": {
            "VWO": "Vanguard FTSE Emerging Markets ETF",
            "EEM": "iShares MSCI Emerging Markets ETF",
            "FXI": "iShares China Large-Cap ETF",
            "INDA": "iShares MSCI India ETF",
            "EWZ": "iShares MSCI Brazil ETF",
            "EWY": "iShares MSCI South Korea ETF",
            "EWT": "iShares MSCI Taiwan ETF",
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
        }
    },
    "Fixed Income": {
        "United States": {
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "AGG": "iShares Core U.S. Aggregate Bond ETF",
            "LQD": "iShares Investment Grade Corporate Bond ETF",
            "HYG": "iShares High Yield Corporate Bond ETF",
            "TIP": "iShares TIPS Bond ETF",
        }
    },
    "Commodities": {
        "Global": {
            "GLD": "SPDR Gold Shares",
            "SLV": "iShares Silver Trust",
            "USO": "United States Oil Fund",
            "UNG": "United States Natural Gas Fund",
            "DBC": "Invesco DB Commodity Index Tracking Fund",
            "DBA": "Invesco DB Agriculture Fund",
            "CPER": "United States Copper Index Fund",
        }
    },
    "Volatility / Alternatives": {
        "United States": {
            "VIXY": "ProShares VIX Short-Term Futures ETF",
            "SVXY": "ProShares Short VIX Short-Term Futures ETF",
            "BTAL": "AGF U.S. Market Neutral Anti-Beta Fund",
            "QAI": "IQ Hedge Multi-Strategy Tracker ETF",
            "MNA": "IQ Merger Arbitrage ETF",
        }
    },
    "Crypto Proxy": {
        "United States": {
            "IBIT": "iShares Bitcoin Trust",
            "FBTC": "Fidelity Wise Origin Bitcoin Fund",
            "BITO": "ProShares Bitcoin Strategy ETF",
            "GBTC": "Grayscale Bitcoin Trust",
            "ETHE": "Grayscale Ethereum Trust",
        }
    },
}

BENCHMARKS = {
    "S&P 500 Index": "^GSPC",           # not SPY
    "Nasdaq 100 Index": "^NDX",
    "Russell 2000 Index": "^RUT",
    "Dow Jones Industrial Average": "^DJI",
    "US Aggregate Bond ETF": "AGG",
    "Gold ETF": "GLD",
    "Emerging Markets ETF": "EEM",
    "Global Equity ETF": "VT",
    "Cash Proxy / 1-3Y Treasury ETF": "SHY",
}

STRESS_SCENARIOS = {
    "Equity Shock -10%": -0.10,
    "Equity Shock -20%": -0.20,
    "Sharp Rally +10%": 0.10,
    "Rate Shock Proxy -5%": -0.05,
    "Liquidity Shock -7.5%": -0.075,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def flatten_universe() -> pd.DataFrame:
    rows = []
    for asset_class, regions in UNIVERSE.items():
        for region, instruments in regions.items():
            for ticker, name in instruments.items():
                rows.append(
                    {
                        "Asset Class": asset_class,
                        "Region": region,
                        "Ticker": ticker,
                        "Name": name,
                    }
                )
    return pd.DataFrame(rows)


UNIVERSE_DF = flatten_universe()


def get_regions(asset_class: str):
    return list(UNIVERSE.get(asset_class, {}).keys())


def get_tickers(asset_class: str, region: str):
    return list(UNIVERSE.get(asset_class, {}).get(region, {}).keys())


def get_name(ticker: str) -> str:
    row = UNIVERSE_DF.loc[UNIVERSE_DF["Ticker"] == ticker]
    if row.empty:
        return ticker
    return str(row.iloc[0]["Name"])


def normalize_date(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")
    if hasattr(x, "strftime"):
        return x.strftime("%Y-%m-%d")
    return str(x)[:10]


def fmt_pct(x, digits=2):
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return "N/A"
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "N/A"


def fmt_num(x, digits=2):
    try:
        if x is None or pd.isna(x) or np.isinf(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"


def status_badge(label: str, ok: bool):
    color = "#166534" if ok else "#991b1b"
    bg = "#dcfce7" if ok else "#fee2e2"
    return f"""
    <span style="background:{bg};color:{color};padding:4px 9px;border-radius:999px;
    font-size:12px;font-weight:700;">{label}</span>
    """


# -----------------------------------------------------------------------------
# Yahoo-only Data Layer
# -----------------------------------------------------------------------------
@lru_cache(maxsize=512)
def fetch_ohlcv_cached(ticker: str, start: str, end: str, cache_bucket: int) -> pd.DataFrame:
    """
    Yahoo-only OHLCV fetch.
    No synthetic/fallback price data is generated. If Yahoo returns no usable data,
    an empty DataFrame is returned and the UI reports it.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=25,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance can return MultiIndex even for a single ticker.
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if "Close" not in df.columns:
            return pd.DataFrame()

        # Do not synthesize prices. For OHLC components only, if missing, mirror Close
        # only to allow chart functions. Close itself remains original Yahoo data.
        for col in required:
            if col not in df.columns:
                if col == "Volume":
                    df[col] = np.nan
                else:
                    df[col] = df["Close"]

        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Close"])

        if len(df) < 2:
            return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


def cache_bucket():
    return int(time.time() // CACHE_TTL_SECONDS)


def fetch_ohlcv(ticker: str, start, end) -> pd.DataFrame:
    start_s = normalize_date(start)
    end_s = normalize_date(end)
    return fetch_ohlcv_cached(ticker, start_s, end_s, cache_bucket()).copy()


def fetch_price_matrix(tickers, start, end) -> pd.DataFrame:
    series = {}
    rejected = []

    for ticker in tickers:
        df = fetch_ohlcv(ticker, start, end)
        if df.empty:
            rejected.append(ticker)
            continue
        series[ticker] = df["Close"]

    if not series:
        return pd.DataFrame()

    prices = pd.DataFrame(series).sort_index()
    prices = prices.dropna(how="all")
    prices = prices.ffill(limit=3)

    # Keep only matched, sufficiently populated series.
    min_count = max(MIN_OBS, int(len(prices) * 0.70))
    prices = prices.dropna(axis=1, thresh=min_count)
    prices = prices.dropna()

    return prices


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Log Return"] = np.log(d["Close"] / d["Close"].shift(1))
    d["Cumulative Return"] = (1 + d["Return"].fillna(0)).cumprod() - 1

    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    d["Vol21"] = d["Return"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    d["Vol63"] = d["Return"].rolling(63).std() * np.sqrt(TRADING_DAYS)

    wealth = (1 + d["Return"].fillna(0)).cumprod()
    d["Drawdown"] = wealth / wealth.cummax() - 1

    if TALIB_AVAILABLE:
        try:
            close = d["Close"].astype(float).values
            high = d["High"].astype(float).values
            low = d["Low"].astype(float).values

            d["RSI"] = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            d["MACD"] = macd
            d["MACD Signal"] = signal
            d["MACD Hist"] = hist

            upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            d["BB Upper"] = upper
            d["BB Mid"] = mid
            d["BB Lower"] = lower
            d["ATR"] = talib.ATR(high, low, close, timeperiod=14)

            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3,
            )
            d["Stoch K"] = slowk
            d["Stoch D"] = slowd
            d["Indicator Engine"] = "TA-Lib"
            return d
        except Exception:
            pass

    # Indicator fallback only. Price data remains Yahoo-only.
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD Hist"] = d["MACD"] - d["MACD Signal"]

    d["BB Mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB Upper"] = d["BB Mid"] + 2 * bb_std
    d["BB Lower"] = d["BB Mid"] - 2 * bb_std

    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - d["Close"].shift()).abs(),
            (d["Low"] - d["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch K"] = 100 * (d["Close"] - low14) / (high14 - low14)
    d["Stoch D"] = d["Stoch K"].rolling(3).mean()
    d["Indicator Engine"] = "Formula fallback"

    return d


# -----------------------------------------------------------------------------
# Risk Engine
# -----------------------------------------------------------------------------
def risk_metrics(returns: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) < 30:
        return {k: np.nan for k in [
            "Ann Return", "Ann Vol", "Sharpe", "Sortino", "Max Drawdown",
            "VaR 95", "CVaR 95", "VaR 99", "CVaR 99", "Skew", "Kurtosis",
            "Win Rate", "Calmar", "Omega"
        ]}

    ann_return = (1 + r.mean()) ** TRADING_DAYS - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_return - rf) / downside if downside and downside > 0 else np.nan

    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    max_dd = dd.min()

    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean()
    var99 = r.quantile(0.01)
    cvar99 = r[r <= var99].mean()

    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    omega = gains / losses if losses > 0 else np.nan

    return {
        "Ann Return": ann_return,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "VaR 95": var95,
        "CVaR 95": cvar95,
        "VaR 99": var99,
        "CVaR 99": cvar99,
        "Skew": r.skew(),
        "Kurtosis": r.kurtosis(),
        "Win Rate": (r > 0).mean(),
        "Calmar": calmar,
        "Omega": omega,
    }


def active_metrics(asset_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    joined = pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(joined) < 30:
        return {k: np.nan for k in ["Tracking Error", "Information Ratio", "Beta", "Alpha", "Correlation"]}

    active = joined["asset"] - joined["bench"]
    te = active.std() * np.sqrt(TRADING_DAYS)
    ir = (active.mean() * TRADING_DAYS) / te if te > 0 else np.nan

    cov = np.cov(joined["asset"], joined["bench"])[0, 1]
    var = np.var(joined["bench"])
    beta = cov / var if var > 0 else np.nan
    alpha = (joined["asset"].mean() * TRADING_DAYS) - beta * (joined["bench"].mean() * TRADING_DAYS) if pd.notna(beta) else np.nan
    corr = joined["asset"].corr(joined["bench"])

    return {
        "Tracking Error": te,
        "Information Ratio": ir,
        "Beta": beta,
        "Alpha": alpha,
        "Correlation": corr,
    }


# -----------------------------------------------------------------------------
# Styling / Layout
# -----------------------------------------------------------------------------
def css():
    return f"""
    <style>
    body, .bk-root, .bk, .bk-input, .bk-btn {{
        font-family: {FONT_STACK};
    }}
    .qfa-header {{
        background: linear-gradient(90deg, #0f172a, #172554, #1e293b);
        color: white;
        padding: 24px 28px;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(15,23,42,.18);
        margin-bottom: 16px;
    }}
    .qfa-title {{
        font-size: 30px;
        font-weight: 850;
        letter-spacing: -0.03em;
    }}
    .qfa-subtitle {{
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 7px;
        line-height: 1.5;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(160px, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }}
    .kpi-card {{
        background: #ffffff;
        border: 1px solid #dbe4ef;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 2px 12px rgba(15,23,42,.055);
    }}
    .kpi-card.pos {{ background: #f0fdf4; border-color: #86efac; }}
    .kpi-card.neg {{ background: #fff1f2; border-color: #fca5a5; }}
    .kpi-card.warn {{ background: #fffbeb; border-color: #fcd34d; }}
    .kpi-label {{
        color: #64748b;
        font-size: 12px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .04em;
    }}
    .kpi-value {{
        color: #0f172a;
        font-size: 22px;
        font-weight: 850;
        margin-top: 6px;
        white-space: nowrap;
    }}
    .qfa-note {{
        background: #f8fafc;
        border: 1px solid #dbe4ef;
        border-radius: 14px;
        padding: 12px 14px;
        color: #334155;
        font-size: 13px;
        line-height: 1.45;
    }}
    </style>
    """


def make_kpi_cards(ticker, benchmark_label, metrics, active=None, engine_label=""):
    active = active or {}
    cards = [
        ("Instrument", ticker, ""),
        ("Benchmark", benchmark_label, ""),
        ("Annual Return", fmt_pct(metrics.get("Ann Return")), "pos" if metrics.get("Ann Return", 0) > 0 else "neg"),
        ("Annual Volatility", fmt_pct(metrics.get("Ann Vol")), ""),
        ("Sharpe @ RF 4.5%", fmt_num(metrics.get("Sharpe"), 2), "pos" if metrics.get("Sharpe", 0) > 1 else "warn"),
        ("Sortino", fmt_num(metrics.get("Sortino"), 2), "pos" if metrics.get("Sortino", 0) > 1 else "warn"),
        ("Max Drawdown", fmt_pct(metrics.get("Max Drawdown")), "neg"),
        ("CVaR 95", fmt_pct(metrics.get("CVaR 95")), "neg"),
        ("Tracking Error", fmt_pct(active.get("Tracking Error")), ""),
        ("Information Ratio", fmt_num(active.get("Information Ratio"), 2), "pos" if active.get("Information Ratio", 0) > 0 else "warn"),
        ("Beta", fmt_num(active.get("Beta"), 2), ""),
        ("TA Engine", engine_label, ""),
    ]

    html = '<div class="kpi-grid">'
    for label, value, tone in cards:
        html += f"""
        <div class="kpi-card {tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """
    html += "</div>"
    return pn.pane.HTML(html, sizing_mode="stretch_width")


def empty_state(title, detail):
    return pn.Column(
        pn.pane.HTML(
            f"""
            <div class="qfa-note">
                <b>{title}</b><br>{detail}
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def chart_layout(fig, title, height=720):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=20)),
        template="plotly_white",
        height=height,
        margin=dict(l=46, r=28, t=78, b=48),
        font=dict(family="DejaVu Sans, Liberation Sans, Segoe UI, Helvetica, sans-serif", size=12, color="#1e293b"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(203,213,225,.48)", zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(203,213,225,.48)", zeroline=False)
    return fig


# -----------------------------------------------------------------------------
# Build Blocks
# -----------------------------------------------------------------------------
def build_kpi(ticker, benchmark_label, start, end):
    bench = BENCHMARKS[benchmark_label]
    asset = add_indicators(fetch_ohlcv(ticker, start, end))
    benchmark = add_indicators(fetch_ohlcv(bench, start, end))

    if asset.empty:
        return empty_state("No Yahoo data", f"{ticker} could not be downloaded from Yahoo Finance for the selected period.")

    metrics = risk_metrics(asset["Return"], RISK_FREE_RATE)
    act = {}
    if not benchmark.empty:
        act = active_metrics(asset["Return"], benchmark["Return"])

    engine = str(asset["Indicator Engine"].iloc[-1]) if "Indicator Engine" in asset.columns else ("TA-Lib" if TALIB_AVAILABLE else "Formula fallback")
    return make_kpi_cards(ticker, benchmark_label, metrics, act, engine)


def build_price_chart(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, start, end))
    if df.empty:
        return empty_state("No Yahoo data", f"No original Yahoo data was available for {ticker}.")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.50, 0.17, 0.18, 0.15],
        subplot_titles=["OHLC + Bollinger + MA50/MA200", "RSI 14", "MACD", "Stochastic"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC", increasing_line_color="#166534", decreasing_line_color="#991b1b",
            hoverinfo="x+y+name",
        ),
        row=1, col=1,
    )

    for col, name, dash, width in [
        ("BB Upper", "BB Upper", "dash", 1.1),
        ("BB Mid", "BB Mid", "dot", 1.1),
        ("BB Lower", "BB Lower", "dash", 1.1),
        ("MA50", "MA50", "solid", 1.7),
        ("MA200", "MA200", "solid", 2.0),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name, mode="lines",
                           line=dict(dash=dash, width=width),
                           hovertemplate="%{y:.4f}<extra></extra>"),
                row=1, col=1,
            )

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines",
                             hovertemplate="%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#64748b")
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines",
                             hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD Signal"], name="Signal", mode="lines",
                             hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD Hist"], name="Histogram", opacity=.42,
                         hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch K"], name="Stoch K", mode="lines",
                             hovertemplate="%{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch D"], name="Stoch D", mode="lines",
                             hovertemplate="%{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_hline(y=80, row=4, col=1, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=20, row=4, col=1, line_dash="dash", line_color="#64748b")

    return pn.pane.Plotly(chart_layout(fig, f"{ticker} | TA-Lib Technical Dashboard", 920), config={"responsive": True})


def build_risk_chart(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, start, end))
    if df.empty:
        return empty_state("No Yahoo data", f"No original Yahoo data was available for {ticker}.")

    r = df["Return"].dropna()
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=["Cumulative Return", "Rolling Volatility", "Drawdown", "Daily Return Distribution"],
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Cumulative Return"]*100, name="Cumulative Return %",
                             mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol21"]*100, name="Vol 21D %",
                             mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol63"]*100, name="Vol 63D %",
                             mode="lines", hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"]*100, name="Drawdown %",
                             mode="lines", fill="tozeroy", hovertemplate="%{y:.2f}%<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Histogram(x=r*100, nbinsx=70, name="Daily Returns %",
                               hovertemplate="%{x:.2f}%<extra></extra>"), row=4, col=1)
    return pn.pane.Plotly(chart_layout(fig, f"{ticker} | Hedge Fund Risk Diagnostics", 880), config={"responsive": True})


def build_benchmark_relative(ticker, benchmark_label, start, end):
    bench = BENCHMARKS[benchmark_label]
    asset = fetch_ohlcv(ticker, start, end)
    benchmark = fetch_ohlcv(bench, start, end)

    if asset.empty or benchmark.empty:
        return empty_state("Benchmark relative data unavailable", f"Yahoo data missing for {ticker} or {bench}.")

    joined = pd.concat([asset["Close"].rename(ticker), benchmark["Close"].rename(bench)], axis=1).dropna()
    if len(joined) < MIN_OBS:
        return empty_state("Insufficient matched data", f"Matched Yahoo data count is below {MIN_OBS} observations.")

    ret = joined.pct_change().dropna()
    cum = (1 + ret).cumprod() - 1
    active = ret[ticker] - ret[bench]
    active_cum = (1 + active).cumprod() - 1
    rolling_te = active.rolling(63).std() * np.sqrt(TRADING_DAYS)
    rolling_ir = (active.rolling(63).mean() * TRADING_DAYS) / rolling_te

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=["Cumulative Return Comparison", "Active Cumulative Return", "Rolling Tracking Error", "Rolling Information Ratio"],
    )
    fig.add_trace(go.Scatter(x=cum.index, y=cum[ticker]*100, name=ticker, mode="lines",
                             hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum.index, y=cum[bench]*100, name=benchmark_label, mode="lines",
                             hovertemplate="%{y:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_cum.index, y=active_cum*100, name="Active Return", mode="lines",
                             hovertemplate="%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_te.index, y=rolling_te*100, name="Rolling TE 63D", mode="lines",
                             hovertemplate="%{y:.2f}%<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=rolling_ir.index, y=rolling_ir, name="Rolling IR 63D", mode="lines",
                             hovertemplate="%{y:.2f}<extra></extra>"), row=4, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=0, row=4, col=1, line_dash="dash", line_color="#64748b")
    return pn.pane.Plotly(chart_layout(fig, f"{ticker} vs {benchmark_label} | Benchmark Relative Analytics", 900), config={"responsive": True})


def build_universe_board(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)
    if prices.empty:
        return empty_state("Universe unavailable", "No matched original Yahoo data was available for the selected universe.")

    returns = prices.pct_change().dropna()
    rows = []
    for t in returns.columns:
        m = risk_metrics(returns[t], RISK_FREE_RATE)
        total_return = prices[t].iloc[-1] / prices[t].iloc[0] - 1
        rows.append({
            "Ticker": t,
            "Name": get_name(t),
            "Total Return": total_return,
            "Ann Return": m["Ann Return"],
            "Ann Vol": m["Ann Vol"],
            "Sharpe": m["Sharpe"],
            "Max DD": m["Max Drawdown"],
            "CVaR 95": m["CVaR 95"],
        })

    board = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.13,
        subplot_titles=["Sharpe Ranking", "Annual Return vs Annual Volatility Map"],
    )
    fig.add_trace(go.Bar(x=board["Ticker"], y=board["Sharpe"], name="Sharpe",
                         hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=board["Ann Vol"]*100, y=board["Ann Return"]*100,
            mode="markers+text", text=board["Ticker"], textposition="top center",
            marker=dict(size=np.clip((board["Sharpe"].fillna(0).abs()+0.6)*14, 9, 32)),
            name="Risk/Return",
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title="Ticker", row=1, col=1)
    fig.update_xaxes(title="Annualized Volatility (%)", row=2, col=1)
    fig.update_yaxes(title="Sharpe", row=1, col=1)
    fig.update_yaxes(title="Annualized Return (%)", row=2, col=1)

    table_df = board.copy()
    for col in ["Total Return", "Ann Return", "Ann Vol", "Max DD", "CVaR 95"]:
        table_df[col] = table_df[col].map(lambda x: fmt_pct(x))
    table_df["Sharpe"] = table_df["Sharpe"].map(lambda x: fmt_num(x, 2))

    return pn.Column(
        pn.pane.Plotly(chart_layout(fig, f"{asset_class} | {region} | Universe Board", 820), config={"responsive": True}),
        pn.widgets.Tabulator(table_df, height=360, pagination="remote", page_size=12, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


def build_optimizer(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)

    if prices.empty or prices.shape[1] < 3:
        return empty_state("Optimizer unavailable", "At least 3 instruments with matched Yahoo data are required.")

    if not PYPFOPT_AVAILABLE:
        return empty_state(
            "PyPortfolioOpt unavailable",
            "PyPortfolioOpt is not installed. Check requirements.txt and Render build logs.",
        )

    try:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        outputs = []

        ef = EfficientFrontier(mu, S)
        w_sharpe = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        clean_sharpe = ef.clean_weights()
        ret_s, vol_s, sharpe_s = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        ef2 = EfficientFrontier(mu, S)
        w_minvol = ef2.min_volatility()
        clean_minvol = ef2.clean_weights()
        ret_m, vol_m, sharpe_m = ef2.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        weights_df = pd.DataFrame({
            "Max Sharpe": pd.Series(clean_sharpe),
            "Min Volatility": pd.Series(clean_minvol),
        }).fillna(0)
        weights_df = weights_df[(weights_df.abs().sum(axis=1) > 0.0001)].copy()

        summary = pd.DataFrame([
            {"Strategy": "Max Sharpe", "Expected Return": fmt_pct(ret_s), "Volatility": fmt_pct(vol_s), "Sharpe": fmt_num(sharpe_s, 2)},
            {"Strategy": "Min Volatility", "Expected Return": fmt_pct(ret_m), "Volatility": fmt_pct(vol_m), "Sharpe": fmt_num(sharpe_m, 2)},
        ])

        fig = go.Figure()
        for col in weights_df.columns:
            fig.add_trace(go.Bar(x=weights_df.index, y=weights_df[col]*100, name=col,
                                 hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
        fig.update_layout(barmode="group", yaxis_title="Weight (%)")
        fig = chart_layout(fig, f"{asset_class} | {region} | PyPortfolioOpt Weights", 620)

        return pn.Column(
            pn.pane.HTML(
                f"""
                <div class="qfa-note">
                <b>Optimizer engine:</b> PyPortfolioOpt | Covariance: Ledoit-Wolf shrinkage |
                Risk-free rate: {RISK_FREE_RATE:.2%}.<br>
                The optimizer uses only original matched Yahoo close prices. No synthetic price data.
                </div>
                """,
                sizing_mode="stretch_width",
            ),
            pn.widgets.Tabulator(summary, height=150, sizing_mode="stretch_width"),
            pn.pane.Plotly(fig, config={"responsive": True}),
            pn.widgets.Tabulator((weights_df*100).round(2), height=360, pagination="remote", page_size=15, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    except Exception as e:
        return empty_state("Optimizer failed", f"{type(e).__name__}: {str(e)}")


def build_tearsheet(ticker, benchmark_label, start, end):
    if not QUANTSTATS_AVAILABLE:
        return empty_state("QuantStats unavailable", "QuantStats is not installed. Check requirements.txt and Render build logs.")

    bench = BENCHMARKS[benchmark_label]
    asset = fetch_ohlcv(ticker, start, end)
    benchmark = fetch_ohlcv(bench, start, end)

    if asset.empty or benchmark.empty:
        return empty_state("Tearsheet unavailable", f"Yahoo data missing for {ticker} or {benchmark_label} ({bench}).")

    returns = asset["Close"].pct_change().rename(ticker)
    bench_returns = benchmark["Close"].pct_change().rename(bench)
    matched = pd.concat([returns, bench_returns], axis=1).dropna()

    if len(matched) < MIN_OBS:
        return empty_state("Insufficient matched observations", f"QuantStats requires more matched observations. Current count: {len(matched)}.")

    file_name = f"tearsheet_{ticker.replace('^','IDX').replace('/','_')}_vs_{bench.replace('^','IDX').replace('/','_')}.html"
    output_path = OUTPUT_DIR / file_name

    try:
        qs.reports.html(
            matched[ticker],
            benchmark=matched[bench],
            rf=RISK_FREE_RATE,
            output=str(output_path),
            title=f"{ticker} vs {benchmark_label} | QFA Hedge Fund Tearsheet",
            compounded=True,
        )
        html = output_path.read_text(encoding="utf-8", errors="ignore")

        header = f"""
        <div class="qfa-note">
        <b>Tearsheet generated dynamically on demand.</b><br>
        Instrument: <b>{ticker}</b> | Benchmark: <b>{benchmark_label}</b> ({bench}) |
        RF: <b>{RISK_FREE_RATE:.2%}</b> | Matched observations: <b>{len(matched)}</b><br>
        S&P 500 benchmark uses <b>^GSPC</b>; SPY is not hard-coded as benchmark.
        </div>
        """

        # Render free tier can be memory-sensitive. QuantStats HTML is embedded only after
        # the Tearsheet tab is opened because Tabs are dynamic=True.
        return pn.Column(
            pn.pane.HTML(header, sizing_mode="stretch_width"),
            pn.pane.HTML(html, height=1050, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    except Exception as e:
        return empty_state("QuantStats generation failed", f"{type(e).__name__}: {str(e)}")


def build_stress(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)
    if prices.empty:
        return empty_state("Stress panel unavailable", "No matched Yahoo data for selected universe.")

    returns = prices.pct_change().dropna()
    vol = returns.std() * np.sqrt(TRADING_DAYS)

    rows = []
    for scenario, shock in STRESS_SCENARIOS.items():
        for ticker in returns.columns:
            beta_to_universe = returns[ticker].corr(returns.mean(axis=1))
            beta_to_universe = 1.0 if pd.isna(beta_to_universe) else beta_to_universe
            impact = shock * beta_to_universe
            severity = abs(impact) / max(vol[ticker], 1e-9)
            rows.append({
                "Scenario": scenario,
                "Ticker": ticker,
                "Name": get_name(ticker),
                "Estimated Impact": fmt_pct(impact),
                "Volatility": fmt_pct(vol[ticker]),
                "Severity Score": round(severity, 2),
            })

    table = pd.DataFrame(rows).sort_values("Severity Score", ascending=False)
    return pn.Column(
        pn.pane.HTML(
            """
            <div class="qfa-note">
            Scenario engine is a deterministic sensitivity panel based on matched Yahoo returns.
            It does not create synthetic price history.
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        pn.widgets.Tabulator(table, height=560, pagination="remote", page_size=20, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


# -----------------------------------------------------------------------------
# Batch Report Button
# -----------------------------------------------------------------------------
def generate_selected_report(ticker, benchmark_label, start, end):
    try:
        component = build_tearsheet(ticker, benchmark_label, start, end)
        path = OUTPUT_DIR / f"QFA_Selected_Report_{ticker.replace('^','IDX')}_{BENCHMARKS[benchmark_label].replace('^','IDX')}.html"

        # Create a small durable wrapper report. QuantStats block remains generated separately.
        wrapper = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <title>QFA Selected Report - {ticker}</title>
        {css()}
        </head>
        <body style="background:#f8fafc;padding:22px;font-family:{FONT_STACK};">
        <div class="qfa-header">
            <div class="qfa-title">QFA Selected ETF Report</div>
            <div class="qfa-subtitle">Instrument: {ticker} | Benchmark: {benchmark_label} | RF: {RISK_FREE_RATE:.2%}</div>
        </div>
        <div class="qfa-note">Full interactive dashboard remains in the Panel app. This file records the selected report trigger.</div>
        </body>
        </html>
        """
        path.write_text(wrapper, encoding="utf-8")
        return f"Generated: {path}"
    except Exception as e:
        return f"Report generation failed: {type(e).__name__}: {str(e)}"


# -----------------------------------------------------------------------------
# App Factory: pure pn.widgets + pn.bind
# -----------------------------------------------------------------------------
def make_app():
    default_asset = "Equity ETF"
    default_region = get_regions(default_asset)[0]
    default_ticker = get_tickers(default_asset, default_region)[0]

    asset_class = pn.widgets.Select(name="Asset Class", options=list(UNIVERSE.keys()), value=default_asset)
    region = pn.widgets.Select(name="Region", options=get_regions(default_asset), value=default_region)
    ticker = pn.widgets.Select(name="Instrument", options=get_tickers(default_asset, default_region), value=default_ticker)
    benchmark = pn.widgets.Select(name="Benchmark", options=list(BENCHMARKS.keys()), value="S&P 500 Index")

    start_date = pn.widgets.DatePicker(name="Start Date", value=datetime(2018, 1, 1))
    end_date = pn.widgets.DatePicker(name="End Date", value=datetime.now())

    report_button = pn.widgets.Button(name="Generate Selected ETF HTML Report", button_type="primary")
    report_status = pn.pane.Markdown("")

    def update_regions(event=None):
        regions = get_regions(asset_class.value)
        region.options = regions
        region.value = regions[0] if regions else None

    def update_tickers(event=None):
        tickers = get_tickers(asset_class.value, region.value)
        ticker.options = tickers
        ticker.value = tickers[0] if tickers else None

    asset_class.param.watch(lambda e: (update_regions(e), update_tickers(e)), "value")
    region.param.watch(update_tickers, "value")

    def on_report_click(event):
        report_status.object = "Generating selected report..."
        msg = generate_selected_report(ticker.value, benchmark.value, start_date.value, end_date.value)
        report_status.object = msg

    report_button.on_click(on_report_click)

    sidebar = pn.Column(
        pn.pane.HTML(
            f"""
            <div style="padding:8px 0 12px 0;">
                <div style="font-size:25px;font-weight:850;color:#0f172a;">QFA Hedge Fund</div>
                <div style="font-size:13px;color:#64748b;line-height:1.45;">
                    Live reactive dashboard powered by Yahoo Finance, QuantStats and PyPortfolioOpt.
                </div>
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        pn.pane.Markdown("### Investment Universe"),
        asset_class,
        region,
        ticker,
        pn.pane.Markdown("### Benchmark & Period"),
        benchmark,
        start_date,
        end_date,
        pn.Spacer(height=8),
        report_button,
        report_status,
        pn.pane.HTML(
            f"""
            <div class="qfa-note">
                {status_badge("TA-Lib Active", TALIB_AVAILABLE)}
                {status_badge("QuantStats Active", QUANTSTATS_AVAILABLE)}
                {status_badge("PyPortfolioOpt Active", PYPFOPT_AVAILABLE)}
                <br><br>
                <b>RF:</b> {RISK_FREE_RATE:.2%}<br>
                <b>Data policy:</b> Yahoo-only matched data. No synthetic price fallback.<br>
                <b>S&P 500 benchmark:</b> ^GSPC, not SPY.
            </div>
            """,
            sizing_mode="stretch_width",
        ),
        width=340,
        min_height=760,
        sizing_mode="stretch_height",
        styles={
            "background": "#f8fafc",
            "padding": "20px",
            "border-right": "1px solid #dbe4ef",
            "overflow-y": "auto",
        },
    )

    header = pn.pane.HTML(
        f"""
        <div class="qfa-header">
            <div class="qfa-title">{APP_TITLE}</div>
            <div class="qfa-subtitle">
                Hedge-fund-grade analytics: live KPI scorecard, TA-Lib technicals, VaR/CVaR,
                benchmark-relative tracking error, QuantStats Tearsheet, PyPortfolioOpt construction,
                universe ranking and stress sensitivity.
            </div>
        </div>
        """,
        sizing_mode="stretch_width",
    )

    bound_kpi = pn.bind(build_kpi, ticker, benchmark, start_date, end_date)
    bound_price = pn.bind(build_price_chart, ticker, start_date, end_date)
    bound_risk = pn.bind(build_risk_chart, ticker, start_date, end_date)
    bound_relative = pn.bind(build_benchmark_relative, ticker, benchmark, start_date, end_date)
    bound_universe = pn.bind(build_universe_board, asset_class, region, start_date, end_date)
    bound_optimizer = pn.bind(build_optimizer, asset_class, region, start_date, end_date)
    bound_tearsheet = pn.bind(build_tearsheet, ticker, benchmark, start_date, end_date)
    bound_stress = pn.bind(build_stress, asset_class, region, start_date, end_date)

    tabs = pn.Tabs(
        ("Executive KPI Dashboard", bound_kpi),
        ("Price & TA-Lib", bound_price),
        ("Risk Metrics", bound_risk),
        ("Benchmark Relative", bound_relative),
        ("Investment Universe", bound_universe),
        ("Portfolio Optimizer", bound_optimizer),
        ("Stress Testing", bound_stress),
        ("Tearsheet", bound_tearsheet),
        dynamic=True,
        sizing_mode="stretch_width",
    )

    main = pn.Column(
        pn.pane.HTML(css(), sizing_mode="stretch_width"),
        header,
        tabs,
        sizing_mode="stretch_width",
        styles={"padding": "18px", "background": "#ffffff"},
    )

    return pn.Row(sidebar, main, sizing_mode="stretch_width")


app = make_app()
app.servable(title=APP_TITLE)
