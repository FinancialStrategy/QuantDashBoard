# =============================================================================
# QFA Dashboard FinTECH - Render Rebuilt V2
# Panel + Plotly + Yahoo Finance + QuantStats + PyPortfolioOpt
# Reactive architecture: pn.widgets + pn.bind, no synthetic price fallback
# =============================================================================

import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

# Optional dependencies ---------------------------------------------------------
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    qs = None
    QUANTSTATS_AVAILABLE = False

try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

# Constants --------------------------------------------------------------------
APP_TITLE = "QFA Dashboard FinTECH"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TRADING_DAYS = 252
RISK_FREE_RATE = 0.045  # Fixed 4.5%; later can be replaced with 13W T-Bill

# S&P 500 benchmark must NOT be SPY in Tearsheet. Use index ticker ^GSPC.
BENCHMARKS = {
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Nasdaq 100 Index (^NDX)": "^NDX",
    "Russell 2000 Index (^RUT)": "^RUT",
    "US Aggregate Bond ETF (BND)": "BND",
    "Gold ETF (GLD)": "GLD",
    "Emerging Markets ETF (EEM)": "EEM",
}

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
        }
    },
    "Fixed Income": {
        "United States": {
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "HYG": "iShares High Yield Corporate Bond ETF",
            "LQD": "iShares Investment Grade Corporate Bond ETF",
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
        }
    },
}

# Helpers ----------------------------------------------------------------------
def regions_for(asset_class: str):
    return list(UNIVERSE[asset_class].keys())

def tickers_for(asset_class: str, region: str):
    return list(UNIVERSE[asset_class][region].keys())

def name_for(ticker: str) -> str:
    for regions in UNIVERSE.values():
        for instruments in regions.values():
            if ticker in instruments:
                return instruments[ticker]
    return ticker

def pct(x, d=2):
    return "N/A" if pd.isna(x) else f"{100*x:.{d}f}%"

def num(x, d=2):
    return "N/A" if pd.isna(x) else f"{x:.{d}f}"

@pn.cache(ttl=900, max_items=128)
def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Yahoo Finance only. No synthetic price fallback."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            return pd.DataFrame()
        for c in ["Open", "High", "Low"]:
            if c not in df.columns:
                return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception:
        return pd.DataFrame()

def fetch_price_matrix(tickers, start, end):
    data = {}
    rejected = []
    for t in tickers:
        df = fetch_ohlcv(t, start, end)
        if df.empty:
            rejected.append(t)
        else:
            data[t] = df["Close"]
    if not data:
        return pd.DataFrame(), rejected
    px = pd.DataFrame(data).sort_index().ffill(limit=3).dropna(axis=1, thresh=max(30, int(0.70 * len(pd.DataFrame(data)))))
    return px, rejected

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Cumulative Return"] = (1 + d["Return"].fillna(0)).cumprod() - 1
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["Vol21"] = d["Return"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    d["Vol63"] = d["Return"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + d["Return"].fillna(0)).cumprod()
    d["Drawdown"] = wealth / wealth.cummax() - 1
    if TALIB_AVAILABLE:
        close, high, low = d["Close"].values.astype(float), d["High"].values.astype(float), d["Low"].values.astype(float)
        d["RSI"] = talib.RSI(close, 14)
        macd, sig, hist = talib.MACD(close, 12, 26, 9)
        d["MACD"], d["MACD Signal"], d["MACD Hist"] = macd, sig, hist
        up, mid, lowb = talib.BBANDS(close, 20, 2, 2)
        d["BB Upper"], d["BB Mid"], d["BB Lower"] = up, mid, lowb
        d["ATR"] = talib.ATR(high, low, close, 14)
    else:
        delta = d["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["RSI"] = 100 - 100 / (1 + rs)
        ema12 = d["Close"].ewm(span=12, adjust=False).mean()
        ema26 = d["Close"].ewm(span=26, adjust=False).mean()
        d["MACD"] = ema12 - ema26
        d["MACD Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
        d["MACD Hist"] = d["MACD"] - d["MACD Signal"]
        d["BB Mid"] = d["Close"].rolling(20).mean()
        std = d["Close"].rolling(20).std()
        d["BB Upper"] = d["BB Mid"] + 2 * std
        d["BB Lower"] = d["BB Mid"] - 2 * std
        tr = pd.concat([(d["High"]-d["Low"]), (d["High"]-d["Close"].shift()).abs(), (d["Low"]-d["Close"].shift()).abs()], axis=1).max(axis=1)
        d["ATR"] = tr.rolling(14).mean()
    return d

def risk_metrics(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 30:
        return {k: np.nan for k in ["Ann Return","Ann Vol","Sharpe","Sortino","Max Drawdown","VaR95","CVaR95","Win Rate"]}
    ann_ret = (1 + r.mean()) ** TRADING_DAYS - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    var = r.quantile(0.05)
    return {
        "Ann Return": ann_ret,
        "Ann Vol": ann_vol,
        "Sharpe": (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol else np.nan,
        "Sortino": (ann_ret - RISK_FREE_RATE) / downside if downside else np.nan,
        "Max Drawdown": dd.min(),
        "VaR95": var,
        "CVaR95": r[r <= var].mean(),
        "Win Rate": (r > 0).mean(),
    }

def layout(fig, title, height=760):
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.01), height=height,
                      margin=dict(l=50, r=30, t=80, b=50), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(210,220,230,.5)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(210,220,230,.5)")
    return fig

def no_data(msg):
    return pn.pane.HTML(f"<div style='padding:20px;border:1px solid #d8e1ec;border-radius:12px;background:#fff7ed'><b>{msg}</b></div>")

# Components -------------------------------------------------------------------
def kpi_view(ticker, benchmark_label, start, end, asset_class, region):
    df = add_indicators(fetch_ohlcv(ticker, str(start), str(end)))
    if df.empty:
        return no_data(f"No Yahoo Finance data found for {ticker}.")
    m = risk_metrics(df["Return"])
    bmk = BENCHMARKS[benchmark_label]
    cards = [
        ("Instrument", f"{ticker}<br><small>{name_for(ticker)}</small>", "#f8fafc"),
        ("Class / Region", f"{asset_class}<br><small>{region}</small>", "#f8fafc"),
        ("Benchmark", f"{bmk}<br><small>{benchmark_label}</small>", "#f8fafc"),
        ("RF", "4.50%", "#f8fafc"),
        ("Annual Return", pct(m["Ann Return"]), "#ecfdf5" if m["Ann Return"] > 0 else "#fef2f2"),
        ("Annual Vol", pct(m["Ann Vol"]), "#f8fafc"),
        ("Sharpe", num(m["Sharpe"]), "#ecfdf5" if m["Sharpe"] > 1 else "#fff7ed"),
        ("Sortino", num(m["Sortino"]), "#ecfdf5" if m["Sortino"] > 1 else "#fff7ed"),
        ("Max Drawdown", pct(m["Max Drawdown"]), "#fef2f2"),
        ("VaR 95", pct(m["VaR95"]), "#fef2f2"),
        ("CVaR 95", pct(m["CVaR95"]), "#fef2f2"),
        ("Win Rate", pct(m["Win Rate"]), "#ecfdf5" if m["Win Rate"] > .5 else "#fff7ed"),
    ]
    html = "<div style='display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px'>"
    for title, value, bg in cards:
        html += f"<div style='background:{bg};border:1px solid #d8e1ec;border-radius:14px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.04)'><div style='font-size:12px;color:#64748b;font-weight:700'>{title}</div><div style='font-size:20px;font-weight:800;margin-top:6px;color:#0f172a'>{value}</div></div>"
    html += "</div>"
    return pn.pane.HTML(html)

def price_tab(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, str(start), str(end)))
    if df.empty:
        return no_data(f"No Yahoo data for {ticker}.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.055, row_heights=[.55,.22,.23], subplot_titles=["OHLC + Bollinger + Moving Averages", "RSI", "MACD"])
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="OHLC", increasing_line_color="#15803d", decreasing_line_color="#b91c1c", hoverinfo="x+y+name"), row=1, col=1)
    for c, n, dash in [("BB Upper","BB Upper","dash"),("BB Mid","BB Mid","dot"),("BB Lower","BB Lower","dash"),("MA50","MA50","solid"),("MA200","MA200","solid")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=n, mode="lines", line=dict(dash=dash), hovertemplate="%{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", mode="lines"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="gray"); fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD Signal"], name="Signal", mode="lines"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD Hist"], name="Hist", opacity=.45), row=3, col=1)
    return pn.pane.Plotly(layout(fig, f"{ticker} | Price & TA-Lib Technicals", 850))

def risk_tab(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, str(start), str(end)))
    if df.empty:
        return no_data(f"No Yahoo data for {ticker}.")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.055, subplot_titles=["Cumulative Return", "Rolling Volatility", "Drawdown", "Daily Return Distribution"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Cumulative Return"]*100, name="Cumulative Return %", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.Vol21*100, name="Vol 21D %", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.Vol63*100, name="Vol 63D %", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.Drawdown*100, name="Drawdown %", mode="lines", fill="tozeroy"), row=3, col=1)
    fig.add_trace(go.Histogram(x=df.Return.dropna()*100, name="Daily Returns %", nbinsx=70), row=4, col=1)
    return pn.pane.Plotly(layout(fig, f"{ticker} | Institutional Risk Metrics", 900))

def benchmark_tab(ticker, benchmark_label, start, end):
    bmk = BENCHMARKS[benchmark_label]
    a = fetch_ohlcv(ticker, str(start), str(end))
    b = fetch_ohlcv(bmk, str(start), str(end))
    if a.empty or b.empty:
        return no_data(f"Missing Yahoo data for {ticker} or benchmark {bmk}.")
    px = pd.concat([a.Close.rename(ticker), b.Close.rename(bmk)], axis=1).dropna()
    if px.empty:
        return no_data("No overlapping dates between selected instrument and benchmark.")
    r = px.pct_change().dropna(); cum = (1+r).cumprod()-1; active = r[ticker] - r[bmk]
    ir = (active.rolling(63).mean()*TRADING_DAYS)/(active.rolling(63).std()*np.sqrt(TRADING_DAYS))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=["Cumulative Return Comparison", "Active Return", "Rolling Information Ratio 63D"])
    fig.add_trace(go.Scatter(x=cum.index, y=cum[ticker]*100, name=ticker, mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum.index, y=cum[bmk]*100, name=bmk, mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=active.index, y=((1+active).cumprod()-1)*100, name="Active Return", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ir.index, y=ir, name="Rolling IR", mode="lines"), row=3, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray"); fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray")
    return pn.pane.Plotly(layout(fig, f"{ticker} vs {bmk} | Benchmark Relative Analytics", 850))

def universe_tab(asset_class, region, start, end):
    tickers = tickers_for(asset_class, region)
    px, rejected = fetch_price_matrix(tickers, str(start), str(end))
    if px.empty:
        return no_data("No matched Yahoo Finance data for selected universe.")
    r = px.pct_change().dropna()
    rows = []
    for t in r.columns:
        m = risk_metrics(r[t])
        rows.append({"Ticker": t, "Name": name_for(t), "Ann Return": m["Ann Return"], "Ann Vol": m["Ann Vol"], "Sharpe": m["Sharpe"], "Max DD": m["Max Drawdown"]})
    board = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=.14, subplot_titles=["Universe Sharpe Ranking", "Risk / Return Map"])
    fig.add_trace(go.Bar(x=board.Ticker, y=board.Sharpe, name="Sharpe"), row=1, col=1)
    fig.add_trace(go.Scatter(x=board["Ann Vol"]*100, y=board["Ann Return"]*100, text=board.Ticker, mode="markers+text", textposition="top center", name="Risk/Return"), row=2, col=1)
    pane = pn.pane.Plotly(layout(fig, f"{asset_class} | {region} | Universe Board", 820))
    if rejected:
        return pn.Column(pn.pane.Markdown(f"**Yahoo rejected / no matched data:** {', '.join(rejected)}"), pane)
    return pane

def optimizer_tab(asset_class, region, start, end):
    if not PYPFOPT_AVAILABLE:
        return no_data("PyPortfolioOpt is not installed or failed to import. Check requirements build logs.")
    px, rejected = fetch_price_matrix(tickers_for(asset_class, region), str(start), str(end))
    if px.shape[1] < 2:
        return no_data("Optimizer requires at least two assets with matched Yahoo data.")
    try:
        mu = expected_returns.mean_historical_return(px, frequency=TRADING_DAYS)
        S = risk_models.CovarianceShrinkage(px).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        w_sharpe = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        perf_sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        ef2 = EfficientFrontier(mu, S)
        w_minvol = ef2.min_volatility()
        perf_minvol = ef2.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        rows = []
        for label, weights, perf in [("Max Sharpe", w_sharpe, perf_sharpe), ("Min Vol", w_minvol, perf_minvol)]:
            for k, v in weights.items():
                if abs(v) > 1e-4:
                    rows.append({"Strategy": label, "Ticker": k, "Weight": f"{100*v:.2f}%", "Exp Return": pct(perf[0]), "Vol": pct(perf[1]), "Sharpe": num(perf[2])})
        return pn.widgets.Tabulator(pd.DataFrame(rows), height=430, pagination="remote", page_size=20)
    except Exception as e:
        return no_data(f"PyPortfolioOpt failed: {e}")

def tearsheet_tab(ticker, benchmark_label, start, end):
    if not QUANTSTATS_AVAILABLE:
        return no_data("QuantStats is not available. Check requirements build logs.")
    bmk = BENCHMARKS[benchmark_label]
    a = fetch_ohlcv(ticker, str(start), str(end))
    b = fetch_ohlcv(bmk, str(start), str(end))
    if a.empty or b.empty:
        return no_data(f"Missing Yahoo data for {ticker} or benchmark {bmk}. Tearsheet was not generated.")
    returns = a.Close.pct_change().rename("asset")
    bench = b.Close.pct_change().rename("benchmark")
    joined = pd.concat([returns, bench], axis=1).dropna()
    if joined.empty:
        return no_data("No overlapping return dates for Tearsheet.")
    safe_t = ticker.replace("^", "IDX_").replace("/", "_")
    safe_b = bmk.replace("^", "IDX_").replace("/", "_")
    out = OUTPUT_DIR / f"QFA_Tearsheet_{safe_t}_vs_{safe_b}.html"
    try:
        qs.reports.html(joined.asset, benchmark=joined.benchmark, rf=RISK_FREE_RATE, output=str(out), title=f"{ticker} vs {bmk} | QFA Tearsheet")
        html = out.read_text(encoding="utf-8", errors="ignore")
        return pn.Column(pn.pane.Markdown(f"### Tearsheet: {ticker} vs {bmk} | RF 4.5%"), pn.pane.HTML(html, height=900, sizing_mode="stretch_width"))
    except Exception as e:
        return no_data(f"QuantStats Tearsheet failed: {e}")

# App --------------------------------------------------------------------------
asset_select = pn.widgets.Select(name="Instrument Class", options=list(UNIVERSE.keys()), value="Equity ETF")
region_select = pn.widgets.Select(name="Region", options=regions_for(asset_select.value), value=regions_for(asset_select.value)[0])
ticker_select = pn.widgets.Select(name="Instrument", options=tickers_for(asset_select.value, region_select.value), value=tickers_for(asset_select.value, region_select.value)[0])
benchmark_select = pn.widgets.Select(name="Benchmark", options=list(BENCHMARKS.keys()), value="S&P 500 Index (^GSPC)")
start_picker = pn.widgets.DatePicker(name="Start Date", value=datetime(2018,1,1))
end_picker = pn.widgets.DatePicker(name="End Date", value=datetime.now())


def _asset_changed(event):
    regs = regions_for(event.new)
    region_select.options = regs
    region_select.value = regs[0]
    ticks = tickers_for(event.new, region_select.value)
    ticker_select.options = ticks
    ticker_select.value = ticks[0]

def _region_changed(event):
    ticks = tickers_for(asset_select.value, event.new)
    ticker_select.options = ticks
    ticker_select.value = ticks[0]

asset_select.param.watch(_asset_changed, "value")
region_select.param.watch(_region_changed, "value")

sidebar = pn.Column(
    pn.pane.HTML("""
    <div style='padding:10px 0 16px 0'>
      <div style='font-size:26px;font-weight:900;color:#0f172a'>QFA Prime</div>
      <div style='font-size:13px;color:#64748b;font-weight:600'>FinTECH Institutional Platform</div>
    </div>
    """),
    asset_select, region_select, ticker_select, benchmark_select, start_picker, end_picker,
    pn.pane.Markdown(f"""
---
**RF:** 4.5% fixed  
**Data:** Yahoo Finance only  
**TA-Lib:** {'Active' if TALIB_AVAILABLE else 'Formula fallback only'}  
**QuantStats:** {'Active' if QUANTSTATS_AVAILABLE else 'Missing'}  
**PyPortfolioOpt:** {'Active' if PYPFOPT_AVAILABLE else 'Missing'}
"""),
    width=340,
    sizing_mode="fixed",
    styles={"background":"#f4f7fb","padding":"20px","border-right":"1px solid #d8e1ec","height":"100vh","overflow-y":"auto"},
)

header = pn.pane.HTML("""
<div style='background:linear-gradient(90deg,#0f172a,#1e293b);padding:24px 28px;border-radius:16px;color:white;margin-bottom:16px;box-shadow:0 4px 14px rgba(0,0,0,.14)'>
  <div style='font-size:30px;font-weight:900'>QFA Dashboard FinTECH</div>
  <div style='font-size:14px;color:#cbd5e1;margin-top:7px'>Reactive Render-ready analytics: KPI, TA-Lib technicals, risk metrics, benchmark relative analytics, PyPortfolioOpt and QuantStats Tearsheet.</div>
</div>
""")

bound_args = dict(ticker=ticker_select, benchmark_label=benchmark_select, start=start_picker, end=end_picker, asset_class=asset_select, region=region_select)

tabs = pn.Tabs(
    ("Executive KPI Dashboard", pn.bind(kpi_view, **bound_args)),
    ("Price & TA-Lib Technicals", pn.bind(price_tab, ticker_select, start_picker, end_picker)),
    ("Institutional Risk Metrics", pn.bind(risk_tab, ticker_select, start_picker, end_picker)),
    ("Benchmark Relative Analytics", pn.bind(benchmark_tab, ticker_select, benchmark_select, start_picker, end_picker)),
    ("Investment Universe Board", pn.bind(universe_tab, asset_select, region_select, start_picker, end_picker)),
    ("PyPortfolioOpt Optimizer", pn.bind(optimizer_tab, asset_select, region_select, start_picker, end_picker)),
    ("Tearsheet", pn.bind(tearsheet_tab, ticker_select, benchmark_select, start_picker, end_picker)),
    dynamic=False,
    sizing_mode="stretch_width",
)

main = pn.Column(header, tabs, sizing_mode="stretch_width", styles={"padding":"18px","background":"#fff"})
app = pn.Row(sidebar, main, sizing_mode="stretch_width")
app.servable(title=APP_TITLE)
