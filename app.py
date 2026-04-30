# =============================================================================
# QFA Hedge Fund Dashboard – PRODUCTION LEVEL v10 Render Memory Profiled Edition
# Fully reactive: pn.widgets + pn.bind, Yahoo-only data, ^GSPC benchmark, RF 4.5%
# Advanced Risk Analytics (Historical / Parametric / Monte Carlo VaR/CVaR,
# Rolling Beta, VaR/Nav ratio, Historical Stress Testing, Backtesting)
# Expanded Portfolio Optimizer: all core PyPortfolioOpt strategies explained
# Ultra memory-optimised for Render free tier (512 MB): lazy tabs, no startup tearsheet/optimizer
# =============================================================================
import os, io, time, math, json, warnings, traceback, gc
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf

import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matplotlib is kept in Agg mode only for optional report generation.
# Do NOT force-build the font cache on Render Free; it causes a RAM spike.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.rcParams["figure.dpi"] = 72

# Heavy libraries are not imported at startup on Render Free.
qs = None
QUANTSTATS_AVAILABLE = False

# Heavy optional libraries are lazy-loaded only when their tab/function is used.
# This keeps Render Free 512 MB stable while preserving TA-Lib and PyPortfolioOpt features.
talib = None
TALIB_AVAILABLE = False

def load_talib() -> bool:
    """Render-safe TA-Lib loader. Imported only when indicators are computed."""
    global talib, TALIB_AVAILABLE
    if talib is not None:
        return True
    try:
        import talib as _talib
        talib = _talib
        TALIB_AVAILABLE = True
    except Exception:
        talib = None
        TALIB_AVAILABLE = False
    return TALIB_AVAILABLE

expected_returns = None
risk_models = None
EfficientFrontier = None
HRPOpt = None
PYPFOPT_AVAILABLE = False

def load_pypfopt() -> bool:
    """Render-safe PyPortfolioOpt loader. Imported only when Portfolio Optimizer is opened."""
    global expected_returns, risk_models, EfficientFrontier, HRPOpt, PYPFOPT_AVAILABLE
    if EfficientFrontier is not None and expected_returns is not None and risk_models is not None:
        PYPFOPT_AVAILABLE = True
        return True
    try:
        from pypfopt import expected_returns as _expected_returns, risk_models as _risk_models
        from pypfopt.efficient_frontier import EfficientFrontier as _EfficientFrontier
        from pypfopt.hierarchical_portfolio import HRPOpt as _HRPOpt
        expected_returns = _expected_returns
        risk_models = _risk_models
        EfficientFrontier = _EfficientFrontier
        HRPOpt = _HRPOpt
        PYPFOPT_AVAILABLE = True
    except Exception:
        expected_returns = None
        risk_models = None
        EfficientFrontier = None
        HRPOpt = None
        PYPFOPT_AVAILABLE = False
    return PYPFOPT_AVAILABLE

# -----------------------------------------------------------------------------
# Panel Extension
# -----------------------------------------------------------------------------
pn.extension("plotly", "tabulator", sizing_mode="stretch_width", notifications=True)

# -----------------------------------------------------------------------------
# Global Institutional Configuration
# -----------------------------------------------------------------------------
APP_TITLE = "QFA Hedge Fund Dashboard - PRODUCTION v10"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045          # 4.5% USD
MIN_OBS = 90
CACHE_TTL_SECONDS = 600
MAX_RENDER_UNIVERSE_TICKERS = int(os.getenv("QFA_MAX_RENDER_TICKERS", "10"))

FONT_STACK = "Inter, DejaVu Sans, Segoe UI, Helvetica, Arial, sans-serif"

def memory_mb() -> float:
    """Best-effort RSS memory usage in MB. Works without psutil."""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    except Exception:
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(rss / 1024)
        except Exception:
            return float("nan")

def cleanup_memory() -> None:
    try:
        plt.close("all")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def memory_note(label: str = "Memory"):
    mb = memory_mb()
    txt = "N/A" if (isinstance(mb, float) and math.isnan(mb)) else f"{mb:.0f} MB"
    return pn.pane.HTML(f'<div class="qfa-note"><b>{label}:</b> current worker RSS ≈ {txt}. Render command must use <code>--num-procs 1</code>.</div>', sizing_mode="stretch_width")

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
    "S&P 500 Index": "^GSPC",
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
                rows.append({"Asset Class": asset_class, "Region": region, "Ticker": ticker, "Name": name})
    return pd.DataFrame(rows)

UNIVERSE_DF = flatten_universe()

def get_regions(asset_class: str):
    return list(UNIVERSE.get(asset_class, {}).keys())

def get_tickers(asset_class: str, region: str):
    return list(UNIVERSE.get(asset_class, {}).get(region, {}).keys())

def get_name(ticker: str) -> str:
    row = UNIVERSE_DF.loc[UNIVERSE_DF["Ticker"] == ticker]
    return str(row.iloc[0]["Name"]) if not row.empty else ticker

def normalize_date(x):
    if x is None: return None
    if isinstance(x, datetime): return x.strftime("%Y-%m-%d")
    if hasattr(x, "strftime"): return x.strftime("%Y-%m-%d")
    return str(x)[:10]

def fmt_pct(x, digits=2):
    try:
        if x is None or pd.isna(x) or np.isinf(x): return "N/A"
        return f"{x*100:.{digits}f}%"
    except: return "N/A"

def fmt_num(x, digits=2):
    try:
        if x is None or pd.isna(x) or np.isinf(x): return "N/A"
        return f"{x:.{digits}f}"
    except: return "N/A"

def status_badge(label: str, ok: bool):
    color = "#166534" if ok else "#991b1b"
    bg = "#dcfce7" if ok else "#fee2e2"
    return f"""<span style="background:{bg};color:{color};padding:4px 9px;border-radius:999px;font-size:12px;font-weight:700;">{label}</span>"""

# -----------------------------------------------------------------------------
# Yahoo‑only Data Layer
# -----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def fetch_ohlcv_cached(ticker: str, start: str, end: str, cache_bucket: int) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=False, threads=False, timeout=15)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)
        required = ["Open","High","Low","Close","Volume"]
        if "Close" not in df.columns: return pd.DataFrame()
        for col in required:
            if col not in df.columns:
                df[col] = df["Close"] if col != "Volume" else np.nan
        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["Close"])
        return df if len(df) >= 2 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def cache_bucket() -> int:
    return int(time.time() // CACHE_TTL_SECONDS)

def fetch_ohlcv(ticker: str, start, end) -> pd.DataFrame:
    start_s = normalize_date(start); end_s = normalize_date(end)
    return fetch_ohlcv_cached(ticker, start_s, end_s, cache_bucket()).copy()

def fetch_price_matrix(tickers, start, end) -> pd.DataFrame:
    tickers = list(dict.fromkeys(list(tickers)))[:MAX_RENDER_UNIVERSE_TICKERS]
    series = {}
    for t in tickers:
        df = fetch_ohlcv(t, start, end)
        if not df.empty: series[t] = df["Close"]
    if not series: return pd.DataFrame()
    prices = pd.DataFrame(series).sort_index().ffill(limit=3)
    min_count = max(MIN_OBS, int(len(prices)*0.70))
    return prices.dropna(axis=1, thresh=min_count).dropna()

# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
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

    if load_talib():
        try:
            close = d["Close"].astype(float).values
            high = d["High"].astype(float).values
            low = d["Low"].astype(float).values
            d["RSI"] = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            d["MACD"], d["MACD Signal"], d["MACD Hist"] = macd, signal, hist
            upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            d["BB Upper"], d["BB Mid"], d["BB Lower"] = upper, mid, lower
            d["ATR"] = talib.ATR(high, low, close, timeperiod=14)
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            d["Stoch K"], d["Stoch D"] = slowk, slowd
            d["Indicator Engine"] = "TA-Lib Lazy"
            return d
        except: pass

    # Fallback indicators
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100/(1+rs))
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD Hist"] = d["MACD"] - d["MACD Signal"]
    d["BB Mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB Upper"] = d["BB Mid"] + 2*bb_std
    d["BB Lower"] = d["BB Mid"] - 2*bb_std
    tr = pd.concat([d["High"]-d["Low"], (d["High"]-d["Close"].shift()).abs(), (d["Low"]-d["Close"].shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch K"] = 100*(d["Close"]-low14)/(high14-low14)
    d["Stoch D"] = d["Stoch K"].rolling(3).mean()
    d["Indicator Engine"] = "Formula fallback"
    return d

# -----------------------------------------------------------------------------
# Risk Engine
# -----------------------------------------------------------------------------
def risk_metrics(returns: pd.Series, rf=RISK_FREE_RATE) -> dict:
    r = pd.Series(returns).replace([np.inf,-np.inf], np.nan).dropna()
    if len(r) < 30: return {k: np.nan for k in ["Ann Return","Ann Vol","Sharpe","Sortino","Max Drawdown","VaR 95","CVaR 95","VaR 99","CVaR 99","Skew","Kurtosis","Win Rate","Calmar","Omega"]}
    ann_ret = (1+r.mean())**TRADING_DAYS - 1
    ann_vol = r.std()*np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - rf)/ann_vol if ann_vol else np.nan
    downside = r[r<0].std()*np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - rf)/downside if downside else np.nan
    wealth = (1+r).cumprod()
    dd = wealth/wealth.cummax() - 1
    max_dd = dd.min()
    var95 = r.quantile(0.05); cvar95 = r[r<=var95].mean()
    var99 = r.quantile(0.01); cvar99 = r[r<=var99].mean()
    calmar = ann_ret/abs(max_dd) if max_dd else np.nan
    gains = r[r>0].sum(); losses = abs(r[r<0].sum())
    omega = gains/losses if losses else np.nan
    return {
        "Ann Return": ann_ret, "Ann Vol": ann_vol, "Sharpe": sharpe,
        "Sortino": sortino, "Max Drawdown": max_dd,
        "VaR 95": var95, "CVaR 95": cvar95, "VaR 99": var99, "CVaR 99": cvar99,
        "Skew": r.skew(), "Kurtosis": r.kurtosis(), "Win Rate": (r>0).mean(),
        "Calmar": calmar, "Omega": omega,
    }

def active_metrics(asset_returns: pd.Series, bench_returns: pd.Series) -> dict:
    joined = pd.concat([asset_returns.rename("asset"), bench_returns.rename("bench")], axis=1).dropna()
    if len(joined) < 30: return {"Tracking Error":np.nan,"Information Ratio":np.nan,"Beta":np.nan,"Alpha":np.nan,"Correlation":np.nan}
    active = joined["asset"] - joined["bench"]
    te = active.std()*np.sqrt(TRADING_DAYS)
    ir = (active.mean()*TRADING_DAYS)/te if te else np.nan
    cov = np.cov(joined["asset"], joined["bench"])[0,1]
    var = np.var(joined["bench"])
    beta = cov/var if var else np.nan
    alpha = (joined["asset"].mean()*TRADING_DAYS) - beta*(joined["bench"].mean()*TRADING_DAYS) if pd.notna(beta) else np.nan
    corr = joined["asset"].corr(joined["bench"])
    return {"Tracking Error":te,"Information Ratio":ir,"Beta":beta,"Alpha":alpha,"Correlation":corr}

# -----------------------------------------------------------------------------
# Advanced VaR / CVaR Methods
# -----------------------------------------------------------------------------
def historical_var(returns: pd.Series, conf: float): return returns.quantile(1-conf)
def historical_cvar(returns: pd.Series, conf: float):
    v = historical_var(returns, conf); return returns[returns<=v].mean()
def parametric_var(returns: pd.Series, conf: float):
    from scipy import stats
    return returns.mean() + returns.std()*stats.norm.ppf(1-conf)
def parametric_cvar(returns: pd.Series, conf: float):
    from scipy import stats
    mu = returns.mean(); sigma = returns.std()
    return mu - sigma*stats.norm.pdf(stats.norm.ppf(1-conf))/(1-conf)
def monte_carlo_var(returns: pd.Series, conf: float, n_sim=1500, dist='normal'):
    from scipy import stats
    np.random.seed(42)
    if len(returns) < 30: return np.nan
    if dist == 'normal':
        mu, sigma = returns.mean(), returns.std()
        sim = np.random.normal(mu, sigma, n_sim)
    else:
        params = stats.t.fit(returns.dropna())
        sim = stats.t.rvs(df=params[0], loc=params[1], scale=params[2], size=n_sim)
    return np.percentile(sim, 100*(1-conf))
def monte_carlo_cvar(returns: pd.Series, conf: float, n_sim=1500, dist='normal'):
    from scipy import stats
    np.random.seed(42)
    if len(returns) < 30: return np.nan
    if dist == 'normal':
        mu, sigma = returns.mean(), returns.std()
        sim = np.random.normal(mu, sigma, n_sim)
    else:
        params = stats.t.fit(returns.dropna())
        sim = stats.t.rvs(df=params[0], loc=params[1], scale=params[2], size=n_sim)
    v = np.percentile(sim, 100*(1-conf))
    return sim[sim<=v].mean()

def compute_rolling_var(returns, window, conf, method):
    func = {'historical':historical_var, 'parametric':parametric_var, 'montecarlo':monte_carlo_var}[method]
    return returns.rolling(window, min_periods=max(window//2,63)).apply(lambda x: func(x, conf), raw=False)
def compute_rolling_cvar(returns, window, conf, method):
    func = {'historical':historical_cvar, 'parametric':parametric_cvar, 'montecarlo':monte_carlo_cvar}[method]
    return returns.rolling(window, min_periods=max(window//2,63)).apply(lambda x: func(x, conf), raw=False)

def compute_var_nav_ratio(price, returns, window_var, window_nav, conf, method):
    roll_var = compute_rolling_var(returns, window_var, conf, method).shift(1)
    dollar_var = price * abs(roll_var)
    nav_avg = price.rolling(window_nav, min_periods=max(window_nav//2,21)).mean()
    return (dollar_var / nav_avg)*100

def rolling_beta(asset_rets, bench_rets, window):
    joined = pd.concat([asset_rets.rename('a'), bench_rets.rename('b')], axis=1).dropna()
    cov = joined.rolling(window).cov().unstack()['a']['b']
    var = joined['b'].rolling(window).var()
    return cov / var

def find_gspc_drawdown_periods(start, end, threshold=-0.20):
    df = fetch_ohlcv("^GSPC", start, end)
    if df.empty: return pd.DataFrame()
    wealth = df["Close"]/df["Close"].iloc[0]
    dd = wealth/wealth.cummax() - 1
    is_stressed = dd < threshold
    groups = (is_stressed != is_stressed.shift()).cumsum()[is_stressed]
    periods = []
    for _, group in dd[is_stressed].groupby(groups):
        start_d = group.index.min(); end_d = group.index.max()
        min_dd = group.min()
        periods.append({"Start":start_d, "End":end_d, "Max DD":min_dd})
    return pd.DataFrame(periods).sort_values("Max DD")

def build_historical_stress_table(ticker, start, end, threshold=-0.20):
    spx = find_gspc_drawdown_periods(start, end, threshold)
    if spx.empty: return pd.DataFrame()
    asset = fetch_ohlcv(ticker, start, end)
    if asset.empty: return pd.DataFrame()
    aret = asset["Close"].pct_change()
    rows = []
    for _, row in spx.iterrows():
        mask = (aret.index >= row["Start"]) & (aret.index <= row["End"])
        if mask.sum() > 1:
            cum = (1+aret[mask]).prod() - 1
            rows.append({
                "Start": row["Start"].strftime("%Y-%m-%d"),
                "End": row["End"].strftime("%Y-%m-%d"),
                "GSPC Max DD": fmt_pct(row["Max DD"]),
                f"{ticker} Cumulative Return": fmt_pct(cum),
                "Days": mask.sum()
            })
    return pd.DataFrame(rows).sort_values("Start") if rows else pd.DataFrame()

# -----------------------------------------------------------------------------
# Styling & Layout
# -----------------------------------------------------------------------------
def css():
    return f"""
    <style>
    body, .bk-root, .bk, .bk-input, .bk-btn {{ font-family: {FONT_STACK}; }}
    .qfa-header {{ background: linear-gradient(90deg, #0f172a, #172554, #1e293b); color: white; padding: 24px 28px; border-radius: 18px; box-shadow: 0 8px 30px rgba(15,23,42,.18); margin-bottom: 16px; }}
    .qfa-title {{ font-size: 30px; font-weight: 850; letter-spacing: -0.03em; }}
    .qfa-subtitle {{ color: #cbd5e1; font-size: 14px; margin-top: 7px; line-height: 1.5; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .kpi-card {{ background: #ffffff; border: 1px solid #dbe4ef; border-radius: 16px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(15,23,42,.055); }}
    .kpi-card.pos {{ background: #f0fdf4; border-color: #86efac; }}
    .kpi-card.neg {{ background: #fff1f2; border-color: #fca5a5; }}
    .kpi-card.warn {{ background: #fffbeb; border-color: #fcd34d; }}
    .kpi-label {{ color: #64748b; font-size: 12px; font-weight: 750; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi-value {{ color: #0f172a; font-size: 22px; font-weight: 850; margin-top: 6px; white-space: nowrap; }}
    .qfa-note {{ background: #f8fafc; border: 1px solid #dbe4ef; border-radius: 14px; padding: 12px 14px; color: #334155; font-size: 13px; line-height: 1.45; }}
    </style>"""

def make_kpi_cards(ticker, benchmark_label, metrics, active=None, engine_label=""):
    active = active or {}
    cards = [
        ("Instrument", ticker, ""), ("Benchmark", benchmark_label, ""),
        ("Annual Return", fmt_pct(metrics.get("Ann Return")), "pos" if metrics.get("Ann Return",0)>0 else "neg"),
        ("Annual Volatility", fmt_pct(metrics.get("Ann Vol")), ""),
        ("Sharpe @ RF 4.5%", fmt_num(metrics.get("Sharpe"),2), "pos" if metrics.get("Sharpe",0)>1 else "warn"),
        ("Sortino", fmt_num(metrics.get("Sortino"),2), "pos" if metrics.get("Sortino",0)>1 else "warn"),
        ("Max Drawdown", fmt_pct(metrics.get("Max Drawdown")), "neg"),
        ("CVaR 95", fmt_pct(metrics.get("CVaR 95")), "neg"),
        ("Tracking Error", fmt_pct(active.get("Tracking Error")), ""),
        ("Information Ratio", fmt_num(active.get("Information Ratio"),2), "pos" if active.get("Information Ratio",0)>0 else "warn"),
        ("Beta", fmt_num(active.get("Beta"),2), ""),
        ("TA Engine", engine_label, ""),
    ]
    html = '<div class="kpi-grid">'
    for label, value, tone in cards:
        html += f'<div class="kpi-card {tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    html += '</div>'
    return pn.pane.HTML(html, sizing_mode="stretch_width")

def empty_state(title, detail):
    return pn.Column(pn.pane.HTML(f'<div class="qfa-note"><b>{title}</b><br>{detail}</div>', sizing_mode="stretch_width"), sizing_mode="stretch_width")

def chart_layout(fig, title, height=720):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=20)),
        template="plotly_white", height=height,
        margin=dict(l=46, r=28, t=78, b=48),
        font=dict(family="DejaVu Sans, Segoe UI, Helvetica, sans-serif", size=12, color="#1e293b"),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(203,213,225,.48)", zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(203,213,225,.48)", zeroline=False)
    return fig

# -----------------------------------------------------------------------------
# KPI, Price, Risk, Benchmark, Universe, Tearsheet, Stress builders
# -----------------------------------------------------------------------------
def build_kpi(ticker, benchmark_label, start, end):
    bench = BENCHMARKS[benchmark_label]
    asset = add_indicators(fetch_ohlcv(ticker, start, end))
    if asset.empty: return empty_state("No Yahoo data", f"{ticker} data missing.")
    metrics = risk_metrics(asset["Return"])
    base = add_indicators(fetch_ohlcv(bench, start, end))
    act = {} if base.empty else active_metrics(asset["Return"], base["Return"])
    engine = asset["Indicator Engine"].iloc[-1] if "Indicator Engine" in asset.columns else ("TA-Lib Lazy" if load_talib() else "Formula fallback")
    return make_kpi_cards(ticker, benchmark_label, metrics, act, engine)

def build_price_chart(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, start, end))
    if df.empty: return empty_state("No Yahoo data", f"No data for {ticker}.")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=[0.5,0.17,0.18,0.15],
                        subplot_titles=["OHLC + Bollinger + MA50/MA200", "RSI 14", "MACD", "Stochastic"])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                name="OHLC", increasing_line_color="#166534", decreasing_line_color="#991b1b"), row=1, col=1)
    for col, name, dash, w in [("BB Upper","BB Upper","dash",1.1),("BB Mid","BB Mid","dot",1.1),("BB Lower","BB Lower","dash",1.1),
                               ("MA50","MA50","solid",1.7),("MA200","MA200","solid",2.0)]:
        if col in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, mode="lines", line=dict(dash=dash,width=w)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#64748b")
    fig.update_yaxes(range=[0,100], row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD Signal"], name="Signal"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD Hist"], name="Histogram", opacity=0.42), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch K"], name="Stoch K"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch D"], name="Stoch D"), row=4, col=1)
    fig.add_hline(y=80, row=4, col=1, line_dash="dash", line_color="#64748b")
    fig.add_hline(y=20, row=4, col=1, line_dash="dash", line_color="#64748b")
    return pn.pane.Plotly(chart_layout(fig, f"{ticker} | TA‑Lib Technical Dashboard", 920), config={"responsive":True})

def build_risk_chart(ticker, start, end):
    df = add_indicators(fetch_ohlcv(ticker, start, end))
    if df.empty: return empty_state("No Yahoo data", "")
    r = df["Return"].dropna()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=["Cumulative Return", "Rolling Volatility", "Drawdown", "Daily Return Distribution"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Cumulative Return"]*100, name="Cumulative Return %", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol21"]*100, name="Vol 21D %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol63"]*100, name="Vol 63D %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"]*100, name="Drawdown %", fill="tozeroy"), row=3, col=1)
    fig.add_trace(go.Histogram(x=r*100, nbinsx=70, name="Daily Returns %"), row=4, col=1)
    return pn.pane.Plotly(chart_layout(fig, f"{ticker} | Hedge Fund Risk Diagnostics", 880), config={"responsive":True})

def build_benchmark_relative(ticker, benchmark_label, start, end):
    bench = BENCHMARKS[benchmark_label]
    asset = fetch_ohlcv(ticker, start, end)
    base = fetch_ohlcv(bench, start, end)
    if asset.empty or base.empty: return empty_state("Benchmark relative data unavailable", "")
    joined = pd.concat([asset["Close"].rename(ticker), base["Close"].rename(bench)], axis=1).dropna()
    if len(joined) < MIN_OBS: return empty_state("Insufficient matched data", "")
    ret = joined.pct_change().dropna()
    cum = (1+ret).cumprod()-1
    active = ret[ticker]-ret[bench]
    active_cum = (1+active).cumprod()-1
    rolling_te = active.rolling(63).std()*np.sqrt(TRADING_DAYS)
    rolling_ir = (active.rolling(63).mean()*TRADING_DAYS)/rolling_te
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=["Cumulative Return Comparison","Active Cumulative Return","Rolling Tracking Error","Rolling Information Ratio"])
    fig.add_trace(go.Scatter(x=cum.index, y=cum[ticker]*100, name=ticker), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum.index, y=cum[bench]*100, name=benchmark_label), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_cum.index, y=active_cum*100, name="Active Return"), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_te.index, y=rolling_te*100, name="Rolling TE 63D"), row=3, col=1)
    fig.add_trace(go.Scatter(x=rolling_ir.index, y=rolling_ir, name="Rolling IR 63D"), row=4, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash="dash"); fig.add_hline(y=0, row=4, col=1, line_dash="dash")
    return pn.pane.Plotly(chart_layout(fig, f"{ticker} vs {benchmark_label} | Benchmark Relative", 900), config={"responsive":True})

def build_universe_board(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)
    if prices.empty: return empty_state("Universe unavailable", "No matched Yahoo data.")
    returns = prices.pct_change().dropna()
    rows = []
    for t in returns.columns:
        m = risk_metrics(returns[t])
        total = prices[t].iloc[-1]/prices[t].iloc[0]-1
        rows.append({"Ticker":t, "Name":get_name(t), "Total Return":total, "Ann Return":m["Ann Return"],
                     "Ann Vol":m["Ann Vol"], "Sharpe":m["Sharpe"], "Max DD":m["Max Drawdown"], "CVaR 95":m["CVaR 95"]})
    board = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.13, subplot_titles=["Sharpe Ranking", "Risk/Return Map"])
    fig.add_trace(go.Bar(x=board["Ticker"], y=board["Sharpe"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=board["Ann Vol"]*100, y=board["Ann Return"]*100, mode="markers+text", text=board["Ticker"],
                             textposition="top center", marker=dict(size=np.clip((board["Sharpe"].fillna(0).abs()+0.6)*14,9,32))), row=2, col=1)
    table_df = board.copy()
    for col in ["Total Return","Ann Return","Ann Vol","Max DD","CVaR 95"]:
        table_df[col] = table_df[col].map(fmt_pct)
    table_df["Sharpe"] = table_df["Sharpe"].map(lambda x: fmt_num(x,2))
    return pn.Column(
        pn.pane.Plotly(chart_layout(fig, f"{asset_class} | {region} | Universe Board", 820), config={"responsive":True}),
        pn.widgets.Tabulator(table_df, height=360, pagination="remote", page_size=12, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

def build_tearsheet(ticker, benchmark_label, start, end):
    try:
        import quantstats as qs
    except Exception:
        return empty_state("Tearsheet unavailable in ULTRA STABLE mode", "Install quantstats and move this behind a button if you need it on a paid Render instance.")
    bench = BENCHMARKS[benchmark_label]
    asset = fetch_ohlcv(ticker, start, end); base = fetch_ohlcv(bench, start, end)
    if asset.empty or base.empty: return empty_state("Tearsheet unavailable", "")
    ret = asset["Close"].pct_change().rename(ticker)
    bret = base["Close"].pct_change().rename(bench)
    matched = pd.concat([ret, bret], axis=1).dropna()
    if len(matched) < MIN_OBS: return empty_state("Insufficient matched observations", "")
    out = OUTPUT_DIR / f"tearsheet_{ticker.replace('^','IDX')}_vs_{bench.replace('^','IDX')}.html"
    qs.reports.html(matched[ticker], benchmark=matched[bench], rf=RISK_FREE_RATE, output=str(out),
                    title=f"{ticker} vs {benchmark_label} | QFA Hedge Fund Tearsheet", compounded=True)
    plt.close('all')  # free memory after generating image‑heavy report
    html = out.read_text(encoding="utf-8", errors="ignore")
    header = f'<div class="qfa-note"><b>Tearsheet generated.</b><br>{ticker} vs {benchmark_label} ({bench}), RF {RISK_FREE_RATE:.2%}, {len(matched)} obs.</div>'
    return pn.Column(pn.pane.HTML(header, sizing_mode="stretch_width"),
                     pn.pane.HTML(html, height=1050, sizing_mode="stretch_width"), sizing_mode="stretch_width")

def build_stress(asset_class, region, start, end):
    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)
    if prices.empty: return empty_state("Stress panel unavailable", "No matched data.")
    ret = prices.pct_change().dropna()
    vol = ret.std()*np.sqrt(TRADING_DAYS)
    rows = []
    for sc, shock in STRESS_SCENARIOS.items():
        for t in ret.columns:
            beta_to_univ = ret[t].corr(ret.mean(axis=1))
            beta_to_univ = 1.0 if pd.isna(beta_to_univ) else beta_to_univ
            impact = shock * beta_to_univ
            severity = abs(impact) / max(vol[t], 1e-9)
            rows.append({"Scenario":sc, "Ticker":t, "Name":get_name(t), "Estimated Impact":fmt_pct(impact),
                         "Volatility":fmt_pct(vol[t]), "Severity Score":round(severity,2)})
    table = pd.DataFrame(rows).sort_values("Severity Score", ascending=False)
    return pn.Column(pn.pane.HTML('<div class="qfa-note">Sensitivity based on matched Yahoo returns.</div>', sizing_mode="stretch_width"),
                     pn.widgets.Tabulator(table, height=560, pagination="remote", page_size=20, sizing_mode="stretch_width"), sizing_mode="stretch_width")

# =============================================================================
# ENHANCED PORTFOLIO OPTIMIZER – all core strategies with institutional explanations
# =============================================================================
def build_optimizer(asset_class, region, start, end):
    if not load_pypfopt():
        return empty_state("PyPortfolioOpt unavailable", "Install PyPortfolioOpt + cvxpy solvers. Optimizer imports are lazy-loaded for Render stability.")

    tickers = get_tickers(asset_class, region)
    prices = fetch_price_matrix(tickers, start, end)
    if prices.empty or prices.shape[1] < 3:
        return empty_state("Optimizer unavailable", "At least 3 instruments with matched Yahoo data required.")

    try:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        rf = RISK_FREE_RATE

        strategies = {}

        # 1. Max Sharpe
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=rf)
        w = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rf)
        div = np.dot(list(w.values()), np.sqrt(np.diag(S))) / np.sqrt(np.dot(list(w.values()), S @ list(w.values())))
        strategies["Max Sharpe"] = {"weights": w, "ret": ret, "vol": vol, "sharpe": sharpe, "div": div}

        # 2. Min Volatility
        ef2 = EfficientFrontier(mu, S)
        ef2.min_volatility()
        w2 = ef2.clean_weights()
        ret2, vol2, sharpe2 = ef2.portfolio_performance(risk_free_rate=rf)
        div2 = np.dot(list(w2.values()), np.sqrt(np.diag(S))) / np.sqrt(np.dot(list(w2.values()), S @ list(w2.values())))
        strategies["Min Volatility"] = {"weights": w2, "ret": ret2, "vol": vol2, "sharpe": sharpe2, "div": div2}

        # 3. Maximum Diversification (custom)
        def max_div_portfolio(mu, S):
            n = len(mu)
            def neg_div_ratio(w):
                w = np.array(w)
                port_vol = np.sqrt(w @ S @ w)
                div = np.dot(w, np.sqrt(np.diag(S))) / port_vol
                return -div
            from scipy.optimize import minimize
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0,1) for _ in range(n)]
            result = minimize(neg_div_ratio, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)
            w = result.x
            ret = np.dot(w, mu)
            vol = np.sqrt(w @ S @ w)
            sharpe = (ret - rf) / vol
            div = np.dot(w, np.sqrt(np.diag(S))) / vol
            return dict(zip(tickers, w)), ret, vol, sharpe, div
        try:
            w3, ret3, vol3, sharpe3, div3 = max_div_portfolio(mu.values, S.values)
            strategies["Max Diversification"] = {"weights": w3, "ret": ret3, "vol": vol3, "sharpe": sharpe3, "div": div3}
        except Exception:
            strategies["Max Diversification"] = None

        # 4. Efficient Risk (target vol 15%)
        try:
            ef4 = EfficientFrontier(mu, S)
            ef4.efficient_risk(target_volatility=0.15)
            w4 = ef4.clean_weights()
            ret4, vol4, sharpe4 = ef4.portfolio_performance(risk_free_rate=rf)
            div4 = np.dot(list(w4.values()), np.sqrt(np.diag(S))) / np.sqrt(np.dot(list(w4.values()), S @ list(w4.values())))
            strategies["Efficient Risk (15% vol)"] = {"weights": w4, "ret": ret4, "vol": vol4, "sharpe": sharpe4, "div": div4}
        except Exception:
            strategies["Efficient Risk (15% vol)"] = None

        # 5. Hierarchical Risk Parity
        try:
            hrp = HRPOpt(returns=prices.pct_change().dropna())
            w5 = hrp.optimize(linkage_method='single')
            ret5 = np.dot(list(w5.values()), mu)
            vol5 = np.sqrt(np.dot(list(w5.values()), S @ list(w5.values())))
            sharpe5 = (ret5 - rf) / vol5
            div5 = np.dot(list(w5.values()), np.sqrt(np.diag(S))) / vol5
            strategies["Hierarchical Risk Parity"] = {"weights": w5, "ret": ret5, "vol": vol5, "sharpe": sharpe5, "div": div5}
        except Exception:
            strategies["Hierarchical Risk Parity"] = None

        # 6. Equal Weight
        n = len(tickers)
        w6 = {t: 1/n for t in tickers}
        ret6 = np.dot(list(w6.values()), mu)
        vol6 = np.sqrt(np.dot(list(w6.values()), S @ list(w6.values())))
        sharpe6 = (ret6 - rf) / vol6
        div6 = np.dot(list(w6.values()), np.sqrt(np.diag(S))) / vol6
        strategies["Equal Weight"] = {"weights": w6, "ret": ret6, "vol": vol6, "sharpe": sharpe6, "div": div6}

        # ---------- Summary table ----------
        summary_rows = []
        for name, s in strategies.items():
            if s is None: continue
            summary_rows.append({
                "Strategy": name,
                "Expected Annual Return": fmt_pct(s["ret"]),
                "Annual Volatility": fmt_pct(s["vol"]),
                "Sharpe Ratio": fmt_num(s["sharpe"], 2),
                "Diversification Ratio": fmt_num(s["div"], 2),
            })
        summary_df = pd.DataFrame(summary_rows)

        # ---------- Individual weight charts + descriptions ----------
        descriptions = {
            "Max Sharpe": "Tangency portfolio: maximizes the Sharpe ratio (excess return per unit of risk) under the assumption of a risk‑free rate of 4.5%. This is the classic mean‑variance optimal portfolio.",
            "Min Volatility": "Global minimum variance portfolio: minimizes volatility without any expected return target. Ideal for investors seeking the lowest possible risk.",
            "Max Diversification": "Maximises the diversification ratio (weighted‑average asset volatility divided by portfolio volatility). Seeks the most balanced risk contribution across assets, often leading to improved risk‑adjusted returns.",
            "Efficient Risk (15% vol)": "Efficient portfolio that targets exactly 15% annual volatility. All other portfolios on the efficient frontier are derived from this constraint.",
            "Hierarchical Risk Parity": "HRP uses hierarchical clustering to allocate risk parity among clusters of assets. It is robust to estimation errors and avoids inversion of the covariance matrix.",
            "Equal Weight": "Naive 1/N allocation. Despite its simplicity, it often performs surprisingly well out‑of‑sample and serves as a benchmark."
        }

        charts = []
        for name, s in strategies.items():
            if s is None: continue
            wdf = pd.DataFrame.from_dict(s["weights"], orient="index", columns=["Weight"]).sort_values("Weight", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=wdf.index, y=wdf["Weight"]*100, marker_color="#1e3a8a", name="Weight %"))
            fig = chart_layout(fig, f"{name} – Allocation", 400)
            desc_html = f'<div class="qfa-note"><b>{name}</b>: {descriptions.get(name, "")}</div>'
            charts.append(pn.Column(pn.pane.HTML(desc_html, sizing_mode="stretch_width"),
                                    pn.pane.Plotly(fig, config={"responsive":True}), sizing_mode="stretch_width"))

        return pn.Column(
            pn.pane.HTML(f'<div class="qfa-note"><b>Optimizer engine:</b> PyPortfolioOpt | Covariance: Ledoit‑Wolf shrinkage | RF: {rf:.2%}<br>All portfolios use original Yahoo close prices; no synthetic data.</div>', sizing_mode="stretch_width"),
            pn.pane.Markdown("## Strategy Comparison", sizing_mode="stretch_width"),
            pn.widgets.Tabulator(summary_df, height=200, sizing_mode="stretch_width"),
            pn.pane.Markdown("## Detailed Allocations & Rationale", sizing_mode="stretch_width"),
            *charts,
            sizing_mode="stretch_width",
        )

    except Exception as e:
        return empty_state("Optimizer failed", f"{type(e).__name__}: {str(e)}")

# =============================================================================
# ADVANCED RISK ANALYTICS TAB
# =============================================================================
def build_advanced_risk(ticker, benchmark_label, start, end):
    """Production v10 advanced risk: no rolling Monte Carlo loop, no startup render."""
    bench = BENCHMARKS[benchmark_label]
    asset = add_indicators(fetch_ohlcv(ticker, start, end))
    base = add_indicators(fetch_ohlcv(bench, start, end))
    if asset.empty:
        return empty_state("No Yahoo data", f"{ticker} data missing.")
    returns = asset["Return"].dropna()
    if returns.empty or len(returns) < 126:
        return empty_state("Insufficient data", "At least 126 daily returns needed.")
    price = asset["Close"]
    base_returns = base["Return"].dropna() if not base.empty else None
    window_var = min(252, max(126, len(returns)//2)); window_nav = 63
    var95_hist = compute_rolling_var(returns, window_var, 0.95, 'historical')
    var99_hist = compute_rolling_var(returns, window_var, 0.99, 'historical')
    var95_param = compute_rolling_var(returns, window_var, 0.95, 'parametric')
    var99_param = compute_rolling_var(returns, window_var, 0.99, 'parametric')
    ratio_hist = compute_var_nav_ratio(price, returns, window_var, window_nav, 0.95, 'historical')
    ratio_param = compute_var_nav_ratio(price, returns, window_var, window_nav, 0.95, 'parametric')
    beta_series = rolling_beta(returns, base_returns, window_var) if base_returns is not None else pd.Series(dtype=float)
    recent = returns.iloc[-window_var:]
    backtest = []
    for method, conf, var_ser in [('Historical 95%', 0.95, var95_hist), ('Parametric 95%', 0.95, var95_param), ('Historical 99%', 0.99, var99_hist), ('Parametric 99%', 0.99, var99_param)]:
        aligned = pd.concat([recent, var_ser.shift(1)], axis=1).dropna()
        total = len(aligned)
        viol = int((aligned.iloc[:,0] < aligned.iloc[:,1]).sum()) if total else 0
        backtest.append({"Method": method, "Expected Violations": int((1-conf)*total), "Actual Violations": viol, "Ratio": f"{(viol/total if total else 0):.2%}"})
    backtest_df = pd.DataFrame(backtest)
    stress_df = build_historical_stress_table(ticker, start, end)
    fig1 = make_subplots(rows=2, cols=1, subplot_titles=["Rolling VaR 95%", "Rolling VaR 99%"])
    for ser, name, row in [(var95_hist, 'Historical 95%', 1), (var95_param, 'Parametric 95%', 1), (var99_hist, 'Historical 99%', 2), (var99_param, 'Parametric 99%', 2)]:
        fig1.add_trace(go.Scatter(x=ser.index, y=ser*100, name=name, mode="lines"), row=row, col=1)
    fig1.update_yaxes(title="VaR (%)", row=1, col=1); fig1.update_yaxes(title="VaR (%)", row=2, col=1)
    fig1 = chart_layout(fig1, f"{ticker} | Advanced Risk Lite / Render Safe", 760)
    fig2 = go.Figure()
    if not ratio_hist.empty: fig2.add_trace(go.Scatter(x=ratio_hist.index, y=ratio_hist, name="Historical"))
    if not ratio_param.empty: fig2.add_trace(go.Scatter(x=ratio_param.index, y=ratio_param, name="Parametric"))
    fig2 = chart_layout(fig2, "VaR / 3-Month NAV Ratio (95%)", 560)
    fig3 = go.Figure()
    if not beta_series.empty:
        fig3.add_trace(go.Scatter(x=beta_series.index, y=beta_series, name="Rolling Beta")); fig3.add_hline(y=1, line_dash="dash", line_color="#64748b")
    fig3 = chart_layout(fig3, f"Rolling {window_var}-Day Beta vs {benchmark_label}", 500)
    stress_pane = pn.widgets.Tabulator(stress_df, height=300, sizing_mode="stretch_width") if not stress_df.empty else pn.pane.HTML("<div class='qfa-note'>No major GSPC drawdowns >20% found.</div>")
    col = pn.Column(pn.pane.HTML('<div class="qfa-note"><b>Advanced Risk Analytics v10:</b> Historical + Parametric rolling VaR/NAV, beta, backtesting, and historical stress. Rolling Monte Carlo is disabled on Render Free to avoid RAM/CPU blowups.</div>', sizing_mode="stretch_width"), memory_note("Before advanced risk render"), pn.pane.Plotly(fig1, config={"responsive": True}), pn.pane.Plotly(fig2, config={"responsive": True}), pn.pane.Plotly(fig3, config={"responsive": True}), pn.pane.Markdown("### VaR Backtest"), pn.widgets.Tabulator(backtest_df, height=240, sizing_mode="stretch_width"), pn.pane.Markdown("### Historical Stress Scenarios (GSPC drawdowns >20%)"), stress_pane, sizing_mode="stretch_width")
    cleanup_memory()
    return col

# -----------------------------------------------------------------------------
# Batch Report Button
# -----------------------------------------------------------------------------
def generate_selected_report(ticker, benchmark_label, start, end):
    try:
        component = build_tearsheet(ticker, benchmark_label, start, end)
        path = OUTPUT_DIR / f"QFA_Selected_Report_{ticker.replace('^','IDX')}_{BENCHMARKS[benchmark_label].replace('^','IDX')}.html"
        wrapper = f"""
        <!DOCTYPE html><html><head><meta charset="utf-8"><title>QFA Report - {ticker}</title>{css()}</head>
        <body style="background:#f8fafc;padding:22px;font-family:{FONT_STACK};">
        <div class="qfa-header"><div class="qfa-title">QFA Selected ETF Report</div><div class="qfa-subtitle">Instrument: {ticker} | Benchmark: {benchmark_label} | RF: {RISK_FREE_RATE:.2%}</div></div>
        <div class="qfa-note">Full dashboard available in the Panel app.</div></body></html>
        """
        path.write_text(wrapper, encoding="utf-8")
        return f"Generated: {path}"
    except Exception as e:
        return f"Report generation failed: {type(e).__name__}: {e}"

# -----------------------------------------------------------------------------
# App Factory
# -----------------------------------------------------------------------------
def make_app():
    default_asset = "Equity ETF"
    default_region = get_regions(default_asset)[0]
    default_ticker = get_tickers(default_asset, default_region)[0]

    asset_class = pn.widgets.Select(name="Asset Class", options=list(UNIVERSE.keys()), value=default_asset)
    region = pn.widgets.Select(name="Region", options=get_regions(default_asset), value=default_region)
    ticker = pn.widgets.Select(name="Instrument", options=get_tickers(default_asset, default_region), value=default_ticker)
    benchmark = pn.widgets.Select(name="Benchmark", options=list(BENCHMARKS.keys()), value="S&P 500 Index")
    start_date = pn.widgets.DatePicker(name="Start Date", value=datetime(2020,1,1))
    end_date = pn.widgets.DatePicker(name="End Date", value=datetime.now())
    report_button = pn.widgets.Button(name="Generate Selected ETF HTML Report", button_type="primary")
    report_status = pn.pane.Markdown("")

    def update_regions(event=None):
        regions = get_regions(asset_class.value)
        region.options = regions; region.value = regions[0] if regions else None
    def update_tickers(event=None):
        tickers = get_tickers(asset_class.value, region.value)
        ticker.options = tickers; ticker.value = tickers[0] if tickers else None
    asset_class.param.watch(lambda e: (update_regions(e), update_tickers(e)), "value")
    region.param.watch(update_tickers, "value")

    def on_report_click(event):
        report_status.object = "Generating..."
        msg = generate_selected_report(ticker.value, benchmark.value, start_date.value, end_date.value)
        report_status.object = msg
    report_button.on_click(on_report_click)

    sidebar = pn.Column(
        pn.pane.HTML(f'<div style="padding:8px 0 12px 0;"><div style="font-size:25px;font-weight:850;color:#0f172a;">QFA Hedge Fund</div><div style="font-size:13px;color:#64748b;">Live reactive dashboard</div></div>', sizing_mode="stretch_width"),
        pn.pane.Markdown("### Investment Universe"),
        asset_class, region, ticker,
        pn.pane.Markdown("### Benchmark & Period"),
        benchmark, start_date, end_date,
        pn.Spacer(height=8), report_button, report_status,
        pn.pane.HTML(f'<div class="qfa-note">{status_badge("TA-Lib Lazy", TALIB_AVAILABLE)} {status_badge("QuantStats Active", QUANTSTATS_AVAILABLE)} {status_badge("PyPortfolioOpt Lazy", PYPFOPT_AVAILABLE)}<br><br><b>RF:</b> {RISK_FREE_RATE:.2%}<br><b>Mode:</b> PRODUCTION v10 / Memory-profiled Render Free + Lazy TA-Lib + Lazy PyPortfolioOpt<br><b>Data policy:</b> Yahoo-only matched data.<br><b>Benchmark:</b> ^GSPC</div>', sizing_mode="stretch_width"),
        width=340, height=980, sizing_mode="fixed",
        styles={"background":"#f8fafc","padding":"20px","border-right":"1px solid #dbe4ef","overflow-y":"auto"}
    )

    header = pn.pane.HTML(f'<div class="qfa-header"><div class="qfa-title">{APP_TITLE}</div><div class="qfa-subtitle">Institutional hedge-fund analytics optimized for Render Free 512 MB: KPI scorecard, TA-Lib technicals, risk metrics, benchmark relative, universe board, stress testing and lazy PyPortfolioOpt optimizer. Heavy engines are never rendered at startup. Use Render start command with --num-procs 1 only.</div></div>', sizing_mode="stretch_width")

    # ULTRA STABLE v10: dynamic=True prevents Panel from rendering every tab at startup.
    # TA-Lib and PyPortfolioOpt are preserved with lazy imports and lazy tab rendering.
    tabs = pn.Tabs(
        ("Executive KPI Dashboard", pn.bind(build_kpi, ticker, benchmark, start_date, end_date)),
        ("Price & TA-Lib", pn.bind(build_price_chart, ticker, start_date, end_date)),
        ("Risk Metrics", pn.bind(build_risk_chart, ticker, start_date, end_date)),
        ("Benchmark Relative", pn.bind(build_benchmark_relative, ticker, benchmark, start_date, end_date)),
        ("Advanced Risk", pn.bind(build_advanced_risk, ticker, benchmark, start_date, end_date)),
        ("Investment Universe", pn.bind(build_universe_board, asset_class, region, start_date, end_date)),
        ("Stress Testing", pn.bind(build_stress, asset_class, region, start_date, end_date)),
        ("Portfolio Optimizer", pn.bind(build_optimizer, asset_class, region, start_date, end_date)),
        ("Runtime Profile", pn.Column(memory_note("Runtime profile"), pn.pane.HTML("<div class='qfa-note'><b>Production rule:</b> Render must run exactly one Bokeh process. Correct command: <code>panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=* --num-procs 1</code></div>", sizing_mode="stretch_width"))),
        dynamic=True,
        sizing_mode="stretch_width",
    )

    main = pn.Column(pn.pane.HTML(css(), sizing_mode="stretch_width"), header, tabs,
                     sizing_mode="stretch_width", styles={"padding":"18px","background":"#ffffff"})
    return pn.Row(sidebar, main, sizing_mode="stretch_width")

app = make_app()
app.servable(title=APP_TITLE)