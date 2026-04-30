# =============================================================================
# QFA PRIME HEDGE FUND PLATFORM V3 - RENDER READY
# Reactive Panel architecture | Yahoo-only data | Institutional risk dashboard
# =============================================================================

import os
import time
import math
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import yfinance as yf
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pn.extension('plotly', 'tabulator', sizing_mode='stretch_width', notifications=True)

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
APP_TITLE = 'QFA Prime Finance Platform'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
RF = float(os.getenv('QFA_RISK_FREE_RATE', '0.045'))
MIN_OBS = int(os.getenv('QFA_MIN_OBS', '90'))
CACHE_TTL_SECONDS = int(os.getenv('QFA_CACHE_TTL_SECONDS', '900'))
MAX_TICKERS = int(os.getenv('QFA_MAX_TICKERS', '12'))
MC_SIMULATIONS = int(os.getenv('QFA_MC_SIMULATIONS', '3000'))
ADVANCED_RISK_WINDOW = int(os.getenv('QFA_ADVANCED_RISK_WINDOW', '252'))
FONT_STACK = 'DejaVu Sans, Liberation Sans, Segoe UI, Helvetica, Arial, sans-serif'

# -----------------------------------------------------------------------------
# Optional libraries
# -----------------------------------------------------------------------------
try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except Exception:
    LedoitWolf = None
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    minimize = None
    stats = None
    SCIPY_AVAILABLE = False

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

# -----------------------------------------------------------------------------
# Universe: stable, liquid Yahoo instruments only. Turkish equities removed.
# -----------------------------------------------------------------------------
UNIVERSE = {
    'Core US Equity ETFs': {
        'United States': {
            'SPY': 'SPDR S&P 500 ETF', 'QQQ': 'Invesco Nasdaq 100 ETF',
            'IWM': 'iShares Russell 2000 ETF', 'DIA': 'SPDR Dow Jones Industrial Average ETF',
            'VTI': 'Vanguard Total Stock Market ETF', 'RSP': 'Invesco S&P 500 Equal Weight ETF',
            'VUG': 'Vanguard Growth ETF', 'VTV': 'Vanguard Value ETF',
        }
    },
    'US Sector ETFs': {
        'United States': {
            'XLK': 'Technology Select Sector SPDR', 'XLF': 'Financials Select Sector SPDR',
            'XLE': 'Energy Select Sector SPDR', 'XLV': 'Health Care Select Sector SPDR',
            'XLI': 'Industrials Select Sector SPDR', 'XLY': 'Consumer Discretionary Select Sector SPDR',
            'XLP': 'Consumer Staples Select Sector SPDR', 'XLU': 'Utilities Select Sector SPDR',
            'XLB': 'Materials Select Sector SPDR', 'XLRE': 'Real Estate Select Sector SPDR',
            'XLC': 'Communication Services Select Sector SPDR',
        }
    },
    'Global Equity ETFs': {
        'Global': {
            'VT': 'Vanguard Total World Stock ETF', 'VEA': 'Vanguard Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF', 'EEM': 'iShares MSCI Emerging Markets ETF',
            'VGK': 'Vanguard FTSE Europe ETF', 'FEZ': 'SPDR EURO STOXX 50 ETF',
            'EWJ': 'iShares MSCI Japan ETF', 'FXI': 'iShares China Large-Cap ETF',
            'INDA': 'iShares MSCI India ETF', 'EWZ': 'iShares MSCI Brazil ETF',
        }
    },
    'Fixed Income ETFs': {
        'United States': {
            'SHY': 'iShares 1-3 Year Treasury Bond ETF', 'IEF': 'iShares 7-10 Year Treasury Bond ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF', 'BND': 'Vanguard Total Bond Market ETF',
            'AGG': 'iShares Core U.S. Aggregate Bond ETF', 'LQD': 'iShares Investment Grade Corporate Bond ETF',
            'HYG': 'iShares High Yield Corporate Bond ETF', 'TIP': 'iShares TIPS Bond ETF',
        }
    },
    'Commodities & Alternatives': {
        'Global': {
            'GLD': 'SPDR Gold Shares', 'SLV': 'iShares Silver Trust',
            'USO': 'United States Oil Fund', 'DBC': 'Invesco DB Commodity Index Tracking Fund',
            'DBA': 'Invesco DB Agriculture Fund', 'CPER': 'United States Copper Index Fund',
            'QAI': 'IQ Hedge Multi-Strategy Tracker ETF', 'MNA': 'IQ Merger Arbitrage ETF',
        }
    },
    'Crypto Proxy ETFs': {
        'United States': {
            'IBIT': 'iShares Bitcoin Trust', 'FBTC': 'Fidelity Wise Origin Bitcoin Fund',
            'BITO': 'ProShares Bitcoin Strategy ETF', 'GBTC': 'Grayscale Bitcoin Trust',
            'ETHE': 'Grayscale Ethereum Trust',
        }
    },
}

BENCHMARKS = {
    'S&P 500 Index (^GSPC)': '^GSPC',
    'Nasdaq 100 Index (^NDX)': '^NDX',
    'Russell 2000 Index (^RUT)': '^RUT',
    'Dow Jones Industrial Average (^DJI)': '^DJI',
    'Global Equity ETF (VT)': 'VT',
    'US Aggregate Bond ETF (AGG)': 'AGG',
    'Gold ETF (GLD)': 'GLD',
    'Cash Proxy 1-3Y Treasury (SHY)': 'SHY',
}

STRESS_SCENARIOS = pd.DataFrame([
    ('Crisis', 'Global equity shock -20%', -0.20, 1.20),
    ('Crisis', 'Liquidity shock -10%', -0.10, 1.60),
    ('Inflation', 'Rates/inflation shock -8%', -0.08, 1.10),
    ('Banking Stress', 'Credit stress -12%', -0.12, 1.40),
    ('Sharp Selloff', 'Risk asset selloff -15%', -0.15, 1.25),
    ('Sharp Rally', 'Risk asset rally +10%', 0.10, 0.80),
], columns=['Family', 'Scenario', 'Shock', 'Severity Multiplier'])

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def all_tickers():
    rows = []
    for asset_class, regions in UNIVERSE.items():
        for region, d in regions.items():
            for ticker, name in d.items():
                rows.append({'Asset Class': asset_class, 'Region': region, 'Ticker': ticker, 'Name': name})
    return pd.DataFrame(rows)

UNIVERSE_DF = all_tickers()

def get_regions(asset_class):
    return list(UNIVERSE.get(asset_class, {}).keys())

def get_tickers(asset_class, region):
    return list(UNIVERSE.get(asset_class, {}).get(region, {}).keys())

def name_for(ticker):
    m = UNIVERSE_DF.loc[UNIVERSE_DF['Ticker'].eq(ticker), 'Name']
    return ticker if m.empty else str(m.iloc[0])

def pct(x, d=2):
    if x is None or not np.isfinite(x):
        return '—'
    return f'{x*100:.{d}f}%'

def num(x, d=2):
    if x is None or not np.isfinite(x):
        return '—'
    return f'{x:.{d}f}'

def today_bucket(refresh_token=0):
    # TTL bucket + explicit refresh token guarantees Render does not show stale Yahoo data forever.
    return int(time.time() // CACHE_TTL_SECONDS) + int(refresh_token) * 10_000_000

@lru_cache(maxsize=2048)
def _download_one(ticker, start_s, end_s, bucket):
    try:
        df = yf.download(ticker, start=start_s, end=end_s, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
        out = df[keep].copy()
        out['Close'] = pd.to_numeric(out[price_col], errors='coerce')
        for c in ['Open', 'High', 'Low', 'Volume']:
            if c not in out.columns:
                out[c] = out['Close']
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=['Close'])
        out = out[~out.index.duplicated(keep='last')].sort_index()
        return out
    except Exception:
        return pd.DataFrame()

def fetch_one(ticker, start, end, refresh_token=0):
    s = pd.Timestamp(start).strftime('%Y-%m-%d')
    e = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return _download_one(str(ticker), s, e, today_bucket(refresh_token)).copy()

def fetch_matrix(tickers, start, end, refresh_token=0):
    frames = []
    for t in list(tickers)[:MAX_TICKERS]:
        d = fetch_one(t, start, end, refresh_token)
        if not d.empty:
            frames.append(d['Close'].rename(t))
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1).sort_index().ffill(limit=3).dropna(how='all')
    px = px.dropna(axis=1, thresh=max(MIN_OBS, int(len(px) * 0.60)))
    return px.dropna()

def add_features(df):
    """TA-Lib-first technical indicator layer with Render-safe fallback."""
    if df.empty:
        return df
    out = df.copy()
    out['Return'] = out['Close'].pct_change()
    out['Log Return'] = np.log(out['Close'] / out['Close'].shift(1))
    out['CumReturn'] = (1 + out['Return'].fillna(0)).cumprod() - 1
    out['MA20'] = out['Close'].rolling(20).mean()
    out['MA50'] = out['Close'].rolling(50).mean()
    out['MA200'] = out['Close'].rolling(200).mean()
    out['Vol21'] = out['Return'].rolling(21).std() * math.sqrt(TRADING_DAYS)
    out['Vol63'] = out['Return'].rolling(63).std() * math.sqrt(TRADING_DAYS)
    wealth = (1 + out['Return'].fillna(0)).cumprod()
    out['Drawdown'] = wealth / wealth.cummax() - 1
    used_talib = False
    if TALIB_AVAILABLE:
        try:
            close = out['Close'].astype(float).values
            high = out['High'].astype(float).values
            low = out['Low'].astype(float).values
            out['RSI'] = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            out['MACD'], out['MACD Signal'], out['MACD Hist'] = macd, signal, hist
            upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            out['BB Upper'], out['BB Mid'], out['BB Lower'] = upper, mid, lower
            out['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            out['Stoch K'], out['Stoch D'] = slowk, slowd
            used_talib = True
        except Exception:
            used_talib = False
    if not used_talib:
        mid = out['Close'].rolling(20).mean()
        sd = out['Close'].rolling(20).std()
        out['BB Upper'], out['BB Mid'], out['BB Lower'] = mid + 2 * sd, mid, mid - 2 * sd
        delta = out['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out['RSI'] = 100 - (100 / (1 + rs))
        ema12 = out['Close'].ewm(span=12, adjust=False).mean()
        ema26 = out['Close'].ewm(span=26, adjust=False).mean()
        out['MACD'] = ema12 - ema26
        out['MACD Signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
        out['MACD Hist'] = out['MACD'] - out['MACD Signal']
        tr = pd.concat([out['High'] - out['Low'], (out['High'] - out['Close'].shift()).abs(), (out['Low'] - out['Close'].shift()).abs()], axis=1).max(axis=1)
        out['ATR'] = tr.rolling(14).mean()
        low14 = out['Low'].rolling(14).min()
        high14 = out['High'].rolling(14).max()
        out['Stoch K'] = 100 * (out['Close'] - low14) / (high14 - low14).replace(0, np.nan)
        out['Stoch D'] = out['Stoch K'].rolling(3).mean()
    ew22 = out['Return'].ewm(span=22, adjust=False).std() * math.sqrt(TRADING_DAYS)
    ew33 = out['Return'].ewm(span=33, adjust=False).std() * math.sqrt(TRADING_DAYS)
    ew99 = out['Return'].ewm(span=99, adjust=False).std() * math.sqrt(TRADING_DAYS)
    out['EWMA Risk Signal'] = ew22 / ((ew33 + ew99) / 2).replace(0, np.nan)
    out['Indicator Engine'] = 'TA-Lib' if used_talib else 'Formula fallback'
    return out

def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 5:
        return {}
    ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * math.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * math.sqrt(TRADING_DAYS)
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    var95 = r.quantile(0.05)
    cvar95 = r[r <= var95].mean() if len(r[r <= var95]) else np.nan
    return {
        'Ann Return': ann_ret,
        'Ann Vol': ann_vol,
        'Sharpe': (ann_ret - RF) / ann_vol if ann_vol else np.nan,
        'Sortino': (ann_ret - RF) / downside if downside else np.nan,
        'Max Drawdown': dd.min(),
        'VaR 95': var95,
        'CVaR 95': cvar95,
        'Hit Ratio': (r > 0).mean(),
        'Obs': len(r),
    }

def active_metrics(asset_r, bench_r):
    x = pd.concat([asset_r.rename('asset'), bench_r.rename('bench')], axis=1).dropna()
    if len(x) < MIN_OBS:
        return {}
    active = x['asset'] - x['bench']
    te = active.std() * math.sqrt(TRADING_DAYS)
    alpha = active.mean() * TRADING_DAYS
    beta = x['asset'].cov(x['bench']) / x['bench'].var() if x['bench'].var() else np.nan
    return {'Tracking Error': te, 'Information Ratio': alpha / te if te else np.nan, 'Active Return': alpha, 'Beta': beta}

def layout(fig, title, height=720):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor='left', font=dict(size=21, color='#0f172a')),
        template='plotly_white', height=height, autosize=True,
        margin=dict(l=58, r=32, t=82, b=60),
        font=dict(family=FONT_STACK, size=12, color='#1e293b'),
        paper_bgcolor='white', plot_bgcolor='white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(203,213,225,.45)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(203,213,225,.45)', zeroline=False)
    return fig

def html_note(title, body):
    return pn.pane.HTML(f'<div class="qfa-note"><b>{title}</b><br>{body}</div>', sizing_mode='stretch_width')

def empty(title, body):
    return pn.pane.HTML(f'<div class="qfa-empty"><h3>{title}</h3><p>{body}</p></div>', sizing_mode='stretch_width')

def css():
    return f'''
    <style>
    body, .bk-root {{ font-family:{FONT_STACK}; background:#f8fafc; }}
    .qfa-header {{ background:linear-gradient(135deg,#07111f,#0f2742); color:white; border-radius:24px; padding:28px 32px; margin-bottom:18px; box-shadow:0 16px 36px rgba(15,23,42,.18); }}
    .qfa-title {{ font-size:40px; line-height:1.05; font-weight:900; letter-spacing:-.025em; }}
    .qfa-subtitle {{ font-size:15px; color:#cbd5e1; margin-top:10px; max-width:1180px; line-height:1.55; }}
    .qfa-note {{ background:#f8fafc; border:1px solid #dbe4ef; border-left:5px solid #0f2742; padding:14px 16px; border-radius:16px; color:#334155; font-size:13px; line-height:1.45; margin:8px 0; }}
    .qfa-empty {{ background:#fff7ed; border:1px solid #fed7aa; border-radius:18px; padding:24px; color:#7c2d12; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(180px,1fr)); gap:14px; margin-bottom:18px; }}
    .kpi-card {{ background:white; border:1px solid #dbe4ef; border-radius:20px; padding:18px 18px; box-shadow:0 10px 26px rgba(15,23,42,.07); }}
    .kpi-label {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#64748b; font-weight:800; }}
    .kpi-value {{ font-size:27px; font-weight:900; color:#0f172a; margin-top:5px; }}
    .tone-green {{ border-top:5px solid #166534; }} .tone-red {{ border-top:5px solid #991b1b; }} .tone-amber {{ border-top:5px solid #b45309; }} .tone-blue {{ border-top:5px solid #1d4ed8; }}
    @media (max-width:1200px) {{ .kpi-grid {{ grid-template-columns:repeat(2,minmax(160px,1fr)); }} .qfa-title {{ font-size:32px; }} }}
    </style>
    '''

def kpi_cards(items):
    html = '<div class="kpi-grid">'
    for label, value, tone in items:
        html += f'<div class="kpi-card tone-{tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    html += '</div>'
    return pn.pane.HTML(html, sizing_mode='stretch_width')

# -----------------------------------------------------------------------------
# Advanced risk engine: VaR/CVaR, VaR/NAV, rolling beta, historical stress regimes
# -----------------------------------------------------------------------------
def historical_var(r, confidence):
    r = pd.Series(r).dropna()
    return r.quantile(1 - confidence) if len(r) else np.nan

def historical_cvar(r, confidence):
    r = pd.Series(r).dropna()
    v = historical_var(r, confidence)
    tail = r[r <= v]
    return tail.mean() if len(tail) else np.nan

def parametric_var(r, confidence):
    r = pd.Series(r).dropna()
    if len(r) < 5 or stats is None:
        return np.nan
    return r.mean() + r.std() * stats.norm.ppf(1 - confidence)

def parametric_cvar(r, confidence):
    r = pd.Series(r).dropna()
    if len(r) < 5 or stats is None:
        return np.nan
    z = stats.norm.ppf(1 - confidence)
    return r.mean() - r.std() * stats.norm.pdf(z) / (1 - confidence)

def monte_carlo_tail(r, confidence, n=None, dist='t', seed=42):
    r = pd.Series(r).dropna()
    n = int(n or MC_SIMULATIONS)
    if len(r) < 30 or stats is None:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    try:
        if dist == 'normal':
            sim = rng.normal(r.mean(), r.std(), n)
        else:
            df, loc, scale = stats.t.fit(r)
            if not np.isfinite(df) or not np.isfinite(scale) or scale <= 0:
                sim = rng.normal(r.mean(), r.std(), n)
            else:
                sim = stats.t.rvs(df=df, loc=loc, scale=scale, size=n, random_state=rng)
        var = np.percentile(sim, 100 * (1 - confidence))
        cvar = sim[sim <= var].mean()
        return float(var), float(cvar)
    except Exception:
        return np.nan, np.nan

def rolling_var_series(r, window, confidence, method):
    r = pd.Series(r).dropna()
    minp = max(60, int(window * 0.70))
    if method == 'historical':
        return r.rolling(window, min_periods=minp).quantile(1 - confidence)
    if method == 'parametric':
        mean = r.rolling(window, min_periods=minp).mean()
        sd = r.rolling(window, min_periods=minp).std()
        z = stats.norm.ppf(1 - confidence) if stats is not None else np.nan
        return mean + sd * z
    tail = r.tail(max(window + 170, 420))
    out = pd.Series(index=r.index, dtype=float)
    vals = []
    for i in range(len(tail)):
        x = tail.iloc[max(0, i-window+1):i+1]
        vals.append(monte_carlo_tail(x, confidence, seed=1000 + i)[0] if len(x) >= minp else np.nan)
    out.loc[tail.index] = vals
    return out

def var_nav_ratio(price, r, window=ADVANCED_RISK_WINDOW, confidence=.95, method='historical'):
    v = rolling_var_series(r, window, confidence, method).shift(1).abs()
    nav = price.rolling(63, min_periods=21).mean().replace(0, np.nan)
    return (price * v / nav) * 100

def rolling_beta_series(asset_r, bench_r, window=ADVANCED_RISK_WINDOW):
    joined = pd.concat([asset_r.rename('asset'), bench_r.rename('bench')], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    minp = max(60, int(window * .70))
    cov = joined['asset'].rolling(window, min_periods=minp).cov(joined['bench'])
    var = joined['bench'].rolling(window, min_periods=minp).var()
    return cov / var.replace(0, np.nan)

def historical_drawdown_periods(benchmark_px, threshold=-0.20):
    if benchmark_px.empty:
        return pd.DataFrame()
    wealth = benchmark_px / benchmark_px.iloc[0]
    dd = wealth / wealth.cummax() - 1
    in_stress = dd < threshold
    if not in_stress.any():
        return pd.DataFrame()
    groups = (in_stress != in_stress.shift()).cumsum()[in_stress]
    rows = []
    for _, g in dd[in_stress].groupby(groups):
        rows.append({'Start': g.index.min(), 'End': g.index.max(), 'Benchmark Max DD': g.min(), 'Days': len(g)})
    return pd.DataFrame(rows).sort_values('Benchmark Max DD')

def advanced_risk(ticker, benchmark_label, start, end, refresh_token):
    b = BENCHMARKS[benchmark_label]
    asset = add_features(fetch_one(ticker, start, end, refresh_token))
    bench = add_features(fetch_one(b, start, end, refresh_token))
    if asset.empty:
        return empty('Advanced risk unavailable', f'Yahoo returned no usable data for {ticker}.')
    r = asset['Return'].dropna()
    if len(r) < 126:
        return empty('Insufficient data', 'At least 126 daily returns are required for advanced rolling risk analytics.')
    window = min(ADVANCED_RISK_WINDOW, max(126, len(r)//2)) if len(r) < ADVANCED_RISK_WINDOW else ADVANCED_RISK_WINDOW
    var95_h = rolling_var_series(r, window, .95, 'historical'); var99_h = rolling_var_series(r, window, .99, 'historical')
    var95_p = rolling_var_series(r, window, .95, 'parametric'); var99_p = rolling_var_series(r, window, .99, 'parametric')
    var95_m = rolling_var_series(r, window, .95, 'montecarlo'); var99_m = rolling_var_series(r, window, .99, 'montecarlo')
    ratio_h = var_nav_ratio(asset['Close'], r, window, .95, 'historical')
    ratio_p = var_nav_ratio(asset['Close'], r, window, .95, 'parametric')
    ratio_m = var_nav_ratio(asset['Close'], r, window, .95, 'montecarlo')
    beta = rolling_beta_series(r, bench['Return'].dropna(), window) if not bench.empty else pd.Series(dtype=float)
    bt_rows = []
    recent = r.tail(min(252, len(r)))
    for label, ser, conf in [('Historical 95%', var95_h, .95), ('Parametric 95%', var95_p, .95), ('Monte Carlo 95%', var95_m, .95), ('Historical 99%', var99_h, .99), ('Parametric 99%', var99_p, .99), ('Monte Carlo 99%', var99_m, .99)]:
        aligned = pd.concat([recent.rename('return'), ser.shift(1).rename('var')], axis=1, join='inner').dropna()
        n = len(aligned); violations = int((aligned['return'] < aligned['var']).sum()) if n else 0
        bt_rows.append({'Method': label, 'Expected Violations': round((1-conf)*n, 1), 'Actual Violations': violations, 'Violation Ratio': (violations/n if n else np.nan)})
    shown_bt = pd.DataFrame(bt_rows); shown_bt['Violation Ratio'] = shown_bt['Violation Ratio'].map(lambda x: pct(x))
    stress_rows = []
    if not bench.empty:
        periods = historical_drawdown_periods(bench['Close'], -0.20)
        ar = asset['Close'].pct_change().dropna()
        for _, row in periods.iterrows():
            mask = (ar.index >= row['Start']) & (ar.index <= row['End'])
            if mask.sum() > 1:
                stress_rows.append({'Start': row['Start'].strftime('%Y-%m-%d'), 'End': row['End'].strftime('%Y-%m-%d'), 'Benchmark Max DD': row['Benchmark Max DD'], f'{ticker} Cumulative Return': (1 + ar[mask]).prod() - 1, 'Days': int(mask.sum())})
    shown_stress = pd.DataFrame(stress_rows)
    if not shown_stress.empty:
        shown_stress['Benchmark Max DD'] = shown_stress['Benchmark Max DD'].map(lambda x: pct(x))
        shown_stress[f'{ticker} Cumulative Return'] = shown_stress[f'{ticker} Cumulative Return'].map(lambda x: pct(x))
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, subplot_titles=['Rolling VaR 95%: Historical / Parametric / Monte Carlo', 'Rolling VaR 99%: Historical / Parametric / Monte Carlo'])
    for ser, name, row in [(var95_h,'Historical 95%',1),(var95_p,'Parametric 95%',1),(var95_m,'Monte Carlo 95%',1),(var99_h,'Historical 99%',2),(var99_p,'Parametric 99%',2),(var99_m,'Monte Carlo 99%',2)]:
        fig1.add_trace(go.Scatter(x=ser.index, y=ser*100, name=name, mode='lines'), row=row, col=1)
    fig1.update_yaxes(title='Daily VaR (%)', row=1, col=1); fig1.update_yaxes(title='Daily VaR (%)', row=2, col=1)
    fig2 = go.Figure()
    for ser, name in [(ratio_h,'Historical'),(ratio_p,'Parametric'),(ratio_m,'Monte Carlo')]:
        fig2.add_trace(go.Scatter(x=ser.index, y=ser, name=name, mode='lines'))
    fig2.update_yaxes(title='VaR / 3M NAV (%)')
    fig3 = go.Figure()
    if not beta.empty:
        fig3.add_trace(go.Scatter(x=beta.index, y=beta, name=f'Rolling Beta vs {benchmark_label}', mode='lines')); fig3.add_hline(y=1.0, line_dash='dash')
    fig3.update_yaxes(title='Beta')
    latest_h_var, latest_h_cvar = historical_var(r, .95), historical_cvar(r, .95)
    latest_p_var = parametric_var(r, .95)
    latest_m_var, latest_m_cvar = monte_carlo_tail(r.tail(window), .95)
    cards = [('Hist VaR 95', pct(latest_h_var), 'red'), ('Hist CVaR 95', pct(latest_h_cvar), 'red'), ('Param VaR 95', pct(latest_p_var), 'amber'), ('MC VaR 95', pct(latest_m_var), 'amber'), ('MC CVaR 95', pct(latest_m_cvar), 'red'), ('Latest Beta', num(beta.dropna().iloc[-1] if beta.dropna().size else np.nan), 'blue'), ('TA Engine', str(asset['Indicator Engine'].iloc[-1]) if 'Indicator Engine' in asset else 'N/A', 'blue'), ('MC Sims', f'{MC_SIMULATIONS:,}', 'green')]
    return pn.Column(html_note('Advanced risk methodology', f'Historical, parametric-normal and Monte Carlo t-distribution VaR/CVaR. Rolling window: <b>{window}</b> trading days. Monte Carlo is capped by QFA_MC_SIMULATIONS for Render stability.'), kpi_cards(cards), pn.pane.Plotly(layout(fig1, f'{ticker} | Rolling VaR Backtest', 820), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Plotly(layout(fig2, f'{ticker} | VaR / 3-Month NAV Ratio', 560), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Plotly(layout(fig3, f'{ticker} | Rolling Beta vs {benchmark_label}', 500), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Markdown('### VaR Violation Backtest'), pn.widgets.Tabulator(shown_bt, height=260, sizing_mode='stretch_width'), pn.pane.Markdown('### Historical Benchmark Drawdown Regimes'), pn.widgets.Tabulator(shown_stress if not shown_stress.empty else pd.DataFrame([{'Message': 'No benchmark drawdown regime below -20% in selected period.'}]), height=300, sizing_mode='stretch_width'), sizing_mode='stretch_width')

# -----------------------------------------------------------------------------
# Render blocks
# -----------------------------------------------------------------------------
def executive(ticker, benchmark_label, start, end, refresh_token):
    asset = add_features(fetch_one(ticker, start, end, refresh_token))
    bench = add_features(fetch_one(BENCHMARKS[benchmark_label], start, end, refresh_token))
    if asset.empty:
        return empty('Yahoo data unavailable', f'{ticker} could not be downloaded. Retry, shorten universe size, or click Refresh Data.')
    m = metrics(asset['Return'])
    a = active_metrics(asset['Return'], bench['Return']) if not bench.empty else {}
    cards = [
        ('Annual Return', pct(m.get('Ann Return')), 'green' if m.get('Ann Return',0) > 0 else 'red'),
        ('Annual Volatility', pct(m.get('Ann Vol')), 'amber'),
        ('Sharpe Ratio', num(m.get('Sharpe')), 'green' if m.get('Sharpe',0) > 1 else 'amber'),
        ('Max Drawdown', pct(m.get('Max Drawdown')), 'red'),
        ('CVaR 95', pct(m.get('CVaR 95')), 'red'),
        ('Tracking Error', pct(a.get('Tracking Error')), 'blue'),
        ('Information Ratio', num(a.get('Information Ratio')), 'green' if a.get('Information Ratio',0) > 0 else 'amber'),
        ('Beta vs Benchmark', num(a.get('Beta')), 'blue'),
    ]
    return pn.Column(
        html_note('Reactive engine status', f'Instrument: <b>{ticker}</b> — {name_for(ticker)} | Benchmark: <b>{benchmark_label}</b> | Observations: <b>{m.get("Obs",0)}</b> | TA Engine: <b>{asset["Indicator Engine"].iloc[-1] if "Indicator Engine" in asset else "N/A"}</b> | Cache TTL: <b>{CACHE_TTL_SECONDS}s</b> | Refresh token: <b>{refresh_token}</b>'),
        kpi_cards(cards),
        sizing_mode='stretch_width'
    )

def technical_chart(ticker, start, end, refresh_token):
    df = add_features(fetch_one(ticker, start, end, refresh_token))
    if df.empty:
        return empty('No price data', f'Yahoo returned no data for {ticker}.')
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.038,
                        row_heights=[.46,.13,.16,.13,.12], subplot_titles=['OHLC + MA + Bollinger Bands','RSI 14','MACD','Stochastic Oscillator','ATR / EWMA Risk Signal'])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    for col in ['MA50','MA200','BB Upper','BB Mid','BB Lower']:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', mode='lines'), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash='dash'); fig.add_hline(y=30, row=2, col=1, line_dash='dash')
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD Signal'], name='Signal', mode='lines'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD Hist'], name='Histogram', opacity=.45), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch K'], name='Stoch K', mode='lines'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch D'], name='Stoch D', mode='lines'), row=4, col=1)
    fig.add_hline(y=80, row=4, col=1, line_dash='dash'); fig.add_hline(y=20, row=4, col=1, line_dash='dash')
    fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR', mode='lines'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EWMA Risk Signal'], name='EWMA Risk Signal', mode='lines'), row=5, col=1)
    fig.add_hline(y=1.0, row=5, col=1, line_dash='dash')
    return pn.pane.Plotly(layout(fig, f'{ticker} | Institutional TA-Lib Technical Dashboard', 1060), config={'responsive': True}, sizing_mode='stretch_width')

def risk_chart(ticker, start, end, refresh_token):
    df = add_features(fetch_one(ticker, start, end, refresh_token))
    if df.empty:
        return empty('Risk chart unavailable', 'No usable price data.')
    r = df['Return'].dropna()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=.08,
                        subplot_titles=['Cumulative Return','Rolling Volatility 21D / 63D','Drawdown','Daily Return Distribution'])
    fig.add_trace(go.Scatter(x=df.index, y=df['CumReturn']*100, name='Cumulative Return %', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol21']*100, name='Vol 21D %', mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol63']*100, name='Vol 63D %', mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown']*100, name='Drawdown %', fill='tozeroy', mode='lines'), row=3, col=1)
    fig.add_trace(go.Histogram(x=r*100, nbinsx=70, name='Daily Returns %'), row=4, col=1)
    return pn.pane.Plotly(layout(fig, f'{ticker} | Risk Diagnostics', 900), config={'responsive': True}, sizing_mode='stretch_width')

def relative_chart(ticker, benchmark_label, start, end, refresh_token):
    b = BENCHMARKS[benchmark_label]
    px = fetch_matrix([ticker, b], start, end, refresh_token)
    if px.empty or ticker not in px or b not in px:
        return empty('Benchmark-relative panel unavailable', f'Missing matched Yahoo data for {ticker} or {b}.')
    ret = px.pct_change().dropna()
    cum = (1 + ret).cumprod() - 1
    active = ret[ticker] - ret[b]
    active_cum = (1 + active).cumprod() - 1
    te = active.rolling(63).std() * math.sqrt(TRADING_DAYS)
    ir = active.rolling(63).mean() * TRADING_DAYS / te
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.055,
                        subplot_titles=['Cumulative Return vs Benchmark','Active Cumulative Return','Rolling Tracking Error','Rolling Information Ratio'])
    fig.add_trace(go.Scatter(x=cum.index, y=cum[ticker]*100, name=ticker, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum.index, y=cum[b]*100, name=benchmark_label, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_cum.index, y=active_cum*100, name='Active Return %', mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=te.index, y=te*100, name='Tracking Error %', mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=ir.index, y=ir, name='Information Ratio', mode='lines'), row=4, col=1)
    fig.add_hline(y=0, row=2, col=1, line_dash='dash'); fig.add_hline(y=0, row=4, col=1, line_dash='dash')
    return pn.pane.Plotly(layout(fig, f'{ticker} vs {benchmark_label} | Benchmark Relative Analytics', 900), config={'responsive': True}, sizing_mode='stretch_width')

def universe_board(asset_class, region, start, end, refresh_token):
    tickers = get_tickers(asset_class, region)
    px = fetch_matrix(tickers, start, end, refresh_token)
    if px.empty:
        return empty('Universe unavailable', 'Yahoo returned no matched data for the selected universe.')
    ret = px.pct_change().dropna()
    rows = []
    for t in ret.columns:
        m = metrics(ret[t])
        rows.append({'Ticker': t, 'Name': name_for(t), 'Ann Return': m.get('Ann Return'), 'Ann Vol': m.get('Ann Vol'), 'Sharpe': m.get('Sharpe'), 'Max DD': m.get('Max Drawdown'), 'CVaR 95': m.get('CVaR 95')})
    board = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=.14, subplot_titles=['Sharpe Ranking','Annual Return vs Annual Volatility'])
    fig.add_trace(go.Bar(x=board['Ticker'], y=board['Sharpe'], name='Sharpe'), row=1, col=1)
    fig.add_trace(go.Scatter(x=board['Ann Vol']*100, y=board['Ann Return']*100, text=board['Ticker'], mode='markers+text', textposition='top center', name='Risk/Return'), row=2, col=1)
    shown = board.copy()
    for c in ['Ann Return','Ann Vol','Max DD','CVaR 95']:
        shown[c] = shown[c].map(lambda x: pct(x))
    shown['Sharpe'] = shown['Sharpe'].map(lambda x: num(x))
    return pn.Column(
        pn.pane.Plotly(layout(fig, f'{asset_class} | {region} | Universe Board', 800), config={'responsive': True}, sizing_mode='stretch_width'),
        pn.widgets.Tabulator(shown, height=380, pagination='remote', page_size=12, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

def optimize_weights(ret):
    cols = list(ret.columns)
    if len(cols) < 2:
        return pd.Series(1.0, index=cols)
    mu = ret.mean().values * TRADING_DAYS
    if SKLEARN_AVAILABLE:
        cov = LedoitWolf().fit(ret.fillna(0).values).covariance_ * TRADING_DAYS
    else:
        cov = ret.cov().values * TRADING_DAYS
    n = len(cols)
    def neg_sharpe(w):
        p_ret = float(w @ mu)
        p_vol = float(np.sqrt(w @ cov @ w))
        return -((p_ret - RF) / p_vol) if p_vol > 0 else 1e6
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1},)
    bounds = [(0, .40)] * n
    x0 = np.ones(n) / n
    if SCIPY_AVAILABLE:
        res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
        w = res.x if res.success else x0
    else:
        w = x0
    return pd.Series(np.maximum(w,0), index=cols).pipe(lambda s: s / s.sum())

def optimizer(asset_class, region, start, end, refresh_token):
    px = fetch_matrix(get_tickers(asset_class, region), start, end, refresh_token)
    if px.empty or px.shape[1] < 2:
        return empty('Optimizer unavailable', 'At least two assets with matched Yahoo history are required.')
    ret = px.pct_change().dropna()
    w = optimize_weights(ret).sort_values(ascending=False)
    port = ret[w.index] @ w
    m = metrics(port)
    fig = go.Figure(go.Bar(x=w.index, y=w.values*100, text=[f'{v*100:.1f}%' for v in w.values], textposition='outside', name='Weight %'))
    return pn.Column(
        kpi_cards([('Optimized Ann Return', pct(m.get('Ann Return')), 'green'), ('Optimized Volatility', pct(m.get('Ann Vol')), 'amber'), ('Optimized Sharpe', num(m.get('Sharpe')), 'green'), ('Optimized Max DD', pct(m.get('Max Drawdown')), 'red')]),
        pn.pane.Plotly(layout(fig, f'{asset_class} | {region} | Long-only Max Sharpe Weights', 620), config={'responsive': True}, sizing_mode='stretch_width'),
        html_note('Optimization policy', 'Long-only, max 40% single-asset cap, Ledoit-Wolf covariance when sklearn is available. Falls back safely when optimizer or covariance estimation fails.'),
        sizing_mode='stretch_width'
    )

def stress(asset_class, region, start, end, family, min_severity, refresh_token):
    px = fetch_matrix(get_tickers(asset_class, region), start, end, refresh_token)
    if px.empty:
        return empty('Stress testing unavailable', 'No matched universe data.')
    ret = px.pct_change().dropna()
    vol = ret.std() * math.sqrt(TRADING_DAYS)
    universe = ret.mean(axis=1)
    rows = []
    scenarios = STRESS_SCENARIOS if family == 'All' else STRESS_SCENARIOS[STRESS_SCENARIOS['Family'].eq(family)]
    for _, s in scenarios.iterrows():
        for t in ret.columns:
            beta = ret[t].corr(universe)
            beta = 1.0 if not np.isfinite(beta) else beta
            impact = s['Shock'] * beta
            sev = abs(impact) / max(float(vol[t]), 1e-9) * s['Severity Multiplier']
            if sev >= min_severity:
                rows.append({'Family': s['Family'], 'Scenario': s['Scenario'], 'Ticker': t, 'Name': name_for(t), 'Estimated Impact': impact, 'Volatility': vol[t], 'Severity Score': sev})
    table = pd.DataFrame(rows).sort_values('Severity Score', ascending=False) if rows else pd.DataFrame(columns=['Family','Scenario','Ticker','Name','Estimated Impact','Volatility','Severity Score'])
    worst = table['Estimated Impact'].min() if not table.empty else np.nan
    avg_sev = table['Severity Score'].mean() if not table.empty else np.nan
    cards = [('Worst Impact', pct(worst), 'red'), ('Average Severity', num(avg_sev), 'amber'), ('Scenario Rows', str(len(table)), 'blue'), ('Assets Covered', str(ret.shape[1]), 'green')]
    shown = table.copy()
    if not shown.empty:
        shown['Estimated Impact'] = shown['Estimated Impact'].map(lambda x: pct(x))
        shown['Volatility'] = shown['Volatility'].map(lambda x: pct(x))
        shown['Severity Score'] = shown['Severity Score'].map(lambda x: num(x))
    return pn.Column(kpi_cards(cards), html_note('Stress methodology', 'Deterministic sensitivity stress based on matched Yahoo returns, universe beta and annualized volatility. No synthetic historical price data is generated.'), pn.widgets.Tabulator(shown, height=560, pagination='remote', page_size=20, sizing_mode='stretch_width'), sizing_mode='stretch_width')

def tearsheet(ticker, benchmark_label, start, end, refresh_token):
    if not QUANTSTATS_AVAILABLE:
        return empty('Tearsheet unavailable', 'quantstats is not installed in this environment.')
    b = BENCHMARKS[benchmark_label]
    px = fetch_matrix([ticker, b], start, end, refresh_token)
    if px.empty or ticker not in px or b not in px:
        return empty('Tearsheet unavailable', 'Missing matched asset/benchmark data.')
    ret = px.pct_change().dropna()
    path = OUTPUT_DIR / f'tearsheet_{ticker.replace("^","IDX")}_vs_{b.replace("^","IDX")}.html'
    try:
        qs.reports.html(ret[ticker], benchmark=ret[b], rf=RF, output=str(path), title=f'{ticker} vs {benchmark_label}', compounded=True)
        return pn.Column(html_note('Generated tearsheet', f'Instrument: <b>{ticker}</b> | Benchmark: <b>{benchmark_label}</b> | RF: <b>{RF:.2%}</b>'), pn.pane.HTML(path.read_text(encoding='utf-8', errors='ignore'), height=1050, sizing_mode='stretch_width'), sizing_mode='stretch_width')
    except Exception as e:
        return empty('Tearsheet generation failed', f'{type(e).__name__}: {e}')

# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------
def make_app():
    default_asset = list(UNIVERSE.keys())[0]
    default_region = get_regions(default_asset)[0]
    default_ticker = get_tickers(default_asset, default_region)[0]

    asset_class = pn.widgets.Select(name='Asset Class', options=list(UNIVERSE.keys()), value=default_asset)
    region = pn.widgets.Select(name='Region', options=get_regions(default_asset), value=default_region)
    ticker = pn.widgets.Select(name='Instrument', options=get_tickers(default_asset, default_region), value=default_ticker)
    benchmark = pn.widgets.Select(name='Benchmark', options=list(BENCHMARKS.keys()), value='S&P 500 Index (^GSPC)')
    start = pn.widgets.DatePicker(name='Start Date', value=datetime(2019,1,1))
    end = pn.widgets.DatePicker(name='End Date', value=datetime.now())
    family = pn.widgets.Select(name='Stress Family', options=['All'] + sorted(STRESS_SCENARIOS['Family'].unique().tolist()), value='All')
    min_severity = pn.widgets.FloatSlider(name='Minimum Severity Threshold', start=0, end=3, step=.1, value=0)
    refresh = pn.widgets.IntInput(name='Refresh Token', value=0, visible=False)
    refresh_btn = pn.widgets.Button(name='Refresh Data / Clear Stale Cache', button_type='primary')

    def update_regions(event=None):
        opts = get_regions(asset_class.value)
        region.options = opts
        region.value = opts[0]
    def update_tickers(event=None):
        opts = get_tickers(asset_class.value, region.value)
        ticker.options = opts
        ticker.value = opts[0]
    asset_class.param.watch(lambda e: (update_regions(e), update_tickers(e)), 'value')
    region.param.watch(update_tickers, 'value')
    refresh_btn.on_click(lambda e: setattr(refresh, 'value', refresh.value + 1))

    sidebar = pn.Column(
        pn.pane.HTML('<div style="font-size:26px;font-weight:900;color:#0f172a;line-height:1.08;">QFA Prime<br>Finance Platform</div><div style="font-size:13px;color:#64748b;margin-top:8px;line-height:1.45;">Render-ready institutional hedge fund dashboard. Yahoo-only data pipeline.</div>'),
        pn.pane.Markdown('### Investment Universe'), asset_class, region, ticker,
        pn.pane.Markdown('### Benchmark & Period'), benchmark, start, end,
        pn.pane.Markdown('### Stress Filters'), family, min_severity,
        pn.Spacer(height=8), refresh_btn, refresh,
        html_note('System Status', f'RF: <b>{RF:.2%}</b><br>Max tickers/request: <b>{MAX_TICKERS}</b><br>Ledoit-Wolf: <b>{SKLEARN_AVAILABLE}</b><br>QuantStats: <b>{QUANTSTATS_AVAILABLE}</b><br>TA-Lib: <b>{TALIB_AVAILABLE}</b><br>MC simulations: <b>{MC_SIMULATIONS:,}</b><br>Data policy: <b>Yahoo only; no synthetic fallback.</b>'),
        width=360, sizing_mode='stretch_height', styles={'background':'#f8fafc','padding':'22px','border-right':'1px solid #dbe4ef','overflow-y':'auto'}
    )
    header = pn.pane.HTML(f'<div class="qfa-header"><div class="qfa-title">{APP_TITLE}</div><div class="qfa-subtitle">Reactive portfolio intelligence: KPI scorecard, risk analytics, benchmark-relative tracking error, optimizer, TA-Lib/fallback technicals, advanced VaR/CVaR, VaR/NAV, rolling beta, historical stress testing and tear-sheet generation. Instrument changes recompute all panes through explicit widget dependencies and refresh-token cache invalidation.</div></div>', sizing_mode='stretch_width')

    tabs = pn.Tabs(
        ('Executive Dashboard', pn.bind(executive, ticker, benchmark, start, end, refresh)),
        ('Technical Dashboard', pn.bind(technical_chart, ticker, start, end, refresh)),
        ('Risk Analytics', pn.bind(risk_chart, ticker, start, end, refresh)),
        ('Benchmark Relative', pn.bind(relative_chart, ticker, benchmark, start, end, refresh)),
        ('Advanced Risk Analytics', pn.bind(advanced_risk, ticker, benchmark, start, end, refresh)),
        ('Universe Board', pn.bind(universe_board, asset_class, region, start, end, refresh)),
        ('Optimizer', pn.bind(optimizer, asset_class, region, start, end, refresh)),
        ('Stress Testing', pn.bind(stress, asset_class, region, start, end, family, min_severity, refresh)),
        ('Tearsheet', pn.bind(tearsheet, ticker, benchmark, start, end, refresh)),
        dynamic=True, sizing_mode='stretch_width'
    )
    main = pn.Column(pn.pane.HTML(css(), sizing_mode='stretch_width'), header, tabs, sizing_mode='stretch_width', styles={'padding':'20px','background':'white'})
    return pn.Row(sidebar, main, sizing_mode='stretch_width')

app = make_app()
app.servable(title=APP_TITLE)
