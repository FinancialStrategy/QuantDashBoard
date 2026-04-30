# =============================================================================
# QFA PRIME FINANCE PLATFORM V7 - TA-LIB PRO + RISK CONTRIBUTION RENDER BUILD
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
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Render/Linux font hard-lock: avoid Arial lookup loops from Matplotlib/QuantStats.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Segoe UI', 'Helvetica', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0,
})
plt.rcParams.update(mpl.rcParams)

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
APP_TITLE = 'QFA Prime Finance Platform V7'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
RF = float(os.getenv('QFA_RISK_FREE_RATE', '0.045'))
MIN_OBS = int(os.getenv('QFA_MIN_OBS', '90'))
CACHE_TTL_SECONDS = int(os.getenv('QFA_CACHE_TTL_SECONDS', '900'))
MAX_TICKERS = int(os.getenv('QFA_MAX_TICKERS', '12'))
MC_SIMULATIONS = int(os.getenv('QFA_MC_SIMULATIONS', '2500'))
ADVANCED_RISK_WINDOW = int(os.getenv('QFA_ADVANCED_RISK_WINDOW', '252'))
ENABLE_ROLLING_MONTE_CARLO = os.getenv('QFA_ENABLE_ROLLING_MC', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
FONT_STACK = 'DejaVu Sans, Liberation Sans, Segoe UI, Helvetica, sans-serif'

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
    # QuantStats uses Matplotlib internally; keep Render-safe fonts.
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'sans-serif']})
    plt.rcParams.update(mpl.rcParams)
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
            out['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            out['CCI'] = talib.CCI(high, low, close, timeperiod=20)
            out['Williams R'] = talib.WILLR(high, low, close, timeperiod=14)
            out['MOM'] = talib.MOM(close, timeperiod=10)
            out['OBV'] = talib.OBV(close, out['Volume'].fillna(0).astype(float).values)
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
        up_move = out['High'].diff()
        down_move = -out['Low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr14 = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14).mean() / atr14
        minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14).mean() / atr14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        out['ADX'] = dx.rolling(14).mean()
        typical = (out['High'] + out['Low'] + out['Close']) / 3
        mean_dev = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0, np.nan)
        out['CCI'] = (typical - typical.rolling(20).mean()) / (0.015 * mean_dev)
        out['Williams R'] = -100 * (high14 - out['Close']) / (high14 - low14).replace(0, np.nan)
        out['MOM'] = out['Close'] - out['Close'].shift(10)
        direction = np.sign(out['Close'].diff()).fillna(0)
        out['OBV'] = (direction * out['Volume'].fillna(0)).cumsum()
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
    var99 = r.quantile(0.01)
    cvar99 = r[r <= var99].mean() if len(r[r <= var99]) else np.nan
    ulcer = np.sqrt(np.mean(np.square(np.minimum(dd, 0)))) if len(dd) else np.nan
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if np.isfinite(max_dd) and max_dd < 0 else np.nan
    gains = r[r > 0].sum(); losses = abs(r[r < 0].sum())
    omega = gains / losses if losses > 0 else np.nan
    return {
        'Ann Return': ann_ret,
        'Ann Vol': ann_vol,
        'Sharpe': (ann_ret - RF) / ann_vol if ann_vol else np.nan,
        'Sortino': (ann_ret - RF) / downside if downside else np.nan,
        'Max Drawdown': max_dd,
        'VaR 95': var95,
        'CVaR 95': cvar95,
        'VaR 99': var99,
        'CVaR 99': cvar99,
        'Skew': r.skew(),
        'Kurtosis': r.kurtosis(),
        'Calmar': calmar,
        'Omega': omega,
        'Ulcer Index': ulcer,
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
    # Production safety: rolling Monte Carlo is intentionally disabled by default on Render.
    # It can trigger CPU spikes and WebSocket disconnects because every window fits/simulates
    # a distribution. Default behavior uses a parametric rolling proxy for the line;
    # Monte Carlo is still computed as a latest point-in-time snapshot in KPI cards.
    if not ENABLE_ROLLING_MONTE_CARLO:
        mean = r.rolling(window, min_periods=minp).mean()
        sd = r.rolling(window, min_periods=minp).std()
        z = stats.norm.ppf(1 - confidence) if stats is not None else np.nan
        return mean + sd * z

    tail = r.tail(max(window + 60, 320))
    out = pd.Series(index=r.index, dtype=float)
    vals = []
    step = max(1, int(os.getenv('QFA_ROLLING_MC_STEP', '5')))
    last_val = np.nan
    for i in range(len(tail)):
        x = tail.iloc[max(0, i-window+1):i+1]
        if len(x) >= minp and (i % step == 0 or i == len(tail)-1):
            last_val = monte_carlo_tail(x, confidence, seed=1000 + i)[0]
        vals.append(last_val)
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
    for label, ser, conf in [('Historical 95%', var95_h, .95), ('Parametric 95%', var95_p, .95), ('MC/Proxy 95%', var95_m, .95), ('Historical 99%', var99_h, .99), ('Parametric 99%', var99_p, .99), ('MC/Proxy 99%', var99_m, .99)]:
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
    for ser, name, row in [(var95_h,'Historical 95%',1),(var95_p,'Parametric 95%',1),(var95_m,'MC Rolling Proxy 95%',1),(var99_h,'Historical 99%',2),(var99_p,'Parametric 99%',2),(var99_m,'MC Rolling Proxy 99%',2)]:
        fig1.add_trace(go.Scatter(x=ser.index, y=ser*100, name=name, mode='lines'), row=row, col=1)
    fig1.update_yaxes(title='Daily VaR (%)', row=1, col=1); fig1.update_yaxes(title='Daily VaR (%)', row=2, col=1)
    fig2 = go.Figure()
    for ser, name in [(ratio_h,'Historical'),(ratio_p,'Parametric'),(ratio_m,'MC Rolling Proxy')]:
        fig2.add_trace(go.Scatter(x=ser.index, y=ser, name=name, mode='lines'))
    fig2.update_yaxes(title='VaR / 3M NAV (%)')
    fig3 = go.Figure()
    if not beta.empty:
        fig3.add_trace(go.Scatter(x=beta.index, y=beta, name=f'Rolling Beta vs {benchmark_label}', mode='lines')); fig3.add_hline(y=1.0, line_dash='dash')
    fig3.update_yaxes(title='Beta')
    latest_h_var, latest_h_cvar = historical_var(r, .95), historical_cvar(r, .95)
    latest_p_var = parametric_var(r, .95)
    latest_m_var, latest_m_cvar = monte_carlo_tail(r.tail(window), .95)
    cards = [('Hist VaR 95', pct(latest_h_var), 'red'), ('Hist CVaR 95', pct(latest_h_cvar), 'red'), ('Param VaR 95', pct(latest_p_var), 'amber'), ('MC VaR 95', pct(latest_m_var), 'amber'), ('MC CVaR 95', pct(latest_m_cvar), 'red'), ('Latest Beta', num(beta.dropna().iloc[-1] if beta.dropna().size else np.nan), 'blue'), ('TA Engine', str(asset['Indicator Engine'].iloc[-1]) if 'Indicator Engine' in asset else 'N/A', 'blue'), ('MC Sims', f'{MC_SIMULATIONS:,}', 'green'), ('Rolling MC', 'ON' if ENABLE_ROLLING_MONTE_CARLO else 'OFF / Proxy', 'blue')]
    return pn.Column(html_note('Advanced risk methodology', f'Historical, parametric-normal and Monte Carlo t-distribution VaR/CVaR. Rolling window: <b>{window}</b> trading days. Monte Carlo is point-in-time by default for Render stability; rolling MC lines use a parametric proxy unless QFA_ENABLE_ROLLING_MC=1.'), kpi_cards(cards), pn.pane.Plotly(layout(fig1, f'{ticker} | Rolling VaR Backtest', 820), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Plotly(layout(fig2, f'{ticker} | VaR / 3-Month NAV Ratio', 560), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Plotly(layout(fig3, f'{ticker} | Rolling Beta vs {benchmark_label}', 500), config={'responsive': True}, sizing_mode='stretch_width'), pn.pane.Markdown('### VaR Violation Backtest'), pn.widgets.Tabulator(shown_bt, height=260, sizing_mode='stretch_width'), pn.pane.Markdown('### Historical Benchmark Drawdown Regimes'), pn.widgets.Tabulator(shown_stress if not shown_stress.empty else pd.DataFrame([{'Message': 'No benchmark drawdown regime below -20% in selected period.'}]), height=300, sizing_mode='stretch_width'), sizing_mode='stretch_width')

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


def risk_metric_dashboard(ticker, benchmark_label, start, end, refresh_token):
    df = add_features(fetch_one(ticker, start, end, refresh_token))
    bench = add_features(fetch_one(BENCHMARKS[benchmark_label], start, end, refresh_token))
    if df.empty:
        return empty('Risk metrics unavailable', f'Yahoo returned no usable data for {ticker}.')
    r = df['Return'].dropna()
    m = metrics(r)
    a = active_metrics(df['Return'], bench['Return']) if not bench.empty else {}
    cards = [
        ('VaR 95', pct(m.get('VaR 95')), 'red'), ('CVaR 95', pct(m.get('CVaR 95')), 'red'),
        ('VaR 99', pct(m.get('VaR 99')), 'red'), ('CVaR 99', pct(m.get('CVaR 99')), 'red'),
        ('Skewness', num(m.get('Skew')), 'amber'), ('Kurtosis', num(m.get('Kurtosis')), 'amber'),
        ('Calmar', num(m.get('Calmar')), 'blue'), ('Omega', num(m.get('Omega')), 'blue'),
        ('Ulcer Index', pct(m.get('Ulcer Index')), 'red'), ('Hit Ratio', pct(m.get('Hit Ratio')), 'green'),
        ('Tracking Error', pct(a.get('Tracking Error')), 'blue'), ('Information Ratio', num(a.get('Information Ratio')), 'blue'),
    ]
    table = pd.DataFrame([{
        'Metric': k,
        'Value': pct(v) if k in ['Ann Return','Ann Vol','Max Drawdown','VaR 95','CVaR 95','VaR 99','CVaR 99','Hit Ratio','Ulcer Index'] else num(v)
    } for k, v in m.items()])
    var_levels = pd.Series({'VaR 95': m.get('VaR 95'), 'CVaR 95': m.get('CVaR 95'), 'VaR 99': m.get('VaR 99'), 'CVaR 99': m.get('CVaR 99')})
    fig = make_subplots(rows=2, cols=1, vertical_spacing=.14, subplot_titles=['Return Distribution with Tail-Risk Markers','Core Tail-Risk Levels'])
    fig.add_trace(go.Histogram(x=r*100, nbinsx=80, name='Daily Returns %'), row=1, col=1)
    for label, value in var_levels.items():
        if np.isfinite(value):
            fig.add_vline(x=value*100, line_dash='dash', annotation_text=label, row=1, col=1)
    fig.add_trace(go.Bar(x=var_levels.index, y=var_levels.values*100, name='Risk Level %'), row=2, col=1)
    fig.update_yaxes(title='Frequency', row=1, col=1)
    fig.update_yaxes(title='Daily return loss threshold (%)', row=2, col=1)
    return pn.Column(
        html_note('Professional RiskMetrics panel', 'RiskMetrics now includes VaR/CVaR 95/99, skewness, kurtosis, Calmar, Omega, Ulcer Index, hit ratio, tracking error and information ratio. All values are computed from original Yahoo close-to-close returns.'),
        kpi_cards(cards),
        pn.pane.Plotly(layout(fig, f'{ticker} | Professional RiskMetrics', 780), config={'responsive': True}, sizing_mode='stretch_width'),
        pn.widgets.Tabulator(table, height=360, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )

def ta_pro_dashboard(ticker, start, end, refresh_token):
    df = add_features(fetch_one(ticker, start, end, refresh_token))
    if df.empty:
        return empty('TA-Lib PRO unavailable', f'Yahoo returned no usable OHLCV data for {ticker}.')
    engine = str(df['Indicator Engine'].iloc[-1]) if 'Indicator Engine' in df else 'N/A'
    latest = df.dropna().iloc[-1] if len(df.dropna()) else df.iloc[-1]
    cards = [
        ('Indicator Engine', engine, 'green' if engine == 'TA-Lib' else 'amber'),
        ('RSI 14', num(latest.get('RSI')), 'amber'),
        ('ADX 14', num(latest.get('ADX')), 'blue'),
        ('CCI 20', num(latest.get('CCI')), 'blue'),
        ('Williams %R', num(latest.get('Williams R')), 'amber'),
        ('Momentum 10', num(latest.get('MOM')), 'blue'),
        ('ATR 14', num(latest.get('ATR')), 'red'),
        ('EWMA Risk Signal', num(latest.get('EWMA Risk Signal')), 'red' if latest.get('EWMA Risk Signal', 0) > 1.15 else 'green'),
    ]
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=.035,
        row_heights=[.30,.14,.14,.14,.14,.14],
        subplot_titles=['Close + Bollinger Bands','ADX Trend Strength','CCI Mean-Reversion','Williams %R','OBV Volume Flow','Momentum + ATR']
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', mode='lines'), row=1, col=1)
    for col in ['BB Upper','BB Mid','BB Lower']:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', mode='lines'), row=2, col=1)
    fig.add_hline(y=25, row=2, col=1, line_dash='dash')
    fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], name='CCI', mode='lines'), row=3, col=1)
    fig.add_hline(y=100, row=3, col=1, line_dash='dash'); fig.add_hline(y=-100, row=3, col=1, line_dash='dash')
    fig.add_trace(go.Scatter(x=df.index, y=df['Williams R'], name='Williams %R', mode='lines'), row=4, col=1)
    fig.add_hline(y=-20, row=4, col=1, line_dash='dash'); fig.add_hline(y=-80, row=4, col=1, line_dash='dash')
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', mode='lines'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MOM'], name='Momentum 10', mode='lines'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR 14', mode='lines'), row=6, col=1)
    return pn.Column(
        html_note('TA-Lib PRO signal layer', 'Adds ADX, CCI, Williams %R, OBV, Momentum, ATR and EWMA Risk Signal. TA-Lib is used when installed; otherwise formulas are computed from real Yahoo OHLCV data only. No synthetic price data is created.'),
        kpi_cards(cards),
        pn.pane.Plotly(layout(fig, f'{ticker} | TA-Lib PRO Signal Dashboard', 1120), config={'responsive': True}, sizing_mode='stretch_width'),
        sizing_mode='stretch_width'
    )


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


def _covariance_matrix(ret: pd.DataFrame) -> np.ndarray:
    """Annualized covariance estimated only from matched Yahoo return history."""
    clean = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.eye(ret.shape[1]) * 1e-6
    if SKLEARN_AVAILABLE and len(clean) >= max(30, ret.shape[1] * 3):
        cov = LedoitWolf().fit(clean.values).covariance_ * TRADING_DAYS
    else:
        cov = clean.cov().values * TRADING_DAYS
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-8:
        cov += np.eye(cov.shape[0]) * (1e-8 - eigvals.min())
    return cov

def _portfolio_stats(weights, mu, cov):
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0)))
    sharpe = (ret - RF) / vol if vol > 0 else np.nan
    asset_vol = np.sqrt(np.maximum(np.diag(cov), 0))
    div = float(np.dot(w, asset_vol) / vol) if vol > 0 else np.nan
    return ret, vol, sharpe, div

def _solve_strategy(objective, n, bounds=None, x0=None, max_weight=0.40):
    bounds = bounds or [(0.0, max_weight)] * n
    x0 = np.ones(n) / n if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    x0 = x0 / x0.sum() if x0.sum() > 0 else np.ones(n) / n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    if not SCIPY_AVAILABLE:
        return x0, False, 'scipy unavailable'
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 800, 'ftol': 1e-10})
    if not getattr(res, 'success', False) or not np.all(np.isfinite(res.x)):
        return x0, False, str(getattr(res, 'message', 'optimizer did not converge'))
    w = np.maximum(res.x, 0)
    w = w / w.sum() if w.sum() > 0 else x0
    return w, True, 'ok'

def _hrp_like_weights(ret: pd.DataFrame, cov: np.ndarray) -> np.ndarray:
    """Render-stable risk allocation using inverse-vol clustering proxy from real Yahoo returns."""
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    inv_vol = 1 / vols
    w = inv_vol / inv_vol.sum()
    return np.asarray(w, dtype=float)

def build_strategy_set(ret: pd.DataFrame):
    clean = ret.replace([np.inf, -np.inf], np.nan).dropna()
    cols = list(clean.columns)
    n = len(cols)
    if n < 2 or clean.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    mu = clean.mean().values * TRADING_DAYS
    cov = _covariance_matrix(clean)
    max_weight = 0.40 if n >= 3 else 1.0
    bounds = [(0.0, max_weight)] * n
    if n * max_weight < 1.0:
        bounds = [(0.0, 1.0)] * n

    strategies = {}

    def add_strategy(name, weights, ok=True, note='ok'):
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
        r, v, sh, div = _portfolio_stats(weights, mu, cov)
        port_ret = clean[cols] @ pd.Series(weights, index=cols)
        m = metrics(port_ret)
        strategies[name] = {
            'weights': pd.Series(weights, index=cols),
            'return': r,
            'vol': v,
            'sharpe': sh,
            'div': div,
            'max_dd': m.get('Max Drawdown'),
            'cvar95': m.get('CVaR 95'),
            'ok': ok,
            'note': note,
        }

    def neg_sharpe(w):
        sh = _portfolio_stats(w, mu, cov)[2]
        return -sh if np.isfinite(sh) else 1e6
    w, ok, note = _solve_strategy(neg_sharpe, n, bounds=bounds, max_weight=max_weight)
    add_strategy('Max Sharpe', w, ok, note)

    def port_vol(w):
        return _portfolio_stats(w, mu, cov)[1]
    w, ok, note = _solve_strategy(port_vol, n, bounds=bounds, max_weight=max_weight)
    add_strategy('Min Volatility', w, ok, note)

    def neg_div_ratio(w):
        div = _portfolio_stats(w, mu, cov)[3]
        return -div if np.isfinite(div) else 1e6
    w, ok, note = _solve_strategy(neg_div_ratio, n, bounds=bounds, max_weight=max_weight)
    add_strategy('Max Diversification', w, ok, note)

    target_vol = 0.15
    def target_risk_objective(w):
        r, v, sh, _ = _portfolio_stats(w, mu, cov)
        return (v - target_vol) ** 2 - 0.02 * r
    w, ok, note = _solve_strategy(target_risk_objective, n, bounds=bounds, max_weight=max_weight)
    add_strategy('Efficient Risk 15% Vol', w, ok, note)

    w = _hrp_like_weights(clean, cov)
    if max_weight < 1.0 and w.max() > max_weight:
        w = np.minimum(w, max_weight)
        remainder = 1 - w.sum()
        free = w < max_weight - 1e-9
        if remainder > 0 and free.any():
            w[free] += remainder * w[free] / w[free].sum()
        w = w / w.sum()
    add_strategy('Hierarchical Risk Parity', w, True, 'HRP-style inverse-vol risk allocation from matched Yahoo returns')

    add_strategy('Equal Weight', np.ones(n) / n, True, '1/N benchmark allocation')

    summary_rows = []
    for name, st in strategies.items():
        summary_rows.append({
            'Strategy': name,
            'Expected Annual Return': st['return'],
            'Annual Volatility': st['vol'],
            'Sharpe Ratio': st['sharpe'],
            'Diversification Ratio': st['div'],
            'Backtested Max DD': st['max_dd'],
            'Backtested CVaR 95': st['cvar95'],
        })
    summary = pd.DataFrame(summary_rows).sort_values('Sharpe Ratio', ascending=False)
    weights = pd.DataFrame({name: st['weights'] for name, st in strategies.items()}).fillna(0)
    return strategies, summary, weights

def optimizer(asset_class, region, start, end, refresh_token):
    px = fetch_matrix(get_tickers(asset_class, region), start, end, refresh_token)
    if px.empty or px.shape[1] < 2:
        return empty('Optimizer unavailable', 'At least two assets with matched original Yahoo history are required.')
    ret = px.pct_change().dropna()
    strategies, summary, weights = build_strategy_set(ret)
    if not strategies or summary.empty:
        return empty('Optimizer unavailable', 'Matched Yahoo return history was insufficient for portfolio construction.')

    best_name = str(summary.iloc[0]['Strategy'])
    best = strategies[best_name]
    cards = [
        ('Best Strategy', best_name, 'blue'),
        ('Best Sharpe', num(best['sharpe']), 'green' if np.isfinite(best['sharpe']) and best['sharpe'] > 1 else 'amber'),
        ('Best Ann Return', pct(best['return']), 'green' if np.isfinite(best['return']) and best['return'] > 0 else 'red'),
        ('Best Volatility', pct(best['vol']), 'amber'),
        ('Best Max DD', pct(best['max_dd']), 'red'),
        ('Strategies Tested', str(len(strategies)), 'blue'),
        ('Assets Used', str(ret.shape[1]), 'green'),
        ('Data Source', 'Yahoo only', 'green'),
    ]

    shown = summary.copy()
    for c in ['Expected Annual Return', 'Annual Volatility', 'Backtested Max DD', 'Backtested CVaR 95']:
        shown[c] = shown[c].map(lambda x: pct(x))
    for c in ['Sharpe Ratio', 'Diversification Ratio']:
        shown[c] = shown[c].map(lambda x: num(x))

    fig_cmp = make_subplots(rows=2, cols=1, vertical_spacing=.14, subplot_titles=['Strategy Sharpe Comparison', 'Strategy Risk / Return Map'])
    fig_cmp.add_trace(go.Bar(x=summary['Strategy'], y=summary['Sharpe Ratio'], name='Sharpe'), row=1, col=1)
    fig_cmp.add_trace(go.Scatter(x=summary['Annual Volatility'] * 100, y=summary['Expected Annual Return'] * 100, mode='markers+text', text=summary['Strategy'], textposition='top center', name='Risk/Return'), row=2, col=1)
    fig_cmp.update_yaxes(title='Sharpe', row=1, col=1)
    fig_cmp.update_xaxes(title='Annual Volatility (%)', row=2, col=1)
    fig_cmp.update_yaxes(title='Expected Annual Return (%)', row=2, col=1)

    descriptions = {
        'Max Sharpe': 'Tangency portfolio: maximizes excess return per unit of risk, using the configured 4.5% risk-free rate.',
        'Min Volatility': 'Global minimum variance portfolio: minimizes annualized volatility with long-only constraints.',
        'Max Diversification': 'Maximizes diversification ratio: weighted average asset volatility divided by total portfolio volatility.',
        'Efficient Risk 15% Vol': 'Targets a 15% annual volatility profile while preserving return preference when the exact target is not feasible.',
        'Hierarchical Risk Parity': 'Risk-parity style allocation designed to be robust to covariance estimation noise; built only from matched Yahoo returns.',
        'Equal Weight': '1/N allocation used as a transparent benchmark for all optimized strategies.',
    }

    allocation_blocks = []
    for name in summary['Strategy'].tolist():
        w = weights[name].sort_values(ascending=False)
        w = w[w > 1e-4]
        fig = go.Figure(go.Bar(x=w.index, y=w.values * 100, text=[f'{v*100:.1f}%' for v in w.values], textposition='outside', name='Weight %'))
        fig.update_yaxes(title='Weight (%)')
        allocation_blocks.append(
            pn.Column(
                html_note(name, descriptions.get(name, 'Institutional allocation strategy built from matched Yahoo close-to-close returns.')),
                pn.pane.Plotly(layout(fig, f'{name} | Allocation', 430), config={'responsive': True}, sizing_mode='stretch_width'),
                sizing_mode='stretch_width'
            )
        )

    return pn.Column(
        kpi_cards(cards),
        html_note('Optimizer methodology', 'Institutional multi-strategy construction using matched original Yahoo close prices only. No fallback price data and no synthetic historical data are created. Covariance is Ledoit-Wolf when available; otherwise the sample covariance from the same Yahoo return matrix is used.'),
        pn.widgets.Tabulator(shown, height=235, sizing_mode='stretch_width'),
        pn.pane.Plotly(layout(fig_cmp, f'{asset_class} | {region} | Portfolio Strategy Comparison', 760), config={'responsive': True}, sizing_mode='stretch_width'),
        pn.pane.Markdown('### Detailed Allocation Rationale'),
        *allocation_blocks,
        sizing_mode='stretch_width'
    )


def risk_contribution(asset_class, region, start, end, refresh_token):
    px = fetch_matrix(get_tickers(asset_class, region), start, end, refresh_token)
    if px.empty or px.shape[1] < 2:
        return empty('Risk contribution unavailable', 'At least two assets with matched original Yahoo history are required.')
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return empty('Risk contribution unavailable', 'Matched Yahoo return matrix is empty after cleaning.')
    strategies, summary, weights = build_strategy_set(ret)
    if weights.empty:
        return empty('Risk contribution unavailable', 'Portfolio weights could not be computed from Yahoo returns.')
    selected_strategy = str(summary.iloc[0]['Strategy']) if not summary.empty else weights.columns[0]
    cov = _covariance_matrix(ret[weights.index])
    rows = []
    for strategy in weights.columns:
        w = weights[strategy].reindex(ret.columns).fillna(0).values
        port_var = float(w @ cov @ w)
        port_vol = math.sqrt(max(port_var, 0))
        marginal = cov @ w
        pct_contrib = (w * marginal / port_var) if port_var > 0 else np.zeros_like(w)
        vol_contrib = pct_contrib * port_vol
        for ticker, wi, pc, vc in zip(ret.columns, w, pct_contrib, vol_contrib):
            if wi > 1e-6 or abs(pc) > 1e-6:
                rows.append({
                    'Strategy': strategy,
                    'Ticker': ticker,
                    'Name': name_for(ticker),
                    'Weight': wi,
                    'Risk Contribution %': pc,
                    'Vol Contribution': vc,
                    'Marginal Risk': marginal[list(ret.columns).index(ticker)],
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return empty('Risk contribution unavailable', 'No non-zero contribution rows were produced.')
    best_table = table[table['Strategy'].eq(selected_strategy)].copy().sort_values('Risk Contribution %', ascending=False)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=.14, subplot_titles=[f'{selected_strategy} | Risk Contribution %', f'{selected_strategy} | Weight vs Risk Contribution'])
    fig.add_trace(go.Bar(x=best_table['Ticker'], y=best_table['Risk Contribution %']*100, name='Risk Contribution %', text=[f'{x*100:.1f}%' for x in best_table['Risk Contribution %']], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(x=best_table['Ticker'], y=best_table['Weight']*100, name='Weight %'), row=2, col=1)
    fig.add_trace(go.Scatter(x=best_table['Ticker'], y=best_table['Risk Contribution %']*100, name='Risk Contribution %', mode='lines+markers'), row=2, col=1)
    fig.update_yaxes(title='Contribution (%)', row=1, col=1)
    fig.update_yaxes(title='Weight / Contribution (%)', row=2, col=1)
    shown = table.copy()
    for c in ['Weight', 'Risk Contribution %', 'Vol Contribution', 'Marginal Risk']:
        shown[c] = shown[c].map(lambda x: pct(x) if c != 'Marginal Risk' else num(x, 6))
    cards = [
        ('Selected Strategy', selected_strategy, 'blue'),
        ('Largest Risk Contributor', str(best_table.iloc[0]['Ticker']), 'red'),
        ('Largest Risk Share', pct(best_table.iloc[0]['Risk Contribution %']), 'red'),
        ('Assets Included', str(ret.shape[1]), 'green'),
        ('Covariance', 'Ledoit-Wolf' if SKLEARN_AVAILABLE else 'Sample', 'blue'),
        ('Data Source', 'Yahoo only', 'green'),
    ]
    return pn.Column(
        html_note('Risk contribution methodology', 'Risk contribution is computed from the annualized covariance matrix of matched Yahoo returns. Contribution_i = w_i × (Σw)_i / portfolio variance. No fallback, substitute or synthetic price history is used.'),
        kpi_cards(cards),
        pn.pane.Plotly(layout(fig, f'{asset_class} | {region} | Portfolio Risk Contribution', 780), config={'responsive': True}, sizing_mode='stretch_width'),
        pn.widgets.Tabulator(shown, height=520, pagination='remote', page_size=20, sizing_mode='stretch_width'),
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
        html_note('System Status', f'RF: <b>{RF:.2%}</b><br>Max tickers/request: <b>{MAX_TICKERS}</b><br>Ledoit-Wolf: <b>{SKLEARN_AVAILABLE}</b><br>QuantStats: <b>{QUANTSTATS_AVAILABLE}</b><br>TA-Lib: <b>{TALIB_AVAILABLE}</b><br>MC simulations: <b>{MC_SIMULATIONS:,}</b><br>Rolling MC: <b>{'ON' if ENABLE_ROLLING_MONTE_CARLO else 'OFF / Proxy'}</b><br>Data policy: <b>Yahoo only; no synthetic or substitute price data.</b>'),
        width=360, sizing_mode='stretch_height', styles={'background':'#f8fafc','padding':'22px','border-right':'1px solid #dbe4ef','overflow-y':'auto'}
    )
    header = pn.pane.HTML(f'<div class="qfa-header"><div class="qfa-title">{APP_TITLE}</div><div class="qfa-subtitle">Reactive portfolio intelligence: KPI scorecard, risk analytics, benchmark-relative tracking error, optimizer, TA-Lib PRO/formula technicals, Professional RiskMetrics, risk contribution, advanced VaR/CVaR, VaR/NAV, rolling beta, advanced historical stress testing and tear-sheet generation. Instrument changes recompute all panes through explicit widget dependencies and refresh-token cache invalidation.</div></div>', sizing_mode='stretch_width')

    tabs = pn.Tabs(
        ('Executive Dashboard', pn.bind(executive, ticker, benchmark, start, end, refresh)),
        ('Technical Dashboard', pn.bind(technical_chart, ticker, start, end, refresh)),
        ('TA-Lib PRO Signals', pn.bind(ta_pro_dashboard, ticker, start, end, refresh)),
        ('Risk Analytics', pn.bind(risk_chart, ticker, start, end, refresh)),
        ('Professional RiskMetrics', pn.bind(risk_metric_dashboard, ticker, benchmark, start, end, refresh)),
        ('Benchmark Relative', pn.bind(relative_chart, ticker, benchmark, start, end, refresh)),
        ('Advanced Risk Analytics', pn.bind(advanced_risk, ticker, benchmark, start, end, refresh)),
        ('Universe Board', pn.bind(universe_board, asset_class, region, start, end, refresh)),
        ('Optimizer', pn.bind(optimizer, asset_class, region, start, end, refresh)),
        ('Risk Contribution', pn.bind(risk_contribution, asset_class, region, start, end, refresh)),
        ('Stress Testing', pn.bind(stress, asset_class, region, start, end, family, min_severity, refresh)),
        ('Tearsheet', pn.bind(tearsheet, ticker, benchmark, start, end, refresh)),
        dynamic=True, sizing_mode='stretch_width'
    )
    main = pn.Column(pn.pane.HTML(css(), sizing_mode='stretch_width'), header, tabs, sizing_mode='stretch_width', styles={'padding':'20px','background':'white'})
    return pn.Row(sidebar, main, sizing_mode='stretch_width')

app = make_app()
app.servable(title=APP_TITLE)
