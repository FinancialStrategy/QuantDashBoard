# -*- coding: utf-8 -*-
"""
QFA V13 Institutional Terminal
Render-ready FastAPI + Plotly dashboard.

Core policy:
- NO Panel / NO Bokeh
- Benchmark fixed to ^GSPC
- Risk-free rate fixed to 4.5%
- Yahoo Finance only
- No synthetic data
- No benchmark fallback
- If Yahoo data is insufficient, fail transparently

Author: MK FinTECH LabGEN@2026
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from scipy import stats
from scipy.optimize import minimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

try:
    from pypfopt import expected_returns, risk_models, objective_functions
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion
    PYPFOPT_AVAILABLE = True
except Exception:
    expected_returns = None
    risk_models = None
    objective_functions = None
    EfficientFrontier = None
    HRPOpt = None
    BlackLittermanModel = None
    market_implied_risk_aversion = None
    PYPFOPT_AVAILABLE = False

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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
CACHE_DIR = Path(os.getenv("QFA_CACHE_DIR", "/tmp/qfa_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "QFA V13 Institutional Terminal"
BENCHMARK_SYMBOL = "^GSPC"
BENCHMARK_LABEL = "S&P 500 Index (^GSPC)"
RISK_FREE_RATE = 0.045
TRADING_DAYS = 252
MIN_OBS = 252
DEFAULT_MC_SIMULATIONS = 10_000
CACHE_TTL_SECONDS = int(os.getenv("QFA_CACHE_TTL", "900"))
MAX_TICKERS = int(os.getenv("QFA_MAX_TICKERS", "18"))

ETF_UNIVERSE: Dict[str, List[str]] = {
    "US Broad Equity": ["SPY", "IVV", "VOO", "VTI", "DIA", "IWM", "MDY", "RSP"],
    "US Growth / Value": ["QQQ", "VUG", "IWF", "VTV", "IWD", "SCHG", "SCHV"],
    "US Sectors": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLC", "XLRE"],
    "International Developed": ["VEA", "IEFA", "EFA", "VGK", "EWJ", "EWG", "EWU", "EWC"],
    "Emerging Markets": ["VWO", "IEMG", "EEM", "EWZ", "INDA", "FXI", "MCHI", "EWT", "EIDO", "EZA"],
    "Fixed Income": ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "BIL", "SGOV"],
    "Real Assets": ["GLD", "IAU", "SLV", "DBC", "VNQ", "REET", "GSG", "PDBC"],
    "Alternatives / Factors": ["MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SCHD", "BTAL", "QAI", "MNA"],
    "Crypto Proxies": ["IBIT", "FBTC", "BITO", "GBTC", "ETHE"],
}

STRESS_SCENARIOS = [
    {"family": "crisis", "name": "COVID Crash", "start": "2020-02-19", "end": "2020-03-23"},
    {"family": "inflation", "name": "2022 Inflation Shock", "start": "2022-01-03", "end": "2022-10-14"},
    {"family": "banking stress", "name": "2023 Banking Stress", "start": "2023-03-08", "end": "2023-03-31"},
    {"family": "sharp rally", "name": "2024 Q1 Rally", "start": "2024-01-02", "end": "2024-03-28"},
    {"family": "sharp selloff", "name": "2024 April Selloff", "start": "2024-04-01", "end": "2024-04-19"},
    {"family": "crisis", "name": "Volmageddon 2018", "start": "2018-01-26", "end": "2018-02-09"},
    {"family": "crisis", "name": "Q4 2018 Risk-Off", "start": "2018-10-03", "end": "2018-12-24"},
]


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        if pd.isna(obj) and not isinstance(obj, (list, tuple, dict, pd.Series, pd.DataFrame, np.ndarray)):
            return None
    except Exception:
        pass
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(row) for row in obj.replace([np.inf, -np.inf], np.nan).to_dict("records")]
    if isinstance(obj, pd.Series):
        return [to_jsonable({"date": idx, "value": val}) for idx, val in obj.replace([np.inf, -np.inf], np.nan).items()]
    if isinstance(obj, pd.Index):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def pct(x: float) -> float:
    return float(x) if x is not None and np.isfinite(x) else np.nan


class ComputeRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=3)
    start_date: str = Field("2019-01-01")
    end_date: Optional[str] = None
    initial_capital: float = Field(10_000_000, gt=0)
    expected_return_method: str = Field("ema_historical")
    covariance_method: str = Field("ledoit_wolf")
    best_strategy_rule: str = Field("balanced_score")
    max_weight: float = Field(0.20, ge=0.01, le=1.0)
    max_category_weight: float = Field(0.45, ge=0.05, le=1.0)
    tracking_error_target: float = Field(0.06, ge=0.005, le=1.0)
    mc_simulations: int = Field(DEFAULT_MC_SIMULATIONS, ge=10000, le=50000)
    rolling_window: int = Field(63, ge=20, le=252)
    stress_family: str = Field("All")
    min_severity: float = Field(0.0, ge=0.0, le=10.0)

    @validator("tickers", pre=True)
    def clean_tickers(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [x.strip() for x in value.replace(";", ",").split(",")]
        out: List[str] = []
        for t in value or []:
            s = str(t).strip().upper()
            if s and s != BENCHMARK_SYMBOL and s not in out:
                out.append(s)
        if len(out) < 3:
            raise ValueError("Select at least 3 unique instruments.")
        if len(out) > MAX_TICKERS:
            raise ValueError(f"Render-safe limit is {MAX_TICKERS} tickers. Reduce selected instruments.")
        return out

    @validator("start_date")
    def clean_start(cls, value: str) -> str:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            raise ValueError("Invalid start_date.")
        return dt.strftime("%Y-%m-%d")

    @validator("end_date")
    def clean_end(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            raise ValueError("Invalid end_date.")
        return dt.strftime("%Y-%m-%d")

    @validator("expected_return_method")
    def clean_exp(cls, value: str) -> str:
        allowed = {"historical_mean", "ema_historical", "capm"}
        v = value.strip().lower()
        if v not in allowed:
            raise ValueError(f"expected_return_method must be one of {sorted(allowed)}")
        return v

    @validator("covariance_method")
    def clean_cov(cls, value: str) -> str:
        allowed = {"ledoit_wolf", "sample", "shrinkage"}
        v = value.strip().lower()
        if v not in allowed:
            raise ValueError(f"covariance_method must be one of {sorted(allowed)}")
        return v


@dataclass
class StrategyResult:
    name: str
    weights: Dict[str, float]
    description: str
    diagnostics: Dict[str, Any]


def category_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cat, tickers in ETF_UNIVERSE.items():
        for t in tickers:
            out[t] = cat
    return out


@lru_cache(maxsize=64)
def fetch_prices_cached(tickers_key: str, start_date: str, end_date: str, bucket: int) -> str:
    tickers = tickers_key.split(",")
    data = yf.download(
        tickers,
        start=start_date,
        end=None if end_date == "" else end_date,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
        group_by="column",
    )
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no data. No fallback or synthetic data is allowed.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            raise ValueError("Yahoo response has no Close/Adj Close field.")
    else:
        if "Close" in data.columns and len(tickers) == 1:
            prices = data[["Close"]].copy()
            prices.columns = tickers
        elif "Adj Close" in data.columns and len(tickers) == 1:
            prices = data[["Adj Close"]].copy()
            prices.columns = tickers
        else:
            raise ValueError("Yahoo response shape not recognized.")

    prices.index = pd.to_datetime(prices.index).tz_localize(None) if getattr(prices.index, "tz", None) is not None else pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(axis=1, how="all")
    return prices.to_json(date_format="iso")


def load_yahoo_prices(tickers: List[str], start_date: str, end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK_SYMBOL]))
    bucket = int(time.time() // CACHE_TTL_SECONDS)
    text = fetch_prices_cached(",".join(all_tickers), start_date, end_date or "", bucket)
    prices = pd.read_json(text)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    missing = [t for t in all_tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Yahoo did not return required instruments: {missing}. No fallback allowed.")

    valid_ratio = prices.notna().mean()
    too_sparse = valid_ratio[valid_ratio < 0.80].index.tolist()
    if too_sparse:
        raise ValueError(f"Insufficient Yahoo daily history for {too_sparse}. No fallback allowed.")

    prices = prices[all_tickers].ffill(limit=3).dropna()
    if len(prices) < MIN_OBS:
        raise ValueError(f"Only {len(prices)} daily observations after strict alignment. Need at least {MIN_OBS}.")

    # Strict daily audit: reject lower frequency samples
    gaps = prices.index.to_series().diff().dt.days.dropna()
    median_gap = float(gaps.median()) if len(gaps) else 1.0
    if median_gap > 3.5:
        raise ValueError("Input series does not look like Yahoo daily data. No lower-frequency data allowed.")

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if BENCHMARK_SYMBOL not in returns.columns:
        raise ValueError("Benchmark ^GSPC is missing. No benchmark fallback allowed.")

    asset_cols = [t for t in tickers if t in returns.columns]
    if len(asset_cols) < 3:
        raise ValueError("Fewer than 3 valid asset return series after Yahoo cleanup.")

    return prices[asset_cols + [BENCHMARK_SYMBOL]], returns[asset_cols + [BENCHMARK_SYMBOL]]


def build_expected_returns(asset_prices: pd.DataFrame, benchmark_prices: pd.Series, method: str) -> pd.Series:
    if not PYPFOPT_AVAILABLE:
        # PyPortfolioOpt is required for this platform; fail explicitly.
        raise ValueError("PyPortfolioOpt is not installed. Install requirements.txt. No optimizer fallback used.")
    if method == "historical_mean":
        mu = expected_returns.mean_historical_return(asset_prices, frequency=TRADING_DAYS)
    elif method == "capm":
        mu = expected_returns.capm_return(
            asset_prices,
            market_prices=benchmark_prices.to_frame("benchmark"),
            risk_free_rate=RISK_FREE_RATE,
            frequency=TRADING_DAYS,
        )
    else:
        mu = expected_returns.ema_historical_return(asset_prices, frequency=TRADING_DAYS)
    return mu.replace([np.inf, -np.inf], np.nan).dropna()


def build_covariance(asset_prices: pd.DataFrame, method: str) -> pd.DataFrame:
    if not PYPFOPT_AVAILABLE:
        raise ValueError("PyPortfolioOpt is not installed. Install requirements.txt. No covariance fallback used.")
    if method == "sample":
        cov = risk_models.sample_cov(asset_prices, frequency=TRADING_DAYS)
    elif method == "shrinkage":
        cov = risk_models.CovarianceShrinkage(asset_prices, frequency=TRADING_DAYS).shrunk_covariance(0.20)
    else:
        cov = risk_models.CovarianceShrinkage(asset_prices, frequency=TRADING_DAYS).ledoit_wolf()
    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eigvals, eigvecs = np.linalg.eigh(cov.values)
    if eigvals.min() <= 1e-10:
        eigvals = np.maximum(eigvals, 1e-8)
        cov = pd.DataFrame(eigvecs @ np.diag(eigvals) @ eigvecs.T, index=cov.index, columns=cov.columns)
    return cov


def normalize_weights(weights: Dict[str, float], assets: List[str]) -> Dict[str, float]:
    s = pd.Series(weights, dtype=float).reindex(assets).fillna(0.0).clip(lower=0.0)
    total = float(s.sum())
    if total <= 0:
        raise ValueError("Weights sum to zero.")
    s = s / total
    return {k: float(v) for k, v in s.items() if v > 1e-8}


def add_category_constraints(ef: EfficientFrontier, assets: List[str], max_category_weight: float) -> None:
    cmap = category_map()
    groups: Dict[str, List[int]] = {}
    for i, asset in enumerate(assets):
        groups.setdefault(cmap.get(asset, "Other"), []).append(i)
    for _, idxs in groups.items():
        ef.add_constraint(lambda w, idxs=idxs: max_category_weight - sum(w[i] for i in idxs))


def make_ef(mu: pd.Series, cov: pd.DataFrame, max_weight: float, max_category_weight: float) -> EfficientFrontier:
    assets = list(mu.index)
    ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, max_weight))
    ef.add_objective(objective_functions.L2_reg, gamma=0.01)
    add_category_constraints(ef, assets, max_category_weight)
    return ef


def portfolio_returns(asset_returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float).reindex(asset_returns.columns).fillna(0.0)
    w = w / w.sum()
    return asset_returns.mul(w, axis=1).sum(axis=1).rename("portfolio")


def strategy_equal_weight(assets: List[str]) -> Dict[str, float]:
    return {a: 1.0 / len(assets) for a in assets}


def strategy_inverse_vol(asset_returns: pd.DataFrame) -> Dict[str, float]:
    vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
    inv = 1 / vol.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
    inv = inv / inv.sum()
    return {k: float(v) for k, v in inv.items()}


def strategy_max_sharpe(mu: pd.Series, cov: pd.DataFrame, max_weight: float, max_category_weight: float) -> Dict[str, float]:
    ef = make_ef(mu, cov, max_weight, max_category_weight)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    return normalize_weights(ef.clean_weights(), list(mu.index))


def strategy_min_vol(mu: pd.Series, cov: pd.DataFrame, max_weight: float, max_category_weight: float) -> Dict[str, float]:
    ef = make_ef(mu, cov, max_weight, max_category_weight)
    ef.min_volatility()
    return normalize_weights(ef.clean_weights(), list(mu.index))


def strategy_efficient_risk(mu: pd.Series, cov: pd.DataFrame, max_weight: float, max_category_weight: float, target_vol: float = 0.15) -> Dict[str, float]:
    ef = make_ef(mu, cov, max_weight, max_category_weight)
    try:
        ef.efficient_risk(target_volatility=target_vol)
    except Exception:
        ef.min_volatility()
    return normalize_weights(ef.clean_weights(), list(mu.index))


def strategy_erc(cov: pd.DataFrame, max_weight: float) -> Dict[str, float]:
    assets = list(cov.index)
    n = len(assets)
    S = cov.values
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        port_var = float(w @ S @ w)
        if port_var <= 0:
            return 1e6
        mrc = S @ w / np.sqrt(port_var)
        rc = w * mrc
        target = np.mean(rc)
        return float(np.sum((rc - target) ** 2))

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        raise ValueError(res.message)
    return normalize_weights({a: float(w) for a, w in zip(assets, res.x)}, assets)


def strategy_max_diversification(cov: pd.DataFrame, max_weight: float) -> Dict[str, float]:
    assets = list(cov.index)
    n = len(assets)
    S = cov.values
    vols = np.sqrt(np.diag(S))
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        port_vol = float(np.sqrt(max(w @ S @ w, 0.0)))
        if port_vol <= 0:
            return 1e6
        return -float((w @ vols) / port_vol)

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        raise ValueError(res.message)
    return normalize_weights({a: float(w) for a, w in zip(assets, res.x)}, assets)


def strategy_hrp(asset_returns: pd.DataFrame) -> Dict[str, float]:
    if HRPOpt is not None:
        hrp = HRPOpt(returns=asset_returns)
        w = hrp.optimize(linkage_method="single")
        return normalize_weights(w, list(asset_returns.columns))
    corr = asset_returns.corr().clip(-1, 1)
    dist = np.sqrt((1 - corr) / 2)
    link = hierarchy.linkage(squareform(dist.values, checks=False), method="ward")
    # fallback HRP-style equal over ordered assets
    order = hierarchy.leaves_list(link)
    ordered = corr.index[order].tolist()
    return {a: 1.0 / len(ordered) for a in ordered}


def strategy_black_litterman(mu: pd.Series, cov: pd.DataFrame, asset_returns: pd.DataFrame, benchmark_returns: pd.Series, max_weight: float, max_category_weight: float) -> Dict[str, float]:
    if BlackLittermanModel is None:
        raise ValueError("BlackLittermanModel unavailable.")
    bench_price_like = (1 + benchmark_returns).cumprod().to_frame("benchmark")
    try:
        delta = market_implied_risk_aversion(bench_price_like, frequency=TRADING_DAYS)
    except Exception:
        delta = 2.5
    w_mkt = pd.Series(1.0 / len(mu), index=mu.index)
    pi = delta * (cov @ w_mkt)
    views: Dict[str, float] = {}
    for candidate, floor in [("GLD", 0.03), ("QQQ", 0.06), ("XLK", 0.06), ("TLT", 0.025)]:
        if candidate in mu.index:
            views[candidate] = float(max(mu.loc[candidate], floor))
    bl = BlackLittermanModel(cov, pi=pi, absolute_views=views if views else None, tau=0.05)
    ef = make_ef(bl.bl_returns(), bl.bl_cov(), max_weight, max_category_weight)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    return normalize_weights(ef.clean_weights(), list(mu.index))


def strategy_tracking_error(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, mu: pd.Series, max_weight: float, target_te: float) -> Dict[str, float]:
    assets = list(asset_returns.columns)
    n = len(assets)
    R = asset_returns[assets].values
    b = benchmark_returns.reindex(asset_returns.index).values
    x0 = np.repeat(1.0 / n, n)
    if "SPY" in assets:
        x0 = np.zeros(n); x0[assets.index("SPY")] = 1.0
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        p = R @ w
        te = float(np.std(p - b) * np.sqrt(TRADING_DAYS))
        ret = float(w @ mu.reindex(assets).fillna(0.0).values)
        penalty = 100.0 * max(te - target_te, 0.0) ** 2
        return -ret + penalty

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        raise ValueError(res.message)
    return normalize_weights({a: float(w) for a, w in zip(assets, res.x)}, assets)


def run_strategies(asset_prices: pd.DataFrame, asset_returns: pd.DataFrame, benchmark_returns: pd.Series, req: ComputeRequest) -> Tuple[Dict[str, StrategyResult], pd.Series, pd.DataFrame]:
    benchmark_prices = (1 + benchmark_returns).cumprod()
    mu = build_expected_returns(asset_prices, benchmark_prices, req.expected_return_method)
    cov = build_covariance(asset_prices[mu.index], req.covariance_method)
    common = list(mu.index.intersection(cov.index).intersection(asset_returns.columns))
    if len(common) < 3:
        raise ValueError("Too few common assets for optimizer.")
    mu = mu.loc[common]
    cov = cov.loc[common, common]
    asset_returns = asset_returns[common]
    strategies: Dict[str, Tuple[Any, str]] = {
        "Max Sharpe": (lambda: strategy_max_sharpe(mu, cov, req.max_weight, req.max_category_weight), "PyPortfolioOpt tangency portfolio using RF 4.5%."),
        "Minimum Volatility": (lambda: strategy_min_vol(mu, cov, req.max_weight, req.max_category_weight), "Global minimum-variance portfolio."),
        "Efficient Risk 15%": (lambda: strategy_efficient_risk(mu, cov, req.max_weight, req.max_category_weight, 0.15), "Efficient portfolio targeting 15% annual volatility where feasible."),
        "Equal Weight": (lambda: strategy_equal_weight(common), "Naive equal-weight institutional benchmark."),
        "Inverse Volatility": (lambda: strategy_inverse_vol(asset_returns), "Allocates inversely to realized volatility."),
        "Equal Risk Contribution": (lambda: strategy_erc(cov, req.max_weight), "Risk parity / equal contribution allocation."),
        "Maximum Diversification": (lambda: strategy_max_diversification(cov, req.max_weight), "Maximizes diversification ratio."),
        "Hierarchical Risk Parity": (lambda: strategy_hrp(asset_returns), "Cluster-based HRP allocation."),
        "Black-Litterman": (lambda: strategy_black_litterman(mu, cov, asset_returns, benchmark_returns, req.max_weight, req.max_category_weight), "Black-Litterman posterior return optimizer with simple institutional views."),
        "Tracking Error Optimal": (lambda: strategy_tracking_error(asset_returns, benchmark_returns, mu, req.max_weight, req.tracking_error_target), "Active portfolio constrained by tracking error target."),
    }
    results: Dict[str, StrategyResult] = {}
    for name, (fn, desc) in strategies.items():
        try:
            w = fn()
            w = normalize_weights(w, common)
            results[name] = StrategyResult(name=name, weights=w, description=desc, diagnostics={"status": "ok"})
        except Exception as exc:
            results[name] = StrategyResult(name=name, weights={}, description=desc, diagnostics={"status": "failed", "error": str(exc)})
    ok = {k: v for k, v in results.items() if v.weights}
    if not ok:
        raise ValueError("All portfolio strategies failed. Check constraints, max weights, and Yahoo data.")
    return results, mu, cov


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return (wealth / wealth.cummax() - 1).rename("drawdown")


def ulcer_index(dd: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(dd.clip(upper=0.0).values))))


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([asset.rename("p"), benchmark.rename("b")], axis=1).dropna()
    cov = aligned["p"].rolling(window).cov(aligned["b"])
    var = aligned["b"].rolling(window).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).dropna().rename("rolling_beta")


def rolling_sharpe(asset: pd.Series, window: int) -> pd.Series:
    sr = ((asset.rolling(window).mean() - RISK_FREE_RATE / TRADING_DAYS) / asset.rolling(window).std()) * np.sqrt(TRADING_DAYS)
    return sr.replace([np.inf, -np.inf], np.nan).dropna().rename("rolling_sharpe")


def rolling_vol(asset: pd.Series, window: int) -> pd.Series:
    return (asset.rolling(window).std() * np.sqrt(TRADING_DAYS)).replace([np.inf, -np.inf], np.nan).dropna().rename("rolling_vol")


def metrics_for_returns(r: pd.Series, benchmark: pd.Series, initial_capital: float) -> Dict[str, Any]:
    aligned = pd.concat([r.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
    p = aligned["portfolio"]
    b = aligned["benchmark"]
    years = len(p) / TRADING_DAYS
    wealth = (1 + p).cumprod()
    bench_wealth = (1 + b).cumprod()
    total_return = float(wealth.iloc[-1] - 1)
    ann_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else np.nan
    ann_vol = float(p.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((ann_return - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else np.nan
    downside = p[p < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = float((ann_return - RISK_FREE_RATE) / downside) if downside and downside > 0 else np.nan
    dd = drawdown_series(p)
    max_dd = float(dd.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan
    beta = float(p.cov(b) / b.var()) if b.var() > 0 else np.nan
    bench_total_return = float(bench_wealth.iloc[-1] - 1)
    bench_ann_return = float((1 + bench_total_return) ** (1 / years) - 1) if years > 0 else np.nan
    alpha = float(ann_return - (RISK_FREE_RATE + beta * (bench_ann_return - RISK_FREE_RATE))) if np.isfinite(beta) else np.nan
    active = p - b
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    ir = float((active.mean() * TRADING_DAYS) / te) if te > 0 else np.nan
    skew = float(p.skew())
    kurt = float(p.kurtosis())
    var95 = -float(np.percentile(p, 5))
    cvar95 = -float(p[p <= -var95].mean())
    return {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "alpha": alpha,
        "beta": beta,
        "tracking_error": te,
        "information_ratio": ir,
        "win_rate": float((p > 0).mean()),
        "win_rate_vs_benchmark": float((p > b).mean()),
        "skewness": skew,
        "kurtosis": kurt,
        "ulcer_index": ulcer_index(dd),
        "var_95_historical": var95,
        "cvar_95_historical": cvar95,
        "final_value": float(initial_capital * wealth.iloc[-1]),
        "benchmark_final_value": float(initial_capital * bench_wealth.iloc[-1]),
        "total_return": total_return,
        "benchmark_total_return": bench_total_return,
    }


def historical_var_cvar(r: pd.Series, cl: float) -> Tuple[float, float]:
    q = np.percentile(r, (1 - cl) * 100)
    tail = r[r <= q]
    return -float(q), -float(tail.mean()) if len(tail) else np.nan


def parametric_var_cvar(r: pd.Series, cl: float) -> Tuple[float, float]:
    mu = float(r.mean())
    sigma = float(r.std())
    z = stats.norm.ppf(1 - cl)
    var = -(mu + sigma * z)
    cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - cl))
    return float(var), float(cvar)


def monte_carlo_var_cvar(r: pd.Series, cl: float, n_sim: int, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed + int(cl * 1000))
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma <= 0:
        return 0.0, 0.0
    sims = rng.normal(mu, sigma, size=int(n_sim))
    q = np.percentile(sims, (1 - cl) * 100)
    tail = sims[sims <= q]
    return -float(q), -float(tail.mean()) if len(tail) else np.nan


def var_cvar_table(r: pd.Series, initial_capital: float, mc_simulations: int) -> List[Dict[str, Any]]:
    rows = []
    methods = {
        "Historical": historical_var_cvar,
        "Parametric Normal": parametric_var_cvar,
    }
    for method_name, fn in methods.items():
        for cl in [0.95, 0.99]:
            var, cvar = fn(r, cl)
            rows.append({
                "Method": method_name,
                "Confidence": f"{int(cl * 100)}%",
                "VaR Daily %": var,
                "CVaR Daily %": cvar,
                "VaR USD": var * initial_capital,
                "CVaR USD": cvar * initial_capital,
                "Simulations": None,
                "Interpretation": "Historical tail" if method_name == "Historical" else "Normal parametric estimate",
            })
    for cl in [0.95, 0.99]:
        var, cvar = monte_carlo_var_cvar(r, cl, mc_simulations)
        rows.append({
            "Method": "Monte Carlo Normal",
            "Confidence": f"{int(cl * 100)}%",
            "VaR Daily %": var,
            "CVaR Daily %": cvar,
            "VaR USD": var * initial_capital,
            "CVaR USD": cvar * initial_capital,
            "Simulations": int(mc_simulations),
            "Interpretation": f"{mc_simulations:,} simulations",
        })
    return rows


def rolling_var_nav_series(r: pd.Series, initial_capital: float, window: int = 63) -> pd.DataFrame:
    nav = initial_capital * (1 + r).cumprod()
    roll_var = r.rolling(window).quantile(0.05).abs()
    dollar_var = nav * roll_var
    ratio = dollar_var / nav
    out = pd.DataFrame({
        "Date": ratio.index,
        "Rolling 3M VaR/NAV": ratio.values,
        "Rolling 3M VaR USD": dollar_var.values,
        "NAV": nav.values,
    }).dropna()
    return out


def risk_contributions(weights: Dict[str, float], cov: pd.DataFrame) -> List[Dict[str, Any]]:
    assets = list(weights.keys())
    w = pd.Series(weights, dtype=float).reindex(assets).fillna(0.0).values
    S = cov.reindex(index=assets, columns=assets).fillna(0.0).values
    port_var = float(w @ S @ w)
    if port_var <= 0:
        return []
    port_vol = math.sqrt(port_var)
    mrc = S @ w / port_vol
    trc = w * mrc
    pct_contrib = trc / trc.sum() if trc.sum() else trc
    rows = []
    for a, weight, marginal, total, pctc in zip(assets, w, mrc, trc, pct_contrib):
        rows.append({"Asset": a, "Weight": float(weight), "Marginal Risk": float(marginal), "Total Risk Contribution": float(total), "Contribution %": float(pctc)})
    return sorted(rows, key=lambda x: x["Contribution %"], reverse=True)


def score_table(strategy_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(strategy_metrics).T
    if df.empty:
        return df
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    def z_pos(s):
        s = pd.to_numeric(s, errors="coerce")
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0
    def z_neg(s):
        return -z_pos(s)
    df["balanced_score"] = (
        0.23 * z_pos(df["sharpe_ratio"]) +
        0.18 * z_pos(df["sortino_ratio"]) +
        0.16 * z_pos(df["calmar_ratio"]) +
        0.16 * z_pos(df["information_ratio"]) +
        0.12 * z_pos(df["annual_return"]) +
        0.10 * z_neg(df["max_drawdown"].abs()) +
        0.05 * z_neg(df["var_95_historical"])
    )
    return df


def select_best_strategy(metrics_df: pd.DataFrame, rule: str) -> str:
    rule = (rule or "balanced_score").strip().lower()
    mapping = {
        "max_sharpe": ("sharpe_ratio", "max"),
        "min_vol": ("annual_volatility", "min"),
        "max_return": ("annual_return", "max"),
        "max_sortino": ("sortino_ratio", "max"),
        "max_calmar": ("calmar_ratio", "max"),
        "min_drawdown": ("max_drawdown", "max"),  # less negative is max
        "max_information_ratio": ("information_ratio", "max"),
        "lowest_tracking_error": ("tracking_error", "min"),
        "balanced_score": ("balanced_score", "max"),
    }
    col, direction = mapping.get(rule, ("balanced_score", "max"))
    s = pd.to_numeric(metrics_df[col], errors="coerce")
    if direction == "min":
        return str(s.idxmin())
    return str(s.idxmax())


def advanced_stress_tests(portfolio_returns_series: pd.Series, benchmark_returns: pd.Series, family_filter: str, min_severity: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    pr = portfolio_returns_series.copy()
    br = benchmark_returns.reindex(pr.index)
    for sc in STRESS_SCENARIOS:
        if family_filter != "All" and sc["family"] != family_filter:
            continue
        s = pd.Timestamp(sc["start"])
        e = pd.Timestamp(sc["end"])
        mask = (pr.index >= s) & (pr.index <= e)
        p = pr.loc[mask]
        b = br.loc[mask].dropna()
        p = p.reindex(b.index).dropna()
        if len(p) < 5:
            continue
        p_return = float((1 + p).prod() - 1)
        b_return = float((1 + b).prod() - 1)
        rel = p_return - b_return
        dd = float(drawdown_series(p).min())
        vol = float(p.std() * np.sqrt(TRADING_DAYS))
        severity = abs(min(p_return, rel, dd)) / max(vol, 1e-9)
        if severity < min_severity:
            continue
        rows.append({
            "Family": sc["family"],
            "Scenario": sc["name"],
            "Start": sc["start"],
            "End": sc["end"],
            "Portfolio Return": p_return,
            "Benchmark Return": b_return,
            "Relative Return": rel,
            "Worst Drawdown": dd,
            "Annualized Volatility": vol,
            "Severity Score": float(severity),
            "Trading Days": int(len(p)),
        })
    rows = sorted(rows, key=lambda x: x["Severity Score"], reverse=True)
    if rows:
        worst = min(rows, key=lambda x: x["Relative Return"])
        kpis = {
            "Worst Scenario": worst["Scenario"],
            "Worst Relative Return": worst["Relative Return"],
            "Worst Drawdown": min(x["Worst Drawdown"] for x in rows),
            "Average Severity": float(np.mean([x["Severity Score"] for x in rows])),
            "Scenario Count": len(rows),
        }
    else:
        kpis = {"Worst Scenario": "N/A", "Worst Relative Return": None, "Worst Drawdown": None, "Average Severity": None, "Scenario Count": 0}
    return rows, kpis


def pca_analysis(asset_returns: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    X = asset_returns.dropna()
    if X.shape[1] < 2:
        return [], []
    corr = X.corr().fillna(0.0).values
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained = eigvals / eigvals.sum()
    var_rows = [{"Component": f"PC{i+1}", "Explained Variance": float(v)} for i, v in enumerate(explained[: min(6, len(explained))])]
    loading_rows = []
    assets = X.columns.tolist()
    for asset_i, asset in enumerate(assets):
        row = {"Asset": asset}
        for j in range(min(4, eigvecs.shape[1])):
            row[f"PC{j+1}"] = float(eigvecs[asset_i, j])
        loading_rows.append(row)
    return var_rows, loading_rows


def quantstats_metrics(port: pd.Series, benchmark: pd.Series) -> Dict[str, Any]:
    # Internal reliable metrics first. QuantStats availability is reported separately.
    m = metrics_for_returns(port, benchmark, 1.0)
    return {
        "QuantStats Available": QUANTSTATS_AVAILABLE,
        "CAGR": m["annual_return"],
        "Volatility": m["annual_volatility"],
        "Sharpe": m["sharpe_ratio"],
        "Sortino": m["sortino_ratio"],
        "Calmar": m["calmar_ratio"],
        "Max Drawdown": m["max_drawdown"],
        "Ulcer Index": m["ulcer_index"],
        "Skew": m["skewness"],
        "Kurtosis": m["kurtosis"],
    }


def instrument_metadata(tickers: List[str]) -> List[Dict[str, str]]:
    cmap = category_map()
    return [{"Ticker": t, "Category": cmap.get(t, "Custom / Yahoo"), "Data Source": "Yahoo Finance", "Benchmark": BENCHMARK_SYMBOL if t == BENCHMARK_SYMBOL else ""} for t in tickers]


def preview_records(df: pd.DataFrame, n: int = 12) -> List[Dict[str, Any]]:
    out = df.tail(n).copy()
    out.insert(0, "Date", out.index)
    return to_jsonable(out)


app = FastAPI(title=APP_TITLE)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    path = TEMPLATES_DIR / "index.html"
    return path.read_text(encoding="utf-8")


@app.get("/api/universe")
def api_universe():
    return {"benchmark": BENCHMARK_SYMBOL, "risk_free_rate": RISK_FREE_RATE, "universe": ETF_UNIVERSE, "max_tickers": MAX_TICKERS}


@app.get("/api/health")
def api_health():
    return {"status": "ok", "app": APP_TITLE, "benchmark": BENCHMARK_SYMBOL, "rf": RISK_FREE_RATE, "pypfopt": PYPFOPT_AVAILABLE, "quantstats": QUANTSTATS_AVAILABLE, "talib": TALIB_AVAILABLE}


@app.post("/api/compute")
def api_compute(req: ComputeRequest):
    start = time.time()
    try:
        prices, returns = load_yahoo_prices(req.tickers, req.start_date, req.end_date)
        asset_cols = [t for t in req.tickers if t in returns.columns]
        asset_prices = prices[asset_cols]
        asset_returns = returns[asset_cols]
        benchmark_returns = returns[BENCHMARK_SYMBOL]
        strategy_results, mu, cov = run_strategies(asset_prices, asset_returns, benchmark_returns, req)

        strategy_metrics: Dict[str, Dict[str, Any]] = {}
        strategy_returns: Dict[str, pd.Series] = {}
        for name, sr in strategy_results.items():
            if not sr.weights:
                continue
            pr = portfolio_returns(asset_returns, sr.weights)
            strategy_returns[name] = pr
            strategy_metrics[name] = metrics_for_returns(pr, benchmark_returns, req.initial_capital)

        metrics_df = score_table(strategy_metrics)
        best_name = select_best_strategy(metrics_df, req.best_strategy_rule)
        best_weights = strategy_results[best_name].weights
        best_returns = strategy_returns[best_name]
        best_metrics = strategy_metrics[best_name]
        best_var_table = var_cvar_table(best_returns, req.initial_capital, req.mc_simulations)
        rolling_var_nav = rolling_var_nav_series(best_returns, req.initial_capital, req.rolling_window)
        rb = rolling_beta(best_returns, benchmark_returns, req.rolling_window)
        rs = rolling_sharpe(best_returns, req.rolling_window)
        rv = rolling_vol(best_returns, req.rolling_window)
        dd = drawdown_series(best_returns)
        wealth = req.initial_capital * (1 + best_returns).cumprod()
        bench_wealth = req.initial_capital * (1 + benchmark_returns.reindex(best_returns.index)).cumprod()
        stress_rows, stress_kpis = advanced_stress_tests(best_returns, benchmark_returns, req.stress_family, req.min_severity)
        pca_var, pca_load = pca_analysis(asset_returns)
        rc = risk_contributions(best_weights, cov)

        strategy_table = []
        for name, row in metrics_df.iterrows():
            result = strategy_results.get(name)
            strategy_table.append({
                "Strategy": name,
                "Status": result.diagnostics.get("status", "ok") if result else "ok",
                "Description": result.description if result else "",
                "Annual Return": row.get("annual_return"),
                "Volatility": row.get("annual_volatility"),
                "Sharpe": row.get("sharpe_ratio"),
                "Sortino": row.get("sortino_ratio"),
                "Calmar": row.get("calmar_ratio"),
                "Max Drawdown": row.get("max_drawdown"),
                "Tracking Error": row.get("tracking_error"),
                "Information Ratio": row.get("information_ratio"),
                "VaR 95 Hist": row.get("var_95_historical"),
                "CVaR 95 Hist": row.get("cvar_95_historical"),
                "Balanced Score": row.get("balanced_score"),
            })

        failed_strategies = [
            {"Strategy": name, "Error": sr.diagnostics.get("error", "")}
            for name, sr in strategy_results.items() if not sr.weights
        ]

        response = {
            "meta": {
                "app": APP_TITLE,
                "benchmark": BENCHMARK_SYMBOL,
                "benchmark_label": BENCHMARK_LABEL,
                "risk_free_rate": RISK_FREE_RATE,
                "data_policy": "Yahoo Finance daily only. No fallback. No synthetic data.",
                "selected_tickers": asset_cols,
                "observations": int(len(best_returns)),
                "start": str(best_returns.index.min().date()),
                "end": str(best_returns.index.max().date()),
                "compute_seconds": round(time.time() - start, 3),
                "pypfopt_available": PYPFOPT_AVAILABLE,
                "quantstats_available": QUANTSTATS_AVAILABLE,
                "talib_available": TALIB_AVAILABLE,
                "expected_return_method": req.expected_return_method,
                "covariance_method": req.covariance_method,
                "best_strategy_rule": req.best_strategy_rule,
                "best_strategy": best_name,
                "mc_simulations": req.mc_simulations,
            },
            "kpis": {
                "Best Strategy": best_name,
                "Annual Return": best_metrics["annual_return"],
                "Annual Volatility": best_metrics["annual_volatility"],
                "Sharpe": best_metrics["sharpe_ratio"],
                "Sortino": best_metrics["sortino_ratio"],
                "Max Drawdown": best_metrics["max_drawdown"],
                "Information Ratio": best_metrics["information_ratio"],
                "VaR 95 Hist": best_metrics["var_95_historical"],
                "CVaR 95 Hist": best_metrics["cvar_95_historical"],
                "Final Value": best_metrics["final_value"],
                "Latest 3M VaR/NAV": float(rolling_var_nav["Rolling 3M VaR/NAV"].iloc[-1]) if not rolling_var_nav.empty else None,
            },
            "weights": [{"Asset": k, "Weight": v, "Category": category_map().get(k, "Custom / Yahoo")} for k, v in sorted(best_weights.items(), key=lambda x: x[1], reverse=True)],
            "strategy_table": strategy_table,
            "failed_strategies": failed_strategies,
            "var_cvar_table": best_var_table,
            "performance_metrics_table": [{"Metric": k, "Value": v} for k, v in best_metrics.items() if not isinstance(v, (pd.Series, pd.DataFrame))],
            "risk_contribution_table": rc,
            "stress_table": stress_rows,
            "stress_kpis": stress_kpis,
            "pca_variance": pca_var,
            "pca_loadings": pca_load,
            "quantstats_metrics": [{"Metric": k, "Value": v} for k, v in quantstats_metrics(best_returns, benchmark_returns).items()],
            "metadata_table": instrument_metadata(asset_cols + [BENCHMARK_SYMBOL]),
            "data_quality_table": [
                {"Ticker": c, "Valid Ratio": float(prices[c].notna().mean()), "Observations": int(prices[c].notna().sum()), "First Date": str(prices[c].dropna().index.min().date()), "Last Date": str(prices[c].dropna().index.max().date())}
                for c in prices.columns
            ],
            "prices_preview": preview_records(prices[asset_cols + [BENCHMARK_SYMBOL]], 10),
            "returns_preview": preview_records(returns[asset_cols + [BENCHMARK_SYMBOL]], 10),
            "series": {
                "portfolio_equity": [{"Date": idx, "Value": val} for idx, val in wealth.items()],
                "benchmark_equity": [{"Date": idx, "Value": val} for idx, val in bench_wealth.reindex(wealth.index).dropna().items()],
                "drawdown": [{"Date": idx, "Value": val} for idx, val in dd.items()],
                "rolling_beta": [{"Date": idx, "Value": val} for idx, val in rb.items()],
                "rolling_sharpe": [{"Date": idx, "Value": val} for idx, val in rs.items()],
                "rolling_volatility": [{"Date": idx, "Value": val} for idx, val in rv.items()],
                "rolling_var_nav": rolling_var_nav.to_dict("records"),
                "daily_returns": [{"Date": idx, "Portfolio": p, "Benchmark": b} for idx, p, b in zip(best_returns.index, best_returns.values, benchmark_returns.reindex(best_returns.index).values)],
            },
        }
        return JSONResponse(content=to_jsonable(response))
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "trace": traceback.format_exc(limit=3), "policy": "Yahoo-only strict mode: no fallback or synthetic data was used."})

