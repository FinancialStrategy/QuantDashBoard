from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import pandas as pd
import yfinance as yf

from pypfopt import EfficientFrontier, expected_returns, risk_models

# =========================
# CONFIG
# =========================
RF = 0.045
BENCHMARK = "^GSPC"

app = FastAPI()

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =========================
# HOME PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =========================
# DATA LOADER (STRICT)
# =========================
def load_data(tickers):
    data = yf.download(tickers + [BENCHMARK], period="5y")["Adj Close"]

    if data.isnull().all().any():
        raise HTTPException(status_code=400, detail="Yahoo data missing")

    returns = data.pct_change().dropna()

    return returns[tickers], returns[BENCHMARK]


# =========================
# OPTIMIZATION
# =========================
def optimize(returns):

    prices = (1 + returns).cumprod()

    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=RF)

    weights = ef.clean_weights()

    return weights


# =========================
# METRICS
# =========================
def compute_metrics(port, bench):

    ann_ret = (1 + port.mean())**252 - 1
    vol = port.std() * np.sqrt(252)
    sharpe = (ann_ret - RF) / vol if vol > 0 else 0

    beta = np.cov(port, bench)[0][1] / np.var(bench)

    cum = (1 + port).cumprod()
    dd = cum / cum.cummax() - 1

    return {
        "return": float(ann_ret),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "beta": float(beta),
        "max_dd": float(dd.min())
    }


# =========================
# VAR / CVAR ENGINE
# =========================
def var_engine(r):

    out = {}

    for cl in [0.95, 0.99]:

        # Historical
        var_h = -np.percentile(r, (1 - cl) * 100)
        cvar_h = -r[r <= -var_h].mean()

        # Parametric
        mu = r.mean()
        sigma = r.std()

        z = abs(np.percentile(np.random.normal(0,1,100000), (1-cl)*100))

        var_p = -(mu - z * sigma)
        cvar_p = -(mu - sigma * (np.exp(-z**2/2) / (np.sqrt(2*np.pi)*(1-cl))))

        # Monte Carlo
        sims = np.random.normal(mu, sigma, 10000)

        var_mc = -np.percentile(sims, (1-cl)*100)
        cvar_mc = -sims[sims <= -var_mc].mean()

        out[f"var_{int(cl*100)}_hist"] = float(var_h)
        out[f"cvar_{int(cl*100)}_hist"] = float(cvar_h)

        out[f"var_{int(cl*100)}_param"] = float(var_p)
        out[f"cvar_{int(cl*100)}_param"] = float(cvar_p)

        out[f"var_{int(cl*100)}_mc"] = float(var_mc)
        out[f"cvar_{int(cl*100)}_mc"] = float(cvar_mc)

    return out


# =========================
# ROLLING ANALYTICS
# =========================
def rolling_beta(port, bench):
    return (
        port.rolling(63).cov(bench) /
        bench.rolling(63).var()
    ).dropna().tolist()


def var_nav_ratio(port):
    var = port.rolling(63).quantile(0.05)
    nav = (1 + port).cumprod()

    return (var / nav).dropna().tolist()


# =========================
# API
# =========================
@app.post("/api/compute")
async def compute(request: Request):

    body = await request.json()

    tickers = body.get("tickers", ["SPY", "QQQ", "IWM", "TLT"])

    if len(tickers) < 2:
        raise HTTPException(400, "At least 2 tickers required")

    # Load data
    returns, bench = load_data(tickers)

    # Optimize
    weights = optimize(returns)

    # Portfolio returns
    port = returns.mul(pd.Series(weights), axis=1).sum(axis=1)

    # Metrics
    metrics = compute_metrics(port, bench)

    # VaR
    var = var_engine(port)

    return {
        "weights": weights,
        "metrics": metrics,
        "var": var,
        "rolling_beta": rolling_beta(port, bench),
        "var_nav": var_nav_ratio(port)
    }
