# QFA Prime Finance Platform V7

Render-ready institutional hedge fund dashboard.

## V7 Additions

- TA-Lib PRO Signals tab
  - ADX, CCI, Williams %R, OBV, Momentum, ATR, EWMA Risk Signal.
  - Uses TA-Lib when available; otherwise formula calculations are computed from original Yahoo OHLCV data.
- Professional RiskMetrics tab
  - VaR 95/99, CVaR 95/99, skewness, kurtosis, Calmar, Omega, Ulcer Index, hit ratio, tracking error, information ratio.
- Advanced Risk Analytics
  - Historical / parametric / Monte Carlo VaR-CVaR snapshot, rolling VaR, VaR/NAV, rolling beta, violation backtest, historical drawdown regimes.
- Risk Contribution tab
  - Portfolio risk contribution by strategy using the covariance matrix of matched Yahoo returns.
- Multi-strategy optimizer
  - Max Sharpe, Min Volatility, Max Diversification, Efficient Risk 15% Vol, HRP-style, Equal Weight.
- Advanced Stress Testing
  - Crisis, inflation, banking stress, selloff, rally scenario families with severity filters and KPI cards.

## Data Policy

No synthetic price data, no substitute price series, no fallback asset prices. All market calculations use original Yahoo Finance OHLCV / close data only. If Yahoo returns no usable data, the app shows a clear warning and does not fabricate data.

## Render

Build command:

```bash
pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

Start command:

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=${RENDER_EXTERNAL_HOSTNAME} --num-procs 1
```

TA-Lib remains optional because Render free Linux builds often fail without the native C library. V7 still gives institutional technical indicators through formula calculations from real Yahoo OHLCV data when TA-Lib is not installed.
