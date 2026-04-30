# QFA Prime Finance Platform V4 Render Ready

This version integrates the advanced TA-Lib and risk analytics requested from the reference code while preserving the stable V3 Render architecture.

## Added in V4

- Optional TA-Lib indicator engine with formula fallback.
- RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, EWMA risk signal.
- Advanced Risk Analytics tab.
- Historical, parametric-normal, and Monte Carlo t-distribution VaR.
- Historical and Monte Carlo CVaR snapshot cards.
- Rolling 95% / 99% VaR backtest panel.
- VaR / 3-month NAV ratio chart.
- Rolling beta vs selected benchmark.
- Historical benchmark drawdown regime table.
- Render-safe Monte Carlo cap via `QFA_MC_SIMULATIONS`.

## Render notes

TA-Lib is optional. The application does not require TA-Lib to deploy. If TA-Lib is unavailable on Render, the app automatically uses the formula fallback and displays `Formula fallback` in the dashboard.

Recommended environment variables:

```text
QFA_MAX_TICKERS=12
QFA_CACHE_TTL_SECONDS=900
QFA_MC_SIMULATIONS=3000
QFA_ADVANCED_RISK_WINDOW=252
QFA_RISK_FREE_RATE=0.045
```

## Run locally

```bash
pip install -r requirements.txt
panel serve app.py --address 0.0.0.0 --port 10000 --allow-websocket-origin="*"
```
