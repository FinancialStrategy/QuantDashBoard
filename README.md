# QFA V13 Institutional Terminal

Render-ready FastAPI + Plotly institutional quant dashboard.

## Fixed policy

- Benchmark: `^GSPC`
- Risk-free rate: `4.5%`
- Data source: Yahoo Finance daily data only
- No fallback benchmark
- No synthetic data
- No Panel / No Bokeh

## Features

- PyPortfolioOpt strategy engine
- Max Sharpe, Minimum Volatility, Efficient Risk, Equal Weight, Inverse Volatility
- Equal Risk Contribution, Maximum Diversification, HRP, Black-Litterman, Tracking Error Optimal
- User-selected best strategy rule
- Institutional KPI layout
- VaR / CVaR at 95% and 99%
- Historical, Parametric Normal, Monte Carlo with at least 10,000 simulations
- 3-month VaR / NAV historical series
- Rolling Beta vs ^GSPC
- Rolling Sharpe and rolling volatility
- Advanced stress testing with scenario family filters and severity ranking
- QuantStats-style metrics table
- PCA factor diagnostics
- Data QA tables

## Render deploy

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Do not use `panel serve`.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```
