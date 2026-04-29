# QFA Dashboard FinTECH — Render Deployment

Render-ready Panel application with:

- Yahoo Finance original matched data only
- no synthetic price fallback
- selected benchmark-driven QuantStats Tearsheet
- fixed risk-free rate: 4.50%
- TA-Lib if available on the Render image
- PyPortfolioOpt optimizer tab: Max Sharpe, Minimum Volatility, HRP
- S&P 500 benchmark mapped to `^GSPC`; SPY is never forced as benchmark

## Files

```text
app.py
requirements.txt
render.yaml
README.md
```

## Render deployment

1. Upload/push these files to GitHub.
2. In Render, create a **New Web Service** from the repo.
3. Render can detect `render.yaml`, or use manually:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Start command:

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=* --num-procs 1
```

## Important implementation notes

QuantStats Tearsheet is generated inside the Render app, saved under `QFA_Dashboard_FinTECH_reports`, then read back and displayed in the Tearsheet tab.

PyPortfolioOpt uses only matched Yahoo close prices from the selected universe. If a ticker has missing or non-overlapping Yahoo data, it is excluded rather than forward-filled or synthetically generated.

TA-Lib is imported if Render successfully installs it. If TA-Lib is unavailable, the app continues with formula-based technical indicators only. Price data is never replaced or synthesized.
