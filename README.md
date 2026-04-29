# QFA Dashboard FinTECH — Render WebSocket Fixed

This package is configured for a live Panel/Bokeh deployment on Render.

## Critical Render fix

The application must be served as a live Panel server app, not as static HTML.

`render.yaml` uses:

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=${RENDER_EXTERNAL_HOSTNAME} --num-procs 1
```

This is required because Panel widgets update charts through Bokeh WebSocket callbacks. If the WebSocket origin is wrong, the page loads but dropdowns do not trigger Python recomputation.

## Data policy

- Yahoo Finance original market data only.
- No synthetic fallback prices.
- Missing/unmatched tickers are excluded or shown as unavailable.
- TA-Lib is used if available. If not available, only technical indicator formulas fall back; price data never falls back.
- Risk-free rate is fixed at 4.5%.
- S&P 500 benchmark uses `^GSPC`, not SPY.

## Included files

- `app.py` — Panel application
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deployment config with WebSocket origin fix
- `README.md` — deployment notes

## Local run

```bash
pip install -r requirements.txt
panel serve app.py --show --allow-websocket-origin=localhost:5006
```

## Render deployment

Upload this folder or push it to GitHub, then create a Render Web Service using the included `render.yaml`.


## KPI reactive fix
This version uses a persistent `self.kpi_pane` plus explicit watchers on Instrument, Asset Class, Region, Benchmark, Start Date, and End Date. The KPI layout is refreshed immediately when the Instrument selector changes.
