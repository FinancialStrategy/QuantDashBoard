# QFA Dashboard FinTECH - Render Rebuilt V2

Render-ready Panel application with fully reactive widget architecture.

## Key fixes

- Uses `pn.widgets.Select` as the single source of state.
- Uses `pn.bind(...)` for every tab, so Instrument and Benchmark changes trigger recomputation.
- No synthetic/fallback price data. Yahoo Finance matched OHLCV data only.
- S&P 500 benchmark uses `^GSPC`, not SPY.
- Risk-free rate is fixed at 4.5%.
- QuantStats tab is named **Tearsheet**.
- PyPortfolioOpt optimizer tab included.
- TA-Lib is used if installed; otherwise only technical indicator formulas fall back. Price data never falls back.

## Render deployment

1. Upload files to GitHub.
2. Create a new Render Web Service.
3. Use this repository.
4. Render will read `render.yaml`.
5. Start command:

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=${RENDER_EXTERNAL_HOSTNAME} --num-procs 1
```

## Notes

QuantStats report generation may take time on first click because it downloads Yahoo data and builds HTML.
