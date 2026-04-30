# QFA Hedge Fund Dashboard - Render Ready

This is a fully rebuilt reactive Panel dashboard.

## Key architecture

- One source of truth: `pn.widgets`
- All analytic blocks are connected with `pn.bind`
- No mixed `param.Parameterized`, `pn.Param`, static KPI panes, or stale state
- Yahoo-only matched data policy
- S&P 500 benchmark is `^GSPC`, not SPY
- RF is fixed at 4.5%
- QuantStats tab name is `Tearsheet`
- PyPortfolioOpt optimizer is included

## Render deployment

Use the included `render.yaml`.

Start command:

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin=${RENDER_EXTERNAL_HOSTNAME} --num-procs 1
```

## TA-Lib

The app attempts to import TA-Lib. If TA-Lib is available, the technical engine uses TA-Lib.  
If TA-Lib is not available on Render, the app does not crash; it uses formula-based indicator calculations only.  
This fallback is only for indicators. Price data remains Yahoo-only and no synthetic/fallback prices are generated.

To try TA-Lib manually, install:

```bash
pip install -r requirements-talib-optional.txt
```

## Local run

```bash
pip install -r requirements.txt
panel serve app.py --show
```

## Notes

Render free tier may sleep. First load and QuantStats generation can be slower.


## V2 stability changes

- Tabs now use `dynamic=True` to prevent QuantStats and optimizer from rendering at initial page load.
- Matplotlib font-manager warnings are suppressed.
- Arial was removed from the Python/Plotly font stack; DejaVu/Liberation Sans are used on Linux.
- Sidebar sizing was changed to avoid Panel `FIXED_SIZING_MODE` warnings.
- If Render shows `SIGTERM`, it usually means the service was restarted or memory was exceeded during heavy initial rendering; this V2 build avoids eager rendering.
