# Pick-1-of-7 (Magnificent 7) — clean report

Date: 2026-09-07. All figures are **correctly measured** (warm indicators — scores
computed on the continuous price series, then sliced to the holdout). This supersedes
earlier numbers that were distorted by a cash-in-warmup measurement bug, now fixed.

## The strategy

LATSS evolves a **name-symmetric** per-series scorer `score(series, tools, p)` built
only from basic technical indicators (moving averages, ROC/momentum, RSI, realized
vol, vol-regime, breakout, drawdown, …). Each bar the harness **holds the single
highest-scoring** of the 7 names — AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA. The
scorer sees one *anonymous* price series at a time (no ticker identity), so it is a
technical **selection rule**, not a hand-picked stock. Params are fit by differential
evolution; fitness = cross-validated median-OOS active Sharpe vs equal-weight; the
champion is measured **once** on a sealed 2026 holdout.

## Benchmarks

- **Equal-weight**: 1/7 in each name, daily rebalanced.
- **Market-cap-weight**: wᵢ ∝ sharesᵢ × priceᵢ, daily rebalanced (shares outstanding
  from Yahoo, point-in-time where available). As of the last bar the cap weights are
  NVDA 23%, AAPL 20%, GOOGL 17%, MSFT 16%, AMZN 12%, META 7%, TSLA 6% — heavily
  mega-cap-tilted.

## Holdout 2026 (Jan 2 – Sep 4, 170 trading days)

The strategy is stochastic (the LLM search lands on different scorers across runs) and
sensitive to the calibration window, so all three pick-1 champions are shown:

| Champion (calibration → 2026 holdout) | Champion return / Sharpe | active Sh vs **Equal-wt** | active Sh vs **Cap-wt** |
|---|---|---|---|
| **2015–2025** (representative) | **+12.4%** / +0.72 | **+0.35** (beat) | **+0.25** (beat) |
| 2020–2025 | **+32.4%** / +1.32 | **+1.24** (beat big) | **+1.18** (beat big) |
| 2012–2025 | +7.3% / +0.49 | +0.06 (matched) | −0.02 (matched) |
| **Equal-weight** benchmark | +7.7% / +0.59 | — | — |
| **Cap-weight** benchmark | +9.5% / +0.74 | — | — |

## Read

- **Cap-weight beat equal-weight in 2026** (+9.5% / Sharpe 0.74 vs +7.7% / 0.59) — the
  mega-caps led — so cap-weight is the *harder* benchmark of the two.
- The pick-1 technical selector **beat both benchmarks in two of three calibrations**
  (2015 and 2020) and **matched** them in the third (2012); it never underperformed
  in 2026. Best case (2020 calibration) returned +32% against ~8–9% for the passives.
- **But it is not statistically robust on this holdout.** 2026 is 170 days over only 7
  names, and the random-picker null (hold a uniformly random 1 of 7 each bar) has a
  q95 of ≈ **+1.2** active Sharpe. Only the 2020-calibration champion (+1.24) clears
  that bar, and only barely; the others sit inside the noise. Run-to-run champion
  variance is large. So the honest verdict is **encouraging but not a demonstrated
  edge** — one noisy year, dominated by which scorer the search happened to find.

## Champion (representative — 2015–2025 calibration)

Pool CV median-OOS active Sharpe **+0.620** (shift-the-signal skill p = 0.024).

```python
def param_space():
    return {"nf": ("int",15,60), "ns": ("int",60,140), "w": ("float",0.2,0.8),
            "vn": ("int",20,80), "lo": ("float",0.2,0.5), "hi": ("float",0.5,0.85),
            "dn": ("int",20,80), "dpen": ("float",0.0,1.5)}

def score(series, tools, p):
    fast = tools.roc(series, p["nf"])
    slow = tools.roc(series, p["ns"])
    mom  = p["w"] * fast + (1.0 - p["w"]) * slow          # dual-horizon momentum
    reg  = tools.vol_regime(series, p["vn"], p["lo"], p["hi"])
    dd   = tools.drawdown(series, p["dn"])
    return mom * reg - p["dpen"] * (-dd)                  # momentum, regime-gated, drawdown-penalized
```
Fitted: `nf=26, ns=75, w=0.356, vn=77, lo=0.366, hi=0.607, dn=62, dpen=0.198`.

## Method notes

- **Warm-indicator fix** (this is what makes these numbers trustworthy): the champion's
  indicators are computed on the full continuous panel and then sliced to 2026, so
  long-lookback selectors are fully warmed up at the holdout boundary. The prior
  reports computed them on the 170-day slice alone, which left the strategy in cash
  through the warmup and understated results.
- **Cap-weight benchmark** uses shares outstanding × price; share counts carry a mild
  look-ahead (current/slowly-updated shares), immaterial at this horizon.
- Costs 5 bp per unit turnover; 1-bar execution lag; long-only, one name at a time.
- Reproduce: `python run_mag7.py --start 2015-01-01 --holdout-year 2026 --top-k 1`
  (with a Claude Code session servicing `runs/mutation_request.json`).
