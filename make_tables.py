"""
Render the article's result tables as PNGs for Substack (which can't show markdown tables).
Run: python make_tables.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

INK = "#0F172A"; MUTE = "#64748B"; HEADBG = "#1E293B"; ROWBG = "#F8FAFC"
GREEN = "#059669"; GRID = "#E2E8F0"; WHITE = "#FFFFFF"
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": WHITE, "savefig.facecolor": WHITE})


def render(name, headers, rows, bold=(), widths=None, title=None):
    """bold: set of (row, col) data-cell coords to bold+green (0-indexed data rows)."""
    ncol = len(headers)
    widths = widths or [1.0 / ncol] * ncol
    fig, ax = plt.subplots(figsize=(max(9.5, 3.6 * ncol + 2.5),
                                    0.9 * (len(rows) + 1) + (0.5 if title else 0)))
    ax.axis("off")
    tbl = ax.table(cellText=[headers] + rows, colWidths=widths,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(19)
    tbl.scale(1, 2.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.0)
        if r == 0:                                  # header row
            cell.set_facecolor(HEADBG)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(17)
        else:
            cell.set_facecolor(ROWBG if r % 2 else WHITE)
            cell.get_text().set_color(INK)
            if c == 0:
                cell.get_text().set_fontweight("bold")
            if (r - 1, c) in bold:
                cell.get_text().set_color(GREEN)
                cell.get_text().set_fontweight("bold")
    if title:
        ax.set_title(title, fontsize=13, color=MUTE, pad=10)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", p)


def render_ref(name, headers, rows, widths, fig_w=15.0):
    """Left-aligned reference table; first column monospace."""
    fig, ax = plt.subplots(figsize=(fig_w, 0.52 * (len(rows) + 1) + 0.4))
    ax.axis("off")
    tbl = ax.table(cellText=[headers] + rows, colWidths=widths, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(16)
    tbl.scale(1, 2.3)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.0)
        cell.PAD = 0.015
        t = cell.get_text()
        t.set_ha("left")
        if r == 0:
            cell.set_facecolor(HEADBG)
            t.set_color(WHITE)
            t.set_fontweight("bold")
            t.set_fontsize(16)
        else:
            cell.set_facecolor(ROWBG if r % 2 else WHITE)
            t.set_color(INK)
            if c == 0:
                t.set_family("monospace")
                t.set_fontsize(13.5)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", p)


render_ref("tbl_tools.png",
           ["tool", "what it computes"],
           [["sma(x, n)", "simple moving average over n bars"],
            ["ema(x, n)", "exponential moving average (span n)"],
            ["roc(x, n)", "rate of change / momentum:  x_t / x_(t-n) − 1"],
            ["zscore(x, n)", "rolling z-score of x over n bars"],
            ["rsi(x, n)", "Wilder RSI oscillator, 0–100"],
            ["realized_vol(x, n)", "annualized rolling realized volatility"],
            ["vol_target_scale(x, target_vol, n, max_lev)",
             "position multiplier that scales exposure\ninversely to recent vol, toward a target"],
            ["rolling_high(x, n) / rolling_low(x, n)", "highest / lowest value over the prior n bars"],
            ["breakout(x, n)", "+1 on a new n-bar high, −1 on a new n-bar low"],
            ["vol_regime(x, n, low, high)", "realized-vol regime: +1 low vol, −1 high vol"],
            ["pctile_rank(x, n)", "rolling percentile rank in [0, 1]"],
            ["crossover(fast, slow)", "+1 when fast crosses above slow, −1 when below"],
            ["clip_signal(sig)", "clamp a raw signal to a valid position in [−1, 1]"]],
           widths=[0.47, 0.53], fig_w=16.0)

render_ref("tbl_repo.png",
           ["file", "role"],
           [["alpha_tools.py", "fixed, causal indicator library available to strategies"],
            ["data.py", "Yahoo Finance price data (any ticker), cached to disk"],
            ["params.py", "encodes a strategy's param_space() into the optimizer's search box"],
            ["strategy_seed.py", "the trainable param_space() + strategy(data, tools, p) contract and seed"],
            ["prompt.py", "the mutation prompt: whole scored history to one child"],
            ["backtest.py", "backtest, differential-evolution fit, quarter-CV median-OOS fitness, shift-the-signal skill test"],
            ["evolve.py", "islands, parent sampling, model mutation call, evaluation"],
            ["run.py", "orchestration: run_search(...) and CLI"],
            ["llm.py", "provider shim: local Ollama, any OpenAI-compatible endpoint, or Anthropic"],
            ["make_diagrams.py", "regenerates the figures in assets/"]],
           widths=[0.24, 0.76])


# 1. overfitting ladder
render("tbl_overfit.png",
       ["champion", "CV median OOS Sharpe", "2026 holdout Sharpe"],
       [["early", "1.34", "0.37"],
        ["mid",   "1.47", "0.71"],
        ["late",  "1.63", "0.76"]],
       widths=[0.22, 0.42, 0.36])

# 2. objective comparison  (bold the return-max winning column)
render("tbl_objective.png",
       ["out-of-sample (2026)", "Sharpe-max", "Return-max\n(min Sharpe 0.8)", "buy & hold"],
       [["2026 Sharpe", "0.76", "0.92", "0.94"],
        ["2026 return", "+6.2%", "+15.5%", "+18.7%"],
        ["avg exposure", "~29%", "~93%", "100%"]],
       bold={(0, 2), (1, 2)}, widths=[0.30, 0.22, 0.26, 0.22])

# 3. MAR  (bold the return-max drawdown + MAR)
render("tbl_mar.png",
       ["2017–2025 (in-sample)", "CAGR", "max drawdown", "MAR"],
       [["buy & hold", "+62%", "−66%", "0.93"],
        ["return-max strategy", "+66%", "−41%", "1.61"]],
       bold={(1, 2), (1, 3)}, widths=[0.40, 0.18, 0.24, 0.18])

print("\nAll tables written to", OUT)
