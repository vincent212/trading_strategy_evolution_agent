"""
Generate polished PNG diagrams for the article, into assets/.
Run: python make_diagrams.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

INK    = "#0F172A"
MUTE   = "#64748B"
BLUE   = "#2563EB"; BLUEBG  = "#EAF1FE"
GREEN  = "#059669"; GREENBG = "#E6F6F0"
RED    = "#DC2626"; REDBG   = "#FDECEC"
AMBER  = "#D97706"; AMBERBG = "#FEF3E2"
GRAYBG = "#F1F5F9"; GRAYED  = "#CBD5E1"
WHITE  = "#FFFFFF"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": WHITE,
                     "savefig.facecolor": WHITE})


def box(ax, x, y, w, h, title, sub=None, fc=GRAYBG, ec=GRAYED, tc=INK,
        title_fs=15, sub_fs=11, lw=1.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.025",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=title_fs, color=tc, fontweight="bold")
        ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                fontsize=sub_fs, color=MUTE, linespacing=1.35)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, color=tc, fontweight="bold")


def line(ax, x1, y1, x2, y2, color=INK, lw=2.2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round")


def arrow(ax, x1, y1, x2, y2, color=INK, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def label(ax, x, y, text, color=MUTE, fs=11, ha="left", italic=True, bold=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            style="italic" if italic else "normal",
            fontweight="bold" if bold else "normal", linespacing=1.3)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
def fig_loop():
    fig, ax = new_ax(10, 8.5)
    ax.text(0.5, 0.96, "How the strategies evolve", ha="center",
            fontsize=21, fontweight="bold", color=INK)

    bx, bw = 0.12, 0.56
    cx = bx + bw / 2
    box(ax, bx, 0.775, bw, 0.14, "STRATEGY POPULATION",
        "hundreds of little trading programs", fc=BLUEBG, ec=BLUE, tc=BLUE)
    box(ax, bx, 0.545, bw, 0.15, "LANGUAGE  MODEL",
        "writes a new strategy · sees CODE, never prices", fc=WHITE, ec=INK)
    box(ax, bx, 0.315, bw, 0.14, "BACKTEST  +  SCORE",
        "does it actually hold up?", fc=GRAYBG, ec=GRAYED)

    arrow(ax, cx, 0.775, cx, 0.695)
    label(ax, 0.71, 0.735, "pick 2 of the\nbest 'parents'")
    arrow(ax, cx, 0.545, cx, 0.455)
    label(ax, 0.71, 0.500, "a new\n'child' strategy")

    # keep-loop (green), routed up the right channel into the population box
    pop_mid, bt_mid, chan = 0.845, 0.385, 0.90
    line(ax, bx + bw, bt_mid, chan, bt_mid, GREEN)
    line(ax, chan, bt_mid, chan, pop_mid, GREEN)
    arrow(ax, chan, pop_mid, bx + bw, pop_mid, GREEN)
    label(ax, 0.915, 0.62, "keep the\ngood ones", GREEN, fs=12, italic=False, bold=True)

    # discard (red)
    arrow(ax, bx, bt_mid, 0.045, bt_mid, RED)
    label(ax, 0.05, 0.30, "discard if\nweak or broken", RED, ha="center", italic=False)

    ax.text(0.5, 0.115, "Good strategies breed more. Over thousands of rounds, the\n"
            "population climbs toward strategies that work.",
            ha="center", fontsize=13, color=MUTE, linespacing=1.4)
    save(fig, "01_loop.png")


# ---------------------------------------------------------------------------
def fig_data():
    fig, ax = new_ax(10, 8.5)
    ax.text(0.5, 0.96, "Where the data comes from", ha="center",
            fontsize=21, fontweight="bold", color=INK)

    cx = 0.5
    box(ax, 0.31, 0.84, 0.38, 0.085, "Yahoo Finance", fc=INK, ec=INK, tc=WHITE,
        title_fs=16)
    arrow(ax, cx, 0.84, cx, 0.755)
    label(ax, 0.53, 0.798, "free  'yfinance'  library")

    box(ax, 0.13, 0.615, 0.74, 0.14, "Pick ANY ticker",
        "SPY   ·   AAPL   ·   TSLA   ·   QQQ   ·   NVDA   ·   …",
        fc=BLUEBG, ec=BLUE, tc=BLUE)
    arrow(ax, cx, 0.615, cx, 0.545)

    box(ax, 0.10, 0.39, 0.37, 0.145, "Daily adjusted prices",
        "the stock you trade", fc=GREENBG, ec=GREEN, tc=GREEN, title_fs=14)
    box(ax, 0.53, 0.39, 0.37, 0.145, "The VIX",
        "market 'fear gauge'", fc=AMBERBG, ec=AMBER, tc=AMBER, title_fs=14)
    arrow(ax, 0.285, 0.39, 0.45, 0.315, MUTE)
    arrow(ax, 0.715, 0.39, 0.55, 0.315, MUTE)

    box(ax, 0.22, 0.19, 0.56, 0.10, "Cached on disk",
        "downloaded once, reused every run", fc=WHITE, ec=INK, title_fs=14)
    arrow(ax, cx, 0.19, cx, 0.125)
    ax.text(0.5, 0.075, "handed to every strategy as  close  and  vix",
            ha="center", fontsize=13.5, color=INK, fontweight="bold")
    save(fig, "02_data.png")


# ---------------------------------------------------------------------------
def fig_islands():
    fig, ax = new_ax(10, 7.5)
    ax.text(0.5, 0.95, "Four 'islands' keep the search diverse", ha="center",
            fontsize=21, fontweight="bold", color=INK)
    ax.text(0.5, 0.875, "Each population evolves on its own — no cross-breeding",
            ha="center", fontsize=13, color=MUTE)

    islands = [("Island 1", GREEN, GREENBG, "kept"),
               ("Island 2", GREEN, GREENBG, "kept"),
               ("Island 3", RED, REDBG, "wiped"),
               ("Island 4", RED, REDBG, "wiped")]
    w, gap = 0.195, 0.035
    x0 = (1 - (4 * w + 3 * gap)) / 2
    for i, (lab, ec, fc, tag) in enumerate(islands):
        x = x0 + i * (w + gap)
        box(ax, x, 0.58, w, 0.18, lab, "evolves\nseparately", fc=fc, ec=ec, tc=ec,
            title_fs=14, sub_fs=10.5)
        ax.text(x + w / 2, 0.545, tag, ha="center", va="center", fontsize=11,
                color=ec, fontweight="bold")

    box(ax, 0.11, 0.19, 0.78, 0.17, "Every 50 rounds",
        "keep the 2 strongest islands  ·  wipe the 2 weakest,\n"
        "then restart them from the current champion",
        fc="#FFFBEB", ec=AMBER, tc=AMBER, title_fs=15, sub_fs=11.5)
    for i in (2, 3):
        x = x0 + i * (w + gap) + w / 2
        arrow(ax, 0.5, 0.36, x, 0.525, AMBER, lw=1.8, rad=-0.18)
    save(fig, "03_islands.png")


# ---------------------------------------------------------------------------
def numbered(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 0.026, facecolor=color, edgecolor="none"))
    ax.text(x, y, str(n), ha="center", va="center", color=WHITE,
            fontsize=13, fontweight="bold")


def fig_gauntlet():
    fig, ax = new_ax(11.5, 6.8)
    ax.text(0.5, 0.95, "Before I believe a strategy, it must survive all three",
            ha="center", fontsize=19, fontweight="bold", color=INK)

    w, gap = 0.275, 0.045
    x0 = (1 - (3 * w + 2 * gap)) / 2
    y, h = 0.30, 0.46
    cards = [
        (BLUE, BLUEBG, "Consistency", "Did it work in MOST\nperiods — not one\nlucky stretch?"),
        (GREEN, GREENBG, "Unseen data", "Tested once on recent\nyears the search\nnever got to see."),
        (AMBER, AMBERBG, "The null-max bar", "Beat what a strategy\ncould fake on pure\nrandom noise."),
    ]
    for i, (ec, fc, title, body) in enumerate(cards):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.8)
        numbered(ax, x + w / 2, y + h - 0.055, i + 1, ec)
        ax.text(x + w / 2, y + h - 0.135, title, ha="center", fontsize=15.5,
                color=ec, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.42, body, ha="center", va="center",
                fontsize=12, color=INK, linespacing=1.4)

    # glyphs near each card bottom
    # 1: five checked blocks
    bx = x0 + 0.045
    for k in range(5):
        ax.add_patch(FancyBboxPatch((bx + k * 0.038, y + 0.05), 0.03, 0.05,
                     boxstyle="round,pad=0.002", fc=WHITE, ec=BLUE, lw=1.2))
        ax.text(bx + k * 0.038 + 0.015, y + 0.075, "✓", ha="center", va="center",
                color=BLUE, fontsize=10)
    # 2: train | wall | test
    x = x0 + (w + gap)
    ax.add_patch(FancyBboxPatch((x + 0.03, y + 0.05), 0.12, 0.05,
                 boxstyle="round,pad=0.002", fc=WHITE, ec=GREEN, lw=1.2))
    ax.text(x + 0.09, y + 0.075, "train", ha="center", va="center", color=GREEN, fontsize=10)
    ax.plot([x + 0.157, x + 0.157], [y + 0.045, y + 0.105], color=INK, lw=2.6)
    ax.add_patch(FancyBboxPatch((x + 0.165, y + 0.05), 0.07, 0.05,
                 boxstyle="round,pad=0.002", fc=GREENBG, ec=GREEN, lw=1.2))
    ax.text(x + 0.20, y + 0.075, "test", ha="center", va="center", color=GREEN, fontsize=10)
    # 3: noise squiggle
    x = x0 + 2 * (w + gap)
    xs = np.linspace(x + 0.04, x + 0.15, 40)
    ax.plot(xs, y + 0.075 + 0.018 * np.sin(np.linspace(0, 9, 40)), color=AMBER, lw=1.8)
    ax.text(x + 0.205, y + 0.075, "must\nbeat this", ha="center", va="center",
            fontsize=9.5, color=AMBER, linespacing=1.2)

    arrow(ax, 0.055, y + h / 2, x0 - 0.008, y + h / 2, MUTE)
    ax.text(0.05, y + h / 2 + 0.07, "a\nstrategy", ha="center", va="center",
            fontsize=11, color=MUTE, linespacing=1.2)
    arrow(ax, x0 + 3 * w + 2 * gap + 0.008, y + h / 2, 0.965, y + h / 2, GREEN)
    ax.text(0.955, y + h / 2 + 0.06, "KEEP", ha="center", fontsize=13.5,
            color=GREEN, fontweight="bold")
    ax.text(0.5, 0.135, "Fail any one, and it's thrown out — no matter how good the backtest looked.",
            ha="center", fontsize=12.5, color=MUTE)
    save(fig, "04_gauntlet.png")


if __name__ == "__main__":
    fig_loop(); fig_data(); fig_islands(); fig_gauntlet()
    print("\nAll diagrams written to", OUT)
