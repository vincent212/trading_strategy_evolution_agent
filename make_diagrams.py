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
    ax.text(0.5, 0.96, "Search loop", ha="center",
            fontsize=21, fontweight="bold", color=INK)

    bx, bw = 0.12, 0.56
    cx = bx + bw / 2
    box(ax, bx, 0.775, bw, 0.14, "STRATEGY POPULATION",
        "hundreds of little trading programs", fc=BLUEBG, ec=BLUE, tc=BLUE)
    box(ax, bx, 0.545, bw, 0.15, "LANGUAGE  MODEL",
        "writes a new strategy · sees CODE, never prices", fc=WHITE, ec=INK)
    box(ax, bx, 0.315, bw, 0.14, "FIT  +  CROSS-VALIDATE",
        "median out-of-sample Sharpe", fc=GRAYBG, ec=GRAYED)

    arrow(ax, cx, 0.775, cx, 0.695)
    label(ax, 0.71, 0.735, "show it every\nstrategy tried,\nbest first")
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
def fig_islands():
    fig, ax = new_ax(10, 7.5)
    ax.text(0.5, 0.95, "Islands", ha="center",
            fontsize=21, fontweight="bold", color=INK)
    ax.text(0.5, 0.875, "Independent sub-populations; no members are exchanged",
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
    fig, ax = new_ax(13, 6.8)
    ax.text(0.5, 0.95, "The keep / reject decision uses only pre-2026 data",
            ha="center", fontsize=18, fontweight="bold", color=INK)

    w, gap = 0.20, 0.035
    x0 = 0.05
    y, h = 0.46, 0.28
    steps = [
        (BLUE, BLUEBG, "Fit", "tune the params on\n75% of the quarters"),
        (GREEN, GREENBG, "Cross-validate", "median OOS Sharpe\nover 100 splits"),
        (AMBER, AMBERBG, "Null-max bar", "must beat the\nnoise ceiling"),
    ]
    for i, (ec, fc, title, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.8)
        numbered(ax, x + w / 2, y + h - 0.05, i + 1, ec)
        ax.text(x + w / 2, y + h - 0.115, title, ha="center", fontsize=14,
                color=ec, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center",
                fontsize=11, color=INK, linespacing=1.4)
        if i < 2:
            arrow(ax, x + w + 0.003, y + h / 2, x + w + gap - 0.003, y + h / 2, MUTE, lw=1.8)

    # bracket: the three decision steps all run on pre-2026 data
    bx0, bx1 = x0, x0 + 3 * w + 2 * gap
    ax.plot([bx0, bx0, bx1, bx1],
            [y + h + 0.035, y + h + 0.06, y + h + 0.06, y + h + 0.035], color=MUTE, lw=1.2)
    ax.text((bx0 + bx1) / 2, y + h + 0.10, "all on pre-2026 data (2017–2025)",
            ha="center", fontsize=12, color=MUTE, style="italic")

    # decision node — fed only by steps 1-3
    dx, dw = bx1 + gap, 0.16
    box(ax, dx, y + 0.04, dw, h - 0.08, "KEEP\nor REJECT", fc=WHITE, ec=INK, title_fs=14)
    arrow(ax, bx1 + 0.003, y + h / 2, dx - 0.003, y + h / 2, INK, lw=2.0)
    dcx = dx + dw / 2

    # strategy in
    arrow(ax, 0.008, y + h / 2, x0 - 0.006, y + h / 2, MUTE)
    ax.text(0.0, y + h / 2 + 0.075, "a\nstrategy", ha="left", va="center",
            fontsize=10.5, color=MUTE, linespacing=1.2)

    # 2026 held out — SEPARATE, report-only, not wired into the decision
    ty, th = 0.13, 0.12
    box(ax, dcx - 0.12, ty, 0.24, th, "2026  —  held out", "measured once, only if kept",
        fc=GRAYBG, ec=GRAYED, tc=INK, title_fs=13, sub_fs=10.5)
    ax.add_patch(FancyArrowPatch((dcx, y + 0.04), (dcx, ty + th), arrowstyle="-|>",
                 mutation_scale=15, lw=1.6, color=MUTE, linestyle=(0, (4, 3))))
    ax.text(dcx - 0.02, (y + 0.04 + ty + th) / 2, "report only —\nnot part of the\ndecision",
            ha="right", va="center", fontsize=10, color=MUTE, linespacing=1.25)

    ax.text(0.5, 0.03, "2026 never touches the search, the fit, the cross-validation, or the null-max bar — "
            "it only shows how the kept strategy did on an unseen year.",
            ha="center", fontsize=11.5, color=MUTE)
    save(fig, "04_gauntlet.png")


def fig_combine():
    fig, ax = new_ax(12.5, 6.4)
    ax.text(0.5, 0.93, "Two jobs: the model wires the tools, the optimizer sets the numbers",
            ha="center", fontsize=17, fontweight="bold", color=INK)
    yc = 0.55
    lw, lh = 0.27, 0.30

    box(ax, 0.02, yc - 0.10, 0.16, 0.20, "Alpha tools",
        "sma · rsi · vol_regime\nbreakout · … (fixed)",
        fc=GRAYBG, ec=GRAYED, tc=INK, title_fs=12.5, sub_fs=9)

    lx = 0.235
    box(ax, lx, yc - 0.15, lw, lh, "", fc=BLUEBG, ec=BLUE)
    ax.text(lx + lw / 2, yc + 0.10, "LANGUAGE MODEL", ha="center", fontsize=13.5,
            color=BLUE, fontweight="bold")
    ax.text(lx + lw / 2, yc + 0.045, "decides HOW to combine them", ha="center", fontsize=10, color=INK)
    ax.text(lx + lw / 2, yc - 0.02, "w1·trend + w2·(dip × vol_regime)", ha="center",
            fontsize=9, family="monospace", color=INK)
    ax.text(lx + lw / 2, yc - 0.075, "params left as symbols", ha="center", fontsize=8.5,
            color=MUTE, style="italic")
    ax.text(lx + lw / 2, yc + 0.185, "the model's ONLY job", ha="center", fontsize=10.5,
            color=BLUE, style="italic")

    ox = 0.535
    box(ax, ox, yc - 0.15, lw, lh, "", fc=GREENBG, ec=GREEN)
    ax.text(ox + lw / 2, yc + 0.10, "OPTIMIZER", ha="center", fontsize=13.5,
            color=GREEN, fontweight="bold")
    ax.text(ox + lw / 2, yc + 0.045, "differential evolution: the numbers", ha="center", fontsize=10, color=INK)
    ax.text(ox + lw / 2, yc - 0.02, "fast=41  w1=0.10  w2=−0.74", ha="center",
            fontsize=9, family="monospace", color=INK)
    ax.text(ox + lw / 2, yc - 0.075, "fitted per split", ha="center", fontsize=8.5,
            color=MUTE, style="italic")
    ax.text(ox + lw / 2, yc + 0.185, "never the model", ha="center", fontsize=10.5,
            color=GREEN, style="italic")

    box(ax, 0.845, yc - 0.10, 0.13, 0.20, "SCORE", "median OOS\nSharpe",
        fc=AMBERBG, ec=AMBER, tc=AMBER, title_fs=12.5, sub_fs=9.5)

    arrow(ax, 0.18, yc, lx - 0.004, yc, MUTE)
    arrow(ax, lx + lw + 0.004, yc, ox - 0.004, yc, MUTE)
    arrow(ax, ox + lw + 0.004, yc, 0.845 - 0.004, yc, MUTE)

    line(ax, 0.91, yc - 0.10, 0.91, 0.12, AMBER)
    line(ax, 0.91, 0.12, 0.10, 0.12, AMBER)
    arrow(ax, 0.10, 0.12, 0.10, yc - 0.10, AMBER)
    ax.text(0.5, 0.085, "keep the highest-scoring combinations → the model mutates them into new ones",
            ha="center", fontsize=10.5, color=AMBER, style="italic")
    save(fig, "05_combine.png")


if __name__ == "__main__":
    fig_loop(); fig_islands(); fig_gauntlet(); fig_combine()
    print("\nAll diagrams written to", OUT)
