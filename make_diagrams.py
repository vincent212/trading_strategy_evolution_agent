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
    fig, ax = new_ax(13, 6.6)
    ax.text(0.5, 0.95, "Evaluation: fit, cross-validate, gate, hold out",
            ha="center", fontsize=18.5, fontweight="bold", color=INK)

    w, gap = 0.19, 0.028
    x0 = (1 - (4 * w + 3 * gap)) / 2
    y, h = 0.32, 0.44
    cards = [
        (BLUE, BLUEBG, "Fit",
         "Tune the parameters\non 75% of the\nquarters (max Sharpe)."),
        (GREEN, GREENBG, "Cross-validate",
         "Score on the other\n25%. Repeat 100x.\nTake the MEDIAN\nout-of-sample Sharpe."),
        (AMBER, AMBERBG, "Null-max bar",
         "Refit on sign-flipped\nnoise; the real median\nmust beat what noise\ncan fake."),
        (INK, GRAYBG, "2026 held out",
         "Scored once on this\nyear — never seen by\nthe search or the bar."),
    ]
    for i, (ec, fc, title, body) in enumerate(cards):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.8)
        numbered(ax, x + w / 2, y + h - 0.05, i + 1, ec)
        ax.text(x + w / 2, y + h - 0.125, title, ha="center", fontsize=14.5,
                color=ec, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.40, body, ha="center", va="center",
                fontsize=11, color=INK, linespacing=1.4)
        if i < 3:
            arrow(ax, x + w + 0.002, y + h / 2, x + w + gap - 0.002, y + h / 2, MUTE, lw=1.8)

    arrow(ax, 0.045, y + h / 2, x0 - 0.006, y + h / 2, MUTE)
    ax.text(0.032, y + h / 2 + 0.075, "a\nstrategy", ha="center", va="center",
            fontsize=10.5, color=MUTE, linespacing=1.2)
    arrow(ax, x0 + 4 * w + 3 * gap + 0.006, y + h / 2, 0.965, y + h / 2, GREEN)
    ax.text(0.955, y + h / 2 + 0.065, "KEEP", ha="center", fontsize=13,
            color=GREEN, fontweight="bold")
    ax.text(0.5, 0.15, "Steps 1–2 are the fitness the search maximizes; step 3 is the pass/fail gate; step 4 is the final untouched number.",
            ha="center", fontsize=12, color=MUTE)
    save(fig, "04_gauntlet.png")


if __name__ == "__main__":
    fig_loop(); fig_islands(); fig_gauntlet()
    print("\nAll diagrams written to", OUT)
