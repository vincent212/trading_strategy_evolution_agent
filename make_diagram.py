import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.5, 12))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis('off')
BLUE='#2c6fbf'; DARK='#14304f'; GREY='#666'; GREEN='#1a7f4b'
def box(cx, cy, w, h, title, sub=None, fc='#eef3fb', ec=BLUE):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.18", linewidth=1.8, edgecolor=ec, facecolor=fc))
    if sub:
        ax.text(cx, cy+0.20, title, ha='center', va='center', fontsize=12, fontweight='bold', color=DARK)
        ax.text(cx, cy-0.30, sub, ha='center', va='center', fontsize=10.5, color=DARK, family='monospace')
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=12, fontweight='bold', color=DARK)
def arrow(y1, y2, x=5, label=None):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>', mutation_scale=20, linewidth=1.8, color=DARK))
    if label: ax.text(x+0.2, (y1+y2)/2, label, ha='left', va='center', fontsize=9.5, color=GREY, style='italic')
def note(cx, cy, txt): ax.text(cx, cy, txt, ha='left', va='center', fontsize=10, color=GREEN, style='italic')

cx=5; w=5.4
ax.text(cx,14.85,"LATSS loop", ha='center', va='center', fontsize=15, fontweight='bold', color=DARK)
box(cx,14.0,7.0,1.0,"population of candidate scoring functions")
arrow(13.5,12.55,label="history: code + fitness")
box(cx,12.0,w,1.1,"LLM: propose / mutate","score(features, p)"); note(cx+w/2+0.2,12.0,"generic operator\n(FunSearch)")
arrow(11.45,10.35)
box(cx,9.8,w,1.0,"differential evolution"); note(cx+w/2+0.2,9.8,"fit parameters p")
arrow(9.3,8.45)
box(cx,7.9,8.6,1.0,"backtest:  apply strategy to data  →  strategy returns", fc='#fbf6ee', ec='#b9791f')
arrow(7.4,6.35)
box(cx,5.8,w,1.1,"purged CV, median OOS","Sharpe vs benchmark (optional)"); note(cx+w/2+0.2,5.8,"fitness =\nSharpe")
arrow(5.25,4.05,label="keep best → champion")
box(cx,3.5,w,1.0,"empirical Rademacher bar", fc='#f6eef6', ec='#8a2f8a'); note(cx+w/2+0.2,3.5,"certified?")
# feedback loop (left)
lx=cx-w/2-1.4
ax.add_patch(FancyArrowPatch((cx-w/2,5.8),(lx,5.8),arrowstyle='-',linewidth=1.6,color=GREY,linestyle=(0,(5,3))))
ax.add_patch(FancyArrowPatch((lx,5.8),(lx,14.0),arrowstyle='-',linewidth=1.6,color=GREY,linestyle=(0,(5,3))))
ax.add_patch(FancyArrowPatch((lx,14.0),(cx-3.5,14.0),arrowstyle='-|>',mutation_scale=18,linewidth=1.6,color=GREY,linestyle=(0,(5,3))))
ax.text(lx-0.1,9.9,"evolutionary loop:\nbest + failures\nfeed next prompt", ha='right', va='center', fontsize=9.5, color=GREY, style='italic')
plt.savefig('latss_loop.png', dpi=200, bbox_inches='tight', facecolor='white')
print("saved")
