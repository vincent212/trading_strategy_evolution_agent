import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HEAD='#2c6fbf'; HTXT='white'; ALT='#f2f6fc'; EDGE='#c9d6e8'

def render(fname, cols, rows, colw, fs=12, title=None):
    n=len(rows)
    fig, ax = plt.subplots(figsize=(sum(colw), 0.55*(n+1)+ (0.5 if title else 0)))
    ax.axis('off')
    if title: ax.set_title(title, fontsize=13, fontweight='bold', color='#14304f', pad=10)
    t = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center',
                 colWidths=[w/sum(colw) for w in colw])
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.6)
    for (r,c), cell in t.get_celld().items():
        cell.set_edgecolor(EDGE)
        if r==0:
            cell.set_facecolor(HEAD); cell.set_text_props(color=HTXT, fontweight='bold')
        else:
            cell.set_facecolor('white' if r%2 else ALT)
    fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print("saved", fname)

# 1. rho calibration
render('table_calibration.png',
  ['ρ','SNR = ρ/√(1−ρ²)','oracle active Sharpe'],
  [['0.0','0.00','−0.67'],['0.1','0.10','0.00'],['0.2','0.20','+0.66'],['0.3','0.31','+1.23'],
   ['0.4','0.44','+1.66'],['0.5','0.58','+2.06'],['0.6','0.75','+2.47'],['0.7','0.98','+2.81'],
   ['0.8','1.33','+3.24'],['0.9','2.07','+3.62'],['1.0','∞','+3.92']],
  colw=[1.2,2.6,3.0])

# 2. Exp1 results
render('table_exp1.png',
  ['run','champion CV','Rademacher bar','cleared'],
  [['1','+0.500','+0.495','no  (tie, +0.005 — within bar noise)'],
   ['2','+0.405','+0.668','no'],
   ['3','−0.000','+0.621','no']],
  colw=[1.0,2.2,2.6,5.2])

# 3. Exp2 results
render('table_exp2.png',
  ['ρ','~Sharpe','runs cleared','champion CV (per run)','bar range'],
  [['0.05','−0.4','0/1','+0.52','0.56'],
   ['0.10','0.0','0/3','+0.33,  +0.62,  −0.08','0.65–0.73'],
   ['0.15','0.3','1/3','+0.42,  +0.74,  +0.53','0.59–0.63'],
   ['0.20','0.66','3/3','+1.19,  +1.18,  +0.63','0.57–0.77'],
   ['0.25','0.95','2/3','+1.30,  +1.34,  +0.71','0.60–0.78'],
   ['0.30','1.23','3/3','+1.84,  +1.72,  +1.22','0.60–0.73'],
   ['0.35','1.9','3/3','+2.14,  +2.01,  +1.44','0.62–0.68']],
  colw=[1.0,1.4,1.8,4.0,1.8])
