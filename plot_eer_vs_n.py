"""Reproduce Fig. 5 of the paper: training and validation EER vs. the number of MSTE frequencies N.

Usage:
    python plot_eer_vs_n.py [--out outputs/eer_vs_n.pdf]

The EER values are the final train/validation EERs of the N-ablation runs (LSTM, SupConLoss);
re-running that ablation means `python train.py --n-periods N` for N in 2, 4, ..., 128.

The figure is drawn at its printed size and its text is typeset by LaTeX with the paper's
fonts, so a LaTeX installation with the newtx package is required.
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', default='outputs/eer_vs_n.pdf', help='output PDF (a PNG is saved next to it)')
args = parser.parse_args()

# (N, device, validation EER %, training EER %)
records = [
    (2,   'Desktop', 1.48697, 1.41296), (2,   'Mobile', 2.10877, 1.81580),
    (4,   'Desktop', 1.14441, 1.11236), (4,   'Mobile', 2.07825, 1.79215),
    (8,   'Desktop', 1.06812, 0.98267), (8,   'Mobile', 2.10266, 1.77078),
    (16,  'Desktop', 1.09253, 0.97275), (16,  'Mobile', 1.92871, 1.59607),
    (32,  'Desktop', 0.98472, 0.92011), (32,  'Mobile', 1.89209, 1.52588),
    (64,  'Desktop', 0.99792, 0.88806), (64,  'Mobile', 2.00501, 1.44653),
    (128, 'Desktop', 1.01928, 0.90332), (128, 'Mobile', 1.94397, 1.46408),
]
ns = [2, 4, 8, 16, 32, 64, 128]
CHOSEN_N = 32
COLORS = {'Desktop': '#2166ac', 'Mobile': '#c0392b'}

def series(device, split):
    i = 2 if split == 'val' else 3
    return [r[i] for r in records if r[1] == device]

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage[varvw]{newtxmath}',
    'font.family': 'serif',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
})

# 3.6 in = 0.75 of the LNCS text width (122 mm)
fig, ax = plt.subplots(figsize=(3.6, 1.75), layout='constrained')

ax.axvspan(CHOSEN_N / 1.25, CHOSEN_N * 1.25, color='#f0f0f0', zorder=0, lw=0)
ax.text(CHOSEN_N, 2.255, f'$N={CHOSEN_N}$', ha='center', va='bottom', fontsize=7.5, color='#555555')

for device in ('Desktop', 'Mobile'):
    c = COLORS[device]
    val, trn = series(device, 'val'), series(device, 'train')
    ax.plot(ns, val, color=c, lw=1.3, marker='o', ms=4, zorder=4)
    ax.plot(ns, trn, color=c, lw=1.0, ls='--', marker='s', ms=3.4,
            markerfacecolor='white', markeredgewidth=0.9, alpha=0.85, zorder=3)
    best = val[ns.index(CHOSEN_N)]
    ax.annotate(rf'\textbf{{{best:.2f}}}', xy=(CHOSEN_N, best), xytext=(0, 6), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, color=c)

ax.set_xscale('log', base=2)
ax.set_xticks(ns)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v)}'))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.set_xlim(1.6, 160)

ax.set_ylim(0.8, 2.25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

ax.set_xlabel('Number of sinusoidal components $N$')
ax.set_ylabel(r'EER (\%)')
ax.grid(axis='y', lw=0.3, ls=':', color='#bbbbbb', zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

handles = [
    Line2D([], [], color=COLORS['Desktop'], lw=1.6, label='Desktop'),
    Line2D([], [], color=COLORS['Mobile'], lw=1.6, label='Mobile'),
    Line2D([], [], color='#444444', lw=1.3, marker='o', ms=4, label='Validation'),
    Line2D([], [], color='#444444', lw=1.0, ls='--', marker='s', ms=3.4,
           markerfacecolor='white', markeredgewidth=0.9, label='Training'),
]
ax.legend(handles=handles, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.06),
          frameon=False, handlelength=2.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
fig.savefig(args.out)
fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=300)
print(f'Saved {args.out}')
