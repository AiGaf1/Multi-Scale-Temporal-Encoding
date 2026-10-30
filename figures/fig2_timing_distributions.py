"""Reproduce Fig. 2 of the paper: hold-time and flight-time distributions, desktop vs. mobile.

Usage (from the repository root):
    python figures/fig2_timing_distributions.py [--out outputs/timing_distributions.pdf]

Reads the KVC-onGoing development sets from data/raw/. Collecting the timing values
takes a few minutes, so they are cached (unique values at 0.1 ms resolution and their counts)
in outputs/timing_hist_cache.npz; delete that file to recompute.

The figure is drawn at its printed size (LNCS text width) and its text is typeset by LaTeX
with the paper's fonts, so a LaTeX installation with the newtx package is required.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere
from mste.data import extract_timing_features

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', default='outputs/timing_distributions.pdf', help='output PDF (a PNG is saved next to it)')
parser.add_argument('--cache', default='outputs/timing_hist_cache.npz', help='cache of the collected timing values')
args = parser.parse_args()

# -- Collect HT / FT values from the development sets ---------------------------
if not os.path.exists(args.cache):
    hist = {}
    for ds in ('desktop', 'mobile'):
        raw = np.load(f'data/raw/{ds}/{ds}_dev_set.npy', allow_pickle=True).item()
        hold, flight = [], []
        for sessions in raw.values():
            for sess in sessions.values():
                f = extract_timing_features(sess)
                hold.append(f[:, 0])
                flight.append(f[1:, 1])   # first FT is 0 by construction
        del raw
        for name, vals in (('hold', np.concatenate(hold)), ('flight', np.concatenate(flight))):
            hist[f'{ds}_{name}_u'], hist[f'{ds}_{name}_c'] = np.unique(vals.round(4), return_counts=True)
        print(f'collected {ds}')
    os.makedirs(os.path.dirname(args.cache) or '.', exist_ok=True)
    np.savez_compressed(args.cache, **hist)
cache = np.load(args.cache)

# -- Plot ------------------------------------------------------------------------
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage[varvw]{newtxmath}',
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
})
COLORS = {'desktop': '#2166ac', 'mobile': '#c0392b'}
PANELS = [  # (device, feature, x-limits in s, x tick step)
    ('desktop', 'hold',   (0.0, 0.25),   0.1),
    ('mobile',  'hold',   (0.0, 0.05),   0.02),
    ('desktop', 'flight', (-0.2, 0.4),   0.2),
    ('mobile',  'flight', (-0.005, 0.8), 0.2),
]
LABEL = {'hold': r'Hold time HT (s)', 'flight': r'Flight time FT (s)'}

def darker(hex_color, f=0.6):
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return (r * f, g * f, b * f)

fmt = ticker.FuncFormatter(lambda v, _: f'${v:g}$')

# 4.8 in = LNCS text width
fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.42), layout='constrained')
fig.get_layout_engine().set(w_pad=0.02, wspace=0.06)

for k, (ax, (ds, feat, xlim, step)) in enumerate(zip(axes, PANELS)):
    u, c = cache[f'{ds}_{feat}_u'], cache[f'{ds}_{feat}_c']
    p = c / c.sum()
    col = COLORS[ds]

    # one bar per distinct timing value (0.1 ms resolution) -- shows the USB-polling spikes
    keep = (u >= xlim[0]) & (u <= xlim[1])
    ax.bar(u[keep], p[keep] * 100, width=0.001, align='edge', color=col, linewidth=0, zorder=2)
    ax.set_xlim(xlim)
    ax.set_ylim(0, p[keep].max() * 100 * 1.08)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='y', pad=1.5)
    ax.set_xlabel(LABEL[feat], labelpad=1.5)
    ax.set_title(rf'\textbf{{{ds.capitalize()}}}', pad=2.5, color=darker(col, 0.85))
    ax.spines[['top']].set_visible(False)

    # cumulative distribution over all values, on a shared 0-1 scale
    ax2 = ax.twinx()
    ax2.plot(u, np.cumsum(p), color=darker(col), lw=1.0, zorder=4)
    ax2.set_ylim(0, 1.04)
    ax2.set_yticks([0, 0.5, 1])
    ax2.yaxis.set_major_formatter(fmt)
    ax2.tick_params(axis='y', right=(k == 3), labelright=(k == 3), length=2, pad=1.5)
    ax2.spines[['top']].set_visible(False)
    if k == 3:
        ax2.set_ylabel('CDF', labelpad=2)
    if k == 0:
        ax.set_ylabel(r'Frequency (\%)', labelpad=2)

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
fig.savefig(args.out)
fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=300)
print(f'Saved {args.out}')
