"""Reproduce Fig. 3 of the paper: MSTE applied to a 50-keystroke flight-time sequence.

Usage (from the repository root):
    python figures/fig3_mste_components.py <checkpoint.ckpt> [--scenario desktop] [--user 0] [--session 0]
                                    [--out outputs/mste_flight_explanation.pdf]

The figure is drawn at its printed size (LNCS text width) and its text is typeset by
LaTeX with the paper's fonts, so a LaTeX installation with the newtx package is required.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere
from mste.data import extract_timing_features

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('checkpoint', help='Lightning checkpoint of an MSTE model')
parser.add_argument('--scenario', default='desktop', choices=['desktop', 'mobile'])
parser.add_argument('--user', type=int, default=0, help='index of the user in the dev set')
parser.add_argument('--session', type=int, default=0, help='index of the session for that user')
parser.add_argument('--out', default='outputs/mste_flight_explanation.pdf', help='output PDF (a PNG is saved next to it)')
args = parser.parse_args()

# -- Learned frequencies from the checkpoint -----------------------------------
state = torch.load(args.checkpoint, map_location='cpu')['state_dict']

def _pick_state(name):
    # checkpoints saved from a torch.compile'd model prefix keys with "_orig_mod."
    for prefix in ('model._orig_mod.', 'model.'):
        for encoder in ('time_encoders', 'time_encoder'):
            key = f'{prefix}{encoder}.{name}'
            if key in state:
                return state[key]
    raise KeyError(f'MSTE parameter "{name}" not found in checkpoint')

freq = _pick_state('freq').numpy()
scales = torch.sigmoid(_pick_state('scales_raw')).numpy()
eff_freq    = freq * scales           # omega_j, eq. (3)
eff_periods = 2 * np.pi / eff_freq    # learned period p_j, in seconds

# -- One real session, first 50 keystrokes -------------------------------------
raw  = np.load(f'data/raw/{args.scenario}/{args.scenario}_dev_set.npy', allow_pickle=True).item()
user = list(raw.keys())[args.user]
sess = raw[user][list(raw[user].keys())[args.session]]
ft_raw = extract_timing_features(sess)[:50, 1]

n = len(ft_raw)
t = np.arange(n)
peak_i = np.argmax(np.abs(ft_raw))

# -- MSTE, eq. (4) -------------------------------------------------------------
def mste_exact(signal, eff_freq_arr):
    proj = signal[None, :] * eff_freq_arr[:, None]   # (F, N)
    return np.stack([np.sin(proj), np.cos(proj)], axis=1).reshape(-1, len(signal))

ft_eff_per = eff_periods[1]
ft_mste    = mste_exact(ft_raw, eff_freq[1])

def fmt_period(p):
    return rf'{p * 1000:.0f}\,ms' if p < 1 else rf'{p:.2f}\,s'

# Show the learned periods closest to these targets
TARGET_PERIODS = np.array([0.080, 0.300, 0.700, 3.000, 6.000, 10.000, 20.000])  # seconds
SHOW_IDX = []
for p in TARGET_PERIODS:
    for idx in np.argsort(np.abs(np.log(ft_eff_per) - np.log(p))):
        idx = int(idx)
        if idx not in SHOW_IDX:
            SHOW_IDX.append(idx)
            break
N_show = len(SHOW_IDX)
# shortest period at the bottom, darkest colour
show_colors = plt.cm.viridis(np.linspace(0.05, 0.72, N_show))

print('Picked periods:', [f'{ft_eff_per[i]:.3f}s' for i in SHOW_IDX])

# -- Typography: LaTeX with the paper's fonts, drawn at printed size ------------
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage[varvw]{newtxmath}',
    'font.family': 'serif',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
})
PEAK = '#c0392b'

# 4.8 in = LNCS text width (122 mm)
fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(4.8, 3.05), sharex=True, layout='constrained',
    gridspec_kw={'height_ratios': [1, 2.6], 'hspace': 0.06},
)

# -- (a) Raw flight time -------------------------------------------------------
ax_a.axhline(0, color='#999999', lw=0.5, zorder=1)
ax_a.axvline(peak_i, color=PEAK, lw=0.8, ls='--', alpha=0.6, zorder=1)
ax_a.plot(t, ft_raw, color='#333333', lw=0.8, marker='o', ms=2.4,
          markerfacecolor='white', markeredgewidth=0.6, zorder=3)
ax_a.plot(peak_i, ft_raw[peak_i], marker='o', ms=3.2, color=PEAK, zorder=4)
ax_a.annotate(rf'{ft_raw[peak_i]:.2f}\,s', xy=(peak_i, ft_raw[peak_i]), xytext=(-6, -1),
              textcoords='offset points', ha='right', va='center', fontsize=8, color=PEAK)

y_pad = 0.1 * float(np.ptp(ft_raw))
ax_a.set_ylim(float(ft_raw.min()) - y_pad, float(ft_raw.max()) + y_pad)
ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'${v:g}$'))
ax_a.set_ylabel(r'$\mathrm{FT}_i$ (s)')
ax_a.set_title(r'\textbf{(a)} Flight time $\mathrm{FT}_i$, raw signed values', loc='left', pad=3)

# -- (b) MSTE sin/cos components -----------------------------------------------
offset_step = 2.2
for k, idx in enumerate(SHOW_IDX):
    base = k * offset_step
    c    = show_colors[k]
    ax_b.axhline(base, color='#e4e4e4', lw=0.4, zorder=0)
    ax_b.plot(t, base + ft_mste[2*idx],     color=c, lw=0.85, ls='-',  zorder=3)
    ax_b.plot(t, base + ft_mste[2*idx + 1], color=c, lw=0.85, ls=(0, (2.5, 1.5)), alpha=0.8, zorder=2)

ax_b.axvline(peak_i, color=PEAK, lw=0.8, ls='--', alpha=0.6, zorder=1)
ax_b.set_ylim(-1.25, (N_show - 1) * offset_step + 1.25)
ax_b.set_yticks([k * offset_step for k in range(N_show)])
ax_b.set_yticklabels([fmt_period(ft_eff_per[i]) for i in SHOW_IDX])
for k, lbl in enumerate(ax_b.get_yticklabels()):
    lbl.set_color(show_colors[k])
ax_b.tick_params(axis='y', length=0, pad=4)
ax_b.set_ylabel(r'Learned period $p_j$')
ax_b.set_xlabel(r'Keystroke index $i$')
ax_b.set_title(r'\textbf{(b)} MSTE components $\sin(\omega_j \mathrm{FT}_i)$ and $\cos(\omega_j \mathrm{FT}_i)$',
               loc='left', pad=3)
ax_b.legend(handles=[
    Line2D([0], [0], color='#444444', lw=0.9, ls='-', label='sine'),
    Line2D([0], [0], color='#444444', lw=0.9, ls=(0, (2.5, 1.5)), label='cosine'),
], frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(1.0, 1.0),
   handlelength=2.0, columnspacing=1.0, borderaxespad=0.2)

for ax in (ax_a, ax_b):
    ax.spines[['top', 'right']].set_visible(False)
ax_b.set_xlim(-0.8, n - 0.2)
ax_b.xaxis.set_major_locator(ticker.MultipleLocator(10))
fig.align_ylabels([ax_a, ax_b])

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
fig.savefig(args.out)
fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=300)
print(f'Saved {args.out}')
