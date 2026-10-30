"""Reproduce Fig. 3 of the paper: MSTE applied to a 50-keystroke flight-time sequence.

Usage:
    python plot_mste_explanation.py <checkpoint.ckpt> [--scenario desktop] [--user 0] [--session 0]
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.Aalto.preprocessing import extract_features_classic

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('checkpoint', help='Lightning checkpoint of an MSTE model')
parser.add_argument('--scenario', default='desktop', choices=['desktop', 'mobile'])
parser.add_argument('--user', type=int, default=0, help='index of the user in the dev set')
parser.add_argument('--session', type=int, default=0, help='index of the session for that user')
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
raw  = np.load(f'data/Aalto/raw/{args.scenario}/{args.scenario}_dev_set.npy', allow_pickle=True).item()
user = list(raw.keys())[args.user]
sess = raw[user][list(raw[user].keys())[args.session]]
ft_raw = extract_features_classic(sess)[:50, 1]

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
    return f'{p * 1000:.0f} ms' if p < 1 else f'{p:.2f} s'

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
show_colors = plt.cm.viridis(np.linspace(0.08, 0.78, N_show))
period_label_size = 7 if N_show <= 12 else 5.5 if N_show <= 24 else 4.5

print(f'Picked periods: {[fmt_period(ft_eff_per[i]) for i in SHOW_IDX]}')

# -- Typography ----------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'STIXGeneral',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def _add_dummy_cbar(ax):
    # keeps both panels the same width
    make_axes_locatable(ax).append_axes('right', size='2.2%', pad='2%').set_visible(False)

def _tighten_gap(ax_top, ax_bot, frac):
    pt = ax_top.get_position()
    pb = ax_bot.get_position()
    gap = pt.y0 - (pb.y0 + pb.height)
    ax_bot.set_position([pb.x0, pb.y0 + gap * frac, pb.width, pb.height])

freq_panel_height = max(2.2, 0.18 * N_show)
fig = plt.figure(figsize=(9.2, 1.9 + freq_panel_height))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, freq_panel_height], hspace=0.32)

# -- (a) Raw flight time -------------------------------------------------------
y_ms   = ft_raw * 1000
peak_v = y_ms[peak_i]

ax_a = fig.add_subplot(gs[0])
_add_dummy_cbar(ax_a)

ax_a.plot(t, y_ms, color='#333333', lw=0.8,
          marker='o', ms=1.8, markerfacecolor='white', markeredgewidth=0.55, zorder=3)
ax_a.axvline(peak_i, color='#d62728', lw=0.65, ls='--', alpha=0.45, zorder=0)
peak_label = f'{peak_v / 1000:.2f} s' if peak_v >= 1000 else f'{peak_v:.0f} ms'
ax_a.annotate(peak_label,
              xy=(peak_i, peak_v), xytext=(-55, -5), textcoords='offset points',
              fontsize=8, color='#d62728',
              arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.65, shrinkA=2, shrinkB=2))

def _fmt_time(v, _):
    if v >= 1000:
        return f'{v / 1000:.1f} s'
    return f'{v:.0f} ms'

y_pad = max(20.0, 0.08 * float(np.ptp(y_ms)))
ax_a.set_xlim(-0.5, n - 0.5)
ax_a.set_ylim(float(np.min(y_ms)) - y_pad, float(np.max(y_ms)) + y_pad)
ax_a.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))
ax_a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_a.set_ylabel(r'$\mathrm{FT_i}$')
ax_a.tick_params(axis='x', labelbottom=False)
ax_a.set_title(r'(a) Flight Time $\mathrm{FT_i}$, raw signed values', pad=3, loc='left')
ax_a.spines[['top', 'right']].set_visible(False)
ax_a.grid(axis='y', lw=0.2, ls='--', color='#cccccc', zorder=0)
ax_a.legend(handles=[
    Line2D([0], [0], color='#d62728', lw=1.0, ls='--', label=r'peak $|\mathrm{FT}_i|$'),
], fontsize=9, frameon=False, loc='upper right',
   bbox_to_anchor=(1.0, 1.20), ncol=3, columnspacing=0.8, handlelength=1.8)

# -- (b) MSTE sin/cos components -----------------------------------------------
ax_b = fig.add_subplot(gs[1])
_add_dummy_cbar(ax_b)

offset_step = 2.1
for k, idx in enumerate(SHOW_IDX):
    base = k * offset_step
    c    = show_colors[k]
    ax_b.plot(t, base + ft_mste[2*idx],     color=c, lw=0.75, ls='-',  alpha=0.95)
    ax_b.plot(t, base + ft_mste[2*idx + 1], color=c, lw=0.75, ls='--', alpha=0.75)
    ax_b.axhline(base, color='#dddddd', lw=0.25, zorder=0)

ax_b.axvline(peak_i, color='#d62728', lw=0.6, ls='--', alpha=0.55, zorder=3)
ax_b.legend(handles=[
    Line2D([0], [0], color='#333333', lw=1.0, ls='-',  label=r'$\sin(\omega_j \mathrm{FT}_i)$'),
    Line2D([0], [0], color='#333333', lw=1.0, ls='--', label=r'$\cos(\omega_j \mathrm{FT}_i)$'),
], fontsize=9, frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.13),
   ncol=2, columnspacing=0.8, handlelength=1.8, borderaxespad=0.2)

ax_b.set_xlim(-0.5, n - 0.5)
ax_b.set_ylim(-1.15, (N_show - 1) * offset_step + 1.15)
ax_b.set_yticks([k * offset_step for k in range(N_show)])
ax_b.set_yticklabels([fmt_period(ft_eff_per[i]) for i in SHOW_IDX], fontsize=period_label_size)
for k, lbl in enumerate(ax_b.get_yticklabels()):
    lbl.set_color(show_colors[k])

ax_b.set_xlabel('Keystroke index $k$')
ax_b.set_ylabel(r'Learned $p_j$', labelpad=3)
ax_b.set_title(r'(b) MSTE sin/cos components', pad=3, loc='left')
ax_b.spines[['top', 'right']].set_visible(False)
ax_b.grid(axis='x', lw=0.2, ls=':', color='#cccccc', zorder=0)

fig.align_ylabels([ax_a, ax_b])
_tighten_gap(ax_a, ax_b, 0.35)

os.makedirs('outputs', exist_ok=True)
fig.savefig('outputs/mste_flight_explanation.pdf', bbox_inches='tight')
fig.savefig('outputs/mste_flight_explanation.png', bbox_inches='tight', dpi=300)
print('Saved outputs/mste_flight_explanation.{pdf,png}')
