#!/usr/bin/env python3
"""
make_figures.py -- unified figure generation for
"One-port Liouvillian sensing: critical dynamics, measurement information, and collective-mode visibility"

Replaces make_quantum_figures.py, make_feedback_figures.py and the rendering half of
make_multispin_figures.py with a SINGLE typographic system, so that font sizes, legend
style, panel labels and colours are identical across every panel of every figure.

Usage
-----
    python make_figures.py                      # all figures except 2a/S1 raw-data panels
    python make_figures.py --data-root /path    # additionally rebuilds Fig. 2a and Fig. S1
                                                # directly from the archived repository

Fig. 2a and Fig. S1 are the only panels that require the raw repository. When
--data-root is absent, Fig. 2a is rendered from data_processed/fig2a_trace.csv (the
same trace, carried forward) and Fig. S1 is left untouched.
"""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Arc, Ellipse
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multispin_chain import (slow_gap_and_weight, slow_gap_asymptotic, slow_weight_asymptotic,
                             boundary_coefficient, propagate_single_excitation,
                             propagation_length_readout, intrinsic_floor_length,
                             mode_resolvability_length, validation_scan)

HERE = Path(__file__).resolve().parents[1]
OUT  = HERE/'figures'
DATA = HERE/'data_processed'
RES  = HERE/'results'
RAW  = None

# ============================================================ palette
NAVY   = '#18364F'
TEAL   = '#0C8F8B'
ORANGE = '#DD7438'
RED    = '#B23C49'
PURPLE = '#7151A4'
GREEN  = '#458A61'
GREY   = '#67737D'
MID    = '#AAB6BE'
LIGHT  = '#EDF2F5'
DARK   = '#22262A'
RULE   = '#30363B'

# ============================================================ ONE typographic system
#   Every size used anywhere in any figure is listed here. Nothing else sets a fontsize.
FS_TICK   = 7.8     # tick labels
FS_LABEL  = 8.6     # axis labels
FS_LEGEND = 8.4     # all legends            (+1 pt vs the previous revision)
FS_ANNOT  = 8.0     # in-axes annotations and equations
FS_PANEL  = 9.6     # bold panel letters
FS_SKETCH = 7.4     # schematic body text    (fig. 1, 6a, 7a)
FS_SKHEAD = 7.4     # schematic headings     (unified with body: only weight differs)
FS_SKMATH = 8.8     # schematic maths symbols (Omega_F, gamma_s, J, ...)
FS_SKMATH_L = 11.0  # enlarged schematic maths symbols (Fig. 7a only)
FS_INSET  = 6.0     # inset axes labels

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': FS_ANNOT,
    'axes.labelsize': FS_LABEL,
    'legend.fontsize': FS_LEGEND,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'axes.linewidth': 0.85,
    'lines.linewidth': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
    'xtick.minor.size': 2.3, 'ytick.minor.size': 2.3,
    'xtick.major.width': 0.75, 'ytick.major.width': 0.75,
    'xtick.minor.width': 0.6,  'ytick.minor.width': 0.6,
    'legend.handlelength': 1.5,
    'legend.labelspacing': 0.32,
    'legend.borderpad': 0.34,
    'legend.borderaxespad': 0.45,
    'legend.columnspacing': 1.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

# Legend style: an opaque-enough white patch so a legend can sit over a gridline or a
# faint curve without the reader losing either. Used for EVERY legend in the paper.
LEG = dict(frameon=True, facecolor='white', edgecolor='none', framealpha=0.88)
# Same idea for free-standing annotations that sit inside the data area.
def abox(alpha=0.86, pad=1.8):
    return dict(facecolor='white', edgecolor='none', alpha=alpha, pad=pad)


def style(ax, xminor=True, yminor=True):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.85); s.set_color(RULE)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if xminor: ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yminor: ax.yaxis.set_minor_locator(AutoMinorLocator())
    return ax


def twin(ax, colour):
    ax2 = ax.twinx()
    for s in ax2.spines.values():
        s.set_visible(True); s.set_linewidth(0.85); s.set_color(RULE)
    ax2.tick_params(which='both', direction='in', top=True, right=True, colors=colour)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    return ax2


def panel(ax, letter, x=-0.13, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FS_PANEL, fontweight='bold')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=0.025)
    fig.savefig(OUT/f'{name}.png', dpi=350, bbox_inches='tight', pad_inches=0.025)
    plt.close(fig)
    print('  wrote', name)


# ============================================================ Figure 1  (v7 redesign)
def _mini_axes(fig, rect):
    """Small inset-style axes for the Fig. 1 result tiles: one common style."""
    ax = fig.add_axes(rect)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.6); s.set_color('#5B656C')
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=FS_INSET-0.3,
                   length=2, pad=1.5)
    ax.patch.set_facecolor('white')
    return ax


def _tiny_tip(ax, xc, y_top, y_apex, w_top, colour=NAVY):
    """Miniature tip for the Fig. 1 schematic (outline only, apex atom)."""
    h = y_top - y_apex; y_sh = y_top - .40*h; w_neck = w_top*.16
    body = [(xc-w_top, y_top), (xc+w_top, y_top), (xc+w_top*.82, y_sh),
            (xc+w_neck, y_apex+.045*h), (xc-w_neck, y_apex+.045*h), (xc-w_top*.82, y_sh)]
    ax.add_patch(Polygon(body, closed=True, fc=LIGHT, ec=colour, lw=1.0, joinstyle='round', zorder=3))
    ax.add_patch(Circle((xc, y_apex+.02*h), .020*h, fc='white', ec=colour, lw=.8, zorder=4))
    ax.arrow(xc, y_top-.24*h, 0, .13*h, width=.006, head_width=.03, head_length=.035,
             color=colour, length_includes_head=True, zorder=5)


def fig1():
    """Logic of the study (v7).  Left: the one-port sensor, in which drive, dissipation
    and readout enter through one interface.  Right: the three results that this
    constraint produces.  Bottom: the experimental anchor supplied by the archived
    pentacene dataset."""
    fig = plt.figure(figsize=(7.25, 3.80))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 5.25); ax.axis('off')

    # ---------------------------------------------------------------- left: one-port sensor
    x0 = 0.28
    ax.text(x0, 4.97, 'one-port open quantum sensor', fontsize=FS_SKHEAD, fontweight='bold',
            color=NAVY, ha='left', va='center')
    # tip + spins
    xs_sp = np.array([1.00, 1.62, 2.24, 2.86]); ysp = 3.07
    _tiny_tip(ax, xs_sp[0], 4.65, 3.53, .27)
    ax.plot(xs_sp, ysp*np.ones(4), color=PURPLE, lw=1.5, zorder=1)
    for k, x in enumerate(xs_sp):
        col = ORANGE if k == 0 else TEAL
        ax.add_patch(Ellipse((x, ysp), .30, .40, fc='white', ec=col, lw=1.3, zorder=3))
        ax.arrow(x, ysp-.10, 0, .21, width=.010, head_width=.062, head_length=.062,
                 color=col, length_includes_head=True, zorder=4)
    # drive, readout, dissipation through the junction
    ax.add_patch(FancyArrowPatch((xs_sp[0]-.34, 3.97), (xs_sp[0]-.34, 3.37), arrowstyle='-|>',
                                 mutation_scale=9, lw=1.05, color=ORANGE, zorder=4))
    ax.text(xs_sp[0]-.44, 3.69, r'$\Omega_F$', color=ORANGE, fontsize=FS_SKMATH, ha='right', va='center')
    ax.add_patch(FancyArrowPatch((xs_sp[0]+.34, 3.37), (xs_sp[0]+.34, 3.97), arrowstyle='-|>',
                                 mutation_scale=9, lw=1.05, color=NAVY, ls=(0, (2.2, 1.5)), zorder=4))
    ax.text(xs_sp[0]+.45, 3.69, r'$I$', color=NAVY, fontsize=FS_SKMATH, ha='left', va='center')
    ax.text(xs_sp[0], 2.65, r'$\gamma_s$', color=NAVY, fontsize=FS_SKMATH, ha='center', va='center')
    ax.text((xs_sp[1]+xs_sp[2])/2, 3.37, r'$J$', color=PURPLE, fontsize=FS_SKMATH, ha='center', va='bottom')
    ax.text(xs_sp[3], 2.65, r'$\gamma_t$', color=TEAL, fontsize=FS_SKMATH, ha='center', va='center')
    ax.text(x0, 2.21, 'drive, dissipation and readout\nshare one interface',
            fontsize=FS_SKETCH, color=DARK, ha='left', va='top', linespacing=1.25)
    ax.text(x0, 1.50, r'$\dot\rho=-\frac{i}{\hbar}[H(t),\rho]+\sum_\nu\mathcal{D}[L_\nu(t)]\rho$',
            fontsize=FS_SKETCH+0.3, color=NAVY, ha='left', va='center')

    # connector to the result tiles
    ax.add_patch(FancyArrowPatch((3.30, 3.40), (3.78, 3.40), arrowstyle='-|>', mutation_scale=12,
                                 lw=1.3, color='#8B979F', zorder=2))
    ax.text(3.54, 3.57, 'Liouvillian\nmodes', fontsize=FS_SKETCH-0.4, color=GREY, ha='center',
            va='bottom', linespacing=1.15)

    # ---------------------------------------------------------------- right: three result tiles
    tiles = [
        ('critical reset', ORANGE,
         'EP = onset of the maximum\nreset-rate plateau ($\\times16$)'),
        ('information optimum', PURPLE,
         'current information peaks\nat $3.2\\,\\Omega_{\\rm EP}$, not at the EP'),
        ('collective-mode visibility', TEAL,
         'excess decay $=\\gamma_b\\times$ port\nparticipation; budget sums to 1'),
    ]
    tx = [4.05, 6.05, 8.05]; tw = 1.80
    for (head, col, sub), x in zip(tiles, tx):
        ax.text(x+tw/2, 4.97, head, fontsize=FS_SKHEAD, fontweight='bold', color=col,
                ha='center', va='center')
        ax.text(x+tw/2, 2.21, sub, fontsize=FS_SKETCH, color=DARK, ha='center', va='top',
                linespacing=1.25)
    # mini-plot 1: dissipative gap of the reduced model across the EP
    r = json.load(open(DATA/'experimental_summary.json')) if (DATA/'experimental_summary.json').exists() else None
    G1 = 1/138.2308e-9; G2 = 2.2186e8; A = (G1+G2)/2; D = (G2-G1)/2
    om = np.linspace(0, 2.2*D, 400)
    gap = np.where(om < D, A-np.sqrt(np.maximum(D**2-om**2, 0)), A)/G1
    m1 = _mini_axes(fig, [.415, .485, .155, .33])
    m1.plot(om/D, gap, color=ORANGE, lw=1.35)
    m1.axvline(1, color='#AFB9C0', ls=':', lw=.8)
    m1.set_xlim(0, 2.2); m1.set_ylim(0, 19)
    m1.set_xticks([0, 1, 2]); m1.set_yticks([1, 16])
    m1.set_xlabel(r'$\Omega_F/D$', fontsize=FS_INSET, labelpad=1)
    m1.set_ylabel(r'$\Delta_{\mathcal{L}}/\Gamma_1$', fontsize=FS_INSET, labelpad=1)
    m1.text(1.06, 3.2, 'EP', fontsize=FS_INSET, color=GREY)
    # mini-plot 2: counting Fisher-information rate across the extended scan
    met = pd.read_csv(RES/'metrology_scan.csv')
    fi = met.counting_FI_per_rad2s/met.counting_FI_per_rad2s.max()
    m2 = _mini_axes(fig, [.615, .485, .155, .33])
    m2.plot(met.Omega_over_EP, fi, color=PURPLE, lw=1.35)
    m2.axvline(1, color='#AFB9C0', ls=':', lw=.8)
    k = int(np.argmax(fi.to_numpy()))
    m2.plot([met.Omega_over_EP[k]], [fi[k]], 'o', ms=3.2, color=PURPLE)
    m2.set_xscale('log'); m2.set_xlim(.6, 10); m2.set_ylim(0, 1.15)
    m2.set_yticks([0, 1]); m2.set_xticks([1, 3, 10]); m2.set_xticklabels(['1', '3', '10'])
    m2.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    m2.set_xlabel(r'$\Omega_F/\Omega_{\rm EP}$', fontsize=FS_INSET, labelpad=1)
    m2.set_ylabel(r'$\dot{\mathcal{F}}$ (norm.)', fontsize=FS_INSET, labelpad=1)
    # mini-plot 3: boundary-induced slow decay versus N
    Ns = np.unique(np.round(np.logspace(np.log10(2), 2, 40)).astype(int))
    ex = np.array([slow_gap_and_weight(int(N), 1., .25, 0.)[0] for N in Ns])
    m3 = _mini_axes(fig, [.815, .485, .155, .33])
    m3.loglog(Ns, ex, color=TEAL, lw=1.35)
    m3.loglog(Ns[Ns >= 8], [slow_gap_asymptotic(int(N), 1., .25, 0.) for N in Ns[Ns >= 8]],
              '--', color=GREY, lw=.9)
    m3.set_xlim(2, 100); m3.set_ylim(3e-6, 0.3)
    m3.set_xticks([2, 10, 100]); m3.set_xticklabels(['2', '10', '100'])
    m3.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    m3.set_yticks([1e-5, 1e-3, 1e-1]); m3.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    m3.set_xlabel(r'$N$', fontsize=FS_INSET, labelpad=1)
    m3.set_ylabel(r'$(\Delta_{\rm slow}-\gamma_t)/J$', fontsize=FS_INSET, labelpad=1)
    m3.text(22, 4e-3, r'$N^{-3}$', fontsize=FS_INSET, color=GREY)

    # ---------------------------------------------------------------- bottom: experimental anchor
    ry = .50
    ax.text(x0, 1.02, 'experimental anchor: archived pentacene dataset', fontsize=FS_SKHEAD,
            fontweight='bold', color=NAVY, ha='left', va='center')
    rx = np.linspace(.85, 9.30, 5)
    rlab = ['public transients', r'$T_1,\;T_{2,\mathrm{Rabi}},\;\omega_d$',
            r'$\Omega_F/D=1.80^{+0.19}_{-0.18}$', r'$T_{\rm crit}\simeq5$ ns',
            r'design scales ($N_{\rm floor}\simeq7$)']
    ax.plot([rx[0], rx[-1]], [ry, ry], color='#B5C0C7', lw=1.05, zorder=0)
    for i, (xx, ll) in enumerate(zip(rx, rlab)):
        ax.scatter([xx], [ry], s=34, c=[TEAL if i < 4 else GREEN], edgecolor='white', lw=.8, zorder=3)
        ax.text(xx, ry-.19, ll, ha='center', va='top', fontsize=FS_SKETCH, color=DARK)
    save(fig, 'fig1_story')


# ============================================================ Figure 6: tip (v7)
def _draw_tip(ax, xc, y_top, y_apex, w_top=.115, colour=NAVY):
    """A slender STM tip drawn as a single outline with a soft, clipped vector shading
    (no separate highlight patch).  Shank, shoulder, taper, neck, apex atom and the
    tip-magnetisation arrow."""
    h = y_top - y_apex
    y_sh = y_top - .34*h
    w_neck = .017
    body = [(xc-w_top, y_top), (xc+w_top, y_top),
            (xc+w_top*.80, y_sh), (xc+w_neck, y_apex+.055),
            (xc-w_neck, y_apex+.055), (xc-w_top*.80, y_sh)]
    poly = Polygon(body, closed=True, fc='none', ec='none', zorder=3)
    ax.add_patch(poly)
    # left-to-right shading: thin strips clipped to the tip outline (pure vector output)
    n = 40
    xl, xr = xc-w_top, xc+w_top
    base = np.array(mpl.colors.to_rgb(LIGHT)); dark = np.array(mpl.colors.to_rgb('#B9C6CF'))
    for k in range(n):
        u = (k+.5)/n
        shade = .55*(1-np.cos(np.pi*u))**1.1          # lighter at the left, darker at the right
        c = base*(1-shade) + dark*shade
        r = mpl.patches.Rectangle((xl+k*(xr-xl)/n, y_apex), (xr-xl)/n*1.03, h, fc=c, ec='none',
                                  zorder=3)
        r.set_clip_path(poly); ax.add_patch(r)
    ax.add_patch(Polygon(body, closed=True, fc='none', ec=colour, lw=1.15, joinstyle='round', zorder=4))
    # apex atom
    ax.add_patch(Circle((xc, y_apex+.028), .019, fc='#F7F9FA', ec=colour, lw=1.0, zorder=5))
    # tip magnetisation
    ax.arrow(xc, y_top-.105, 0, .062, width=.005, head_width=.028, head_length=.028,
             color=colour, length_includes_head=True, zorder=6)


# ============================================================ Figure 7 (v7)
def fig7():
    J = 1.; gs = .25
    Ns = np.unique(np.round(np.logspace(np.log10(2), 2, 80)).astype(int))
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)

    # ---- a   schematic: symbols only, set at the enlarged schematic size
    ax = axs[0, 0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); panel(ax, 'a', x=-.09)
    xs = np.linspace(.22, .90, 7); y = .50
    ax.plot(xs, y*np.ones_like(xs), color=PURPLE, lw=1.7, zorder=1)
    for k, x in enumerate(xs):
        col = ORANGE if k == 0 else TEAL
        ax.add_patch(Ellipse((x, y), .088, .154, fc='white', ec=col, lw=1.45, zorder=3))
        ax.arrow(x, y-.040, 0, .090, width=.0048, head_width=.027, head_length=.029,
                 color=col, length_includes_head=True, zorder=4)
    ax.add_patch(FancyArrowPatch((xs[0], .89), (xs[0], .61), arrowstyle='-|>',
                                 mutation_scale=13, lw=1.4, color=ORANGE, zorder=4))
    ax.text(xs[0]-.065, .905, r'$\Omega_F$', color=ORANGE, fontsize=FS_SKMATH_L, ha='center', va='bottom')
    ax.add_patch(FancyArrowPatch((xs[0], .18), (xs[0], .39), arrowstyle='<|-|>',
                                 mutation_scale=9, lw=1.15, color=NAVY, zorder=4))
    ax.text(xs[0]-.070, .245, r'$I$', color=NAVY, fontsize=FS_SKMATH_L, ha='center', va='center')
    ax.text(xs[0]+.075, .245, r'$\gamma_s$', color=NAVY, fontsize=FS_SKMATH_L, ha='center', va='center')
    ax.text((xs[2]+xs[3])/2, .625, r'$J$', color=PURPLE, fontsize=FS_SKMATH_L, ha='center', va='bottom')
    ax.text((xs[4]+xs[5])/2, .275, r'$\gamma_t$', color=TEAL, fontsize=FS_SKMATH_L, ha='center', va='center')
    ax.text(.56, .90, r'$\chi_{11}(\omega)$', color=NAVY, fontsize=FS_SKMATH_L, ha='center', va='bottom')
    ax.add_patch(FancyArrowPatch((.34, .62), (.48, .88), arrowstyle='-', lw=.7, color=MID,
                                 connectionstyle='arc3,rad=-.25', zorder=2))

    # ---- b   gap scaling
    ax = axs[0, 1]; style(ax, xminor=False, yminor=False); panel(ax, 'b')
    rows = []
    for gt, c in [(0, NAVY), (.001, TEAL), (.01, ORANGE), (.03, RED)]:
        vals = []
        for N in Ns:
            gap, w, _ = slow_gap_and_weight(N, J, gs, gt); vals.append(gap); rows.append((N, gt, gap, w))
        ax.loglog(Ns, vals, color=c, lw=1.5, label=fr'$\gamma_t/J={gt:g}$')
        if gt > 0: ax.axhline(gt, color=c, ls=':', lw=.85, alpha=.65)
    nref = np.array([15, 100]); cref = slow_gap_asymptotic(30, J, gs, 0)*31**3
    ax.loglog(nref, cref/(nref+1)**3, '--', color=GREY, lw=1.2, label=r'$N^{-3}$')
    ax.set(xlabel='chain length $N$', ylabel=r'$\Delta_{\rm slow}/J$')
    ax.legend(loc='lower left', bbox_to_anchor=(.02, .03), **LEG)

    # ---- c   participation, residue and the rigidity correction
    ax = axs[1, 0]; style(ax, xminor=False, yminor=False); panel(ax, 'c')
    sm = pd.read_csv(RES/'one_port_slow_mode.csv'); sm = sm[sm.case == 'benchmark']
    ax.loglog(sm.N, sm.participation, color=TEAL, lw=1.5, label=r'$w_1=|R_1(1)|^2$')
    ax.loglog(sm.N, sm.residue_abs, color=NAVY, lw=1.1, ls=(0, (4, 2)), label=r'$|\mathrm{Res}_1\chi_{11}|$')
    ax.loglog(sm.N, sm.asymptotic_participation, '--', color=ORANGE, lw=1.3, label=r'asymptotic $N^{-3}$')
    ax.set(xlabel='chain length $N$', ylabel='sensor-site weight')
    ax.legend(loc='lower left', bbox_to_anchor=(.02, .03), **LEG)
    ax.text(.97, .955, r'$\Delta_{\rm slow}-\gamma_t=\gamma_b\,w_1=\gamma_b\,r_1|\mathrm{Res}_1|$',
            transform=ax.transAxes, ha='right', va='top', fontsize=FS_ANNOT+0.6, color=NAVY, bbox=abox(pad=1.2))
    ins = ax.inset_axes([.57, .55, .39, .27])
    ins.loglog(sm.N, 1/sm.rigidity-1, color=PURPLE, lw=1.1)
    nn = np.array([10, 100]); ins.loglog(nn, (1/sm.rigidity.iloc[-1]-1)*(100/nn)**2, ':', color=GREY, lw=.9)
    for s in ins.spines.values(): s.set_visible(True); s.set_linewidth(.6); s.set_color('#555E64')
    ins.tick_params(which='both', direction='in', top=True, right=True, labelsize=FS_INSET-0.4, length=2)
    ins.set_xlabel(r'$N$', fontsize=FS_INSET, labelpad=0)
    ins.set_ylabel(r'$1/r_1-1$', fontsize=FS_INSET, labelpad=1)
    ins.set_yticks([1e-4, 1e-2]); ins.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ins.text(.93, .88, r'$N^{-2}$', transform=ins.transAxes, ha='right', va='top', fontsize=FS_INSET, color=GREY)
    ins.patch.set_facecolor('white'); ins.patch.set_alpha(.95)

    # ---- d   Zeno turnover
    ax = axs[1, 1]; style(ax, xminor=False, yminor=False); panel(ax, 'd')
    gbs = np.logspace(-2, np.log10(4), 160)
    ax.semilogx(gbs, [boundary_coefficient(J, g) for g in gbs], '--',
                color=ORANGE, lw=1.4, label='large-$N$ law')
    gpts = np.logspace(-2, np.log10(4), 11)
    for N, mark, col in [(20, 'o', TEAL), (40, 's', PURPLE), (80, '^', NAVY)]:
        ys = [(N+1)**3*slow_gap_and_weight(N, J, gb, 0)[0] for gb in gpts]
        ax.semilogx(gpts, ys, mark+'-', ms=3.3, lw=1.05, color=col, label=fr'$N={N}$')
    ax.axvline(.5, color=GREY, ls=':', lw=1.0)
    ax.set(xlabel=r'boundary excess decay $\gamma_b/J$',
           ylabel=r'$(N+1)^3\Delta_{\rm slow}/J$')
    ax.set_ylim(-0.35, 6.4)
    ax.text(.035, .955, r'maximum at $\gamma_b=J/2$', transform=ax.transAxes,
            ha='left', va='top', fontsize=FS_ANNOT, color=DARK, bbox=abox())
    ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(.99, .03), **LEG)
    pd.DataFrame(rows, columns=['N', 'gamma_t_over_J', 'gap_over_J', 'sensor_weight'])\
        .to_csv(RES/'multispin_scaling.csv', index=False)
    save(fig, 'fig7_multispin_scaling')


# ============================================================ Figure S2 (v7)
def figS2():
    conv = pd.read_csv(RES/'floquet_convergence.csv'); ep = pd.read_csv(RES/'floquet_ep_scan.csv')
    fit = json.load(open(RES/'floquet_convergence_fit.json'))
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.62), constrained_layout=True)
    ax = style(axs[0], xminor=False, yminor=False); panel(ax, 'a', x=-.20)
    ax.semilogy(conv.steps_per_period, conv.sep_norm, 'o-', color=NAVY, lw=1.4, ms=3.4,
                label=r'$|\mu_1-\mu_2|/(|\mu_1|+|\mu_2|)$')
    ax.semilogy(conv.steps_per_period, conv.phase_rigidity, 's--', color=RED, lw=1.2, ms=3.2,
                label='phase rigidity $r$')
    ax.semilogy(conv.steps_per_period, np.maximum(1-conv.overlap, 1e-16), '^:', color=TEAL, lw=1.1, ms=3.2,
                label=r'$1-|\langle R_1|R_2\rangle|$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_ylim(1e-15, 1e-4)
    ax.set(xlabel=r'$N_t$', ylabel='EP diagnostics at $f_{\\rm EP}(N_t)$')
    ax.legend(loc='center right', bbox_to_anchor=(.99, .55), **LEG)
    ax = style(axs[1]); panel(ax, 'b', x=-.20)
    x = 1/conv.steps_per_period.to_numpy()**2
    ax.plot(x*1e3, conv.fEP_MHz, 'o', color=PURPLE, ms=3.6, label='root at each $N_t$')
    xx = np.linspace(0, x.max()*1.05, 50)
    ax.plot(xx*1e3, fit['f_inf_MHz']+fit['a_MHz']*xx, '-', color=ORANGE, lw=1.2,
            label=r'$f_\infty+a\,N_t^{-2}$')
    ax.set(xlabel=r'$10^3\,N_t^{-2}$', ylabel=r'$f_{\rm EP}$ (MHz)')
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)
    ax = style(axs[2]); panel(ax, 'c', x=-.20)
    ax.plot(ep.aD, ep.mu1_real_Mrad_s, 'o-', color=TEAL, lw=1.45, ms=3.4, label=r'$\mathrm{Re}\,\mu_1$')
    ax.plot(ep.aD, ep.mu2_real_Mrad_s, 's--', color=ORANGE, lw=1.3, ms=3.2, label=r'$\mathrm{Re}\,\mu_2$')
    ax.set(xlabel=r'$a_D$', ylabel=r'$\mathrm{Re}\,\mu$ ($10^6$ s$^{-1}$)')
    ax.legend(loc='upper right', bbox_to_anchor=(.99, .98), **LEG)
    save(fig, 'figS2_numerics')


# ============================================================ Figure S5 (new in v7)
def figS5():
    """One-port response: exact participation-decay identity for every mode, phase
    rigidity of the dark and bright modes, and the modal decomposition of chi_11."""
    from multispin_chain import one_port_modes
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.62), constrained_layout=True)
    # ---- a  identity for all modes
    ax = style(axs[0], xminor=False, yminor=False); panel(ax, 'a', x=-.20)
    for N, mk, col in [(4, 'o', ORANGE), (8, 's', TEAL), (16, '^', PURPLE), (32, 'D', NAVY)]:
        m = one_port_modes(N, 1., .25, .01)
        ax.loglog(m['part'], (-m['lam'].real-.01)/.24, mk, ms=3.4, mfc='none', mew=.9, color=col,
                  label=fr'$N={N}$')
    dd = np.array([1e-4, 1]); ax.loglog(dd, dd, '-', color='#AFB9C0', lw=.9, zorder=0)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set(xlabel=r'participation $|R_m(1)|^2$', ylabel=r'$(-\mathrm{Re}\,\lambda_m-\gamma_t)/\gamma_b$')
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)
    # ---- b  rigidity versus boundary loss
    ax = style(axs[1], xminor=False, yminor=False); panel(ax, 'b', x=-.20)
    rg = pd.read_csv(RES/'one_port_rigidity_scan.csv')
    for N, col in [(6, ORANGE), (10, TEAL), (20, PURPLE), (40, NAVY)]:
        g = rg[rg.N == N]
        ax.semilogx(g.gamma_b_over_J, g.slow_rigidity, '-', color=col, lw=1.4, label=fr'$N={N}$')
        ax.semilogx(g.gamma_b_over_J, g.min_rigidity, ':', color=col, lw=1.1)
    ax.axvline(.5, color=GREY, ls=':', lw=.9)
    ax.set_ylim(0, 1.05)
    ax.set(xlabel=r'$\gamma_b/J$', ylabel='phase rigidity $r_m$')
    ax.text(.035, .50, 'solid: slowest mode\ndotted: minimum over modes', transform=ax.transAxes,
            ha='left', va='center', fontsize=FS_ANNOT-0.6, color=DARK, linespacing=1.2, bbox=abox())
    ax.legend(loc='lower left', bbox_to_anchor=(.02, .03), **LEG)
    # ---- c  one-port spectrum and its modal decomposition
    ax = style(axs[2]); panel(ax, 'c', x=-.20)
    sp = pd.read_csv(RES/'one_port_spectrum.csv'); md = pd.read_csv(RES/'one_port_spectrum_modes.csv')
    ax.plot(sp.delta_over_J, sp.abs_chi, color=NAVY, lw=1.5, label=r'$|\chi_{11}|$')
    cols = [TEAL, ORANGE, PURPLE, GREEN, RED, GREY]
    for k in range(len(md)):
        ax.plot(sp.delta_over_J, sp[f'abs_mode{k+1}'], lw=.9, color=cols[k % 6], alpha=.85)
    ax.set(xlabel=r'detuning $\delta/J$', ylabel=r'$J|\chi_{11}(\delta)|$')
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(0, 5.6)
    ax.legend(loc='upper right', bbox_to_anchor=(.99, .98), **LEG)
    ax.text(.03, .96, f'$N=6$: dark modes carry\n'
            fr'$|\mathrm{{Res}}|={md.abs_residue.iloc[0]:.3f},\;{md.abs_residue.iloc[1]:.3f}$',
            transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANNOT-0.6, color=DARK,
            linespacing=1.2, bbox=abox())
    save(fig, 'figS5_one_port_response')


# ============================================================ Figure 2
def _fig2a_trace():
    """Return (tau, dI, sigma) for the pump-probe panel."""
    if RAW is not None:
        raw = pd.read_csv(RAW/'SM/S16/dI(Delay).csv')
        return (raw['Delay (ns)'].to_numpy(float),
                raw['dI (sig) (fA)'].to_numpy(float),
                raw['dI_err (sig) (fA)'].to_numpy(float))
    d = pd.read_csv(DATA/'fig2a_trace.csv')
    return d.tau_ns.to_numpy(), d.dI_fA.to_numpy(), d.dI_err_fA.to_numpy()


def fig2():
    d  = pd.read_csv(DATA/'experimental_liouvillian_reconstruction.csv').sort_values('bias_mV')
    mc = pd.read_csv(DATA/'experimental_uncertainties_mc.csv').sort_values('bias_mV')
    summary = json.load(open(DATA/'experimental_summary.json'))
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)

    # ---- a  pump-probe relaxation
    ax = style(axs[0, 0]); panel(ax, 'a')
    t, yy, ss = _fig2a_trace()
    T = summary['T1_ns']
    X = np.c_[np.ones_like(t), np.exp(-t/T)]
    b = np.linalg.lstsq(X/ss[:, None], yy/ss, rcond=None)[0]
    tt = np.linspace(t.min(), t.max(), 500)
    ax.errorbar(t, yy, yerr=ss, fmt='o', ms=2.7, mfc='white', mec=NAVY, mew=.7,
                ecolor=MID, elinewidth=.65, capsize=1.1, zorder=2, label='data')
    ax.plot(tt, b[0]+b[1]*np.exp(-tt/T), color=RED, lw=1.7, zorder=3, label='fit')
    ax.set(xlabel=r'$\tau$ (ns)', ylabel=r'$\Delta I$ (fA)')
    ax.legend(loc='lower left', bbox_to_anchor=(.02, .04), **LEG)
    ax.text(.97, .93, fr'$T_1={summary["T1_ns"]:.1f}\pm{summary["T1_err_ns"]:.1f}$ ns'
                      + '\n' + fr'$\chi^2_\nu={summary["T1_reduced_chi2"]:.2f}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=FS_ANNOT, bbox=abox())

    # ---- b  reconstructed rates
    ax = style(axs[0, 1]); panel(ax, 'b'); x = d.bias_mV.to_numpy()
    ax.errorbar(x, d.f_damped_MHz, yerr=d.f_damped_err_MHz, fmt='o', ms=3.9,
                mfc='white', mec=NAVY, mew=.8, ecolor=MID, elinewidth=.7,
                zorder=3, label=r'$\omega_d/2\pi$')
    ax.plot(x, d.f_coherent_MHz, 'o-', color=TEAL, lw=1.5, ms=3.2, zorder=2, label=r'$\Omega_F/2\pi$')
    ax.plot(x, d.EP_halfdifference_Mrad_s/(2*np.pi), '--', color=RED, lw=1.45, zorder=2, label=r'$D/2\pi$')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)', ylabel=r'$f$ (MHz)')
    # data rise steeply on the right; park the legend in the empty upper-left corner
    ax.set_ylim(top=float(d.f_coherent_MHz.max())*1.28)
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)

    # ---- c  distance to criticality
    ax = style(axs[1, 0]); panel(ax, 'c')
    ax.fill_between(mc.bias_mV, mc.ratio_p16, mc.ratio_p84, color=TEAL, alpha=.14, lw=0)
    ax.plot(mc.bias_mV, mc.ratio_med, 'o-', color=TEAL, lw=1.55, ms=3.8, zorder=3)
    ax.axhline(1, color=RED, ls='--', lw=1.15, zorder=2)
    imin = int(np.argmin(mc.ratio_med.to_numpy()))
    bx = float(mc.bias_mV.iloc[imin]); by = float(mc.ratio_med.iloc[imin])
    ax.scatter([bx], [by], s=40, color=ORANGE, edgecolor='white', lw=.7, zorder=5)
    ax.annotate(fr'closest: ${by:.2f}$', xy=(bx, by), xytext=(bx+26, by+9.8),
                ha='left', va='center', fontsize=FS_ANNOT, color=DARK, bbox=abox(),
                arrowprops=dict(arrowstyle='->', lw=.8, color=GREY, shrinkA=8, shrinkB=9))
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)', ylabel=r'$\Omega_F/D$')
    ax.set_ylim(.55, float(max(mc.ratio_p84))*1.10)

    # ---- d  measured vs critical envelope
    ax = style(axs[1, 1]); panel(ax, 'd')
    ax.errorbar(x, d.T2Rabi_ns, yerr=d.T2Rabi_err_ns, fmt='o-', color=NAVY, mfc='white',
                mec=NAVY, mew=.8, ecolor=MID, elinewidth=.7, lw=1.4, ms=3.6,
                zorder=3, label=r'$T_{2,\rm Rabi}$')
    ax.fill_between(mc.bias_mV, mc.Tcrit_p16_ns, mc.Tcrit_p84_ns, color=ORANGE, alpha=.14, lw=0)
    ax.plot(mc.bias_mV, mc.Tcrit_med_ns, 's--', color=ORANGE, lw=1.45, ms=3.2,
            zorder=3, label=r'$T_{\rm crit}$')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)', ylabel=r'$T$ (ns)')
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)
    save(fig, 'fig2_experiment')


# ============================================================ Figure 3
def fig3():
    G1, G2 = 1.0, 3.5
    D = abs(G1-G2)/2; A = (G1+G2)/2
    Om = np.linspace(0, 2.2*D, 900)
    root = np.sqrt((D**2-Om**2)+0j); lp = -A+root; lm = -A-root
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)

    # ---- a
    ax = style(axs[0, 0]); panel(ax, 'a')
    ax.plot(Om/D, lp.real/D, color=TEAL, lw=1.55, label=r'$\mathrm{Re}\,\lambda_+/D$')
    ax.plot(Om/D, lm.real/D, color=NAVY, lw=1.55, label=r'$\mathrm{Re}\,\lambda_-/D$')
    ax.plot(Om/D, np.abs(lp.imag)/D, color=ORANGE, lw=1.55, label=r'$|\mathrm{Im}\,\lambda_\pm|/D$')
    ax.axvline(1, color=RED, ls='--', lw=1.1)
    ax.text(1.05, -0.42, 'EP', color=RED, fontsize=FS_ANNOT, bbox=abox(0.8, 1.2))
    ax.set(xlabel=r'$\Omega_F/D$', ylabel=r'$\lambda/D$')
    ax.set_ylim(-3.25, 2.45)
    ax.legend(ncol=1, loc='upper left', bbox_to_anchor=(.02, .98), **LEG)

    # ---- b
    ax = style(axs[0, 1], xminor=False, yminor=False); panel(ax, 'b')
    eps = np.logspace(-5, -.15, 220)
    split = 2*np.sqrt(2*D**2*eps + eps**2*D**2)
    ax.loglog(eps, split/D, color=NAVY, lw=1.55, label='exact')
    ax.loglog(eps, 2*np.sqrt(2)*np.sqrt(eps), '--', color=ORANGE, lw=1.35, label=r'$\propto\sqrt{\epsilon}$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set(xlabel=r'$\epsilon$', ylabel=r'$|\Delta\lambda|/D$')
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)

    # ---- c
    ax = style(axs[1, 0]); panel(ax, 'c')
    t = np.linspace(0, 4/A, 500)
    yj = (1+A*t)*np.exp(-A*t)
    yu = np.exp(-A*t)*np.cos(1.2*A*t)
    yo = .5*np.exp(-.55*A*t) + .5*np.exp(-1.55*A*t)
    ax.plot(t*A, yu, color=TEAL, lw=1.45, label='underdamped')
    ax.plot(t*A, yj, color=RED,  lw=1.65, label='EP')
    ax.plot(t*A, yo, color=NAVY, lw=1.45, label='overdamped')
    ax.set(xlabel=r'$At$', ylabel=r'$z(t)/z(0)$')
    ax.legend(loc='upper right', bbox_to_anchor=(.98, .98), **LEG)

    # ---- d
    ax = style(axs[1, 1]); panel(ax, 'd')
    ax.plot(lp.real/D, lp.imag/D, color=TEAL, lw=1.5)
    ax.plot(lm.real/D, lm.imag/D, color=NAVY, lw=1.5)
    idx = int(np.argmin(abs(Om-D)))
    xc, yc = lp.real[idx]/D, lp.imag[idx]/D
    ax.scatter([xc], [yc], s=44, color=RED, edgecolor='white', lw=.7, zorder=5)
    ax.axhline(0, color='#D6DDE2', lw=.8, zorder=1)
    ax.set(xlabel=r'$\mathrm{Re}\,\lambda/D$', ylabel=r'$\mathrm{Im}\,\lambda/D$')
    ax.set_aspect('equal', adjustable='datalim')
    # move 'coalescence' sideways into the empty upper-left quadrant with a clear gap
    ax.annotate('coalescence', xy=(xc, yc), xytext=(-4.30, 1.28),
                fontsize=FS_ANNOT, color=DARK, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=.8, color=GREY, shrinkA=4, shrinkB=8))
    save(fig, 'fig3_ep_anatomy')


# ============================================================ Figure 4
def fig4():
    ep   = pd.read_csv(RES/'floquet_ep_scan.csv')
    conv = pd.read_csv(RES/'floquet_convergence.csv')
    sens = pd.read_csv(RES/'transport_sensitivity.csv')
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)

    # ---- a
    ax = style(axs[0, 0]); panel(ax, 'a')
    ax.plot(ep.aD, ep.fEP_MHz, 'o-', color=TEAL, lw=1.55, ms=3.7)
    ax.set(xlabel=r'$a_D$', ylabel=r'$f_{\rm EP}$ (MHz)')

    # ---- b   gap and current
    ax = style(axs[0, 1]); panel(ax, 'b')
    l1 = ax.plot(ep.aD, ep.gap_Mrad_s, 's-', color=ORANGE, lw=1.5, ms=3.5, label=r'$\Delta_{\mathcal{L}}$')
    ax2 = twin(ax, NAVY)
    l2 = ax2.plot(ep.aD, ep.current_pA, 'o--', color=NAVY, lw=1.3, ms=3.3, label=r'$I$')
    ax.set(xlabel=r'$a_D$', ylabel=r'$\Delta_{\mathcal{L}}$ ($10^6$ s$^{-1}$)')
    ax2.set_ylabel(r'$I$ (pA)', color=NAVY)
    ln = l1+l2
    ax.legend(ln, [z.get_label() for z in ln], loc='upper left', bbox_to_anchor=(.02, .98), **LEG)

    # ---- c   Floquet discretisation
    ax = style(axs[1, 0]); panel(ax, 'c')
    ax.plot(conv.steps_per_period, conv.fEP_MHz, 'o-', color=PURPLE, lw=1.55, ms=3.6)
    fit = json.load(open(RES/'floquet_convergence_fit.json'))
    ax.axhline(fit['f_inf_MHz'], color='#AFB9C0', ls='--', lw=.9)
    ax.text(.97, .10, r'$f_\infty=%.3f$ MHz' % fit['f_inf_MHz'], transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT-0.4, color=GREY)
    ax.set(xlabel=r'$N_t$', ylabel=r'$f_{\rm EP}$ (MHz)')
    axin = ax.inset_axes([.50, .52, .45, .37])
    axin.semilogy(conv.steps_per_period, conv.sep_norm, 'o-', color=RED, ms=2.6, lw=1.0,
                  label=r'$|\Delta\mu|/\Sigma|\mu|$')
    axin.semilogy(conv.steps_per_period, conv.phase_rigidity, 's--', color=NAVY, ms=2.4, lw=.9,
                  label=r'$r$')
    axin.set_ylim(1e-8, 1e-4)
    for s_ in axin.spines.values(): s_.set_visible(True); s_.set_linewidth(.65); s_.set_color('#555E64')
    axin.tick_params(which='both', direction='in', top=True, right=True, labelsize=FS_INSET-0.4, length=2)
    axin.set_xlabel(r'$N_t$', fontsize=FS_INSET, labelpad=0)
    axin.set_ylabel('EP defect', fontsize=FS_INSET, labelpad=1)
    axin.legend(fontsize=FS_INSET-0.6, loc='upper right', frameon=False, handlelength=1.2, borderaxespad=.2)
    axin.patch.set_facecolor('white'); axin.patch.set_alpha(.96)

    # ---- d   tip polarisation / angle grid
    ax = style(axs[1, 1]); panel(ax, 'd')
    for _, grp in sens.groupby('pL'):
        ax.plot(grp.theta_deg, grp.fEP_MHz, color='#C6CDD2', lw=.7, zorder=1)
    sc = ax.scatter(sens.theta_deg, sens.fEP_MHz, c=sens.pL, s=38, cmap='viridis',
                    edgecolor='white', linewidth=.45, zorder=3)
    ax.set(xlabel=r'$\theta$ (deg)', ylabel=r'$f_{\rm EP}$ (MHz)')
    # widen the x-range so the colour bar sits inside the box, clear of every marker
    ax.set_xlim(12.5, 52.0)
    cax = ax.inset_axes([.775, .13, .030, .58])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r'$p_L$', fontsize=FS_TICK, labelpad=3)
    cb.ax.tick_params(labelsize=FS_TICK-0.8, direction='in', length=2.2, pad=1.6)
    cb.outline.set_linewidth(.65)
    save(fig, 'fig4_floquet_transport')


# ============================================================ Figure 5
def fig5():
    m  = pd.read_csv(RES/'metrology_scan.csv')
    x  = m.Omega_over_EP.to_numpy()
    fi = m.counting_FI_per_rad2s.to_numpy()
    imax = int(np.nanargmax(fi)); xopt = x[imax]
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.62), constrained_layout=True)

    # ---- a   coalescence diagnostics
    ax = style(axs[0]); panel(ax, 'a', x=-.20)
    l1 = ax.plot(x, m.pair_sep_Mrad_s, color=NAVY, lw=1.5, label=r'$|\Delta\mu|$')
    ax2 = twin(ax, RED)
    l2 = ax2.plot(x, m.overlap, color=RED, lw=1.4, label=r'$\mathcal{O}_R$')
    ax.axvline(1, color=ORANGE, ls='--', lw=1.05)
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$', ylabel=r'$|\Delta\mu|$ ($10^6$ s$^{-1}$)')
    ax2.set_ylabel(r'$\mathcal{O}_R$', color=RED)
    ax.set_ylim(top=float(m.pair_sep_Mrad_s.max())*1.30)   # open space at the top
    ln = l1+l2
    ax.legend(ln, [z.get_label() for z in ln], loc='upper center',
              bbox_to_anchor=(.50, .99), ncol=2, **LEG)

    # ---- b   counting Fisher information
    ax = style(axs[1]); panel(ax, 'b', x=-.20)
    ax.plot(x, fi/fi[imax], color=TEAL, lw=1.6, label=r'$\dot{\mathcal{F}}_{\rm count}$')
    ax.axvline(1, color=ORANGE, ls='--', lw=1.05, label='EP')
    ax.axvline(xopt, color=PURPLE, ls=':', lw=1.2, label='optimum')
    ax.scatter([xopt], [1], s=30, color=PURPLE, edgecolor='white', lw=.6, zorder=5)
    ax.annotate(fr'$\Omega_F\simeq {xopt:.2f}\,\Omega_{{\rm EP}}$', xy=(xopt, 1.0),
                xytext=(5.0, 1.13), fontsize=FS_ANNOT, ha='left', va='center', color=DARK,
                arrowprops=dict(arrowstyle='->', lw=.8, color=GREY, shrinkA=3, shrinkB=6))
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$',
           ylabel=r'$\dot{\mathcal{F}}_{\rm count}/\dot{\mathcal{F}}_{\max}$', ylim=(0.0, 1.26))
    ax.legend(ncol=1, loc='lower right', bbox_to_anchor=(.99, .03), **LEG)

    # ---- c   current and occupation
    ax = style(axs[2]); panel(ax, 'c', x=-.20)
    l1 = ax.plot(x, m.current_pA, color=NAVY, lw=1.5, label=r'$I$')
    ax2 = twin(ax, GREEN)
    l2 = ax2.plot(x, m.singly_occ, color=GREEN, lw=1.4, label=r'$P_1$')
    ax.axvline(1, color=ORANGE, ls='--', lw=1.05)
    ax.axvline(xopt, color=PURPLE, ls=':', lw=1.0)
    ax.set(xlabel=r'$\Omega_F/\Omega_{\rm EP}$', ylabel=r'$I$ (pA)')
    ax2.set_ylabel(r'$P_1$', color=GREEN)
    # I rises and P1 falls; both curves cross the middle, so open the top and put the
    # legend in the resulting clear band
    lo, hi = float(m.current_pA.min()), float(m.current_pA.max())
    ax.set_ylim(lo-0.06*(hi-lo), hi+0.34*(hi-lo))
    ln = l1+l2
    ax.legend(ln, [z.get_label() for z in ln], loc='upper center',
              bbox_to_anchor=(.50, .99), ncol=2, **LEG)
    save(fig, 'fig5_metrology')


# ============================================================ Figure 6
def _flat_pentacene(ax, x0, y0, n=5, r=.048, squash=.32, colour=TEAL):
    """Five fused hexagons in perspective: a FLAT-LYING molecule on the surface."""
    ang = np.arange(6)*np.pi/3 + np.pi/6
    dx = r*np.sqrt(3)
    for k in range(n):
        cx = x0 + k*dx
        pts = np.c_[cx + r*np.cos(ang), y0 + r*squash*np.sin(ang)]
        ax.add_patch(Polygon(pts, closed=True, fc='white', ec=colour, lw=1.05, zorder=4))
    return x0 + (n-1)*dx


def fig6():
    d = pd.read_csv(RES/'target_gap_scan.csv')
    fit = json.load(open(RES/'target_gap_fit.json'))
    J = d.J_over_2pi_MHz.to_numpy(); gap = d['gap_s-1'].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(7.25, 2.80), constrained_layout=True,
                            width_ratios=[.95, 1.45])

    # ---- a   schematic: flat-lying pentacene, slender tip, well-separated labels
    ax = axs[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); panel(ax, 'a', x=-.09)
    # substrate slab, drawn in perspective
    ax.add_patch(Polygon([[.02, .120], [.98, .120], [.905, .225], [.095, .225]],
                         closed=True, fc='#EFF2F4', ec='#93A0A8', lw=1.0, zorder=1))
    ax.text(.50, .055, 'MgO/Ag(001)', ha='center', fontsize=FS_SKETCH, color=GREY)

    # flat-lying pentacene on the surface, left of centre
    xs0 = .175
    xend = _flat_pentacene(ax, xs0, .250, n=5, r=.047, squash=.32, colour=TEAL)
    xmid = (xs0 + xend)/2
    ax.text(xmid, .175, 'pentacene sensor', ha='center', va='top',
            fontsize=FS_SKETCH, color=TEAL)

    # tip above the molecule
    _draw_tip(ax, xmid, .96, .43)
    # label placed in the clear upper-right area with a short leader
    ax.plot([xmid+.075, .545], [.855, .855], color=MID, lw=.7, zorder=2)
    ax.text(.565, .855, 'spin-polarized\ntip', ha='left', va='center',
            fontsize=FS_SKETCH, color=NAVY, linespacing=1.25)

    # tunnelling current
    ax.add_patch(FancyArrowPatch((xmid, .425), (xmid, .318), arrowstyle='-|>',
                                 mutation_scale=9, lw=1.0, color=NAVY,
                                 ls=(0, (2.2, 1.6)), zorder=6))
    ax.text(xmid-.048, .372, r'$I$', fontsize=FS_SKMATH, color=NAVY,
            ha='center', va='center')

    # sensor spin on the molecule
    ax.arrow(xmid+.052, .258, 0, .080, width=.0045, head_width=.028, head_length=.028,
             color=TEAL, length_includes_head=True, zorder=7)

    # target spin further along the same surface
    tx = .845
    ax.add_patch(Circle((tx, .272), .048, fc='#F2F7F2', ec=GREEN, lw=1.3, zorder=4))
    ax.arrow(tx, .252, .016, .085, width=.0045, head_width=.030, head_length=.030,
             color=GREEN, length_includes_head=True, zorder=5)
    ax.text(tx, .175, 'target spin', ha='center', va='top', fontsize=FS_SKETCH, color=GREEN)

    # exchange coupling
    ax.add_patch(Arc(((xmid+tx)/2, .285), tx-xmid, .26, theta1=16, theta2=164,
                     color=PURPLE, lw=1.15, ls='--', zorder=3))
    ax.text((xmid+tx)/2, .445, r'$J$', ha='center', va='center',
            fontsize=FS_SKMATH, color=PURPLE)

    # ---- b   gap vs coupling
    ax = style(axs[1], xminor=False, yminor=False); panel(ax, 'b', x=-.12)
    pred = fit['gap0_fit_s-1'] + fit['quadratic_coefficient_s']*(2*np.pi*J*1e6)**2
    ax.loglog(J[1:], gap[1:], color=TEAL, lw=1.65, zorder=3, label=r'$\Delta_{\mathcal{L}}$')
    weak = (J > 0) & (J <= 0.03)
    ax.loglog(J[weak], pred[weak], '--', color=ORANGE, lw=1.4, zorder=4, label=r'$\Delta_0+CJ^2$')
    ax.axhline(fit['gap0_fit_s-1'], color=NAVY, ls=':', lw=1.1, zorder=2, label=r'$\Delta_0$')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set(xlabel=r'$J/2\pi$ (MHz)', ylabel=r'$\Delta_{\mathcal{L}}$ (s$^{-1}$)')
    ax.legend(loc='lower right', bbox_to_anchor=(.99, .03), **LEG)
    ins = ax.inset_axes([.10, .55, .37, .36])
    rel = (gap - fit['gap0_fit_s-1'])/fit['gap0_fit_s-1']
    ins.loglog(J[1:], np.maximum(rel[1:], 1e-12), color=PURPLE, lw=1.15)
    ins.axhline(1, color='#ADB7BE', ls='--', lw=.8)
    for s in ins.spines.values(): s.set_visible(True); s.set_linewidth(.6); s.set_color('#555E64')
    ins.tick_params(which='both', direction='in', top=True, right=True,
                    labelsize=FS_INSET-0.4, length=2)
    ins.set_xlabel(r'$J/2\pi$', fontsize=FS_INSET, labelpad=0)
    ins.set_ylabel(r'$(\Delta-\Delta_0)/\Delta_0$', fontsize=FS_INSET, labelpad=1)
    ins.patch.set_facecolor('white'); ins.patch.set_alpha(.95)
    save(fig, 'fig6_target_gap')


# ============================================================ Figure S1 (raw data only)
T1_NS = 138.23; G1 = 1/T1_NS


def kov_model(t, c, A, T, f):
    return c + A*np.exp(-t/T)*np.cos(2*np.pi*f*1e-3*t)


def liouv_model(t, c, A, G2, Om_MHz):
    Om = 2*np.pi*Om_MHz*1e-3; D = .5*(G1-G2); wd2 = Om*Om - D*D
    wd = np.sqrt(np.maximum(wd2, 1e-12)); aa = .5*(G1+G2); q = (G2-G1)/(2*wd)
    return c + A*np.exp(-aa*t)*(np.cos(wd*t) + q*np.sin(wd*t))


def figS1():
    raw = pd.read_csv(RAW/'SM/S21/AllSpectra.csv'); fp = pd.read_csv(RAW/'F4/E-F/FittedParameters.csv')
    biases = sorted([int(c.split('@')[1].split(' mV')[0]) for c in raw.columns if c.startswith('Pulse duration')])
    rows, curves = [], {}
    for V in biases:
        t = raw[f'Pulse duration (ns)@{V} mV'].to_numpy(float); y = raw[f'dI (fA)@{V} mV'].to_numpy(float)
        r = fp.loc[np.isclose(fp['U_DC (mV)'], V)]
        if r.empty: continue
        r = r.iloc[0]; T0 = float(r['T_2 (ns)']); f0 = float(r['Omega/2pi (MHz)'])
        p0 = [np.mean(y[-10:]), y[0]-np.mean(y[-10:]), T0, f0]
        pk, _ = curve_fit(kov_model, t, y, p0=p0, bounds=([-1500, -3000, 1, .1], [1500, 3000, 500, 500]), maxfev=200000)
        G20 = max(1e-5, 2/T0-G1); Om0 = np.sqrt((2*np.pi*f0*1e-3)**2 + (.5*(G1-G20))**2)/(2*np.pi*1e-3)
        pl0 = [np.mean(y[-10:]), y[0]-np.mean(y[-10:]), G20, Om0]
        pl, _ = curve_fit(liouv_model, t, y, p0=pl0, bounds=([-1500, -3000, 1e-5, .1], [1500, 3000, 2, 500]), maxfev=200000)
        n = len(y); k = 4

        def stat(pred):
            rss = float(np.sum((y-pred)**2)); rm = np.sqrt(rss/n)
            aic = n*np.log(rss/n)+2*k; return rm, aic+2*k*(k+1)/(n-k-1), n*np.log(rss/n)+k*np.log(n)
        sk = stat(kov_model(t, *pk)); sl = stat(liouv_model(t, *pl))
        rows.append(dict(bias_mV=V, rmse_published=sk[0], rmse_liouv=sl[0], aicc_published=sk[1],
                         aicc_liouv=sl[1], delta_aicc=sl[1]-sk[1], bic_published=sk[2], bic_liouv=sl[2]))
        tt = np.linspace(t.min(), t.max(), 700)
        curves[V] = (t, y, kov_model(tt, *pk), liouv_model(tt, *pl), tt)
    stats = pd.DataFrame(rows); stats.to_csv(HERE/'fit_comparison_S1.csv', index=False)

    chosen = [-60, -100, -140, -180, -220, -260]
    fig, axs = plt.subplots(2, 1, figsize=(7.25, 5.7), constrained_layout=True,
                            gridspec_kw={'height_ratios': [2.2, 1.0]})
    ax = style(axs[0]); panel(ax, 'a', x=-.06)
    offsets = np.arange(len(chosen))[::-1]*380.0
    for off, V in zip(offsets, chosen):
        t, y, kfit, lfit, tt = curves[V]
        ax.plot(t, y-np.mean(y[-12:])+off, 'o', ms=1.9, mfc='#CBD2D7', mec='#89959E', mew=.35)
        ax.plot(tt, kfit-np.mean(kfit[-20:])+off, '--', color=NAVY, lw=1.15)
        ax.plot(tt, lfit-np.mean(lfit[-20:])+off, color=ORANGE, lw=1.25)
        ax.text(tt.max()*.98, off+22, fr'$U_{{\rm DC}}={V}$ mV', ha='right', va='bottom',
                fontsize=FS_ANNOT, color=DARK)
    ax.plot([], [], 'o', ms=3.0, mfc='#CBD2D7', mec='#89959E', mew=.35, label='data')
    ax.plot([], [], '--', color=NAVY, lw=1.2, label='published fit')
    ax.plot([], [], color=ORANGE, lw=1.3, label='Liouvillian fit')
    ax.set(xlabel=r'$t_p$ (ns)', ylabel=r'offset $\Delta I$ (fA)')
    ax.legend(loc='upper right', bbox_to_anchor=(.99, .99), **LEG)
    ax = style(axs[1]); panel(ax, 'b', x=-.06)
    ax.axhline(0, color='#8F999F', lw=.85)
    pos = stats.delta_aicc < 0
    ax.plot(stats.bias_mV, stats.delta_aicc, 'o-', color=PURPLE, lw=1.25, ms=3.2)
    ax.scatter(stats.loc[pos, 'bias_mV'], stats.loc[pos, 'delta_aicc'], s=30, color=ORANGE,
               zorder=3, label='Liouvillian preferred')
    ax.scatter(stats.loc[~pos, 'bias_mV'], stats.loc[~pos, 'delta_aicc'], s=30, facecolor='white',
               edgecolor=NAVY, lw=.8, zorder=3, label='published fit preferred')
    ax.set(xlabel=r'$U_{\rm DC}$ (mV)', ylabel=r'$\Delta\mathrm{AIC}_c$')
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(.02, .98), **LEG)
    save(fig, 'figS1_rabi_overlays')


# ============================================================ Figure S2
# ============================================================ Figure S3
def figS3():
    J = 1.; gs = .25; N = 24; times = np.linspace(0, 35, 180)
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.62), constrained_layout=True)
    for k, (gt, label) in enumerate([(0, r'$\gamma_t=0$'), (.03, r'$\gamma_t/J=0.03$')]):
        ax = axs[k]; panel(ax, chr(97+k), x=-.20)
        dat = propagate_single_excitation(N, times, J, gs, gt).T
        im = ax.imshow(dat, origin='lower', aspect='auto',
                       extent=[times.min(), times.max(), 1, N], cmap='magma', vmin=0, vmax=1)
        style(ax, xminor=False, yminor=False)
        ax.set(xlabel=r'$Jt$', ylabel='site $j$')
        ax.text(.05, .95, label, transform=ax.transAxes, ha='left', va='top',
                color='white', fontsize=FS_ANNOT)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        cb.set_label(r'$|c_j|^2$', fontsize=FS_TICK)
        cb.ax.tick_params(labelsize=FS_TICK-0.8, direction='in')
    ax = style(axs[2], xminor=False, yminor=False); panel(ax, 'c', x=-.20)
    ratios = np.logspace(1, 5, 160); gt = 1/ratios; gs_arr = .25+gt
    xi = np.array([propagation_length_readout(J, g) for g in gt])
    nf = np.array([intrinsic_floor_length(J, s, t) for s, t in zip(gs_arr, gt)])
    nr = np.array([mode_resolvability_length(J, g) for g in gt])
    ax.loglog(ratios, xi, color=NAVY,   lw=1.5, label=r'$\xi_{\rm read}$')
    ax.loglog(ratios, nf, color=ORANGE, lw=1.5, label=r'$N_{\rm floor}$')
    ax.loglog(ratios, nr, color=TEAL,   lw=1.5, label=r'$N_{\rm res}$')
    ax.set(xlabel=r'$J/\gamma_t$', ylabel='sites')
    ax.legend(loc='upper left', bbox_to_anchor=(.02, .98), **LEG)
    pd.DataFrame({'J_over_gamma_t': ratios, 'xi_read': xi, 'N_floor': nf, 'N_res': nr})\
        .to_csv(RES/'multispin_length_scales.csv', index=False)
    save(fig, 'figS3_multispin_propagation')


# ============================================================ Figure S4
def figS4():
    df = pd.DataFrame(validation_scan()); df.to_csv(RES/'multispin_validation.csv', index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.62), constrained_layout=True)
    colours = {2: NAVY, 3: TEAL, 4: ORANGE, 5: PURPLE}
    for N, g in df.groupby('N'):
        axs[0].loglog(g.Omega_over_J, g.linear_response_error, 'o-', ms=3.0, lw=1.15,
                      color=colours[N], label=fr'$N={N}$')
        axs[1].loglog(g.Omega_over_J, g.excitation_density, 'o-', ms=3.0, lw=1.15,
                      color=colours[N], label=fr'$N={N}$')
        if N <= 4:
            axs[2].loglog(g.Omega_over_J, g.xxz_xy_response_difference, 'o-', ms=3.0, lw=1.15,
                          color=colours[N], label=fr'$N={N}$')
    for i, ax in enumerate(axs):
        style(ax, xminor=False, yminor=False); panel(ax, chr(97+i), x=-.20)
        ax.set_xlabel(r'$\Omega_F/J$')
    axs[0].set_ylabel('linear-response error')
    axs[0].axhline(.05, color=GREY, ls='--', lw=.9)
    axs[0].legend(ncol=2, loc='lower right', bbox_to_anchor=(.99, .03), **LEG)
    axs[1].set_ylabel(r'excitation density $n_{\rm ex}$')
    axs[2].set_ylabel('XXZ-XY response difference')
    axs[2].axhline(.05, color=GREY, ls='--', lw=.9)
    # data occupy the upper band and fall to the right; lower left is the only clear corner
    axs[2].set_ylim(bottom=float(df.xxz_xy_response_difference.min())*0.20)
    axs[2].legend(loc='lower left', bbox_to_anchor=(.02, .03), **LEG)
    save(fig, 'figS4_multispin_validation')


# ============================================================ driver
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=None,
                    help='root of the archived experimental repository; required only for Fig. 2a and Fig. S1')
    a = ap.parse_args()
    if a.data_root: RAW = Path(a.data_root)
    OUT.mkdir(parents=True, exist_ok=True)
    print('Generating figures ->', OUT)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    figS2(); figS3(); figS4(); figS5()
    if RAW is not None:
        figS1()
    else:
        print('  (Fig. S1 skipped: needs --data-root)')
    print('done')
