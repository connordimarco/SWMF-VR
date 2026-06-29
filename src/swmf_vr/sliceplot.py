"""Despeckle + render the y=0 slice maps. Pure numpy/matplotlib (no ParaView),
so both pvbatch and system python3 can import it -- the heavy trace saves the
raw flux to an .npz and the cleanup/plot can be re-tuned offline from that.
"""

import os
import copy
import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Fixed (vmin, vmax) per band so every frame shares one color scale -- required
# for a flicker-free movie. vmax sits just above the global equatorial-omni peak
# over the whole run (the true ceiling); each band spans 4 decades.
LIMITS = {
    '<1 MeV':  (1.5e6, 1.5e10),
    '1-2 MeV': (2.0e4, 2.0e8),
    '>2 MeV':  (1.0e4, 1.0e8),
}


def despeckle(f, floor=1e-10, window=5, drop=1.5, passes=12):
    """Replace ONLY the bad pixels -- ones isolated-dark in a brighter neighbour-
    hood, where a trace got a bad min-|B| and the mapped flux dropped well below
    its surroundings (from a full ~1e-46 collapse down to a mild dip), plus
    embedded holes (NaN surrounded by data). A pixel is bad if it sits more than
    `drop` decades below its local median, or below the absolute `floor`, or is an
    NaN with mostly-finite neighbours. Flag them, blank them, and fill just those
    from the median of good neighbours -- real faint regions (whose local median
    is also low) and the big empty exterior are left untouched."""
    from scipy.ndimage import generic_filter, uniform_filter
    f = np.asarray(f, float).copy()
    fin = np.isfinite(f)
    logf = np.where(fin & (f > 0), np.log10(f), np.nan)
    ref = np.where(fin & (f >= floor), np.log10(f), np.nan)   # garbage excluded from the reference

    def nanmed(w):
        w = w[np.isfinite(w)]
        return np.median(w) if w.size else np.nan

    # local median of the *real-valued* neighbours -- if the garbage were left in,
    # it would drag the median down and pixels next to a garbage cluster (the inner
    # edge) wouldn't register as outliers.
    med = generic_filter(ref, nanmed, size=window, mode='constant', cval=np.nan)
    frac = uniform_filter(fin.astype(float), size=window)
    with np.errstate(invalid='ignore'):
        outlier = fin & np.isfinite(med) & (logf < med - drop)   # far below local median
    bad = outlier | (fin & (f < floor)) | (~fin & (frac >= 0.5))
    f[bad] = np.nan
    r = window // 2
    for _ in range(passes):
        ys, xs = np.where(bad & ~np.isfinite(f))
        if len(ys) == 0:
            break
        nxt = f.copy(); changed = False
        for y, x in zip(ys, xs):
            nb = f[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            v = nb[np.isfinite(nb)]
            if v.size:
                nxt[y, x] = float(np.median(v)); changed = True
        f = nxt
        if not changed:
            break
    return f


# Context strips drawn under the maps: (npz key, label, color).
STRIPS = [('bz', 'IMF Bz (nT)', '#4488ff'),
          ('al', 'AL (nT)', '#22aa55'),
          ('dst', 'Dst (nT)', '#ee4444')]


def _draw_strips(fig, axes, idx, ts):
    """Bz / AL / Dst time series with a red line at this frame's time."""
    base = datetime.datetime.strptime(str(idx['base_iso']), '%Y-%m-%dT%H:%M:%S')
    t_now = (datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S') - base).total_seconds() / 3600.0
    t_end = float(idx['t_end'])
    for ax, (key, lbl, color) in zip(axes, STRIPS):
        ax.plot(idx[key + '_t'], idx[key + '_v'], color=color, lw=0.9)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.axvline(t_now, color='red', lw=2.2, zorder=10)
        ax.set_xlim(0, t_end)
        ax.set_ylabel(lbl, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis='x', color='0.9', lw=0.5)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('hours since {} UT'.format(str(idx['base_iso']).replace('T', ' ')))


def plot_bands(xs, zs, flux, labels, ts, quantity, out_png, floor=1e-10, drop=1.5,
               limits=LIMITS, indices=None):
    """3-panel (one per energy band) log heatmap of the y=0 cut. `limits` fixes
    the color scale per band (movie-safe; pass None to auto-scale). If `indices`
    (path to the indices .npz) exists, add Bz/AL/Dst strips below with a red line
    at this frame's time."""
    flux = [despeckle(f, floor=floor, drop=drop) for f in flux]
    nb = len(flux)

    idx = dict(np.load(indices, allow_pickle=True)) if indices and os.path.exists(indices) else None
    if idx is not None:
        fig = plt.figure(figsize=(6.0 * nb, 11.5))
        gs = fig.add_gridspec(4, nb, height_ratios=[6, 1, 1, 1], hspace=0.55, wspace=0.22)
        axes = [fig.add_subplot(gs[0, c]) for c in range(nb)]
        strip_axes = [fig.add_subplot(gs[r, :]) for r in (1, 2, 3)]
    else:
        fig, axes = plt.subplots(1, nb, figsize=(6.0 * nb, 6.6))
        if nb == 1:
            axes = [axes]
        strip_axes = []

    X, Z = np.meshgrid(xs, zs)
    th = np.linspace(0, 2 * np.pi, 200)
    cmap = copy.copy(plt.get_cmap('inferno')); cmap.set_bad('#101015')
    unit = 'cm^-2 s^-1' + ('' if quantity == 'omni' else ' sr^-1')
    for ax, f, lbl in zip(axes, flux, labels):
        ax.set_aspect('equal'); ax.set_xlabel('X GSM (Re) [Sun ->]')
        fin = f[np.isfinite(f) & (f > 0)]
        if limits and lbl in limits:
            vmin, vmax = limits[lbl]                      # fixed scale (consistent across frames)
        elif fin.size:
            vmax = float(fin.max()); vmin = max(vmax / 1e4, float(fin.min()))
        else:
            ax.set_title('{}\n(nothing mapped)'.format(lbl)); ax.set_facecolor('#101015')
            continue
        pcm = ax.pcolormesh(X, Z, np.ma.masked_invalid(f), cmap=cmap,
                            norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
        ax.fill(np.cos(th), np.sin(th), color='steelblue', zorder=5)
        ax.plot(2 * np.cos(th), 2 * np.sin(th), '--', color='white', lw=0.8, zorder=5)
        ax.set_title('{}  ({} flux)'.format(lbl, quantity))
        cb = fig.colorbar(pcm, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label(unit)
    axes[0].set_ylabel('Z GSM (Re)')

    if idx is not None:
        _draw_strips(fig, strip_axes, idx, ts)

    fig.suptitle('CIMI flux mapped to 3D  --  y=0 GSM cut  --  {}'.format(ts), fontsize=14)
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
