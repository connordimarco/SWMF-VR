#!/usr/bin/env python3
"""Draw the field-line shells in GSM and GEI side by side (system python3).

Reads the traced shells and the exported R_gei, rotates the shells into GEI, and
plots both. The GSM axes show up as R_gei's columns in the GEI panel, so you can
see the rotation. Colored by radius.

  python3 scripts/plot.py [--ts STAMP] [--gsm-root DIR] [--transforms CSV] [--out PNG]
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import frames
from swmf_vr.data import parse_obj_timestamp

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RUN = os.path.join(REPO, 'run')
SURFACES = ['obj_inner', 'obj_outer', 'obj_max']


def read_obj(path):
    """points, the per-vertex color coord U (from vt), and the polylines."""
    pts, us, lines = [], [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith('v '):
                p = ln.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith('vt '):
                us.append(float(ln.split()[1]))
            elif ln.startswith('l '):
                lines.append(np.array([int(t.split('/')[0]) - 1 for t in ln.split()[1:]], dtype=int))
    return np.array(pts), np.array(us), lines


def read_R_gei(ts, csv):
    with open(csv) as f:
        col = {n: i for i, n in enumerate(f.readline().rstrip('\n').split(','))}
        for line in f:
            row = line.rstrip('\n').split(',')
            if row[0] == ts:
                R = np.array([[float(row[col['R_gei_{}{}'.format(r, c)]]) for c in range(3)] for r in range(3)])
                dip = np.array([float(row[col['dipole_gei_' + a]]) for a in 'xyz'])
                return R, dip
    raise ValueError('{} not in {}'.format(ts, csv))


def draw(ax, polylines, radii, norm, title, basis=None, dipole=None):
    cmap = get_cmap('inferno')
    for seg, r in zip(polylines, radii):
        if len(seg) < 2:
            continue
        lc = Line3DCollection(np.stack([seg[:-1], seg[1:]], axis=1), cmap=cmap, norm=norm, linewidths=0.6)
        lc.set_array(0.5 * (r[:-1] + r[1:]))
        ax.add_collection3d(lc)
    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:8j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color='steelblue', alpha=0.5, linewidth=0)
    if basis is not None:
        for k, c in enumerate(['#ff5555', '#55ff55', '#5599ff']):
            vhat = basis[:, k] * 3.0
            ax.plot([0, vhat[0]], [0, vhat[1]], [0, vhat[2]], color=c, lw=2.5)
    if dipole is not None:
        d = dipole / np.linalg.norm(dipole) * 4.0
        ax.plot([-d[0], d[0]], [-d[1], d[1]], [-d[2], d[2]], color='magenta', lw=1.8, linestyle='--')
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=12, azim=-72)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', default='20240510_210000')
    p.add_argument('--gsm-root', default=RUN)
    p.add_argument('--transforms', default=os.path.join(RUN, 'gsm_transforms.csv'))
    p.add_argument('--out', default=None)
    args = p.parse_args()

    ts = args.ts
    out = args.out or os.path.join(RUN, 'shell_renders', 'shells_{}.png'.format(ts))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dt = parse_obj_timestamp('fieldlines_{}.obj'.format(ts))
    R, dip_gei = read_R_gei(ts, args.transforms)
    frames.recalc_for(dt)
    from geopack import geopack
    dip_gsm = np.array(geopack.smgsm(0.0, 0.0, 1.0, 1))

    gsm_poly, gei_poly, vals = [], [], []
    for s in SURFACES:
        path = os.path.join(args.gsm_root, s, 'fieldlines_{}.obj'.format(ts))
        if not os.path.exists(path):
            continue
        pts, us, lines = read_obj(path)
        pts_gei = pts @ R.T
        for ln in lines:
            gsm_poly.append(pts[ln]); gei_poly.append(pts_gei[ln]); vals.append(us[ln])
    if not vals:
        print('no shells under {} for {}'.format(args.gsm_root, ts)); return

    # The OBJ vt.u is already the flux color coordinate in [0,1].
    meta = {}
    cj = os.path.join(args.gsm_root, 'obj_outer', 'colormap.json')
    if os.path.exists(cj):
        meta = json.load(open(cj))
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(16, 8))
    draw(fig.add_subplot(121, projection='3d'), gsm_poly, vals, norm,
         'GSM  (as traced)  {}'.format(ts), basis=np.eye(3), dipole=dip_gsm)
    draw(fig.add_subplot(122, projection='3d'), gei_poly, vals, norm,
         'GEI  (for VR)', basis=R, dipole=dip_gei)

    sm = cm.ScalarMappable(norm=norm, cmap='inferno'); sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02)
    if meta:
        cb.set_label('{}   U=0 -> {:g},  U=1 -> {:g}{}'.format(
            meta.get('scalar', 'flux'), meta.get('vmin', 0), meta.get('vmax', 1),
            ' (log)' if meta.get('log') else ''))
    else:
        cb.set_label('flux color (U)')
    fig.suptitle('Belt shells GSM -> GEI, colored by flux  '
                 '(axes: red=GSM x->Sun, blue=z; dashed=dipole)', fontsize=12)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main()
