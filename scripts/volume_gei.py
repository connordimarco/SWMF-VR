#!/usr/bin/env python3
"""Rotate 3D flux volumes from GSM into GEI (plain python3, no re-tracing).

The tracer samples in GSM; the VR app works in GEI/EME2000. This reads the raw
volume npz (points + flux) written by volume3d.py, applies that frame's
GSM->GEI rotation, and writes the delivery CSV -- same columns and flux values,
coordinates now GEI. (The points are a lattice in GSM, so in GEI they arrive as
a rotated lattice, not axis-aligned.)

  python3 scripts/volume_gei.py run/volume_renders/npz/volume_*_bands.npz
       [--out-dir DIR]
"""

import os
import sys
import glob
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import frames

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def write_csv(path, pts, flux, ts, quantity, step, extent):
    unit = 'particles/(cm^2 s)' if quantity == 'omni' else 'particles/(cm^2 s sr)'
    def fmt(v):
        return 'NaN' if not np.isfinite(v) else '{:.5g}'.format(v)
    with open(path, 'w') as f:
        f.write('# SWMF-VR 3D flux volume  ts={}  frame=GEI/EME2000 (Re, Earth-centered)\n'
                .format(ts))
        f.write('# grid: {} Re lattice in GSM rotated to GEI, 2.05 <= r <= {} Re;  '
                '{} energy-band-integrated flux, {}\n'.format(step, extent, quantity, unit))
        f.write('# NaN = no value (open field line or equatorial footpoint off the CIMI grid)\n')
        f.write('x_re,y_re,z_re,flux_lt1MeV,flux_1to2MeV,flux_gt2MeV\n')
        for (x, y, z), row in zip(pts, flux):
            f.write('{:.4f},{:.4f},{:.4f},{},{},{}\n'.format(
                x, y, z, fmt(row[0]), fmt(row[1]), fmt(row[2])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='+', help='volume_*_bands.npz file(s) or globs')
    p.add_argument('--out-dir',
                   default=os.path.join(REPO, 'run', 'volume_renders', 'csv_gei'))
    args = p.parse_args()

    paths = sorted(set(q for pat in args.npz for q in glob.glob(pat)))
    if not paths:
        print('no npz matched'); return
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:                       # one recalc per frame, never interleaved
        z = np.load(path)
        ts = str(z['ts'])
        frames.recalc_for(datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S'))
        R = frames.rotation_matrix(frames.gsm_to_gei)
        pts = np.asarray(z['points'], dtype=float) @ R.T
        out = os.path.join(args.out_dir, 'volume_{}_bands_gei.csv'.format(ts))
        write_csv(out, pts, z['flux'], ts, str(z['quantity']),
                  float(z['step']), float(z['extent']))
        print('[{}] -> {}'.format(ts, out))


if __name__ == '__main__':
    main()
