#!/usr/bin/env python3
"""Re-export the meridional flux slices as CSV from the saved npz cubes
(plain python3, no pvbatch, ~seconds per frame).

  python3 scripts/fp_slices_csv.py [RANK SIZE] [--ts TS] [--out-dir DIR]
        [--bands all|gt2|lt1,1to2,...] [--drop-zeros] [--split-azimuths]
        [--frame gei|sm|gsm] [--force]

fp_slices.py writes the npz cube (dense, all bands, all cuts) and a CSV in the
same run; this script rewrites the CSV in any layout afterwards, so a change
of layout for the VR team never needs the BATSRUS sampling again. Default
layout: one file per timestamp, all cuts stacked with `azimuth_deg` as the cut
column, all three flux bands, zeros written. Typical delivery for the VR app
(they plot >2 MeV only and consume XYZ point lists):

  python3 scripts/fp_slices_csv.py --bands gt2 --drop-zeros \
          --out-dir run/fp_slices/csv_gt2_sparse

RANK SIZE shard the frame list for a SLURM array (slurm/fp_slices_csv_array.sh).
"""

import os
import sys
import glob
import time
import datetime
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import frames
from swmf_vr.slices_csv import (sm_to_frame_matrix, write_slices_csv,
                                parse_bands, write_grid_json, gei_static_bbox)

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rank', type=int, nargs='?', default=0)
    p.add_argument('size', type=int, nargs='?', default=1)
    p.add_argument('--ts', default=None, help='one timestamp only')
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--out-dir', default=None,
                   help='CSV directory (default <out-root>/fp_slices/csv)')
    p.add_argument('--frame', choices=['gei', 'sm', 'gsm'], default=None,
                   help='output frame (default: the frame recorded in the npz, gei)')
    p.add_argument('--bands', default='all',
                   help='flux columns to write: all (default) or a comma list of lt1,1to2,gt2')
    p.add_argument('--drop-zeros', action='store_true',
                   help='omit rows where every written flux column is 0')
    p.add_argument('--split-azimuths', action='store_true',
                   help='one CSV per cut (slice_azNNN_<ts>.csv) instead of one per timestamp')
    p.add_argument('--integer-flux', action='store_true',
                   help='write flux as plain rounded integers (no 1.5e+05 exponent form; '
                        'the %%.5g default flips notation at exactly 1e5, which broke an importer)')
    p.add_argument('--force', action='store_true', help='rewrite existing CSVs')
    args = p.parse_args()
    bands = parse_bands(args.bands)

    npz_dir = os.path.join(args.out_root, 'fp_slices', 'npz')
    out_dir = args.out_dir or os.path.join(args.out_root, 'fp_slices', 'csv')
    os.makedirs(out_dir, exist_ok=True)
    if args.ts:
        npz_files = [os.path.join(npz_dir, 'slices_{}.npz'.format(args.ts))]
    else:
        npz_files = sorted(glob.glob(os.path.join(npz_dir, 'slices_*.npz')))
    npz_files = npz_files[args.rank::args.size]
    print('shard {}/{}: {} frames -> {}  bands={} drop_zeros={} split={} integer_flux={}'.format(
        args.rank, args.size, len(npz_files), out_dir, ','.join(bands),
        args.drop_zeros, args.split_azimuths, args.integer_flux))

    if npz_files:
        d0 = np.load(npz_files[0], allow_pickle=True)
        gj_frame = args.frame or str(d0['frame'])
        # tight static box over ALL frames (glob the whole dir, not just this
        # shard) so every shard writes an identical grid.json
        all_ts = [os.path.basename(f)[len('slices_'):-len('.npz')]
                  for f in sorted(glob.glob(os.path.join(npz_dir, 'slices_*.npz')))]
        bbox = gei_static_bbox(all_ts, float(np.asarray(d0['rho_re']).max()),
                               float(np.abs(np.asarray(d0['z_re'])).max()), gj_frame)
        write_grid_json(os.path.join(out_dir, 'grid.json'),
                        np.asarray(d0['azimuths_deg'], dtype=float),
                        np.asarray(d0['rho_re'], dtype=float),
                        np.asarray(d0['z_re'], dtype=float),
                        gj_frame, str(d0['quantity']),
                        bands, args.drop_zeros, str(d0['axis']), bbox)

    t0 = time.time(); n_done = 0
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        ts = str(d['ts']); quantity = str(d['quantity'])
        frame = args.frame or str(d['frame'])
        azimuths = np.asarray(d['azimuths_deg'], dtype=float)
        rho = np.asarray(d['rho_re'], dtype=float); z = np.asarray(d['z_re'], dtype=float)
        cube = d['cube']
        if args.split_azimuths:
            outs = [(os.path.join(out_dir, 'slice_az{:03d}_{}.csv'.format(int(round(a)), ts)),
                     azimuths[i:i + 1], cube[i:i + 1]) for i, a in enumerate(azimuths)]
        else:
            outs = [(os.path.join(out_dir, 'slices_{}.csv'.format(ts)), azimuths, cube)]
        if not args.force and all(os.path.exists(o[0]) for o in outs):
            continue
        frames.recalc_for(datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S'))
        R_out = sm_to_frame_matrix(frame)
        for path, az, cb in outs:
            n_kept, n_total = write_slices_csv(path, az, rho, z, cb, R_out, frame, ts,
                                               quantity, bands, args.drop_zeros,
                                               integer_flux=args.integer_flux)
        n_done += 1
        if n_done % 25 == 0 or len(npz_files) == 1:
            print('[{}] {} file(s), {} of {} rows kept, {:.1f} s/frame'.format(
                ts, len(outs), n_kept, n_total, (time.time() - t0) / n_done))
    print('shard {} done: {} frames written in {:.0f} s'.format(args.rank, n_done, time.time() - t0))


if __name__ == '__main__':
    main()
