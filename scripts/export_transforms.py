#!/usr/bin/env python3
"""Export per-timestep GSM->GEO/GEI rotation matrices + dipole axis (system python3).

Reproduces v5/export_gsm_transforms.py. For each fieldlines_*.obj timestamp in an
OBJ dir, write the row-major R_geo / R_gei matrices and the GEI dipole axis.

  python3 scripts/export_transforms.py --obj-dir <dir> --output gsm_transforms.csv
"""

import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import coords
from swmf_vr.timeaxis import parse_obj_timestamp, dt_to_unix


def main():
    parser = argparse.ArgumentParser(
        description='Export GSM->GEO and GSM->GEI rotation matrices per timestep.')
    parser.add_argument('--obj-dir', required=True,
                        help='Directory containing fieldlines_*.obj files')
    parser.add_argument('--output', default='gsm_transforms.csv')
    args = parser.parse_args()

    obj_dir = os.path.abspath(args.obj_dir)
    obj_files = sorted(glob.glob(os.path.join(obj_dir, 'fieldlines_*.obj')))
    if not obj_files:
        print('No fieldlines_*.obj files found in {}'.format(obj_dir))
        return
    print('Found {} OBJ files in {}'.format(len(obj_files), obj_dir))

    geo_cols = ['R_geo_{}{}'.format(r, c) for r in range(3) for c in range(3)]
    gei_cols = ['R_gei_{}{}'.format(r, c) for r in range(3) for c in range(3)]
    dipole_cols = ['dipole_gei_x', 'dipole_gei_y', 'dipole_gei_z']
    header = ['timestamp', 'unix_epoch'] + geo_cols + gei_cols + dipole_cols

    rows = []
    for obj_path in obj_files:
        dt = parse_obj_timestamp(obj_path)
        if dt is None:
            print('Skipping (bad filename): {}'.format(os.path.basename(obj_path)))
            continue
        ut = dt_to_unix(dt)
        coords.recalc_for(dt)
        R_geo = coords.rotation_matrix(coords.gsm_to_geo)
        R_gei = coords.rotation_matrix(coords.gsm_to_gei)
        dip = coords.dipole_gei()

        for name, R in [('GEO', R_geo), ('GEI', R_gei)]:
            err = np.max(np.abs(R @ R.T - np.eye(3)))
            if err > 1e-6:
                print('WARNING: {} {} matrix not orthogonal (max err={:.2e})'.format(
                    dt, name, err))

        row = [dt.strftime('%Y%m%d_%H%M%S'), '{:.1f}'.format(ut)]
        row += ['{:.10f}'.format(v) for v in R_geo.ravel()]
        row += ['{:.10f}'.format(v) for v in R_gei.ravel()]
        row += ['{:.10f}'.format(v) for v in dip]
        rows.append(row)
        if len(rows) % 100 == 0:
            print('  Processed {}/{}...'.format(len(rows), len(obj_files)))

    with open(args.output, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')
    print('Done. Wrote {} rows to {}'.format(len(rows), args.output))


if __name__ == '__main__':
    main()
