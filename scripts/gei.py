#!/usr/bin/env python3
"""GSM field-line OBJs -> GEI OBJs (the frame the VR meshes ship in), and the
per-frame rotation matrices the VR side can use directly (system python3).

  python3 scripts/gei.py --obj-dir DIR --out-dir DIR [--transforms CSV]

Transforms every fieldlines_*.obj in --obj-dir into --out-dir (color preserved,
ramp assets copied). With --transforms, also writes R_geo / R_gei / dipole_gei
per timestamp to that CSV.
"""

import os
import sys
import glob
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import frames, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--obj-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--transforms', default=None, help='also write the matrix CSV here')
    p.add_argument('--no-copy-assets', action='store_true')
    args = p.parse_args()

    obj_dir = os.path.abspath(args.obj_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    objs = sorted(glob.glob(os.path.join(obj_dir, 'fieldlines_*.obj')))
    if not objs:
        print('no fieldlines_*.obj in {}'.format(obj_dir)); return

    geo = ['R_geo_{}{}'.format(r, c) for r in range(3) for c in range(3)]
    gei = ['R_gei_{}{}'.format(r, c) for r in range(3) for c in range(3)]
    header = ['timestamp', 'unix_epoch'] + geo + gei + ['dipole_gei_x', 'dipole_gei_y', 'dipole_gei_z']
    rows = []

    for obj in objs:
        dt = data.parse_obj_timestamp(obj)
        if dt is None:
            continue
        ut = frames.recalc_for(dt)
        data.transform_obj(obj, os.path.join(out_dir, os.path.basename(obj)), frames.gsm_to_gei)
        if args.transforms:
            R_geo = frames.rotation_matrix(frames.gsm_to_geo)
            R_gei = frames.rotation_matrix(frames.gsm_to_gei)
            row = [dt.strftime('%Y%m%d_%H%M%S'), '{:.1f}'.format(ut)]
            row += ['{:.10f}'.format(v) for v in R_geo.ravel()]
            row += ['{:.10f}'.format(v) for v in R_gei.ravel()]
            row += ['{:.10f}'.format(v) for v in frames.dipole_gei()]
            rows.append(row)

    if not args.no_copy_assets:
        for name in ('ramp.png', 'material.mtl', 'colormap.json'):
            src = os.path.join(obj_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, name))

    if args.transforms and rows:
        with open(args.transforms, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(row) + '\n')
        print('wrote {} matrix rows -> {}'.format(len(rows), args.transforms))
    print('GEI: {} objs -> {}'.format(len(objs), out_dir))


if __name__ == '__main__':
    main()
