#!/usr/bin/env python3
"""Transform field-line OBJ files from GSM to GEO or GEI (system python3).

Reproduces v5/transform_obj_coords.py. Only 'v x y z' lines change; vt / mtllib /
usemtl / l connectivity (the per-vertex color) pass through unchanged.

  python3 scripts/transform_obj.py --target gei --obj-dir <dir> --out-dir <dir>
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import coords
from swmf_vr.io.obj_io import transform_obj
from swmf_vr.timeaxis import parse_obj_timestamp


def main():
    parser = argparse.ArgumentParser(
        description='Transform OBJ field lines from GSM to GEO or GEI.')
    parser.add_argument('--target', choices=['geo', 'gei'], required=True)
    parser.add_argument('--obj-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--copy-assets', action='store_true',
                        help='Also copy ramp.png/material.mtl/colormap.json into out-dir')
    args = parser.parse_args()

    transform_fn = coords.transform_fn_for(args.target)
    obj_dir = os.path.abspath(args.obj_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    obj_files = sorted(glob.glob(os.path.join(obj_dir, 'fieldlines_*.obj')))
    if not obj_files:
        print('No fieldlines_*.obj files found in {}'.format(obj_dir))
        return
    print('Found {} OBJ files.'.format(len(obj_files)))
    print('Target frame: {}'.format(args.target.upper()))
    print('Output dir:   {}'.format(out_dir))

    for i, obj_path in enumerate(obj_files):
        dt = parse_obj_timestamp(obj_path)
        if dt is None:
            print('Skipping (bad filename): {}'.format(os.path.basename(obj_path)))
            continue
        coords.recalc_for(dt)
        out_path = os.path.join(out_dir, os.path.basename(obj_path))
        transform_obj(obj_path, out_path, transform_fn)
        if (i + 1) % 50 == 0 or i == 0:
            print('  [{}/{}] {}'.format(i + 1, len(obj_files), os.path.basename(obj_path)))

    if args.copy_assets:
        import shutil
        for name in ('ramp.png', 'material.mtl', 'colormap.json'):
            src = os.path.join(obj_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, name))
        print('Copied color assets into {}'.format(out_dir))

    print('Done. Transformed {} files to {}/'.format(len(obj_files), out_dir))


if __name__ == '__main__':
    main()
