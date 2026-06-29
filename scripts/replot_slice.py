#!/usr/bin/env python3
"""Re-render a saved y=0 slice .npz with a tunable despeckle (system python3).

The heavy trace (slice3d.py) saves raw, unfilled flux; this re-applies the
connected-component fill and replots without re-tracing.

  python3 scripts/replot_slice.py --npz run/slice_renders/slice_y0_..._bands.npz
       [--drop 1.5] [--floor 1e-10] [--out PNG]
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import sliceplot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--drop', type=float, default=1.5,
                   help='decades below the local median to count a pixel as bad')
    p.add_argument('--floor', type=float, default=1e-10,
                   help='absolute flux below this (garbage from bad traces) is replaced')
    p.add_argument('--indices', default=None, help='indices .npz (default: alongside the slice)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    flux = list(d['flux'])
    labels = [str(x) for x in d['labels']]
    ts, quantity = str(d['ts']), str(d['quantity'])
    slice_root = os.path.dirname(os.path.dirname(args.npz))   # .../slice_renders
    if args.out:
        out = args.out
    else:                                            # npz/foo.npz -> png/foo_<q>.png
        png_dir = os.path.join(slice_root, 'png')
        os.makedirs(png_dir, exist_ok=True)
        out = os.path.join(png_dir, os.path.basename(args.npz).replace(
            '.npz', '_{}.png'.format(quantity)))
    indices = args.indices or os.path.join(slice_root, 'indices.npz')
    sliceplot.plot_bands(d['xs'], d['zs'], flux, labels, ts, quantity, out,
                         floor=args.floor, drop=args.drop,
                         indices=indices if os.path.exists(indices) else None)
    print('wrote', out)


if __name__ == '__main__':
    main()
