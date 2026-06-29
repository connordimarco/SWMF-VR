#!/usr/bin/env python3
"""Build the Bz / AL / Dst context strips from the SWMF run logs (system python3).

Writes run/slice_renders/indices.npz, which slice3d.py / replot_slice.py drop in
under the flux maps. Run once before the slice run.

  python3 scripts/parse_indices.py [--run-dir DIR] [--out-root DIR]
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr import indices, config

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=os.environ.get(
        'SWMF_RUN_DIR', '/nfs/turbo/coe-tuija/shared/run_mothersday_ne'))
    p.add_argument('--out-root', default=os.path.join(REPO, 'run'))
    p.add_argument('--frames', type=int, default=1683, help='total frames (sets the x-span)')
    args = p.parse_args()

    cfg = config.CRAConfig()
    gm = os.path.join(args.run_dir, 'GM', 'IO2')
    gm_logs = sorted(glob.glob(os.path.join(gm, 'log_e*.log')))
    geo_logs = sorted(glob.glob(os.path.join(gm, 'geoindex_e*.log')))
    imf = os.path.join(args.run_dir, 'mothersday2_IMF.dat')
    out_npz = os.path.join(args.out_root, 'slice_renders', 'indices.npz')
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    print('gm logs :', [os.path.basename(p) for p in gm_logs])
    print('geo logs:', [os.path.basename(p) for p in geo_logs])
    res = indices.build(out_npz, cfg.time.base_iso, (args.frames - 1) / 60.0,
                        gm_logs, geo_logs, imf)
    for k, name in (('bz', 'Bz'), ('al', 'AL'), ('dst', 'Dst')):
        t, v = res[k + '_t'], res[k + '_v']
        span = '{:.1f}..{:.1f} h'.format(t.min(), t.max()) if len(t) else 'EMPTY'
        rng = '[{:.1f}, {:.1f}] nT'.format(v.min(), v.max()) if len(v) else ''
        print('{:4s}: {:5d} pts  {}  {}'.format(name, len(t), span, rng))
    print('wrote', out_npz)


if __name__ == '__main__':
    main()
