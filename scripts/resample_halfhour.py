#!/usr/bin/env python3
"""Resample an output directory to half-hour resolution via symlinks (system python3).

Reproduces v4/downsample_obj_renders.py for one src/dst pair. Picks the frame
nearest each half-hour center (HH:15 / HH:45) and symlinks it. Works for any
files named like <stem>_YYYYMMDD_HHMMSS.<ext> (obj or png).

  python3 scripts/resample_halfhour.py --src obj_outer --dst obj_outer_halfhour
"""

import os
import re
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr.timeaxis import resample_to_halfhour

# group(1) = full filename, group(2) = YYYYMMDD_HHMMSS stamp. Matches v4's three
# job patterns (fieldlines_*.obj, fieldlines_*.png, flux_boundary_*_*.png).
DEFAULT_PATTERN = re.compile(r'(.*_(\d{8}_\d{6})\.[A-Za-z0-9]+)$')


def main():
    parser = argparse.ArgumentParser(
        description='Half-hour-resolution symlink resampling.')
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--pattern', default=None,
                        help='Override regex; group(2) must be the YYYYMMDD_HHMMSS stamp')
    args = parser.parse_args()

    pattern = re.compile(args.pattern) if args.pattern else DEFAULT_PATTERN
    print('Resampling {} -> {}'.format(args.src, args.dst))
    selected = resample_to_halfhour(os.path.abspath(args.src),
                                    os.path.abspath(args.dst), pattern)
    if selected:
        print('  Created {} symlinks. First: {}  Last: {}'.format(
            len(selected), selected[0], selected[-1]))


if __name__ == '__main__':
    main()
