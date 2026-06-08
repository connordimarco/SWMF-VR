#!/usr/bin/env python3
"""Thin a folder of frames down to half-hour resolution with symlinks.

  python3 scripts/resample.py --src DIR --dst DIR

Picks the frame nearest each HH:15 / HH:45 and symlinks it. Works for any
files named <stem>_YYYYMMDD_HHMMSS.<ext>.
"""

import os
import re
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from swmf_vr.data import resample_to_halfhour

PATTERN = re.compile(r'(.*_(\d{8}_\d{6})\.[A-Za-z0-9]+)$')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--pattern', default=None)
    args = p.parse_args()
    pat = re.compile(args.pattern) if args.pattern else PATTERN
    kept = resample_to_halfhour(os.path.abspath(args.src), os.path.abspath(args.dst), pat)
    if kept:
        print('{} symlinks ({} .. {})'.format(len(kept), kept[0], kept[-1]))


if __name__ == '__main__':
    main()
