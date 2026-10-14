#!/usr/bin/env python3
"""CIMI electron flux (.fls) -> the RBE-style npz the foot-point pipeline reads.

The SWMF-VR pipeline was written against Austin's RBE npz (latN mltN ro mlto bo
irm iba flux time + lat_lvls mlt_lvls E_lvls alpha_lvls). CIMI's `fls e` output
(IM/plots/<date>_e.fls, one file per time with DoSaveSeparateFiles) has the same
physics content on a different grid (75 lat x 48 MLT x 20 E x 18 pitch angles
here), so this script just re-grids the layout:

  header     rc nr ip je ig ntime
             je energies [keV]            (6f14.3)
             ig pitch-angle sines         (6f9.5)
             nr foot-point latitudes [deg] (10f8.3)
  per time   hour L*max parmod(10)
             per lat (outer loop) x MLT: `lat mlt ro mlto bo L*`
                                        then je lines of ig fluxes

Rows beyond a sector's last closed line (irm) are frozen copies of the irm row
in the file (lat repeats), exactly as in RBE; irm is recovered as the count of
real rows and stored the RBE way (constant along the latitude axis). The MLT
axis is rolled so mlt_lvls runs 0 .. 23.5 (CIMI writes it starting at noon),
which RbeFootpointGrid's periodic lookup assumes. `time` is written as float
minutes after --base-iso (what data.load_grids / frame_datetime expect), and
flux as float32 (frames stream through data.iter_flux_frames, which reads the
dtype from the npy header).

Usage (frames with a matching 3D .plt only, the 10-min cadence of the run):
  python3 scripts/cimi_fls_to_npz.py --fls-dir RUN/IM/plots --plt-dir RUN/GM/IO2 \
      --base-iso 2010-04-05T04:00:00 --out RUN/galaxy2010_e_fls.npz
"""
import argparse
import datetime
import glob
import os
import re
import sys

import numpy as np

PLACEHOLDER = 1.0e-50   # CIMI writes this for "no particles"


def parse_fls(path):
    """One CIMI .fls file -> dict of arrays (first time block only)."""
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    head = lines[0].split('!')[0].split()
    rc = float(head[0])
    nr, ip, je, ig = (int(v) for v in head[1:5])
    pos = 1

    def take(n_vals):
        nonlocal pos
        vals = []
        while len(vals) < n_vals:
            vals.extend(float(v) for v in lines[pos].split())
            pos += 1
        return np.array(vals[:n_vals], dtype=float)

    energy = take(je)
    sin_a = take(ig)
    xlat = take(nr)

    # time block
    hdr = lines[pos].split('!')[0].split()
    pos += 1
    hour = float(hdr[0])
    lstar_max = float(hdr[1])

    n_flux_lines = (ig + 11) // 12          # 12 values per line
    lat = np.empty((nr, ip)); mlt = np.empty((nr, ip)); ro = np.empty((nr, ip))
    mlto = np.empty((nr, ip)); bo = np.empty((nr, ip)); lstar = np.empty((nr, ip))
    flux_txt = []
    for i in range(nr):
        for j in range(ip):
            p = lines[pos].split()
            lat[i, j], mlt[i, j], ro[i, j], mlto[i, j], bo[i, j], lstar[i, j] = \
                (float(v) for v in p[:6])
            pos += 1
            flux_txt.extend(lines[pos:pos + je * n_flux_lines])
            pos += je * n_flux_lines
    flux = np.array(' '.join(flux_txt).split(), dtype=np.float32)
    flux = flux.reshape(nr, ip, je, ig)
    return {'rc': rc, 'hour': hour, 'lstar_max': lstar_max, 'energy': energy,
            'sin_a': sin_a, 'xlat': xlat, 'lat': lat, 'mlt': mlt, 'ro': ro,
            'mlto': mlto, 'bo': bo, 'lstar': lstar, 'flux': flux}


def irm_counts(lat_grid):
    """Number of real rows per MLT: the first row whose latitude repeats the
    previous one starts the frozen zone (CIMI copies the irm row upward)."""
    nr, ip = lat_grid.shape
    out = np.full(ip, nr, dtype=int)
    for j in range(ip):
        rep = np.where(np.diff(lat_grid[:, j]) == 0.0)[0]
        if rep.size:
            out[j] = int(rep[0]) + 1
    return out


def fls_timestamp(path):
    m = re.search(r'(\d{8})_(\d{6})_e\.fls$', os.path.basename(path))
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')


def plt_timestamps(plt_dir):
    out = set()
    for p in glob.glob(os.path.join(plt_dir, '*.plt')):
        m = re.search(r'e(\d{8})-(\d{6})', os.path.basename(p))
        if m and os.path.getsize(p) > 0:
            out.add(datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S'))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--fls-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--base-iso', required=True,
                   help='frame-0 reference, e.g. 2010-04-05T04:00:00 (npz time = minutes after)')
    p.add_argument('--plt-dir', default=None,
                   help='keep only times that have a non-empty 3D .plt here')
    p.add_argument('--every', type=int, default=1, help='keep every N-th remaining file')
    p.add_argument('--max-frames', type=int, default=None)
    args = p.parse_args()

    base = datetime.datetime.strptime(args.base_iso, '%Y-%m-%dT%H:%M:%S')
    files = sorted(glob.glob(os.path.join(args.fls_dir, '*_e.fls')))
    stamped = [(fls_timestamp(f), f) for f in files]
    stamped = [(t, f) for t, f in stamped if t is not None]
    if args.plt_dir:
        keep = plt_timestamps(args.plt_dir)
        stamped = [(t, f) for t, f in stamped if t in keep]
        print('{} fls files match a .plt in {}'.format(len(stamped), args.plt_dir))
    stamped = stamped[::args.every]
    if args.max_frames:
        stamped = stamped[:args.max_frames]
    if not stamped:
        sys.exit('no .fls files selected')

    first = parse_fls(stamped[0][1])
    nr, ip, je, ig = first['flux'].shape
    # roll the MLT axis so mlt_lvls is 0 .. 24-step ascending
    mlt_axis = first['mlt'][0, :]
    roll = -int(np.argmin(mlt_axis))
    mlt_lvls = np.roll(mlt_axis, roll)
    assert np.all(np.diff(mlt_lvls) > 0), 'MLT axis not uniform after roll'
    lat_lvls = first['xlat']
    T = len(stamped)
    print('{} frames; grid lat {} x mlt {} x E {} x alpha {}; E = {:.1f}..{:.0f} keV'.format(
        T, nr, ip, je, ig, first['energy'][0], first['energy'][-1]))

    shape2 = (T, nr, ip)
    latN = np.empty(shape2); mltN = np.empty(shape2); ro = np.empty(shape2)
    mlto = np.empty(shape2); bo = np.empty(shape2); irm = np.empty(shape2)
    lstar = np.empty(shape2)
    flux = np.empty((T, nr, ip, je, ig), dtype=np.float32)
    time_min = np.empty(T)

    for k, (t, f) in enumerate(stamped):
        d = first if k == 0 else parse_fls(f)
        assert d['flux'].shape == (nr, ip, je, ig), f
        counts = np.roll(irm_counts(d['lat']), roll)
        latN[k] = np.roll(d['lat'], roll, axis=1)
        mltN[k] = np.roll(d['mlt'], roll, axis=1)
        ro[k] = np.roll(d['ro'], roll, axis=1)
        mlto[k] = np.roll(d['mlto'], roll, axis=1)
        bo[k] = np.roll(d['bo'], roll, axis=1)
        lstar[k] = np.roll(d['lstar'], roll, axis=1)
        irm[k] = counts[None, :]
        flux[k] = np.roll(d['flux'], roll, axis=1)
        time_min[k] = (t - base).total_seconds() / 60.0
        if (k + 1) % 25 == 0 or k == T - 1:
            print('  {}/{} {}  irm range {}..{}'.format(
                k + 1, T, t.strftime('%Y%m%d_%H%M%S'), counts.min(), counts.max()))
            sys.stdout.flush()

    flux[flux <= PLACEHOLDER] = 0.0
    np.savez(args.out,
             latN=latN, mltN=mltN, ro=ro, mlto=mlto, bo=bo, irm=irm, iba=irm.copy(),
             lstar=lstar, times=np.array([float(T)]), lat_lvls=lat_lvls,
             mlt_lvls=mlt_lvls, E_lvls=first['energy'], alpha_lvls=first['sin_a'],
             flux=flux, time=time_min, base_iso=np.array(args.base_iso),
             source=np.array('CIMI fls e, ' + os.path.abspath(args.fls_dir)))
    print('wrote {} ({:.2f} GB)'.format(args.out, os.path.getsize(args.out) / 1e9))


if __name__ == '__main__':
    main()
