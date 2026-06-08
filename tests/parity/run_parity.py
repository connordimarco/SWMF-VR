#!/usr/bin/env python3
"""Parity harness: prove the swmf_vr pure layer reproduces v5/v6 byte-for-byte.

Run under system python3 with PYTHONPATH=../../src (see slurm/env.sh):
    python3 run_parity.py            # all pure-layer steps (1-3)
Steps 4-5 (OBJ shading) need pvbatch and live in run_parity_pvbatch.py.

Compares against the IN-PLACE v5/v6 outputs (the oracle). Exits nonzero on any
mismatch.
"""

import os
import sys
import glob
import datetime

import numpy as np

import swmf_vr.config as config
from swmf_vr.io import npz as npzio
from swmf_vr import boundary, seeding, coords
from swmf_vr.timeaxis import base_datetime, timestamp_string, dt_to_unix
from geopack import geopack

HERE = os.path.dirname(os.path.abspath(__file__))
CRA = os.path.normpath(os.path.join(HERE, '..', '..', '..'))   # .../CRA
# Oracle = the archived v5/v6 outputs. Falls back to in-place v5/v6 if the
# archive move hasn't happened yet, so this harness works before and after.
V6 = os.path.join(CRA, 'archive', 'v6') if os.path.isdir(os.path.join(CRA, 'archive', 'v6')) else os.path.join(CRA, 'v6')
V5 = os.path.join(CRA, 'archive', 'v5') if os.path.isdir(os.path.join(CRA, 'archive', 'v5')) else os.path.join(CRA, 'v5')
SCRATCH = os.path.join(HERE, 'scratch')

PARITY_STAMPS = ['20240510_130000', '20240510_210000']

_fails = []


def check(name, ok, detail=''):
    mark = 'PASS' if ok else 'FAIL'
    print('  [{}] {}{}'.format(mark, name, ('  -- ' + detail) if detail else ''))
    if not ok:
        _fails.append(name)


def cfg_and_grids():
    cfg = config.default_config()
    grids = npzio.load_grids(cfg.paths.npz_path)
    e_idxs, a_idx = npzio.select_energy_alpha(
        grids['E_lvls'], grids['alpha_lvls'],
        cfg.flux.energy_min_kev, cfg.flux.target_alpha_val)
    return cfg, grids, e_idxs, a_idx


def step1(cfg, grids, e_idxs, a_idx):
    print('Step 1 -- data layer + timeaxis')
    base_dt = base_datetime(cfg.time.base_iso)
    # timestamp_string matches existing flux_slices filenames.
    slice_files = glob.glob(os.path.join(V6, 'flux_slices', 'flux_slice_*.png'))
    existing = set(os.path.basename(f)[len('flux_slice_'):-len('.png')]
                   for f in slice_files)
    want = set(PARITY_STAMPS)
    derived = set()
    # Cross-check: streaming index matches a full np.load for a few frames, and
    # timestamp_string reproduces the parity stamps.
    target_idx = {}
    for t_idx, tv in enumerate(grids['time']):
        ts = timestamp_string(tv, base_dt)
        if ts in want:
            target_idx[ts] = t_idx
            derived.add(ts)
    check('timestamp_string reproduces parity stamps',
          want.issubset(derived), 'idx=' + str(target_idx))
    if existing:
        check('parity stamps present in v6/flux_slices', want.issubset(existing))
    # Streaming reader sanity for the parity frames -- expected shape + finite
    # flux. (We deliberately do NOT np.load the full multi-GB flux array; the
    # byte-exact seed-CSV diff in Step 2 is the real proof that frames stream
    # correctly, and a full load OOMs on the login node.)
    ok_shape = True
    seen = 0
    for t_idx, frame in npzio.iter_flux_frames(cfg.paths.npz_path):
        if t_idx in target_idx.values():
            fs = npzio.extract_integrated_flux(
                frame, e_idxs, a_idx, cfg.flux.lat_dim, cfg.flux.mlt_dim)
            if fs.shape != (cfg.flux.lat_dim, cfg.flux.mlt_dim) or not np.isfinite(fs).all():
                ok_shape = False
            seen += 1
        if seen == len(target_idx):
            break
    check('iter_flux_frames yields finite (lat,mlt) flux slices', ok_shape)
    return target_idx


def regen_seeds(cfg, grids, e_idxs, a_idx, ts, t_idx, frame):
    """Reproduce v6 Stage A seed generation for one timestep into SCRATCH."""
    base_dt = base_datetime(cfg.time.base_iso)
    ro_vals = grids['ro']
    mlt_centers = grids['mlto'][0, :]
    flux_slice = npzio.extract_integrated_flux(
        frame, e_idxs, a_idx, cfg.flux.lat_dim, cfg.flux.mlt_dim)

    r_arrs = {}
    for s in cfg.surfaces:
        if s.mode == 'max':
            r_arrs[s.name] = boundary.find_max_boundary(flux_slice, ro_vals)
        else:
            r_arrs[s.name] = boundary.find_boundary(
                flux_slice, ro_vals, s.thresholds[0], s.search)

    cur_dt = base_dt + datetime.timedelta(minutes=float(grids['time'][t_idx]))
    geopack.recalc(dt_to_unix(cur_dt))
    peak_arr = boundary.find_peak_flux(flux_slice, ro_vals)

    out_paths = {}
    for s in cfg.surfaces:
        d = os.path.join(SCRATCH, s.seed_subdir)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'boundary_{}.csv'.format(ts))
        flux_label = peak_arr if s.mode == 'max' else float(s.thresholds[0])
        seeding.write_seed_csv(r_arrs[s.name], mlt_centers, flux_label, p,
                               flux_peak=peak_arr)
        out_paths[s.name] = p
    return out_paths


def step2(cfg, grids, e_idxs, a_idx, target_idx):
    print('Step 2 -- boundary + seeding (text-diff vs v6/out_seeds_*)')
    want_idx = set(target_idx.values())
    regen = {}
    for t_idx, frame in npzio.iter_flux_frames(cfg.paths.npz_path):
        if t_idx in want_idx:
            ts = [k for k, v in target_idx.items() if v == t_idx][0]
            regen[ts] = regen_seeds(cfg, grids, e_idxs, a_idx, ts, t_idx, frame)
        if t_idx >= max(want_idx):
            break
    for ts in PARITY_STAMPS:
        if ts not in regen:
            continue
        for s in cfg.surfaces:
            ref = os.path.join(V6, s.seed_subdir, 'boundary_{}.csv'.format(ts))
            new = regen[ts][s.name]
            if not os.path.exists(ref):
                check('seed {} {}'.format(s.name, ts), False, 'no oracle ' + ref)
                continue
            with open(ref) as f:
                a = f.read()
            with open(new) as f:
                b = f.read()
            check('seed {} {} byte-identical'.format(s.name, ts), a == b,
                  '' if a == b else 'differs')


def step3(cfg):
    print('Step 3 -- coords (recompute vs v5/gsm_transforms.csv rows)')
    ref_csv = os.path.join(V5, 'gsm_transforms.csv')
    if not os.path.exists(ref_csv):
        check('v5/gsm_transforms.csv present', False)
        return
    with open(ref_csv) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    header = lines[0].split(',')
    rows = [ln.split(',') for ln in lines[1:]]
    sample = rows[:3] + rows[-1:]
    for row in sample:
        rec = dict(zip(header, row))
        ts = rec['timestamp']
        dt = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S')
        coords.recalc_for(dt)
        R_geo = coords.rotation_matrix(coords.gsm_to_geo)
        R_gei = coords.rotation_matrix(coords.gsm_to_gei)
        dip = coords.dipole_gei()
        got = ['{:.10f}'.format(v) for v in R_geo.ravel()]
        got += ['{:.10f}'.format(v) for v in R_gei.ravel()]
        got += ['{:.10f}'.format(v) for v in dip]
        want = ([rec['R_geo_{}{}'.format(r, c)] for r in range(3) for c in range(3)]
                + [rec['R_gei_{}{}'.format(r, c)] for r in range(3) for c in range(3)]
                + [rec['dipole_gei_x'], rec['dipole_gei_y'], rec['dipole_gei_z']])
        check('transform row {} matches'.format(ts), got == want)


def main():
    cfg, grids, e_idxs, a_idx = cfg_and_grids()
    target_idx = step1(cfg, grids, e_idxs, a_idx)
    step2(cfg, grids, e_idxs, a_idx, target_idx)
    step3(cfg)
    print()
    if _fails:
        print('PARITY FAILED: {} check(s) -- {}'.format(len(_fails), ', '.join(_fails)))
        sys.exit(1)
    print('ALL PARITY CHECKS PASSED')


if __name__ == '__main__':
    main()
